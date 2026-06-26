"""LoRA fine-tune of a base LLM on the anchora instruction dataset (PEFT).

This is the v0.3 training entrypoint. The heavy ML stack (torch, transformers,
peft, datasets, accelerate) lives behind the ``finetune`` optional-dependency
group and is imported lazily, so the core package, the test suite and the API
stay light and 100% offline. Install it only when you actually train::

    uv sync --extra finetune
    uv run python scripts/finetune_lora.py \\
        --base Qwen/Qwen2.5-3B-Instruct \\
        --data data/finetune/instructions.jsonl \\
        --out artifacts/lora-anchora

Defaults target a small base model so the LoRA pass fits on a single consumer
GPU (or Apple Silicon via MPS). No paid API is involved — training is local.

After training, register the adapter and its eval metrics with
``scripts/compare_evals.py`` so the model registry knows about it.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class TrainConfig:
    base_model: str
    data_path: Path
    out_dir: Path
    epochs: int = 3
    batch_size: int = 1
    grad_accum: int = 8
    learning_rate: float = 2e-4
    max_seq_len: int = 1024
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    validation_ratio: float = 0.2
    early_stopping_patience: int = 4
    early_stopping_threshold: float = 0.0
    seed: int = 42


def _tokenize_completion(
    record: dict[str, str], tokenizer: Any, max_seq_len: int
) -> dict[str, list[int]]:
    """Tokenize prompt + answer while masking the prompt from the loss."""
    prompt = record["instruction"]
    completion = f" {record['output']}{tokenizer.eos_token or ''}"
    completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
    if len(completion_ids) >= max_seq_len:
        completion_ids = completion_ids[:max_seq_len]
        prompt_ids: list[int] = []
    else:
        max_prompt_len = max_seq_len - len(completion_ids)
        previous_truncation_side = tokenizer.truncation_side
        tokenizer.truncation_side = "left"
        try:
            prompt_ids = tokenizer(
                prompt,
                add_special_tokens=True,
                truncation=True,
                max_length=max_prompt_len,
            )["input_ids"]
        finally:
            tokenizer.truncation_side = previous_truncation_side
    input_ids = (prompt_ids + completion_ids)[:max_seq_len]
    prompt_len = min(len(prompt_ids), len(input_ids))
    labels = [-100] * prompt_len + input_ids[prompt_len:]
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


class CompletionCollator:
    """Pad completion-only causal-LM records, keeping prompt labels masked."""

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        max_len = max(len(feature["input_ids"]) for feature in features)
        pad_id = self.tokenizer.pad_token_id
        batch: dict[str, list[list[int]]] = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            pad = max_len - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [pad_id] * pad)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * pad)
            batch["labels"].append(feature["labels"] + [-100] * pad)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def train(cfg: TrainConfig) -> Path:
    """Run the LoRA fine-tune. Imports the heavy stack lazily on first call."""
    # Lazy, local imports: keep the dependency out of the core package.
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        EarlyStoppingCallback,
        Trainer,
        TrainerCallback,
        TrainingArguments,
        set_seed,
    )

    set_seed(cfg.seed)
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Training on device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model: Any = AutoModelForCausalLM.from_pretrained(cfg.base_model)
    lora = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    dataset = load_dataset("json", data_files=str(cfg.data_path), split="train")

    tokenized = dataset.map(
        lambda rec: _tokenize_completion(rec, tokenizer, cfg.max_seq_len),
        remove_columns=dataset.column_names,
    )
    if cfg.validation_ratio > 0:
        split = tokenized.train_test_split(test_size=cfg.validation_ratio, seed=cfg.seed)
        train_dataset = split["train"]
        eval_dataset = split["test"]
        callbacks: list[TrainerCallback] = [
            EarlyStoppingCallback(
                early_stopping_patience=cfg.early_stopping_patience,
                early_stopping_threshold=cfg.early_stopping_threshold,
            )
        ]
        eval_strategy = "epoch"
        load_best_model_at_end = True
    else:
        train_dataset = tokenized
        eval_dataset = None
        callbacks = []
        eval_strategy = "no"
        load_best_model_at_end = False

    args = TrainingArguments(
        output_dir=str(cfg.out_dir),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        logging_steps=1,
        eval_strategy=eval_strategy,
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=[],
        seed=cfg.seed,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=CompletionCollator(tokenizer),
        callbacks=callbacks,
    )
    trainer.train()

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(cfg.out_dir)
    tokenizer.save_pretrained(cfg.out_dir)
    print(f"Saved LoRA adapter to {cfg.out_dir}")
    return cfg.out_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--data", type=Path, default=Path("data/finetune/instructions.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/lora-anchora"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--early-stopping-threshold", type=float, default=0.0)
    args = parser.parse_args(argv)

    cfg = TrainConfig(
        base_model=args.base,
        data_path=args.data,
        out_dir=args.out,
        epochs=args.epochs,
        learning_rate=args.lr,
        validation_ratio=args.validation_ratio,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_threshold=args.early_stopping_threshold,
    )
    train(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
