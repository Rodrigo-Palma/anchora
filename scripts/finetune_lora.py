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
    seed: int = 42


def _format_example(example: dict[str, str]) -> str:
    """Render one record into a single training string (instruction + answer)."""
    instruction = example["instruction"]
    output = example["output"]
    return f"{instruction} {output}"


def train(cfg: TrainConfig) -> Path:
    """Run the LoRA fine-tune. Imports the heavy stack lazily on first call."""
    # Lazy, local imports: keep the dependency out of the core package.
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
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

    model = AutoModelForCausalLM.from_pretrained(cfg.base_model)
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

    # Map row-wise so _format_example sees full records.
    tokenized = dataset.map(
        lambda rec: tokenizer(_format_example(rec), truncation=True, max_length=cfg.max_seq_len),
        remove_columns=dataset.column_names,
    )

    args = TrainingArguments(
        output_dir=str(cfg.out_dir),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        logging_steps=1,
        save_strategy="epoch",
        report_to=[],
        seed=cfg.seed,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
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
    args = parser.parse_args(argv)

    cfg = TrainConfig(
        base_model=args.base,
        data_path=args.data,
        out_dir=args.out,
        epochs=args.epochs,
        learning_rate=args.lr,
    )
    train(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
