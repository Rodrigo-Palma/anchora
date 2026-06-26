"""Evaluate base vs. LoRA-tuned generation on the anchora golden set.

This script is intentionally separate from ``anchora.evals``:

* ``anchora.evals`` is deterministic and cheap enough for CI.
* this script loads a local Hugging Face model and optionally a PEFT adapter,
  so it is a real comparative experiment for v0.3 evidence.

No paid APIs are used. The default model is small enough for a local smoke run;
for a stronger public benchmark, pass a larger local model when hardware allows.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from anchora import metrics
from anchora.evals import load_cases
from anchora.guardrails import validate_output
from anchora.ingest import ingest_dir
from anchora.llm import _PROMPT, build_context
from anchora.rag import retrieve

_ROOT = Path(__file__).resolve().parents[1]
_CORPUS_DIR = _ROOT / "data" / "corpus"
_DEFAULT_OUT = _ROOT / "artifacts" / "finetune-comparison.json"
_PROVIDER = "hash"


@dataclass
class GenerationScore:
    case_id: str
    expected_doc: str
    answer: str
    grounded: bool
    faithfulness: float
    answer_relevance: float


@dataclass
class GenerationReport:
    name: str
    base_model: str
    adapter_path: str | None
    scores: list[GenerationScore]

    @property
    def grounded_rate(self) -> float:
        return _mean(1.0 if score.grounded else 0.0 for score in self.scores)

    @property
    def mean_faithfulness(self) -> float:
        return _mean(score.faithfulness for score in self.scores)

    @property
    def mean_answer_relevance(self) -> float:
        return _mean(score.answer_relevance for score in self.scores)

    def summary(self) -> dict[str, float | str | None]:
        return {
            "name": self.name,
            "base_model": self.base_model,
            "adapter_path": self.adapter_path,
            "grounded_rate": self.grounded_rate,
            "faithfulness": self.mean_faithfulness,
            "answer_relevance": self.mean_answer_relevance,
        }


def evaluate(
    name: str,
    base_model: str,
    adapter_path: Path | None = None,
    *,
    max_new_tokens: int = 96,
    limit: int | None = None,
) -> GenerationReport:
    """Generate answers for the golden set and score them against retrieved context."""
    tokenizer, model = _load_model(base_model, adapter_path)
    store = ingest_dir(_CORPUS_DIR, provider=_PROVIDER)
    scores: list[GenerationScore] = []
    cases = load_cases()
    if limit is not None:
        cases = cases[:limit]

    for case in cases:
        chunks = retrieve(store, case["question"], k=4, provider=_PROVIDER)
        prompt = _PROMPT.format(context=build_context(chunks), question=case["question"])
        answer = _generate(tokenizer, model, prompt, max_new_tokens=max_new_tokens)
        context = build_context(chunks)
        grounded = validate_output(answer).ok
        scores.append(
            GenerationScore(
                case_id=case["id"],
                expected_doc=case["expected_doc"],
                answer=answer,
                grounded=grounded,
                faithfulness=metrics.faithfulness(answer, context),
                answer_relevance=metrics.answer_relevance(answer, case["question"]),
            )
        )
    return GenerationReport(
        name=name,
        base_model=base_model,
        adapter_path=_as_str(adapter_path),
        scores=scores,
    )


def _load_model(base_model: str, adapter_path: Path | None) -> tuple[Any, Any]:
    """Load a local Transformers model, optionally with a PEFT adapter."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if torch.backends.mps.is_available() else torch.float32
    model: Any = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype)
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, adapter_path)
    if torch.backends.mps.is_available():
        model = model.to("mps")
    model.eval()
    return tokenizer, model


def _generate(tokenizer: Any, model: Any, prompt: str, *, max_new_tokens: int) -> str:
    """Generate a short answer and strip the prompt from the decoded output."""
    import torch

    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = output_ids[0][inputs["input_ids"].shape[-1] :]
    return str(tokenizer.decode(generated, skip_special_tokens=True)).strip()


def write_report(reports: list[GenerationReport], out: Path) -> None:
    """Persist full per-case outputs plus a compact summary table."""
    out.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "summary": [report.summary() for report in reports],
        "reports": [
            {
                "name": report.name,
                "base_model": report.base_model,
                "adapter_path": report.adapter_path,
                "scores": [asdict(score) for score in report.scores],
            }
            for report in reports
        ],
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return round(sum(items) / len(items), 4) if items else 0.0


def _as_str(path: Path | None) -> str | None:
    return str(path) if path is not None else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    reports = [
        evaluate(
            "base",
            args.base,
            max_new_tokens=args.max_new_tokens,
            limit=args.limit,
        )
    ]
    if args.adapter is not None:
        reports.append(
            evaluate(
                "lora",
                args.base,
                args.adapter,
                max_new_tokens=args.max_new_tokens,
                limit=args.limit,
            )
        )
    write_report(reports, args.out)
    for report in reports:
        print(report.summary())
    print(f"Wrote comparison to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
