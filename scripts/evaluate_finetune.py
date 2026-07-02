"""Evaluate base vs. few-shot base vs. LoRA-tuned generation on anchora cases.

This script is intentionally separate from ``anchora.evals``:

* ``anchora.evals`` is deterministic and cheap enough for CI.
* this script loads a local Hugging Face model and optionally a PEFT adapter,
  so it is a real comparative experiment for v0.3 evidence.

Two methodology fixes over the first round (see ``docs/finetuning-results.md``):

* ``--golden data/golden/holdout.json`` evaluates on **held-out** questions the
  adapter never trained on, so the numbers measure generalization, not recall of
  the 24 training answers.
* ``--few-shot`` adds a fair third baseline: the *base* model prompted with a few
  worked examples in the same ``PT + [n]`` output contract. This isolates the
  "did fine-tuning teach knowledge, or just the answer format?" variable — the
  exemplars are drawn only from the training golden set, never from the holdout.

Cases may be marked ``answerable: false`` (out-of-corpus). For those the correct
behavior is to abstain with the exact refusal sentence, so they are scored with a
separate abstention metric instead of the lexical answer proxies.

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
from anchora.guardrails import is_abstention, validate_output
from anchora.ingest import ingest_dir
from anchora.llm import _PROMPT, build_context
from anchora.rag import retrieve

_ROOT = Path(__file__).resolve().parents[1]
_CORPUS_DIR = _ROOT / "data" / "corpus"
_GOLDEN_PATH = _ROOT / "data" / "golden" / "golden.json"
_DEFAULT_OUT = _ROOT / "artifacts" / "finetune-comparison.json"
_PROVIDER = "hash"
# The frozen generations in data/eval/ were produced under dense retrieval.
# Replaying them must pin that mode: scoring resolves [n] citations against the
# retrieved list, so a different retrieval mode would silently re-grade a past
# experiment under conditions it never ran in.
_RETRIEVAL_MODE = "dense"

# Few-shot exemplars are taken ONLY from the training golden set, so evaluating
# on the holdout stays leak-free. Two answerable cases (distinct documents) plus
# one synthetic abstention demo teach the full output contract: short Portuguese
# answer + ``[n]`` citation, or the exact refusal sentence when unsupported.
_FEWSHOT_EXEMPLAR_IDS = ("lai-prazo", "8112-estagio")
_FEWSHOT_ABSTENTION = (
    "You are a legal-administrative assistant. Answer the question using ONLY "
    "the numbered context below. Cite sources inline as [n]. If the answer is "
    "not in the context, reply exactly: "
    '"I could not find this information in the provided documents."\n\n'
    "Context:\n[1] (Lei de Acesso à Informação) O acesso à informação é gratuito, "
    "salvo o custo de reprodução de documentos.\n\n"
    "Question: What is the speed limit on Brazilian federal highways?\n"
    "Answer: I could not find this information in the provided documents."
)


@dataclass
class GenerationScore:
    case_id: str
    expected_doc: str
    answerable: bool
    answer: str
    grounded: bool
    citation_correct: float
    abstained: bool
    faithfulness: float
    answer_relevance: float
    reference_overlap: float


@dataclass
class GenerationReport:
    name: str
    base_model: str
    adapter_path: str | None
    few_shot: bool
    scores: list[GenerationScore]

    @property
    def _answerable(self) -> list[GenerationScore]:
        return [s for s in self.scores if s.answerable]

    @property
    def _unanswerable(self) -> list[GenerationScore]:
        return [s for s in self.scores if not s.answerable]

    @property
    def grounded_rate(self) -> float:
        """Share of *answerable* cases that cited a source (the RAG requirement)."""
        return _mean(1.0 if s.grounded else 0.0 for s in self._answerable)

    @property
    def citation_accuracy(self) -> float:
        """Share of *answerable* cases that cited the RIGHT document, not any bracket."""
        return _mean(s.citation_correct for s in self._answerable)

    @property
    def abstention_rate(self) -> float:
        """Share of *out-of-corpus* cases the model correctly refused to answer."""
        unans = self._unanswerable
        if not unans:
            return float("nan")
        return _mean(1.0 if s.abstained else 0.0 for s in unans)

    @property
    def mean_faithfulness(self) -> float:
        return _mean(s.faithfulness for s in self._answerable)

    @property
    def mean_answer_relevance(self) -> float:
        return _mean(s.answer_relevance for s in self._answerable)

    @property
    def mean_reference_overlap(self) -> float:
        return _mean(s.reference_overlap for s in self._answerable)

    def summary(self) -> dict[str, float | str | bool | None]:
        return {
            "name": self.name,
            "base_model": self.base_model,
            "adapter_path": self.adapter_path,
            "few_shot": self.few_shot,
            "n_answerable": len(self._answerable),
            "n_unanswerable": len(self._unanswerable),
            "grounded_rate": self.grounded_rate,
            "citation_accuracy": self.citation_accuracy,
            "abstention_rate": self.abstention_rate,
            "faithfulness": self.mean_faithfulness,
            "answer_relevance": self.mean_answer_relevance,
            "reference_overlap": self.mean_reference_overlap,
        }


def load_cases(path: Path) -> list[dict[str, Any]]:
    """Load evaluation cases; tolerates the optional ``answerable`` flag."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data["cases"])


def score_case(answer: str, case: dict[str, Any], store: Any) -> GenerationScore:
    """Score a single generated ``answer`` against its retrieved context.

    This is the single scoring authority for both paths: the GPU generation loop
    in :func:`evaluate` and the offline frozen re-scoring in
    ``scripts/score_generations.py``. Retrieval uses the deterministic ``hash``
    provider, so given the same ``answer`` string the score is fully reproducible
    with no model and no network.
    """
    answerable = bool(case.get("answerable", True))
    chunks = retrieve(store, case["question"], k=4, provider=_PROVIDER, mode=_RETRIEVAL_MODE)
    context = build_context(chunks)
    retrieved_docs = [chunk.doc_id for chunk in chunks]
    return GenerationScore(
        case_id=case["id"],
        expected_doc=case["expected_doc"],
        answerable=answerable,
        answer=answer,
        grounded=validate_output(answer).ok,
        citation_correct=metrics.citation_correct(answer, retrieved_docs, case["expected_doc"]),
        abstained=is_abstention(answer),
        faithfulness=metrics.faithfulness(answer, context),
        answer_relevance=metrics.answer_relevance(answer, case["question"]),
        reference_overlap=metrics.answer_relevance(answer, case["reference_answer"]),
    )


def _citation_for_expected_doc(retrieved_docs: list[str], expected_doc: str) -> str:
    """Citation marker for the first retrieved chunk from the expected document."""
    for idx, doc_id in enumerate(retrieved_docs, start=1):
        if doc_id == expected_doc:
            return f"[{idx}]"
    return "[1]"


def build_fewshot_prefix(store: Any, *, k: int = 4) -> str:
    """Render the fixed few-shot exemplars (training cases + abstention demo)."""
    train_cases = {case["id"]: case for case in load_cases(_GOLDEN_PATH)}
    blocks: list[str] = []
    for case_id in _FEWSHOT_EXEMPLAR_IDS:
        case = train_cases[case_id]
        chunks = retrieve(store, case["question"], k=k, provider=_PROVIDER, mode=_RETRIEVAL_MODE)
        prompt = _PROMPT.format(context=build_context(chunks), question=case["question"])
        citation = _citation_for_expected_doc([c.doc_id for c in chunks], case["expected_doc"])
        blocks.append(f"{prompt} {case['reference_answer']} {citation}")
    blocks.append(_FEWSHOT_ABSTENTION)
    return "\n\n".join(blocks) + "\n\n"


def evaluate(
    name: str,
    base_model: str,
    adapter_path: Path | None = None,
    *,
    golden_path: Path = _GOLDEN_PATH,
    few_shot: bool = False,
    max_new_tokens: int = 96,
    limit: int | None = None,
) -> GenerationReport:
    """Generate answers for the cases and score them against retrieved context."""
    tokenizer, model = _load_model(base_model, adapter_path)
    store = ingest_dir(_CORPUS_DIR, provider=_PROVIDER)
    prefix = build_fewshot_prefix(store) if few_shot else ""
    max_length = 2048 if few_shot else 1024

    cases = load_cases(golden_path)
    if limit is not None:
        cases = cases[:limit]

    scores: list[GenerationScore] = []
    for case in cases:
        chunks = retrieve(store, case["question"], k=4, provider=_PROVIDER, mode=_RETRIEVAL_MODE)
        context = build_context(chunks)
        prompt = prefix + _PROMPT.format(context=context, question=case["question"])
        answer = _generate(
            tokenizer, model, prompt, max_new_tokens=max_new_tokens, max_length=max_length
        )
        scores.append(score_case(answer, case, store))
    return GenerationReport(
        name=name,
        base_model=base_model,
        adapter_path=_as_str(adapter_path),
        few_shot=few_shot,
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


def _generate(
    tokenizer: Any, model: Any, prompt: str, *, max_new_tokens: int, max_length: int = 1024
) -> str:
    """Generate a short answer and strip the prompt from the decoded output."""
    import torch

    device = next(model.parameters()).device
    previous_truncation_side = tokenizer.truncation_side
    tokenizer.truncation_side = "left"
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(
            device
        )
    finally:
        tokenizer.truncation_side = previous_truncation_side
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
                "few_shot": report.few_shot,
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
    parser.add_argument(
        "--golden",
        type=Path,
        default=_GOLDEN_PATH,
        help="Cases to evaluate on. Use data/golden/holdout.json for the honest test.",
    )
    parser.add_argument(
        "--few-shot",
        action="store_true",
        help="Add a fair base+few-shot baseline that isolates output-format conformance.",
    )
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    reports = [
        evaluate(
            "base",
            args.base,
            golden_path=args.golden,
            max_new_tokens=args.max_new_tokens,
            limit=args.limit,
        )
    ]
    if args.few_shot:
        reports.append(
            evaluate(
                "base-fewshot",
                args.base,
                golden_path=args.golden,
                few_shot=True,
                max_new_tokens=args.max_new_tokens,
                limit=args.limit,
            )
        )
    if args.adapter is not None:
        reports.append(
            evaluate(
                "lora",
                args.base,
                args.adapter,
                golden_path=args.golden,
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
