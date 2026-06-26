"""Build an instruction-tuning dataset (JSONL) from the golden set + corpus.

Each example pairs a question with the *grounded* context the offline retriever
surfaces and the reference (gold) answer. Teaching the model to answer strictly
from retrieved context — citing ``[n]`` or abstaining — is the whole point of
the LoRA pass, so the training format mirrors the serving prompt exactly.

Fully offline and deterministic (``hash`` retriever), so it runs in CI and on
a laptop with no model or network.

Usage::

    uv run python scripts/build_finetune_dataset.py
    uv run python scripts/build_finetune_dataset.py --out data/finetune/instructions.jsonl --k 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from anchora.evals import load_cases
from anchora.ingest import ingest_dir
from anchora.llm import _PROMPT, build_context
from anchora.rag import retrieve

_ROOT = Path(__file__).resolve().parents[1]
_CORPUS_DIR = _ROOT / "data" / "corpus"
_DEFAULT_OUT = _ROOT / "data" / "finetune" / "instructions.jsonl"
_ABSTENTION_PATH = _ROOT / "data" / "finetune" / "abstention_train.json"
_PROVIDER = "hash"

# Exact refusal the prompt instructs the model to emit when the answer is not in
# context. Teaching this on out-of-corpus questions is what fixes the adapter's
# "always answer + bracket" failure mode (see docs/finetuning-results.md).
_REFUSAL = "I could not find this information in the provided documents."


def build_examples(k: int = 4, *, abstention: bool = True) -> list[dict[str, str]]:
    """Return instruction examples: one per golden case, plus abstention cases.

    Answerable cases teach "answer from context and cite ``[n]``"; the optional
    abstention cases teach "refuse when the answer is not in context", so the two
    behaviors are balanced instead of the model learning to always answer.
    """
    store = ingest_dir(_CORPUS_DIR, provider=_PROVIDER)
    examples: list[dict[str, str]] = []
    for case in load_cases():
        question = case["question"]
        chunks = retrieve(store, question, k=k, provider=_PROVIDER)
        prompt = _PROMPT.format(context=build_context(chunks), question=question)
        retrieved_docs = [chunk.doc_id for chunk in chunks]
        citation = _citation_for_expected_doc(retrieved_docs, case["expected_doc"])
        examples.append(
            {
                "id": case["id"],
                "instruction": prompt,
                "input": "",
                "output": f"{case['reference_answer']} {citation}",
            }
        )
    if abstention:
        for case in _load_abstention():
            chunks = retrieve(store, case["question"], k=k, provider=_PROVIDER)
            prompt = _PROMPT.format(context=build_context(chunks), question=case["question"])
            examples.append(
                {
                    "id": case["id"],
                    "instruction": prompt,
                    "input": "",
                    "output": _REFUSAL,
                }
            )
    return examples


def _load_abstention() -> list[dict[str, str]]:
    """Load training-only, out-of-corpus questions (empty list if file absent)."""
    if not _ABSTENTION_PATH.exists():
        return []
    data = json.loads(_ABSTENTION_PATH.read_text(encoding="utf-8"))
    return list(data["questions"])


def _citation_for_expected_doc(retrieved_docs: list[str], expected_doc: str) -> str:
    """Return the citation marker for the first retrieved chunk from the expected document."""
    for idx, doc_id in enumerate(retrieved_docs, start=1):
        if doc_id == expected_doc:
            return f"[{idx}]"
    return "[1]"


def write_jsonl(examples: list[dict[str, str]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument(
        "--no-abstention",
        action="store_true",
        help="Exclude the out-of-corpus abstention examples (answerable-only dataset).",
    )
    args = parser.parse_args(argv)

    examples = build_examples(k=args.k, abstention=not args.no_abstention)
    n_abstention = 0 if args.no_abstention else len(_load_abstention())
    write_jsonl(examples, args.out)
    print(
        f"Wrote {len(examples)} examples to {args.out} "
        f"({len(examples) - n_abstention} answerable + {n_abstention} abstention)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
