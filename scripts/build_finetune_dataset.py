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
_PROVIDER = "hash"


def build_examples(k: int = 4) -> list[dict[str, str]]:
    """Return one chat-style instruction example per golden case."""
    store = ingest_dir(_CORPUS_DIR, provider=_PROVIDER)
    examples: list[dict[str, str]] = []
    for case in load_cases():
        question = case["question"]
        chunks = retrieve(store, question, k=k, provider=_PROVIDER)
        prompt = _PROMPT.format(context=build_context(chunks), question=question)
        examples.append(
            {
                "id": case["id"],
                "instruction": prompt,
                "input": "",
                "output": case["reference_answer"],
            }
        )
    return examples


def write_jsonl(examples: list[dict[str, str]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    parser.add_argument("--k", type=int, default=4)
    args = parser.parse_args(argv)

    examples = build_examples(k=args.k)
    write_jsonl(examples, args.out)
    print(f"Wrote {len(examples)} examples to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
