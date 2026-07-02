"""Retrieval ablation: dense vs. BM25 vs. hybrid (RRF), measured — not assumed.

Runs every retrieval mode over the training golden set and the held-out set
(answerable cases only) with the deterministic ``hash`` provider, so the table
reproduces bit-for-bit on any machine with no model and no network. This is the
evidence behind the default ``retrieval_mode`` in ``anchora.config`` — if a
mode change is proposed, this script is the referee.

Usage::

    uv run python scripts/ablation_retrieval.py             # aligned table
    uv run python scripts/ablation_retrieval.py --markdown  # README-ready
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from anchora.ingest import ingest_dir
from anchora.rag import retrieve
from anchora.store import VectorStore

_ROOT = Path(__file__).resolve().parents[1]
_CORPUS_DIR = _ROOT / "data" / "corpus"
_GOLDEN_PATH = _ROOT / "data" / "golden" / "golden.json"
_HOLDOUT_PATH = _ROOT / "data" / "golden" / "holdout.json"
_PROVIDER = "hash"
_MODES = ("dense", "bm25", "hybrid")
_K = 4


@dataclass
class ModeScore:
    mode: str
    dataset: str
    n_cases: int
    recall: float
    precision: float
    mrr: float


def _load_cases(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [case for case in data["cases"] if case.get("answerable", True)]


def _score_mode(store: VectorStore, cases: list[dict[str, Any]], mode: str, name: str) -> ModeScore:
    recalls: list[float] = []
    precisions: list[float] = []
    reciprocal_ranks: list[float] = []
    for case in cases:
        expected = str(case["expected_doc"])
        docs = [
            chunk.doc_id
            for chunk in retrieve(store, str(case["question"]), k=_K, provider=_PROVIDER, mode=mode)
        ]
        recalls.append(1.0 if expected in docs else 0.0)
        precisions.append(sum(1 for doc in docs if doc == expected) / len(docs) if docs else 0.0)
        rank = next((i for i, doc in enumerate(docs, start=1) if doc == expected), 0)
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
    return ModeScore(
        mode=mode,
        dataset=name,
        n_cases=len(cases),
        recall=_mean(recalls),
        precision=_mean(precisions),
        mrr=_mean(reciprocal_ranks),
    )


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def run() -> list[ModeScore]:
    store = ingest_dir(_CORPUS_DIR, provider=_PROVIDER)
    datasets = (
        ("golden (train, n=24)", _load_cases(_GOLDEN_PATH)),
        ("holdout (unseen)", _load_cases(_HOLDOUT_PATH)),
    )
    return [_score_mode(store, cases, mode, name) for name, cases in datasets for mode in _MODES]


def print_plain(scores: list[ModeScore]) -> None:
    print(f"{'dataset':<22} {'mode':<8} {'recall@4':>9} {'precision@4':>12} {'MRR@4':>7}")
    print("-" * 62)
    for s in scores:
        print(f"{s.dataset:<22} {s.mode:<8} {s.recall:>9.3f} {s.precision:>12.3f} {s.mrr:>7.3f}")


def print_markdown(scores: list[ModeScore]) -> None:
    print("| Dataset | Mode | Recall@4 | Precision@4 | MRR@4 |")
    print("|---|---|---:|---:|---:|")
    for s in scores:
        bold = s.mode == "hybrid"
        mode = f"**{s.mode}**" if bold else s.mode
        print(f"| {s.dataset} | {mode} | {s.recall:.3f} | {s.precision:.3f} | {s.mrr:.3f} |")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markdown", action="store_true", help="print a README-ready table")
    args = parser.parse_args(argv)
    scores = run()
    if args.markdown:
        print_markdown(scores)
    else:
        print_plain(scores)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
