"""Retrieval ablation: dense vs. BM25 vs. hybrid (RRF), measured — not assumed.

Runs every retrieval mode over the training golden set and the held-out set
(answerable cases only) with the deterministic ``hash`` provider, so the table
reproduces bit-for-bit on any machine with no model and no network. This is the
evidence behind the default ``retrieval_mode`` in ``anchora.config`` — if a
mode change is proposed, this script is the referee.

Every cell carries its sample size and a 95% interval (Wilson for recall, a
seeded bootstrap for the per-question means), and ``--markdown`` adds the exact
paired McNemar test on recall between hybrid and each single mode. On 22
held-out questions a one- or two-question difference is not distinguishable.

The default ``hash`` provider is what CI reproduces. ``--provider ollama``
measures the production embedder (``nomic-embed-text``) instead, and
``--freeze`` stores its per-question hits with the model digest in
``data/eval/ablation-ollama.json``, so the comparison between embedders
(``--compare``, exact McNemar on recall per mode) runs offline afterwards.

Usage::

    uv run python scripts/ablation_retrieval.py                       # aligned table
    uv run python scripts/ablation_retrieval.py --markdown            # README-ready
    uv run python scripts/ablation_retrieval.py --provider ollama --freeze
    uv run python scripts/ablation_retrieval.py --compare             # hash vs frozen nomic
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from anchora.config import settings
from anchora.ingest import ingest_dir
from anchora.rag import retrieve
from anchora.stats import format_mean, format_proportion, mcnemar_exact
from anchora.store import VectorStore

_ROOT = Path(__file__).resolve().parents[1]
_CORPUS_DIR = _ROOT / "data" / "corpus"
_GOLDEN_PATH = _ROOT / "data" / "golden" / "golden.json"
_HOLDOUT_PATH = _ROOT / "data" / "golden" / "holdout.json"
_OLLAMA_FROZEN_PATH = _ROOT / "data" / "eval" / "ablation-ollama.json"
_MODES = ("dense", "bm25", "hybrid")
_K = 4


@dataclass(frozen=True)
class ModeScore:
    mode: str
    dataset: str
    n_cases: int
    recall: float
    precision: float
    mrr: float
    hits: tuple[bool, ...] = ()
    precisions: tuple[float, ...] = ()
    reciprocal_ranks: tuple[float, ...] = ()
    case_ids: tuple[str, ...] = ()


def _load_cases(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [case for case in data["cases"] if case.get("answerable", True)]


def _score_mode(
    store: VectorStore, cases: list[dict[str, Any]], mode: str, name: str, provider: str
) -> ModeScore:
    recalls: list[float] = []
    precisions: list[float] = []
    reciprocal_ranks: list[float] = []
    for case in cases:
        expected = str(case["expected_doc"])
        docs = [
            chunk.doc_id
            for chunk in retrieve(store, str(case["question"]), k=_K, provider=provider, mode=mode)
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
        hits=tuple(r == 1.0 for r in recalls),
        precisions=tuple(precisions),
        reciprocal_ranks=tuple(reciprocal_ranks),
        case_ids=tuple(str(case["id"]) for case in cases),
    )


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _datasets() -> tuple[tuple[str, list[dict[str, Any]]], ...]:
    return (
        ("golden (train, n=24)", _load_cases(_GOLDEN_PATH)),
        ("holdout (unseen, n=22)", _load_cases(_HOLDOUT_PATH)),
    )


def dataset_case_ids() -> dict[str, list[str]]:
    return {name: [str(c["id"]) for c in cases] for name, cases in _datasets()}


def run(provider: str = "hash") -> list[ModeScore]:
    store = ingest_dir(_CORPUS_DIR, provider=provider)
    return [
        _score_mode(store, cases, mode, name, provider)
        for name, cases in _datasets()
        for mode in _MODES
    ]


def freeze_ollama(scores: list[ModeScore], path: Path = _OLLAMA_FROZEN_PATH) -> None:
    """Store per-question hits of an Ollama run with the embedder's digest."""
    tags = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=30.0).json()["models"]
    names = {settings.embed_model, f"{settings.embed_model}:latest"}
    digest = next((m["digest"] for m in tags if m["name"] in names), "unknown")
    document = {
        "_comment": (
            "Retrieval ablation measured with the production embedder through Ollama. "
            "Frozen so the hash-vs-nomic comparison runs offline; regenerate with "
            "scripts/ablation_retrieval.py --provider ollama --freeze."
        ),
        "provider": "ollama",
        "embed_model": settings.embed_model,
        "digest": digest,
        "k": _K,
        "rows": [
            {
                "dataset": s.dataset,
                "mode": s.mode,
                "n": s.n_cases,
                "recall_k": sum(s.hits),
                "precision": s.precision,
                "mrr": s.mrr,
                "case_ids": list(s.case_ids),
                "hits": list(s.hits),
                "precisions": list(s.precisions),
                "reciprocal_ranks": list(s.reciprocal_ranks),
            }
            for s in scores
        ],
    }
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def load_frozen_ollama(path: Path = _OLLAMA_FROZEN_PATH) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def scores_from_frozen(frozen: dict[str, Any]) -> list[ModeScore]:
    return [
        ModeScore(
            mode=r["mode"],
            dataset=r["dataset"],
            n_cases=r["n"],
            recall=r["recall_k"] / r["n"],
            precision=r["precision"],
            mrr=r["mrr"],
            hits=tuple(r["hits"]),
            precisions=tuple(r["precisions"]),
            reciprocal_ranks=tuple(r["reciprocal_ranks"]),
            case_ids=tuple(r["case_ids"]),
        )
        for r in frozen["rows"]
    ]


def embedder_comparison(
    hash_scores: list[ModeScore], frozen: dict[str, Any]
) -> list[tuple[str, str, int, int, float]]:
    """Paired McNemar on recall, hash vs nomic, per dataset and mode."""
    nomic = {(s.dataset, s.mode): s for s in scores_from_frozen(frozen)}
    rows: list[tuple[str, str, int, int, float]] = []
    for score in hash_scores:
        other = nomic[(score.dataset, score.mode)]
        if score.case_ids != other.case_ids:
            raise ValueError(f"question order differs for {score.dataset}/{score.mode}")
        only_hash, only_nomic, p = recall_mcnemar(score, other)
        rows.append((score.dataset, score.mode, only_hash, only_nomic, p))
    return rows


def print_comparison(rows: list[tuple[str, str, int, int, float]]) -> None:
    print("| Dataset | Mode | Only hash hit | Only nomic hit | McNemar exact p |")
    print("|---|---|---:|---:|---:|")
    for dataset, mode, only_hash, only_nomic, p in rows:
        print(f"| {dataset} | {mode} | {only_hash} | {only_nomic} | {p:.3f} |")


def print_plain(scores: list[ModeScore]) -> None:
    print(f"{'dataset':<22} {'mode':<8} {'recall@4':>9} {'precision@4':>12} {'MRR@4':>7}")
    print("-" * 62)
    for s in scores:
        print(f"{s.dataset:<22} {s.mode:<8} {s.recall:>9.3f} {s.precision:>12.3f} {s.mrr:>7.3f}")


def recall_mcnemar(a: ModeScore, b: ModeScore) -> tuple[int, int, float]:
    """Paired exact McNemar on recall@k between two modes on the same questions."""
    if a.dataset != b.dataset or len(a.hits) != len(b.hits):
        raise ValueError(f"cannot pair {a.dataset}/{a.mode} with {b.dataset}/{b.mode}")
    only_a = sum(x and not y for x, y in zip(a.hits, b.hits, strict=True))
    only_b = sum(y and not x for x, y in zip(a.hits, b.hits, strict=True))
    return only_a, only_b, mcnemar_exact(only_a, only_b)


def print_markdown(scores: list[ModeScore]) -> None:
    print(
        "| Dataset | Mode | Recall@4 (k/n, Wilson 95%) "
        "| Precision@4 (mean, 95%) | MRR@4 (mean, 95%) |"
    )
    print("|---|---|---|---|---|")
    for s in scores:
        bold = s.mode == "hybrid"
        mode = f"**{s.mode}**" if bold else s.mode
        recall = format_proportion(sum(s.hits), s.n_cases)
        precision = format_mean(s.precisions)
        mrr = format_mean(s.reciprocal_ranks)
        print(f"| {s.dataset} | {mode} | {recall} | {precision} | {mrr} |")
    print()
    print("| Dataset | Pair (recall@4) | Only hybrid hit | Only other hit | McNemar exact p |")
    print("|---|---|---:|---:|---:|")
    by_key = {(s.dataset, s.mode): s for s in scores}
    for dataset in dict.fromkeys(s.dataset for s in scores):
        hybrid = by_key[(dataset, "hybrid")]
        for other in ("dense", "bm25"):
            only_h, only_o, p = recall_mcnemar(hybrid, by_key[(dataset, other)])
            print(f"| {dataset} | hybrid vs {other} | {only_h} | {only_o} | {p:.3f} |")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markdown", action="store_true", help="print a README-ready table")
    parser.add_argument("--provider", choices=("hash", "ollama"), default="hash")
    parser.add_argument("--freeze", action="store_true", help="freeze an ollama run to JSON")
    parser.add_argument("--compare", action="store_true", help="hash vs frozen nomic, McNemar")
    args = parser.parse_args(argv)
    if args.compare:
        print_comparison(embedder_comparison(run("hash"), load_frozen_ollama()))
        return 0
    if args.provider == "hash" and args.freeze:
        parser.error("--freeze stores an Ollama run; use it with --provider ollama")
    scores = run(args.provider)
    if args.freeze:
        freeze_ollama(scores)
    if args.markdown:
        print_markdown(scores)
    else:
        print_plain(scores)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
