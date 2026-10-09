"""Offline tests for the retrieval ablation report (counts and paired test)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

import ablation_retrieval as ab  # noqa: E402


def _score(mode: str, hits: tuple[bool, ...], dataset: str = "d") -> ab.ModeScore:
    rate = sum(hits) / len(hits)
    return ab.ModeScore(mode, dataset, len(hits), rate, rate, rate, hits, (rate,), (rate,))


def test_recall_mcnemar_counts_discordant_pairs() -> None:
    hybrid = _score("hybrid", (True, True, False, True))
    dense = _score("dense", (True, False, False, False))
    assert ab.recall_mcnemar(hybrid, dense) == (2, 0, 0.5)


def test_recall_mcnemar_refuses_unpaired_inputs() -> None:
    with pytest.raises(ValueError):
        ab.recall_mcnemar(_score("hybrid", (True,), "a"), _score("dense", (True,), "b"))


def test_holdout_recall_counts_and_paired_read() -> None:
    """Hybrid and BM25 both hit 20/22 on the holdout; dense hits 19/22 (p=1.0)."""
    scores = {(s.dataset, s.mode): s for s in ab.run()}
    holdout = "holdout (unseen, n=22)"
    assert sum(scores[(holdout, "hybrid")].hits) == 20
    assert sum(scores[(holdout, "bm25")].hits) == 20
    assert sum(scores[(holdout, "dense")].hits) == 19
    only_h, only_d, p = ab.recall_mcnemar(scores[(holdout, "hybrid")], scores[(holdout, "dense")])
    assert (only_h, only_d) == (1, 0)
    assert p == 1.0


# --- production embedder (nomic-embed-text), frozen ---------------------------


def test_frozen_ollama_ablation_is_internally_consistent() -> None:
    """Counts in the frozen nomic run match its per-question hits and the datasets."""
    frozen = ab.load_frozen_ollama()
    assert frozen["provider"] == "ollama"
    assert frozen["embed_model"].startswith("nomic-embed-text")
    assert len(frozen["digest"]) >= 12
    live_ids = ab.dataset_case_ids()
    for row in frozen["rows"]:
        assert row["case_ids"] == live_ids[row["dataset"]]
        assert row["recall_k"] == sum(row["hits"])
        assert row["n"] == len(row["hits"]) == len(row["case_ids"])


def test_bm25_does_not_depend_on_the_embedder() -> None:
    """BM25 is lexical: the frozen nomic run must reproduce the live BM25 hits."""
    frozen = {(r["dataset"], r["mode"]): r for r in ab.load_frozen_ollama()["rows"]}
    for score in ab.run(provider="hash"):
        if score.mode == "bm25":
            assert list(score.hits) == frozen[(score.dataset, "bm25")]["hits"]


def test_embedder_mcnemar_pairs_hash_and_nomic_on_the_same_questions() -> None:
    rows = ab.embedder_comparison(ab.run(provider="hash"), ab.load_frozen_ollama())
    assert rows
    for _dataset, _mode, only_hash, only_nomic, p in rows:
        assert only_hash >= 0 and only_nomic >= 0
        assert 0.0 <= p <= 1.0
