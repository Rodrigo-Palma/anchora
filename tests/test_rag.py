from __future__ import annotations

import json
from pathlib import Path

import pytest

from anchora.rag import retrieve
from anchora.store import VectorStore

_GOLDEN = Path(__file__).resolve().parents[1] / "data" / "golden" / "golden.json"


def _golden_cases() -> list[dict[str, str]]:
    return list(json.loads(_GOLDEN.read_text(encoding="utf-8"))["cases"])


def test_empty_store_returns_nothing() -> None:
    assert retrieve(VectorStore(), "any question", provider="hash") == []


@pytest.mark.parametrize("case", _golden_cases(), ids=lambda c: c["id"])
def test_expected_doc_is_retrieved(store: VectorStore, case: dict[str, str]) -> None:
    """Every golden question must surface its expected document in top-k."""
    chunks = retrieve(store, case["question"], k=4, provider="hash")
    docs = {c.doc_id for c in chunks}
    assert case["expected_doc"] in docs, f"{case['id']}: got {docs}"


@pytest.mark.parametrize("mode", ["dense", "hybrid"])
def test_dense_and_hybrid_fill_k(store: VectorStore, mode: str) -> None:
    chunks = retrieve(store, "What are the bidding modalities?", k=4, provider="hash", mode=mode)
    assert len(chunks) == 4


def test_bm25_returns_only_scoring_chunks(store: VectorStore) -> None:
    """BM25 never pads with zero-score chunks — fewer than k is honest."""
    chunks = retrieve(store, "What are the bidding modalities?", k=4, provider="hash", mode="bm25")
    assert 1 <= len(chunks) <= 4


def test_unknown_mode_raises(store: VectorStore) -> None:
    with pytest.raises(ValueError, match="unknown retrieval mode"):
        retrieve(store, "any question", provider="hash", mode="reranker")


def test_hybrid_is_deterministic(store: VectorStore) -> None:
    question = "What is the appeal deadline under the LAI?"
    first = retrieve(store, question, k=4, provider="hash", mode="hybrid")
    second = retrieve(store, question, k=4, provider="hash", mode="hybrid")
    assert [c.text for c in first] == [c.text for c in second]


def test_bm25_surfaces_exact_statute_vocabulary(store: VectorStore) -> None:
    """A rare exact term must rank its source document first under BM25."""
    chunks = retrieve(store, "informacao ultrassecreta", k=4, provider="hash", mode="bm25")
    assert chunks, "BM25 returned nothing for corpus vocabulary"
    assert "ultrassecreta" in chunks[0].text.lower()
