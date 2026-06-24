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
