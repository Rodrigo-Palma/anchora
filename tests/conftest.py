"""Shared pytest fixtures. Everything uses the deterministic ``hash`` provider
so the suite runs fully offline with no model or network."""

from __future__ import annotations

from pathlib import Path

import pytest

from anchora.ingest import ingest_dir
from anchora.store import VectorStore

_ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = _ROOT / "data" / "corpus"
GOLDEN_PATH = _ROOT / "data" / "golden" / "golden.json"


@pytest.fixture(scope="session")
def corpus_dir() -> Path:
    return CORPUS_DIR


@pytest.fixture(scope="session")
def store() -> VectorStore:
    return ingest_dir(CORPUS_DIR, provider="hash")
