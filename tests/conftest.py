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


@pytest.fixture(autouse=True)
def _no_local_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point Ollama at a closed port so no test can pass by reaching a local model.

    CI has no Ollama; without this, a test that forgets ``provider="hash"`` is
    green on a developer machine running Ollama and red in CI.
    """
    from anchora.config import settings

    monkeypatch.setattr(settings, "ollama_base_url", "http://127.0.0.1:9")
