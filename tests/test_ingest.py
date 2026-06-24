from __future__ import annotations

from pathlib import Path

from anchora.ingest import ingest_dir, parse_front_matter
from anchora.store import VectorStore


def test_parse_front_matter_with_title() -> None:
    title, body = parse_front_matter("title: My Law\n\nBody text.")
    assert title == "My Law"
    assert body == "Body text."


def test_parse_front_matter_without_title() -> None:
    title, body = parse_front_matter("No header here.")
    assert title == ""
    assert body == "No header here."


def test_ingest_corpus(store: VectorStore) -> None:
    assert len(store) >= 8  # at least one chunk per document


def test_ingest_empty_dir(tmp_path: Path) -> None:
    store = ingest_dir(tmp_path, provider="hash")
    assert len(store) == 0


def test_ingest_assigns_title(tmp_path: Path) -> None:
    doc = tmp_path / "doc.md"
    doc.write_text("title: Document X\n\nRelevant content for testing.", encoding="utf-8")
    store = ingest_dir(tmp_path, provider="hash")
    assert len(store) == 1
