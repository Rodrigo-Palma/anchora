from __future__ import annotations

from pathlib import Path

import pytest

from anchora.ingest import IngestLimitError, corpus_files, ingest_dir, parse_front_matter
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


def test_ingest_skips_symlinks_that_escape_the_directory(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.md").write_text("Private content outside the corpus.", encoding="utf-8")
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "doc.md").write_text("Public legal content.", encoding="utf-8")
    (corpus / "link.md").symlink_to(outside / "secret.md")
    assert [p.name for p in corpus_files(corpus)] == ["doc.md"]


def test_ingest_limits_raise_before_embedding(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("abc", encoding="utf-8")
    (tmp_path / "b.md").write_text("abcdef", encoding="utf-8")
    with pytest.raises(IngestLimitError, match="files exceed"):
        ingest_dir(tmp_path, provider="hash", max_files=1)
    with pytest.raises(IngestLimitError, match=r"b\.md"):
        ingest_dir(tmp_path, provider="hash", max_file_bytes=4)
    assert len(ingest_dir(tmp_path, provider="hash", max_files=2, max_file_bytes=6)) == 2
