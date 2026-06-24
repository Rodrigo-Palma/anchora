from __future__ import annotations

import pytest

from anchora.chunking import chunk_text


def test_short_text_is_single_chunk() -> None:
    assert chunk_text("uma frase curta") == ["uma frase curta"]


def test_empty_text_returns_no_chunks() -> None:
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_chunks_overlap() -> None:
    words = " ".join(str(i) for i in range(100))
    chunks = chunk_text(words, size=30, overlap=10)
    assert len(chunks) > 1
    first = chunks[0].split()
    second = chunks[1].split()
    assert first[-10:] == second[:10]


def test_covers_all_words() -> None:
    words = " ".join(str(i) for i in range(100))
    chunks = chunk_text(words, size=30, overlap=10)
    seen = {w for c in chunks for w in c.split()}
    assert seen == {str(i) for i in range(100)}


def test_invalid_params() -> None:
    with pytest.raises(ValueError):
        chunk_text("x", size=0)
    with pytest.raises(ValueError):
        chunk_text("x", size=10, overlap=10)
