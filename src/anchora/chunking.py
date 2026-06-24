"""Split text into overlapping word chunks for retrieval."""

from __future__ import annotations


def chunk_text(text: str, size: int = 180, overlap: int = 40) -> list[str]:
    """Split ``text`` into chunks of ``size`` words with ``overlap`` words shared."""
    if size <= 0:
        raise ValueError("size must be > 0")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be in [0, size)")
    words = text.split()
    if not words:
        return []
    step = size - overlap
    chunks: list[str] = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start : start + size]))
        if start + size >= len(words):
            break
    return chunks
