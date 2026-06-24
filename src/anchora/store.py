"""Tiny in-memory vector store with cosine search and JSON persistence.

Carries per-chunk metadata (e.g. the human title of the source law) so the
agent can cite sources precisely.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Chunk:
    doc_id: str
    text: str
    embedding: list[float]
    title: str = ""
    metadata: dict[str, str] = field(default_factory=dict)


class VectorStore:
    """Holds chunks and ranks them by cosine similarity (vectors are unit-norm)."""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []

    def __len__(self) -> int:
        return len(self._chunks)

    def add(self, chunks: list[Chunk]) -> None:
        self._chunks.extend(chunks)

    def search(self, query_vec: list[float], k: int = 4) -> list[tuple[Chunk, float]]:
        scored = [(chunk, _dot(chunk.embedding, query_vec)) for chunk in self._chunks]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]

    def save(self, path: str | Path) -> None:
        payload = [asdict(chunk) for chunk in self._chunks]
        Path(path).write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> VectorStore:
        store = cls()
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        store.add([Chunk(**item) for item in raw])
        return store


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=False))
