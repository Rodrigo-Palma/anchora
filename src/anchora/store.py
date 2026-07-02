"""Tiny in-memory vector store with cosine search and JSON persistence.

Carries per-chunk metadata (e.g. the human title of the source law) so the
agent can cite sources precisely.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from anchora.embeddings import tokenize
from anchora.lexical import BM25Index


@dataclass
class Chunk:
    doc_id: str
    text: str
    embedding: list[float]
    title: str = ""
    metadata: dict[str, str] = field(default_factory=dict)


class VectorStore:
    """Holds chunks and ranks them by cosine similarity (vectors are unit-norm).

    Also serves BM25 rankings over the same chunks (built lazily, invalidated
    on ``add``) so the hybrid retriever can fuse both views of the corpus.
    """

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._bm25: BM25Index | None = None

    def __len__(self) -> int:
        return len(self._chunks)

    def add(self, chunks: list[Chunk]) -> None:
        self._chunks.extend(chunks)
        self._bm25 = None  # chunk set changed; rebuild lazily on next lexical query

    def chunk_at(self, index: int) -> Chunk:
        return self._chunks[index]

    def search(self, query_vec: list[float], k: int = 4) -> list[tuple[Chunk, float]]:
        return [
            (self._chunks[index], score) for index, score in self.search_indices(query_vec, k=k)
        ]

    def search_indices(self, query_vec: list[float], k: int = 4) -> list[tuple[int, float]]:
        """Dense cosine top-``k`` as ``(chunk_index, score)``, ties broken by index."""
        scored = [
            (index, _dot(chunk.embedding, query_vec)) for index, chunk in enumerate(self._chunks)
        ]
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[:k]

    def lexical_indices(self, query_tokens: list[str], k: int = 4) -> list[tuple[int, float]]:
        """BM25 top-``k`` as ``(chunk_index, score)`` over the same chunks."""
        if self._bm25 is None:
            index = BM25Index()
            index.fit([tokenize(chunk.text) for chunk in self._chunks])
            self._bm25 = index
        return self._bm25.search(query_tokens, k=k)

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
