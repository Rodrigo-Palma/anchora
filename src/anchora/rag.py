"""Retrieval over a vector store."""

from __future__ import annotations

from anchora.embeddings import embed_texts
from anchora.store import Chunk, VectorStore


def retrieve(
    store: VectorStore, query: str, k: int = 4, provider: str | None = None
) -> list[Chunk]:
    """Return the ``k`` chunks most relevant to ``query``."""
    if len(store) == 0:
        return []
    query_vec = embed_texts([query], provider=provider, query=True)[0]
    return [chunk for chunk, _ in store.search(query_vec, k=k)]
