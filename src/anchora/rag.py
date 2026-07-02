"""Retrieval over a vector store: dense, BM25, or hybrid (RRF).

Dense cosine and BM25 make different mistakes — embeddings generalize across
phrasing, BM25 is exact on rare statute vocabulary. ``hybrid`` fuses the two
rankings with Reciprocal Rank Fusion, which needs no score calibration between
the modalities (only ranks), so it stays deterministic and dependency-free.
Measured ablation across all three modes lives in
``scripts/ablation_retrieval.py``; the numbers are in the README.
"""

from __future__ import annotations

from anchora.config import settings
from anchora.embeddings import embed_texts, tokenize
from anchora.store import Chunk, VectorStore

# Each modality contributes a candidate pool larger than the final k so RRF can
# promote a chunk that is mid-ranked in both lists over one that is top-ranked
# in only one. 3x is a common, deliberately unexciting choice.
_POOL_FACTOR = 3


def retrieve(
    store: VectorStore,
    query: str,
    k: int = 4,
    provider: str | None = None,
    mode: str | None = None,
) -> list[Chunk]:
    """Return the ``k`` chunks most relevant to ``query``.

    ``mode`` is ``"dense"``, ``"bm25"`` or ``"hybrid"`` (default from settings).
    Callers replaying frozen experiments must pin the mode they were run under.
    """
    if len(store) == 0:
        return []
    chosen = mode or settings.retrieval_mode
    if chosen == "dense":
        indices = _dense_indices(store, query, k, provider)
    elif chosen == "bm25":
        indices = [index for index, _ in store.lexical_indices(_query_tokens(query), k=k)]
    elif chosen == "hybrid":
        indices = _hybrid_indices(store, query, k, provider)
    else:
        raise ValueError(f"unknown retrieval mode: {chosen!r}")
    return [store.chunk_at(index) for index in indices]


def _dense_indices(store: VectorStore, query: str, k: int, provider: str | None) -> list[int]:
    query_vec = embed_texts([query], provider=provider, query=True)[0]
    return [index for index, _ in store.search_indices(query_vec, k=k)]


def _hybrid_indices(store: VectorStore, query: str, k: int, provider: str | None) -> list[int]:
    """Reciprocal Rank Fusion of the dense and BM25 rankings.

    ``score(chunk) = Σ 1 / (rrf_k + rank)`` over the lists that contain it.
    Ties break on the chunk's best single-list rank, then on chunk index, so
    the fused ranking is fully deterministic.
    """
    pool = max(k * _POOL_FACTOR, k)
    query_vec = embed_texts([query], provider=provider, query=True)[0]
    dense = [index for index, _ in store.search_indices(query_vec, k=pool)]
    lexical = [index for index, _ in store.lexical_indices(_query_tokens(query), k=pool)]

    fused: dict[int, float] = {}
    best_rank: dict[int, int] = {}
    for ranking in (dense, lexical):
        for rank, index in enumerate(ranking, start=1):
            fused[index] = fused.get(index, 0.0) + 1.0 / (settings.rrf_k + rank)
            best_rank[index] = min(best_rank.get(index, rank), rank)

    ordered = sorted(fused, key=lambda index: (-fused[index], best_rank[index], index))
    return ordered[:k]


def _query_tokens(query: str) -> list[str]:
    return tokenize(query, query=True)
