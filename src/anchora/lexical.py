"""Okapi BM25 over the corpus tokens — pure Python, deterministic, no deps.

Dense cosine retrieval and BM25 fail differently: embeddings catch paraphrase
but blur rare exact terms; BM25 nails the distinctive statute vocabulary
("ultrassecreta", "estagio probatorio") but misses reformulations. The hybrid
retriever in :mod:`anchora.rag` fuses both rankings, so this index has to be
as reproducible as the ``hash`` embedding provider: same corpus in, same
ranking out, on any machine, with no model and no network.

Ties are broken by document index so rankings are stable across runs.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from anchora.config import settings


@dataclass
class BM25Index:
    """Okapi BM25 index over pre-tokenized documents.

    ``k1`` saturates term frequency; ``b`` controls document-length
    normalization. Defaults follow the classic Robertson/Lucene values.
    """

    k1: float = field(default_factory=lambda: settings.bm25_k1)
    b: float = field(default_factory=lambda: settings.bm25_b)

    def __post_init__(self) -> None:
        self._doc_freqs: list[Counter[str]] = []
        self._doc_lengths: list[int] = []
        self._idf: dict[str, float] = {}
        self._avg_doc_length: float = 0.0

    def __len__(self) -> int:
        return len(self._doc_freqs)

    def fit(self, documents: list[list[str]]) -> None:
        """Index ``documents`` (each a token list) for scoring."""
        self._doc_freqs = [Counter(tokens) for tokens in documents]
        self._doc_lengths = [len(tokens) for tokens in documents]
        total = sum(self._doc_lengths)
        self._avg_doc_length = total / len(documents) if documents else 0.0
        self._idf = self._compute_idf()

    def search(self, query_tokens: list[str], k: int = 4) -> list[tuple[int, float]]:
        """Top-``k`` ``(document_index, score)`` pairs, deterministically ordered."""
        if not self._doc_freqs or not query_tokens:
            return []
        scored = [
            (index, self._score(query_tokens, index)) for index in range(len(self._doc_freqs))
        ]
        scored = [(index, score) for index, score in scored if score > 0.0]
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[:k]

    def _score(self, query_tokens: list[str], index: int) -> float:
        freqs = self._doc_freqs[index]
        length_norm = (
            1.0
            - self.b
            + self.b
            * (self._doc_lengths[index] / self._avg_doc_length if self._avg_doc_length else 0.0)
        )
        score = 0.0
        for token in set(query_tokens):
            tf = freqs.get(token, 0)
            if tf == 0:
                continue
            idf = self._idf.get(token, 0.0)
            score += idf * (tf * (self.k1 + 1.0)) / (tf + self.k1 * length_norm)
        return score

    def _compute_idf(self) -> dict[str, float]:
        """Lucene-style smoothed IDF — never negative, even for ubiquitous terms."""
        n_docs = len(self._doc_freqs)
        doc_counts: Counter[str] = Counter()
        for freqs in self._doc_freqs:
            doc_counts.update(freqs.keys())
        return {
            token: math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            for token, df in doc_counts.items()
        }
