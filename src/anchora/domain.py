"""Out-of-domain floor: decide when a question cannot be grounded in the corpus.

A grounded agent must abstain on questions the corpus cannot answer instead of
quoting the nearest-by-cosine chunk with an irrelevant citation. This module
computes that decision from two deterministic signals:

1. **Lexical overlap strength** — the number of *distinct* (bridged) query
   tokens that occur in the corpus. The earlier floor abstained only on *zero*
   overlap, so a single incidental token colliding with the Portuguese corpus
   (e.g. ``"melhor"``) was enough to force an answer to an off-domain question
   — the ``ood-008`` adversarial case. Requiring at least ``min_overlap``
   distinct tokens closes that gap. Measured on the golden set, every in-domain
   question overlaps on ≥ 2 tokens while every off-domain probe overlaps on ≤ 1,
   so the default of 2 separates them with an integer margin, offline.

2. **Dense similarity floor** — the top cosine between the query and any corpus
   chunk. This is the semantically correct signal, but only on a real
   multilingual embedder (Ollama). On the deterministic ``hash`` provider the
   bag-of-words cosine does *not* separate in-domain from off-domain (measured:
   in-domain cosine drops to ~0.06, below several off-domain probes), so the
   threshold defaults to 0.0 (disabled) and is meant to be set on the production
   embedding path. See ``docs/adr/0006-out-of-domain-floor.md``.

A question is in-domain only if it clears *both* configured floors, with one
exception on the production embedder (ADR 6, rule pre-registered 2026-10-15):

3. **Overlap-1 dense rule** — when the provider is ``ollama`` and
   ``ood_overlap1_similarity_threshold`` is positive, a question sharing exactly
   one distinct corpus token (one short of the default floor of 2) is in-domain
   if its top cosine reaches that threshold. It loosens the boundary only on the
   dense path; the ``hash`` path (CI, adversarial suite) is unchanged, so the
   frozen similarities behind ``make ood-floor`` are this rule's only gate. If
   the embedder is unreachable the question abstains, as it did before the rule.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from anchora.config import settings
from anchora.embeddings import embed_texts, tokenize
from anchora.store import VectorStore

DENSE_PROVIDER = "ollama"
# The rule rescues only the one-token band directly below the default floor.
_RESCUABLE_OVERLAP = 1


@dataclass(frozen=True)
class DomainVerdict:
    """Why a question was accepted as in-domain or sent to the abstain path."""

    in_domain: bool
    distinct_overlap: int
    top_similarity: float
    reason: str


def assess_domain(
    question: str,
    store: VectorStore,
    *,
    provider: str | None = None,
    min_overlap: int | None = None,
    similarity_threshold: float | None = None,
    overlap1_threshold: float | None = None,
) -> DomainVerdict:
    """Judge whether ``question`` is answerable from ``store``'s corpus.

    ``min_overlap`` and ``similarity_threshold`` fall back to the configured
    defaults; both are exposed so callers and tests can pin them explicitly.
    The similarity floor is only evaluated when its threshold is positive, so
    the default offline path performs no embedding call. ``overlap1_threshold``
    (default: the configured one) drives the overlap-1 dense rule above.
    """
    floor = settings.ood_min_overlap if min_overlap is None else min_overlap
    threshold = (
        settings.ood_similarity_threshold if similarity_threshold is None else similarity_threshold
    )

    if len(store) == 0:
        return DomainVerdict(False, 0, 0.0, "empty corpus")

    rescue = (
        settings.ood_overlap1_similarity_threshold
        if overlap1_threshold is None
        else overlap1_threshold
    )
    is_dense = (provider or settings.embed_provider) == DENSE_PROVIDER

    overlap = _distinct_overlap(question, store)
    if overlap < floor:
        if not (is_dense and _rule_applies(overlap, floor, rescue)):
            return DomainVerdict(False, overlap, 0.0, f"lexical overlap {overlap} < {floor}")
        return _assess_overlap_one(question, store, provider, overlap, rescue, threshold)

    similarity = _top_similarity(question, store, provider) if threshold > 0.0 else 1.0
    if similarity < threshold:
        return DomainVerdict(
            False, overlap, similarity, f"similarity {similarity:.3f} < {threshold:.3f}"
        )

    return DomainVerdict(True, overlap, similarity, "in domain")


def admits_overlap_one(overlap: int, floor: int, similarity: float, threshold: float) -> bool:
    """The overlap-1 dense rule as a pure decision (shared with the calibration)."""
    return _rule_applies(overlap, floor, threshold) and similarity >= threshold


def _rule_applies(overlap: int, floor: int, threshold: float) -> bool:
    return threshold > 0.0 and overlap == _RESCUABLE_OVERLAP and floor == _RESCUABLE_OVERLAP + 1


def _assess_overlap_one(
    question: str,
    store: VectorStore,
    provider: str | None,
    overlap: int,
    rescue: float,
    threshold: float,
) -> DomainVerdict:
    """Overlap-1 question on the dense path: admit only on a high enough cosine."""
    try:
        similarity = _top_similarity(question, store, provider)
    except httpx.HTTPError as exc:
        return DomainVerdict(False, overlap, 0.0, f"overlap 1, embedder unavailable ({exc})")
    if not admits_overlap_one(overlap, _RESCUABLE_OVERLAP + 1, similarity, rescue):
        return DomainVerdict(
            False, overlap, similarity, f"overlap 1, similarity {similarity:.3f} < {rescue:.3f}"
        )
    if similarity < threshold:
        return DomainVerdict(
            False, overlap, similarity, f"similarity {similarity:.3f} < {threshold:.3f}"
        )
    return DomainVerdict(
        True, overlap, similarity, f"overlap 1, similarity {similarity:.3f} >= {rescue:.3f}"
    )


def _distinct_overlap(question: str, store: VectorStore) -> int:
    """Count distinct (bridged) query tokens that occur in the corpus."""
    query_tokens = set(tokenize(question, query=True))
    return len(query_tokens & store.corpus_vocabulary())


def _top_similarity(question: str, store: VectorStore, provider: str | None) -> float:
    """Top cosine between the query and any corpus chunk (0.0 if empty)."""
    query_vec = embed_texts([question], provider=provider, query=True)[0]
    hits = store.search_indices(query_vec, k=1)
    return hits[0][1] if hits else 0.0
