"""The out-of-domain floor must abstain on off-domain probes without abstaining
on genuine corpus questions — including the single-token-collision case that
used to slip through (ood-008)."""

from __future__ import annotations

from anchora.domain import assess_domain
from anchora.store import Chunk, VectorStore

# The exact adversarial probe: off-domain, but "melhor" ("best") incidentally
# occurs in the Portuguese legal corpus, so the old zero-overlap floor let it
# through and answered with an irrelevant citation.
_OOD_008 = "Melhor tempero para churrasco gaúcho?"


def test_single_token_collision_abstains(store: VectorStore) -> None:
    verdict = assess_domain(_OOD_008, store, provider="hash")
    assert not verdict.in_domain
    assert verdict.distinct_overlap <= 1
    assert "overlap" in verdict.reason


def test_in_domain_question_passes(store: VectorStore) -> None:
    verdict = assess_domain("What are the bidding modalities?", store, provider="hash")
    assert verdict.in_domain
    assert verdict.distinct_overlap >= 2


def test_off_domain_zero_overlap_abstains(store: VectorStore) -> None:
    verdict = assess_domain("Recommend me a sci-fi movie tonight.", store, provider="hash")
    assert not verdict.in_domain
    assert verdict.distinct_overlap == 0


def test_min_overlap_is_configurable(store: VectorStore) -> None:
    # Relaxing the floor to a single token reproduces the old (leaky) behavior,
    # proving the floor — not some other check — is what now catches ood-008.
    lenient = assess_domain(_OOD_008, store, provider="hash", min_overlap=1)
    assert lenient.in_domain
    assert lenient.distinct_overlap == 1


def test_empty_store_abstains() -> None:
    verdict = assess_domain("Any question at all?", VectorStore(), provider="hash")
    assert not verdict.in_domain
    assert verdict.reason == "empty corpus"


def test_similarity_floor_can_reject_a_lexically_overlapping_question(store: VectorStore) -> None:
    # With a positive threshold the dense-cosine floor also bites: an in-domain
    # question that clears the overlap floor is still sent to abstain when its
    # top cosine is below the configured threshold. (Offline hash cosine is weak,
    # so this path is opt-in; here we force it with an unreachable threshold.)
    verdict = assess_domain(
        "What are the bidding modalities?",
        store,
        provider="hash",
        similarity_threshold=1.1,
    )
    assert not verdict.in_domain
    assert "similarity" in verdict.reason


def test_similarity_floor_disabled_by_default_skips_embedding(store: VectorStore) -> None:
    # threshold 0.0 (default) reports full similarity without an embedding call.
    verdict = assess_domain("What are the bidding modalities?", store, provider="hash")
    assert verdict.in_domain
    assert verdict.top_similarity == 1.0


def test_corpus_vocabulary_is_cached_and_invalidated() -> None:
    store = VectorStore()
    assert store.corpus_vocabulary() == frozenset()
    store.add([Chunk(doc_id="d1", text="prazo recurso administrativo", embedding=[0.0])])
    vocab = store.corpus_vocabulary()
    assert "prazo" in vocab
    assert store.corpus_vocabulary() is vocab  # cached identity
    store.add([Chunk(doc_id="d2", text="licitacao modalidades", embedding=[0.0])])
    assert "licitacao" in store.corpus_vocabulary()  # invalidated on add
