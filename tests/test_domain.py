"""The out-of-domain floor must abstain on off-domain probes without abstaining
on genuine corpus questions — including the single-token-collision case that
used to slip through (ood-008)."""

from __future__ import annotations

import httpx
import pytest

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


# --- Overlap-1 dense rule (ADR 6, pre-registered 2026-10-15) -----------------
# Only on the production embedder: a question sharing exactly one distinct
# corpus token is in-domain when its top nomic cosine reaches the threshold.
# The cosine is faked so the four branches are deterministic and offline.

_IN_DOMAIN = "What are the bidding modalities?"  # overlap >= 2
_SCI_FI = "Recommend me a sci-fi movie tonight."  # overlap 0


class _FakeCosine:
    def __init__(self, value: float | Exception) -> None:
        self.value = value
        self.calls = 0

    def __call__(self, question: str, store: VectorStore, provider: str | None) -> float:
        self.calls += 1
        if isinstance(self.value, Exception):
            raise self.value
        return self.value


def _fake(monkeypatch: pytest.MonkeyPatch, value: float | Exception) -> _FakeCosine:
    fake = _FakeCosine(value)
    monkeypatch.setattr("anchora.domain._top_similarity", fake)
    return fake


def test_overlap_two_or_more_passes_without_a_cosine_call(
    store: VectorStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _fake(monkeypatch, 0.0)
    verdict = assess_domain(_IN_DOMAIN, store, provider="ollama", overlap1_threshold=0.56)
    assert verdict.in_domain
    assert fake.calls == 0


def test_overlap_one_at_or_above_threshold_is_admitted(
    store: VectorStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake(monkeypatch, 0.56)
    verdict = assess_domain(_OOD_008, store, provider="ollama", overlap1_threshold=0.56)
    assert verdict.in_domain
    assert verdict.distinct_overlap == 1
    assert verdict.top_similarity == 0.56
    assert "overlap 1" in verdict.reason


def test_overlap_one_below_threshold_abstains(
    store: VectorStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake(monkeypatch, 0.549)
    verdict = assess_domain(_OOD_008, store, provider="ollama", overlap1_threshold=0.56)
    assert not verdict.in_domain
    assert verdict.distinct_overlap == 1
    assert verdict.top_similarity == 0.549


def test_overlap_zero_abstains_without_a_cosine_call(
    store: VectorStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _fake(monkeypatch, 0.99)
    verdict = assess_domain(_SCI_FI, store, provider="ollama", overlap1_threshold=0.56)
    assert not verdict.in_domain
    assert fake.calls == 0


def test_overlap_one_rule_never_applies_on_the_hash_provider(
    store: VectorStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _fake(monkeypatch, 0.99)
    verdict = assess_domain(_OOD_008, store, provider="hash", overlap1_threshold=0.56)
    assert not verdict.in_domain
    assert fake.calls == 0


def test_overlap_one_rule_is_off_at_threshold_zero(
    store: VectorStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _fake(monkeypatch, 0.99)
    verdict = assess_domain(_OOD_008, store, provider="ollama", overlap1_threshold=0.0)
    assert not verdict.in_domain
    assert fake.calls == 0


def test_overlap_one_rule_only_extends_the_default_floor_of_two(
    store: VectorStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _fake(monkeypatch, 0.99)
    verdict = assess_domain(
        _OOD_008, store, provider="ollama", min_overlap=3, overlap1_threshold=0.56
    )
    assert not verdict.in_domain
    assert fake.calls == 0


def test_overlap_one_rule_follows_the_configured_provider(
    store: VectorStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    from anchora.config import settings

    monkeypatch.setattr(settings, "embed_provider", "ollama")
    _fake(monkeypatch, 0.6)
    verdict = assess_domain(_OOD_008, store, overlap1_threshold=0.56)
    assert verdict.in_domain


def test_overlap_one_rule_fails_closed_when_the_embedder_is_down(
    store: VectorStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake(monkeypatch, httpx.ConnectError("ollama down"))
    verdict = assess_domain(_OOD_008, store, provider="ollama", overlap1_threshold=0.56)
    assert not verdict.in_domain
    assert "unavailable" in verdict.reason


def test_rescued_question_still_meets_the_general_dense_floor(
    store: VectorStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake(monkeypatch, 0.6)
    verdict = assess_domain(
        _OOD_008,
        store,
        provider="ollama",
        overlap1_threshold=0.56,
        similarity_threshold=0.7,
    )
    assert not verdict.in_domain
    assert "similarity" in verdict.reason
