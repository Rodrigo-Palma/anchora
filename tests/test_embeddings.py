from __future__ import annotations

import math

from anchora.embeddings import embed_texts


def test_hash_embed_is_deterministic() -> None:
    a = embed_texts(["prazo de recurso"], provider="hash")[0]
    b = embed_texts(["prazo de recurso"], provider="hash")[0]
    assert a == b


def test_hash_embed_is_unit_norm() -> None:
    vec = embed_texts(["licitação e contrato"], provider="hash")[0]
    norm = math.sqrt(sum(v * v for v in vec))
    assert math.isclose(norm, 1.0, rel_tol=1e-9)


def test_accent_folding_collides() -> None:
    a = embed_texts(["licitação"], provider="hash")[0]
    b = embed_texts(["licitacao"], provider="hash")[0]
    assert a == b


def test_distinct_texts_differ() -> None:
    a = embed_texts(["prazo de recurso"], provider="hash")[0]
    b = embed_texts(["modalidades de licitação"], provider="hash")[0]
    assert a != b


def test_batch_length() -> None:
    vecs = embed_texts(["a frase um", "outra frase dois"], provider="hash")
    assert len(vecs) == 2
