"""BM25 index: ranking sanity, determinism, and edge cases — all offline."""

from __future__ import annotations

from anchora.lexical import BM25Index

_DOCS = [
    ["prazo", "recurso", "autoridade", "superior"],
    ["licitacao", "modalidades", "pregao", "concorrencia", "leilao"],
    ["servidor", "estagio", "probatorio", "nomeacao"],
    ["prazo", "prazo", "processuais", "uteis"],
]


def _fitted() -> BM25Index:
    index = BM25Index()
    index.fit(_DOCS)
    return index


def test_exact_rare_term_ranks_its_document_first() -> None:
    index = _fitted()
    results = index.search(["licitacao", "modalidades"], k=2)
    assert results[0][0] == 1


def test_term_frequency_breaks_ties_between_matching_docs() -> None:
    index = _fitted()
    # "prazo" appears once in doc 0 and twice in doc 3; doc 3 must rank higher.
    results = index.search(["prazo"], k=4)
    assert [idx for idx, _ in results][:2] == [3, 0]


def test_search_is_deterministic() -> None:
    index = _fitted()
    first = index.search(["prazo", "recurso"], k=4)
    second = index.search(["prazo", "recurso"], k=4)
    assert first == second


def test_no_match_returns_empty() -> None:
    index = _fitted()
    assert index.search(["inexistente"], k=4) == []


def test_empty_query_and_empty_index() -> None:
    index = _fitted()
    assert index.search([], k=4) == []
    empty = BM25Index()
    empty.fit([])
    assert empty.search(["prazo"], k=4) == []


def test_len_reports_document_count() -> None:
    assert len(_fitted()) == len(_DOCS)
