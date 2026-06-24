from __future__ import annotations

from pathlib import Path

from anchora.embeddings import embed_texts
from anchora.store import Chunk, VectorStore


def _chunk(doc_id: str, text: str) -> Chunk:
    vec = embed_texts([text], provider="hash")[0]
    return Chunk(doc_id=doc_id, text=text, embedding=vec, title=doc_id)


def test_add_and_len() -> None:
    store = VectorStore()
    assert len(store) == 0
    store.add([_chunk("a.md", "prazo de recurso administrativo")])
    assert len(store) == 1


def test_search_ranks_relevant_first() -> None:
    store = VectorStore()
    store.add(
        [
            _chunk("lic.md", "modalidades de licitação pregão concorrência leilão"),
            _chunk("cpc.md", "prazo de contestação em dias úteis no processo civil"),
        ]
    )
    query = embed_texts(["quais são as modalidades de licitação"], provider="hash")[0]
    results = store.search(query, k=2)
    assert results[0][0].doc_id == "lic.md"


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    store = VectorStore()
    store.add([_chunk("a.md", "test text"), _chunk("b.md", "other text")])
    path = tmp_path / "store.json"
    store.save(path)
    loaded = VectorStore.load(path)
    assert len(loaded) == 2
