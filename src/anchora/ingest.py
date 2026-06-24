"""Load documents from a directory, chunk, embed, and build a vector store.

Documents may start with a tiny ``title:`` front-matter line so retrieved
chunks carry a human-readable source name for citations.
"""

from __future__ import annotations

from pathlib import Path

from anchora.chunking import chunk_text
from anchora.embeddings import embed_texts
from anchora.store import Chunk, VectorStore

_SUFFIXES = {".md", ".txt"}


def parse_front_matter(raw: str) -> tuple[str, str]:
    """Return ``(title, body)``; ``title`` is empty if no front-matter present.

    Front-matter is a single optional ``title: ...`` line at the very top.
    """
    lines = raw.splitlines()
    if lines and lines[0].lower().startswith("title:"):
        title = lines[0].split(":", 1)[1].strip()
        body = "\n".join(lines[1:]).strip()
        return title, body
    return "", raw


def ingest_dir(directory: str | Path, provider: str | None = None) -> VectorStore:
    """Ingest every ``.md`` / ``.txt`` file under ``directory`` into a store."""
    store = VectorStore()
    paths = sorted(path for path in Path(directory).rglob("*") if path.suffix.lower() in _SUFFIXES)
    for path in paths:
        title, body = parse_front_matter(path.read_text(encoding="utf-8"))
        chunks = chunk_text(body)
        if not chunks:
            continue
        vectors = embed_texts(chunks, provider=provider)
        store.add(
            [
                Chunk(
                    doc_id=path.name,
                    text=text,
                    embedding=vector,
                    title=title or path.stem,
                )
                for text, vector in zip(chunks, vectors, strict=True)
            ]
        )
    return store
