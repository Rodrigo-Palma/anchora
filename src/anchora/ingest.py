"""Load documents from a directory, chunk, embed, and build a vector store.

Documents may start with a tiny ``title:`` front-matter line so retrieved
chunks carry a human-readable source name for citations.

Ingestion is bounded: a file whose real path escapes ``directory`` (a symlink
pointing elsewhere) is skipped, and too many or too large files raise
:class:`IngestLimitError` before anything is embedded.
"""

from __future__ import annotations

from pathlib import Path

from anchora.chunking import chunk_text
from anchora.embeddings import embed_texts
from anchora.store import Chunk, VectorStore

_SUFFIXES = {".md", ".txt"}


class IngestLimitError(ValueError):
    """The directory holds more files, or a larger file, than ingestion allows."""


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


def corpus_files(
    directory: str | Path,
    max_files: int | None = None,
    max_file_bytes: int | None = None,
) -> list[Path]:
    """List the ``.md`` / ``.txt`` files under ``directory`` that ingestion reads.

    Files whose resolved path leaves ``directory`` are dropped. Raises
    :class:`IngestLimitError` when a limit is set and exceeded.
    """
    root = Path(directory).resolve()
    paths = sorted(
        path
        for path in Path(directory).rglob("*")
        if path.suffix.lower() in _SUFFIXES
        and path.is_file()
        and path.resolve().is_relative_to(root)
    )
    if max_files is not None and len(paths) > max_files:
        raise IngestLimitError(f"{len(paths)} files exceed the limit of {max_files}")
    if max_file_bytes is not None:
        for path in paths:
            size = path.stat().st_size
            if size > max_file_bytes:
                raise IngestLimitError(
                    f"{path.name} has {size} bytes, over the limit of {max_file_bytes}"
                )
    return paths


def ingest_dir(
    directory: str | Path,
    provider: str | None = None,
    max_files: int | None = None,
    max_file_bytes: int | None = None,
) -> VectorStore:
    """Ingest every ``.md`` / ``.txt`` file under ``directory`` into a store."""
    store = VectorStore()
    for path in corpus_files(directory, max_files, max_file_bytes):
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
