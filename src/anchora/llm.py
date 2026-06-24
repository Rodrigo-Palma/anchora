"""Answer generation with citations using a local Ollama model.

Returns ``None`` when the model is unavailable so callers can degrade
gracefully (the project stays useful offline).
"""

from __future__ import annotations

import re

import httpx

from anchora.config import settings
from anchora.store import Chunk

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

_PROMPT = (
    "You are a legal-administrative assistant. Answer the question using ONLY "
    "the numbered context below. Cite sources inline as [n]. If the answer is "
    "not in the context, reply exactly: "
    '"I could not find this information in the provided documents."\n\n'
    "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
)


def build_context(chunks: list[Chunk]) -> str:
    """Render retrieved chunks as a numbered, citable context block."""
    return "\n".join(
        f"[{i + 1}] ({chunk.title or chunk.doc_id}) {chunk.text}" for i, chunk in enumerate(chunks)
    )


def answer(question: str, chunks: list[Chunk]) -> str | None:
    """Generate a cited answer from the retrieved chunks via the local model."""
    prompt = _PROMPT.format(context=build_context(chunks), question=question)
    return generate(prompt)


def generate(prompt: str) -> str | None:
    """Call the local model; return cleaned text or ``None`` if unavailable."""
    try:
        response = httpx.post(
            f"{settings.ollama_base_url}/api/generate",
            json={"model": settings.gen_model, "prompt": prompt, "stream": False},
            timeout=settings.request_timeout,
        )
        response.raise_for_status()
        raw = str(response.json()["response"])
        return _THINK_RE.sub("", raw).strip()
    except (httpx.HTTPError, KeyError, ValueError):
        return None
