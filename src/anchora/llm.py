"""Answer generation with citations using a local Ollama model.

Returns ``None`` when the model is unavailable so callers can degrade
gracefully (the project stays useful offline).
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator

import httpx

from anchora.config import settings
from anchora.store import Chunk

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


class LLMUnavailableError(RuntimeError):
    """The local model could not be reached or returned a malformed stream."""


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


def answer_prompt(question: str, chunks: list[Chunk]) -> str:
    return _PROMPT.format(context=build_context(chunks), question=question)


def answer(question: str, chunks: list[Chunk]) -> str | None:
    """Generate a cited answer from the retrieved chunks via the local model."""
    return generate(answer_prompt(question, chunks))


def _http_client() -> httpx.Client:
    """Client used for streaming; a seam so tests can inject a mock transport."""
    return httpx.Client(timeout=settings.request_timeout)


def stream_answer(
    question: str, chunks: list[Chunk], client: httpx.Client | None = None
) -> Iterator[str]:
    """Yield answer text as the local model decodes it (Ollama ``stream: true``).

    ``<think>`` blocks are dropped. Raises :class:`LLMUnavailableError` if the
    model cannot be reached or the stream breaks; callers that already forwarded
    some text must treat what they have as the (possibly partial) answer.
    """
    http = client or _http_client()
    payload = {
        "model": settings.gen_model,
        "prompt": answer_prompt(question, chunks),
        "stream": True,
        "think": False,
    }
    in_think = False
    try:
        with http.stream("POST", f"{settings.ollama_base_url}/api/generate", json=payload) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line.strip():
                    continue
                message = json.loads(line)
                piece = str(message.get("response", ""))
                if piece == _THINK_OPEN:
                    in_think = True
                elif piece == _THINK_CLOSE:
                    in_think = False
                elif piece and not in_think:
                    yield piece
                if message.get("done"):
                    return
    except (httpx.HTTPError, json.JSONDecodeError) as exc:
        raise LLMUnavailableError(str(exc)) from exc


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
