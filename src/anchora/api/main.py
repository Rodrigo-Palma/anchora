"""FastAPI app serving the anchora RAG agent.

Endpoints:

* ``GET  /health``  — liveness + store status;
* ``POST /ingest``  — (re)build the in-memory store from the configured corpus;
* ``POST /ask``     — answer a question with the tool-using agent.

Cross-cutting concerns handled here:

* API-key gate (``settings.api_key``). With no key the API is in dev mode and
  only serves loopback clients; anyone else gets 503 (fail closed);
* ``/ingest`` reads only inside ``settings.corpus_root``, with file-count and
  file-size limits, so a caller cannot index arbitrary server files;
* PII redaction on every inbound question before it touches the model/logs;
* the agent's own input/output guardrails (injection block, grounding).

The store is held in app state so it survives across requests; it is built
lazily from the default corpus on first use if ``/ingest`` was never called.
"""

from __future__ import annotations

import ipaddress
import json
import secrets
import uuid
from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from anchora import llm
from anchora.agent import Agent, AgentResult
from anchora.config import settings
from anchora.guardrails import detect_pii, redact_pii
from anchora.ingest import IngestLimitError, corpus_files, ingest_dir
from anchora.store import VectorStore

_REQUEST_ID_HEADER = "x-request-id"

_DEFAULT_CORPUS = (Path(__file__).resolve().parents[3] / "data" / "corpus").resolve()


class IngestRequest(BaseModel):
    directory: str | None = Field(
        default=None,
        description="Directory to ingest, relative to the configured corpus root "
        "(an absolute path must also resolve inside it); defaults to the root.",
    )
    provider: str | None = Field(
        default=None,
        description='Embedding provider override ("ollama" or "hash").',
    )


class IngestResponse(BaseModel):
    documents_indexed: int
    chunks_indexed: int


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=settings.top_k, ge=1, le=20)
    use_llm: bool = True
    provider: str | None = Field(
        default=None,
        description="Embedding provider for the query; must match how the store "
        'was indexed ("ollama" or "hash").',
    )


class ToolCallOut(BaseModel):
    name: str
    output: str


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]
    grounded: bool
    refused: bool
    pii_redacted: bool
    output_pii_redacted: bool
    tool_calls: list[ToolCallOut]
    trace_id: str
    timing_ms: dict[str, float]


def create_app() -> FastAPI:
    app = FastAPI(
        title="anchora",
        description="Legal-administrative RAG agent (local-first, with guardrails).",
        version=_version(),
    )
    app.state.store = None
    app.state.docs_indexed = 0

    @app.middleware("http")
    async def add_request_id(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Attach a correlation id to every request/response for tracing.

        Honors an inbound ``x-request-id`` (so a caller's id flows through) or
        mints one; always echoes it back in the response header.
        """
        request_id = request.headers.get(_REQUEST_ID_HEADER) or uuid.uuid4().hex[:12]
        response = await call_next(request)
        response.headers[_REQUEST_ID_HEADER] = request_id
        return response

    def require_api_key(
        request: Request, x_api_key: Annotated[str | None, Header()] = None
    ) -> None:
        if not settings.api_key:
            client_host = request.client.host if request.client else ""
            if not _is_loopback(client_host):
                raise HTTPException(
                    status_code=503,
                    detail="no API key configured; set ANCHORA_API_KEY to serve non-local clients",
                )
            return
        if x_api_key is None or not secrets.compare_digest(
            x_api_key.encode(), settings.api_key.encode()
        ):
            raise HTTPException(status_code=401, detail="invalid or missing API key")

    def get_store() -> VectorStore:
        if app.state.store is None:
            root = _corpus_root()
            app.state.store = _bounded_ingest(root)
            app.state.docs_indexed = len(corpus_files(root))
        return app.state.store

    @app.get("/health")
    def health() -> dict[str, object]:
        store = app.state.store
        return {
            "status": "ok",
            "app": settings.app_name,
            "store_loaded": store is not None,
            "chunks_indexed": len(store) if store is not None else 0,
        }

    @app.post("/ingest", response_model=IngestResponse, dependencies=[Depends(require_api_key)])
    def ingest(req: IngestRequest) -> IngestResponse:
        directory = _resolve_inside_root(req.directory)
        try:
            store = _bounded_ingest(directory, provider=req.provider)
        except IngestLimitError as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        app.state.store = store
        app.state.docs_indexed = len(corpus_files(directory))
        return IngestResponse(
            documents_indexed=app.state.docs_indexed,
            chunks_indexed=len(store),
        )

    @app.post("/ask", response_model=AskResponse, dependencies=[Depends(require_api_key)])
    def ask(req: AskRequest) -> AskResponse:
        pii_found = bool(detect_pii(req.question))
        question = redact_pii(req.question) if pii_found else req.question
        store = get_store()
        agent = Agent(store, k=req.k, provider=req.provider, use_llm=req.use_llm)
        result = agent.run(question)
        output_pii = bool(detect_pii(result.answer))
        return AskResponse(
            question=question,
            answer=redact_pii(result.answer) if output_pii else result.answer,
            sources=result.sources,
            grounded=result.grounded,
            refused=result.refused,
            pii_redacted=pii_found,
            output_pii_redacted=output_pii,
            tool_calls=[ToolCallOut(name=tc.name, output=tc.output) for tc in result.tool_calls],
            trace_id=result.trace.trace_id,
            timing_ms={span.name: round(span.duration_ms, 3) for span in result.trace.spans},
        )

    @app.post("/ask/stream", dependencies=[Depends(require_api_key)])
    def ask_stream(req: AskRequest) -> StreamingResponse:
        """Answer via Server-Sent Events: ``token`` events, then ``done``.

        Guardrails, the domain floor and retrieval run first. With ``use_llm``
        the model's tokens are forwarded as Ollama decodes them, and the output
        guardrail runs on the complete text: if it is not grounded, a
        ``retracted`` event carrying the abstention replaces what the client
        has already shown (ADR 8). Refusals, abstentions and the offline
        extractive path are delivered word by word, already PII-redacted. If the
        model is unreachable before the first token, the extractive answer is
        used instead; if the stream breaks after it, ``done`` carries
        ``truncated: true``. Model text that turns out to contain PII is
        retracted and replaced by its redacted form (ADR 8).
        """
        pii_found = bool(detect_pii(req.question))
        question = redact_pii(req.question) if pii_found else req.question
        agent = Agent(get_store(), k=req.k, provider=req.provider, use_llm=req.use_llm)
        prepared = agent.prepare(question)

        def done(result: AgentResult, *, truncated: bool = False, output_pii: bool = False) -> str:
            return _sse(
                "done",
                {
                    "question": question,
                    "sources": result.sources,
                    "grounded": result.grounded,
                    "refused": result.refused,
                    "truncated": truncated,
                    "pii_redacted": pii_found,
                    "output_pii_redacted": output_pii,
                    "trace_id": result.trace.trace_id,
                    "timing_ms": {s.name: round(s.duration_ms, 3) for s in result.trace.spans},
                },
            )

        def words(text: str) -> Iterator[str]:
            for word in text.split():
                yield _sse("token", {"text": word + " "})

        def event_stream() -> Iterator[str]:
            if prepared.result is not None:
                yield from words(prepared.result.answer)
                yield done(prepared.result)
                return
            if not req.use_llm:
                result = agent.finalize(prepared, agent.extractive_answer(prepared))
                output_pii = bool(detect_pii(result.answer))
                yield from words(redact_pii(result.answer))
                yield done(result, output_pii=output_pii)
                return
            parts: list[str] = []
            truncated = False
            streamed_model_text = False
            with prepared.trace.stage("generation"):
                try:
                    for piece in llm.stream_answer(question, prepared.chunks):
                        parts.append(piece)
                        streamed_model_text = True
                        yield _sse("token", {"text": piece})
                except llm.LLMUnavailableError:
                    if parts:
                        truncated = True  # the model died mid-answer
                    else:
                        fallback = agent.extractive_answer(prepared)
                        parts.append(fallback)
                        yield from words(redact_pii(fallback))
            text = "".join(parts)
            if prepared.deadline_fact and prepared.deadline_fact not in text:
                text = f"{text}\n\n{prepared.deadline_fact}"
            result = agent.finalize(prepared, text)
            output_pii = bool(detect_pii(result.answer))
            if result.retraction is not None:
                yield _sse("retracted", {"reason": result.retraction, "answer": result.answer})
            elif output_pii and streamed_model_text:
                # The client already rendered the raw tokens; replace them.
                yield _sse(
                    "retracted",
                    {"reason": "pii in output", "answer": redact_pii(result.answer)},
                )
            yield done(result, truncated=truncated, output_pii=output_pii)

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return app


def _sse(event: str, data: dict[str, object]) -> str:
    """Format one Server-Sent Event frame."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _version() -> str:
    from anchora import __version__

    return __version__


def _is_loopback(host: str) -> bool:
    """True if the peer address is loopback (127.0.0.0/8 or ::1)."""
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _corpus_root() -> Path:
    return Path(settings.corpus_root).resolve() if settings.corpus_root else _DEFAULT_CORPUS


def _resolve_inside_root(directory: str | None) -> Path:
    """Resolve a requested ingest directory, refusing anything outside the corpus root."""
    root = _corpus_root()
    if not directory:
        return root
    resolved = (root / directory).resolve()  # an absolute ``directory`` replaces ``root``
    if not resolved.is_relative_to(root):
        raise HTTPException(status_code=400, detail="directory must be inside the corpus root")
    if not resolved.is_dir():
        raise HTTPException(status_code=400, detail="not a directory inside the corpus root")
    return resolved


def _bounded_ingest(directory: Path, provider: str | None = None) -> VectorStore:
    return ingest_dir(
        directory,
        provider=provider,
        max_files=settings.ingest_max_files,
        max_file_bytes=settings.ingest_max_file_bytes,
    )


app = create_app()
