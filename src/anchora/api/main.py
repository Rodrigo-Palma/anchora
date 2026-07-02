"""FastAPI app serving the anchora RAG agent.

Endpoints:

* ``GET  /health``  — liveness + store status;
* ``POST /ingest``  — (re)build the in-memory store from a corpus directory;
* ``POST /ask``     — answer a question with the tool-using agent.

Cross-cutting concerns handled here:

* optional API-key gate (``settings.api_key``; empty = open dev mode);
* PII redaction on every inbound question before it touches the model/logs;
* the agent's own input/output guardrails (injection block, grounding).

The store is held in app state so it survives across requests; it is built
lazily from the default corpus on first use if ``/ingest`` was never called.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from anchora.agent import Agent
from anchora.config import settings
from anchora.guardrails import detect_pii, redact_pii
from anchora.ingest import ingest_dir
from anchora.store import VectorStore

_REQUEST_ID_HEADER = "x-request-id"

_DEFAULT_CORPUS = Path(__file__).resolve().parents[3] / "data" / "corpus"


class IngestRequest(BaseModel):
    directory: str | None = Field(
        default=None,
        description="Corpus directory to ingest; defaults to the bundled data/corpus.",
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

    def require_api_key(x_api_key: Annotated[str | None, Header()] = None) -> None:
        if settings.api_key and x_api_key != settings.api_key:
            raise HTTPException(status_code=401, detail="invalid or missing API key")

    def get_store() -> VectorStore:
        if app.state.store is None:
            app.state.store = ingest_dir(_DEFAULT_CORPUS)
            app.state.docs_indexed = _count_docs(_DEFAULT_CORPUS)
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
        directory = Path(req.directory) if req.directory else _DEFAULT_CORPUS
        if not directory.is_dir():
            raise HTTPException(status_code=400, detail=f"not a directory: {directory}")
        store = ingest_dir(directory, provider=req.provider)
        app.state.store = store
        app.state.docs_indexed = _count_docs(directory)
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
        return AskResponse(
            question=question,
            answer=result.answer,
            sources=result.sources,
            grounded=result.grounded,
            refused=result.refused,
            pii_redacted=pii_found,
            tool_calls=[ToolCallOut(name=tc.name, output=tc.output) for tc in result.tool_calls],
            trace_id=result.trace.trace_id,
            timing_ms={span.name: round(span.duration_ms, 3) for span in result.trace.spans},
        )

    @app.post("/ask/stream", dependencies=[Depends(require_api_key)])
    def ask_stream(req: AskRequest) -> StreamingResponse:
        """Answer via Server-Sent Events: incremental ``token`` events then ``done``.

        The answer is composed first (guardrails, retrieval, grounding all run)
        and then delivered incrementally word-by-word, so the SSE contract —
        framing, progressive delivery, and a terminal event carrying sources,
        grounding and the trace — is exercised deterministically offline. With
        the local model the same framing carries the model's tokens.
        """
        pii_found = bool(detect_pii(req.question))
        question = redact_pii(req.question) if pii_found else req.question
        store = get_store()
        agent = Agent(store, k=req.k, provider=req.provider, use_llm=req.use_llm)
        result = agent.run(question)

        def event_stream() -> Iterator[str]:
            for word in result.answer.split():
                yield _sse("token", {"text": word + " "})
            yield _sse(
                "done",
                {
                    "question": question,
                    "sources": result.sources,
                    "grounded": result.grounded,
                    "refused": result.refused,
                    "pii_redacted": pii_found,
                    "trace_id": result.trace.trace_id,
                    "timing_ms": {s.name: round(s.duration_ms, 3) for s in result.trace.spans},
                },
            )

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return app


def _sse(event: str, data: dict[str, object]) -> str:
    """Format one Server-Sent Event frame."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _version() -> str:
    from anchora import __version__

    return __version__


def _count_docs(directory: Path) -> int:
    return sum(1 for p in directory.rglob("*") if p.suffix.lower() in {".md", ".txt"})


app = create_app()
