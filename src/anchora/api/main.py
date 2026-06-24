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

from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from anchora.agent import Agent
from anchora.config import settings
from anchora.guardrails import detect_pii, redact_pii
from anchora.ingest import ingest_dir
from anchora.store import VectorStore

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


def create_app() -> FastAPI:
    app = FastAPI(
        title="anchora",
        description="Legal-administrative RAG agent (local-first, with guardrails).",
        version=_version(),
    )
    app.state.store = None
    app.state.docs_indexed = 0

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
        agent = Agent(store, k=req.k, use_llm=req.use_llm)
        result = agent.run(question)
        return AskResponse(
            question=question,
            answer=result.answer,
            sources=result.sources,
            grounded=result.grounded,
            refused=result.refused,
            pii_redacted=pii_found,
            tool_calls=[ToolCallOut(name=tc.name, output=tc.output) for tc in result.tool_calls],
        )

    return app


def _version() -> str:
    from anchora import __version__

    return __version__


def _count_docs(directory: Path) -> int:
    return sum(1 for p in directory.rglob("*") if p.suffix.lower() in {".md", ".txt"})


app = create_app()
