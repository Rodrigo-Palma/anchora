"""Tools the agent can call.

Each tool is a small, deterministic, independently testable function wrapped in
a :class:`Tool`. The retrieval tool is built from a live vector store; the date
tools are pure so the agent stays testable without a model or network.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta

from anchora.rag import retrieve
from anchora.store import Chunk, VectorStore


@dataclass
class Tool:
    name: str
    description: str
    run: Callable[..., str]


@dataclass
class SearchResult:
    chunks: list[Chunk]


def make_search_tool(store: VectorStore, k: int = 4, provider: str | None = None) -> Tool:
    """A retrieval tool bound to ``store``; returns a short text summary."""

    def _run(query: str) -> str:
        chunks = retrieve(store, query, k=k, provider=provider)
        if not chunks:
            return "No relevant document found."
        titles = sorted({chunk.title or chunk.doc_id for chunk in chunks})
        return f"{len(chunks)} excerpts retrieved from: {', '.join(titles)}."

    return Tool(
        name="search_documents",
        description="Searches for relevant excerpts in the legal/administrative corpus.",
        run=_run,
    )


def parse_iso_date(value: str) -> date:
    """Parse a ``YYYY-MM-DD`` string into a date (raises ValueError if invalid)."""
    return date.fromisoformat(value.strip())


def add_business_days(start: date, days: int) -> date:
    """Return the date ``days`` business days after ``start`` (weekends skipped).

    Holidays are not modelled; this is a planning aid, not legal advice.
    """
    if days < 0:
        raise ValueError("days must be >= 0")
    current = start
    remaining = days
    while remaining > 0:
        current += timedelta(days=1)
        if current.weekday() < 5:  # Mon-Fri
            remaining -= 1
    return current


def legal_deadline(start_date: str, days: int, kind: str = "business") -> str:
    """Compute a deadline ``days`` after ``start_date``.

    ``kind="business"`` counts business days (CPC default for procedural deadlines);
    ``kind="calendar"`` counts calendar days.
    """
    start = parse_iso_date(start_date)
    due = start + timedelta(days=days) if kind == "calendar" else add_business_days(start, days)
    label = "calendar days" if kind == "calendar" else "business days"
    return f"Deadline of {days} {label} from {start.isoformat()}: due on {due.isoformat()}."


def make_deadline_tool() -> Tool:
    def _run(start_date: str, days: int, kind: str = "business") -> str:
        return legal_deadline(start_date, days, kind)

    return Tool(
        name="legal_deadline",
        description="Computes a deadline (business or calendar days) from a date.",
        run=_run,
    )
