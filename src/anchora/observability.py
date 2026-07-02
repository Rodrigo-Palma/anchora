"""Lightweight, dependency-free tracing for the agent pipeline.

A ``Trace`` records the wall-clock duration of each pipeline stage (guardrails,
retrieval, generation) under a single ``trace_id``, so a slow answer can be
attributed to a stage instead of guessed at. No OpenTelemetry dependency: the
same span data serializes to a plain dict for the API and to a one-line log.

Timings are wall-clock and therefore not asserted for value in tests — only
that every stage is recorded. The benchmark (``scripts/benchmark.py``) is where
the durations are aggregated into p50/p95.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class Span:
    name: str
    duration_ms: float


@dataclass
class Trace:
    """Collects stage spans under one id. Use :meth:`stage` around each step."""

    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    spans: list[Span] = field(default_factory=list)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.monotonic()
        try:
            yield
        finally:
            self.spans.append(Span(name=name, duration_ms=(time.monotonic() - start) * 1000.0))

    @property
    def total_ms(self) -> float:
        return round(sum(span.duration_ms for span in self.spans), 3)

    def as_dict(self) -> dict[str, object]:
        return {
            "trace_id": self.trace_id,
            "total_ms": self.total_ms,
            "stages": {span.name: round(span.duration_ms, 3) for span in self.spans},
        }
