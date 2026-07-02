"""Latency benchmark for the offline agent pipeline — p50/p95 per stage.

Runs the agent over the golden questions with the deterministic ``hash``
provider and no LLM, so the shape of the pipeline (guardrails, domain check,
retrieval, extractive generation) is measured without a model in the loop.
Absolute numbers depend on the machine; the point is the *breakdown* and that
CI can assert an upper bound so a pathological regression (e.g. an accidental
O(n^2) in retrieval) fails the build.

Usage::

    uv run python scripts/benchmark.py               # human table
    uv run python scripts/benchmark.py --json        # machine-readable
    uv run python scripts/benchmark.py --max-p95-ms 250   # gate on end-to-end p95
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from anchora.agent import Agent
from anchora.evals import load_cases
from anchora.ingest import ingest_dir

_ROOT = Path(__file__).resolve().parents[1]
_CORPUS_DIR = _ROOT / "data" / "corpus"
_PROVIDER = "hash"
_K = 4


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile on a copy of ``values`` (0 <= pct <= 100)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, min(len(ordered), round(pct / 100.0 * len(ordered))))
    return round(ordered[rank - 1], 3)


def run(warmup: int = 1, repeats: int = 5) -> dict[str, object]:
    store = ingest_dir(_CORPUS_DIR, provider=_PROVIDER)
    agent = Agent(store, k=_K, provider=_PROVIDER, use_llm=False)
    questions = [case["question"] for case in load_cases()]

    for _ in range(warmup):  # prime caches (e.g. lazy BM25 build) before timing
        for question in questions:
            agent.run(question)

    totals: list[float] = []
    per_stage: dict[str, list[float]] = defaultdict(list)
    for _ in range(repeats):
        for question in questions:
            trace = agent.run(question).trace
            totals.append(trace.total_ms)
            for span in trace.spans:
                per_stage[span.name].append(span.duration_ms)

    return {
        "n_questions": len(questions),
        "repeats": repeats,
        "samples": len(totals),
        "end_to_end": {"p50_ms": _percentile(totals, 50), "p95_ms": _percentile(totals, 95)},
        "stages": {
            name: {"p50_ms": _percentile(times, 50), "p95_ms": _percentile(times, 95)}
            for name, times in sorted(per_stage.items())
        },
    }


def print_table(report: dict[str, object]) -> None:
    e2e = report["end_to_end"]
    assert isinstance(e2e, dict)
    print(f"samples: {report['samples']} ({report['n_questions']} questions x {report['repeats']})")
    print(f"{'stage':<20} {'p50 (ms)':>10} {'p95 (ms)':>10}")
    print("-" * 42)
    stages = report["stages"]
    assert isinstance(stages, dict)
    for name, pct in stages.items():
        print(f"{name:<20} {pct['p50_ms']:>10.3f} {pct['p95_ms']:>10.3f}")
    print("-" * 42)
    print(f"{'end-to-end':<20} {e2e['p50_ms']:>10.3f} {e2e['p95_ms']:>10.3f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--max-p95-ms",
        type=float,
        default=None,
        help="fail if end-to-end p95 exceeds this (regression gate)",
    )
    args = parser.parse_args(argv)

    report = run(repeats=args.repeats)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_table(report)

    if args.max_p95_ms is not None:
        e2e = report["end_to_end"]
        assert isinstance(e2e, dict)
        p95 = float(e2e["p95_ms"])
        if p95 > args.max_p95_ms:
            print(f"\nBENCH GATE FAILED: end-to-end p95 {p95:.3f}ms > {args.max_p95_ms:.3f}ms")
            return 1
        print(f"\nBENCH GATE PASSED: end-to-end p95 {p95:.3f}ms <= {args.max_p95_ms:.3f}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
