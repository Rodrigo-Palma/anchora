"""Calibrate the deterministic lexical proxy against an LLM judge.

The CI gate uses a free, deterministic lexical proxy for faithfulness. That is
honest only if we know *how well the proxy tracks a real judge* — otherwise the
green build is measuring the wrong thing. This script quantifies that: over the
frozen held-out generations, it scores each answer with both the lexical proxy
(``anchora.metrics.faithfulness``) and an LLM judge (Ollama), then reports their
agreement (Pearson, Spearman, mean absolute error, binary agreement at 0.5).

The judge needs a local model, so this is a *local analysis tool, not a CI
gate*: with no Ollama reachable it explains that and exits 0. The correlation
math is dependency-free and unit-tested with an injected deterministic judge
(``tests/test_calibration.py``), so the methodology itself is covered offline.

Blind spots the proxy is known to have (documented in docs/eval-calibration.md):
negation, paraphrase without lexical overlap, and numeric correctness. The point
of calibration is to keep those honest, not to hide them.

Usage::

    uv run python scripts/calibrate_judge.py            # needs Ollama
    uv run python scripts/calibrate_judge.py --json
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable
from pathlib import Path

import httpx

from anchora import metrics
from anchora.config import settings
from anchora.ingest import ingest_dir
from anchora.llm import build_context
from anchora.rag import retrieve

_ROOT = Path(__file__).resolve().parents[1]
_CORPUS_DIR = _ROOT / "data" / "corpus"
_HOLDOUT_PATH = _ROOT / "data" / "golden" / "holdout.json"
_GENERATIONS_PATH = _ROOT / "data" / "eval" / "holdout-generations.json"
_PROVIDER = "hash"
_RETRIEVAL_MODE = "dense"  # match the mode the frozen generations were produced under

JudgeFn = Callable[[str, str], float | None]

_JUDGE_PROMPT = (
    "You are grading whether an ANSWER is faithful to the CONTEXT — i.e. every "
    "claim in the answer is supported by the context. Reply with ONLY a number "
    "between 0.0 (unsupported) and 1.0 (fully supported).\n\n"
    "CONTEXT:\n{context}\n\nANSWER:\n{answer}\n\nScore:"
)


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    vx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    vy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if vx == 0.0 or vy == 0.0:
        return float("nan")
    return round(cov / (vx * vy), 4)


def _ranks(values: list[float]) -> list[float]:
    """Average (fractional) ranks, so ties do not distort Spearman."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    return _pearson(_ranks(xs), _ranks(ys))


def correlate(proxy: list[float], judge: list[float]) -> dict[str, float]:
    """Agreement metrics between two score vectors (same length)."""
    n = len(proxy)
    if n == 0:
        return {"n": 0}
    mae = sum(abs(p - j) for p, j in zip(proxy, judge, strict=True)) / n
    agree = sum(1 for p, j in zip(proxy, judge, strict=True) if (p >= 0.5) == (j >= 0.5)) / n
    return {
        "n": n,
        "pearson": _pearson(proxy, judge),
        "spearman": _spearman(proxy, judge),
        "mae": round(mae, 4),
        "binary_agreement@0.5": round(agree, 4),
        "mean_proxy": round(sum(proxy) / n, 4),
        "mean_judge": round(sum(judge) / n, 4),
    }


def ollama_judge(answer: str, context: str) -> float | None:
    """Ask the local model for a 0..1 faithfulness score; None if unreachable."""
    prompt = _JUDGE_PROMPT.format(context=context, answer=answer)
    try:
        response = httpx.post(
            f"{settings.ollama_base_url}/api/generate",
            json={"model": settings.gen_model, "prompt": prompt, "stream": False},
            timeout=settings.request_timeout,
        )
        response.raise_for_status()
        raw = str(response.json()["response"])
    except (httpx.HTTPError, KeyError, ValueError):
        return None
    return _parse_score(raw)


def _parse_score(raw: str) -> float | None:
    import re

    match = re.search(r"\d(?:\.\d+)?", raw)
    if not match:
        return None
    return max(0.0, min(1.0, float(match.group())))


def calibrate(judge_fn: JudgeFn) -> dict[str, object]:
    """Score frozen generations with proxy and judge; return the agreement report.

    Skips any case the judge could not score (``None``) so a partial Ollama
    outage degrades sample size instead of poisoning the correlation.
    """
    fixture = json.loads(_GENERATIONS_PATH.read_text(encoding="utf-8"))
    cases = {c["id"]: c for c in json.loads(_HOLDOUT_PATH.read_text(encoding="utf-8"))["cases"]}
    store = ingest_dir(_CORPUS_DIR, provider=_PROVIDER)

    proxy_scores: list[float] = []
    judge_scores: list[float] = []
    skipped = 0
    for arm in fixture["arms"].values():
        for case_id, answer in arm["generations"].items():
            case = cases.get(case_id)
            if case is None:
                continue
            context = build_context(
                retrieve(store, case["question"], k=4, provider=_PROVIDER, mode=_RETRIEVAL_MODE)
            )
            judged = judge_fn(answer, context)
            if judged is None:
                skipped += 1
                continue
            proxy_scores.append(metrics.faithfulness(answer, context))
            judge_scores.append(judged)

    report: dict[str, object] = dict(correlate(proxy_scores, judge_scores))
    report["skipped"] = skipped
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    report = calibrate(ollama_judge)
    if report.get("n", 0) == 0:
        print(
            "No cases scored — the LLM judge needs a reachable Ollama "
            f"({settings.ollama_base_url}, model {settings.gen_model}).\n"
            "This is a local analysis tool, not a CI gate; the correlation math "
            "is unit-tested offline in tests/test_calibration.py."
        )
        return 0

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("Proxy vs. LLM-judge faithfulness (frozen held-out generations)")
        print("-" * 56)
        for key, value in report.items():
            print(f"  {key:<22} {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
