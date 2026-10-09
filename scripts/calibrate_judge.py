"""Calibrate the deterministic lexical proxy against local LLM judges.

The CI gate uses a free, deterministic lexical proxy for faithfulness. That is
honest only if we know how well the proxy tracks a real judge. This script
measures it in two steps:

1. ``--run LABEL --model MODEL`` (needs Ollama) scores every answerable frozen
   held-out generation with an LLM judge at temperature 0 and freezes the
   per-item scores, the model digest and the prompt hash in
   ``data/eval/judge-scores.json``.
2. With no flags (or ``--check`` in CI) it recomputes every statistic from that
   frozen file, with no model and no network: proxy vs judge (Spearman with a
   question-clustered bootstrap interval, MAE, binary agreement with Wilson),
   judge vs judge (the ceiling any proxy can reach), and the proxy cutoff that
   best reproduces the judge's verdict.

Generations that abstain are excluded: faithfulness is undefined for a refusal
that makes no claim, and the proxy would score it low by construction.

Usage::

    uv run python scripts/calibrate_judge.py --run qwen3-32b --model qwen3:32b
    uv run python scripts/calibrate_judge.py            # report from frozen scores
    uv run python scripts/calibrate_judge.py --check    # fail on drift
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from anchora import metrics
from anchora.config import settings
from anchora.guardrails import is_abstention
from anchora.ingest import ingest_dir
from anchora.llm import build_context
from anchora.rag import retrieve
from anchora.stats import (
    cluster_bootstrap_interval,
    cohen_kappa,
    min_detectable_correlation,
    pearson,
    spearman,
    wilson_interval,
)

_ROOT = Path(__file__).resolve().parents[1]
_CORPUS_DIR = _ROOT / "data" / "corpus"
_HOLDOUT_PATH = _ROOT / "data" / "golden" / "holdout.json"
_GENERATIONS_PATH = _ROOT / "data" / "eval" / "holdout-generations.json"
SCORES_PATH = _ROOT / "data" / "eval" / "judge-scores.json"
_PROVIDER = "hash"
_RETRIEVAL_MODE = "dense"  # the mode the frozen generations were produced under
_K = 4

# A judge score >= this (on the 0..1 scale, i.e. rubric 3 or 4) counts as faithful.
JUDGE_FAITHFUL = 0.75
# Proxy cutoffs reported next to the derived one: 0.5 and the CI floor.
_REPORTED_CUTOFFS = (0.5, 0.7)

JudgeFn = Callable[[str, str], float | None]

JUDGE_PROMPT = (
    "You grade whether an ANSWER is faithful to a CONTEXT of legal passages. "
    "Faithful means every claim in the answer is supported by the context. The "
    "answer may be in English and the context in Portuguese: judge meaning, not "
    "wording. Any text after the answer (invented follow-up questions, unrelated "
    "continuations) counts as claims too. Citation markers like [1] are not claims.\n\n"
    "Score on this scale:\n"
    "4 = every claim is supported by the context\n"
    "3 = almost every claim is supported; one minor detail is not\n"
    "2 = roughly half of the claims are supported\n"
    "1 = most claims are unsupported\n"
    "0 = no claim is supported, or the answer contradicts the context\n\n"
    'Reply with JSON only: {{"score": <0-4>}}\n\n'
    "CONTEXT:\n{context}\n\nANSWER:\n{answer}"
)
_SCORE_SCHEMA = {
    "type": "object",
    "properties": {"score": {"type": "integer", "enum": [0, 1, 2, 3, 4]}},
    "required": ["score"],
}
_OPTIONS = {"temperature": 0.0, "seed": 0}

# Values the frozen judge scores must reproduce (see docs/eval-calibration.md).
# A divergence is a real change in the data or the math, not a number to edit.
_EXPECTED: dict[str, float] = {
    "qwen3-32b.n": 100.0,
    "qwen3-32b.judge_faithful": 74.0,
    "qwen3-32b.spearman": 0.4983,
    "qwen3-32b.mae": 0.2898,
    "qwen3-32b.agree@0.7": 66.0,
    "qwen3-32b.kappa@0.7": 0.357,
    "qwen3-32b.best_cutoff": 0.1579,
    "qwen3-32b.kappa_best": 0.4283,
    "gemma4-31b.n": 100.0,
    "gemma4-31b.judge_faithful": 65.0,
    "gemma4-31b.spearman": 0.4897,
    "gemma4-31b.mae": 0.2802,
    "gemma4-31b.agree@0.7": 71.0,
    "gemma4-31b.kappa@0.7": 0.4402,
    "gemma4-31b.best_cutoff": 1.0,
    "gemma4-31b.kappa_best": 0.4906,
    "qwen3-32b-repeat.n": 100.0,
    "qwen3-32b-repeat.judge_faithful": 74.0,
    "qwen3-32b-repeat.spearman": 0.4983,
    "qwen3-32b-repeat.mae": 0.2898,
    "qwen3-32b-repeat.agree@0.7": 66.0,
    "qwen3-32b-repeat.kappa@0.7": 0.357,
    "qwen3-32b-repeat.best_cutoff": 0.1579,
    "qwen3-32b-repeat.kappa_best": 0.4283,
    "qwen3-32b~gemma4-31b.kappa": 0.6028,
    "qwen3-32b~gemma4-31b.verdict_agree": 83.0,
    "qwen3-32b~qwen3-32b-repeat.kappa": 1.0,
    "qwen3-32b~qwen3-32b-repeat.verdict_agree": 100.0,
    "gemma4-31b~qwen3-32b-repeat.kappa": 0.6028,
    "gemma4-31b~qwen3-32b-repeat.verdict_agree": 83.0,
}


@dataclass(frozen=True)
class Item:
    """One answerable generation scored by the proxy (and later by a judge)."""

    item_id: str
    arm: str
    case_id: str
    answer: str
    context: str
    proxy: float


def prompt_sha256() -> str:
    return hashlib.sha256(JUDGE_PROMPT.encode("utf-8")).hexdigest()


def collect_items() -> tuple[list[Item], list[str]]:
    """Answerable generations with their retrieved context and proxy score.

    Returns the items and the ids excluded because the generation abstained.
    """
    fixture = json.loads(_GENERATIONS_PATH.read_text(encoding="utf-8"))
    cases = {c["id"]: c for c in json.loads(_HOLDOUT_PATH.read_text(encoding="utf-8"))["cases"]}
    store = ingest_dir(_CORPUS_DIR, provider=_PROVIDER)
    contexts: dict[str, str] = {}
    items: list[Item] = []
    excluded: list[str] = []
    for arm, spec in fixture["arms"].items():
        for case_id, answer in spec["generations"].items():
            case = cases.get(case_id)
            if case is None or not case["answerable"]:
                continue
            item_id = f"{arm}/{case_id}"
            if is_abstention(answer):
                excluded.append(item_id)
                continue
            if case_id not in contexts:
                chunks = retrieve(
                    store, case["question"], k=_K, provider=_PROVIDER, mode=_RETRIEVAL_MODE
                )
                contexts[case_id] = build_context(chunks)
            context = contexts[case_id]
            proxy = metrics.faithfulness(answer, context)
            items.append(Item(item_id, arm, case_id, answer, context, proxy))
    return items, excluded


def parse_score(raw: str) -> float | None:
    """Map the judge's ``{"score": 0..4}`` reply to [0, 1]; None if malformed."""
    try:
        value = json.loads(raw)["score"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 4:
        return None
    return value / 4.0


def ollama_judge(model: str, client: httpx.Client | None = None) -> JudgeFn:
    """A judge backed by a local Ollama model; returns None on any failure."""
    http = client or httpx.Client(timeout=max(settings.request_timeout, 300.0))

    def judge(answer: str, context: str) -> float | None:
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": JUDGE_PROMPT.format(context=context, answer=answer)}
            ],
            "stream": False,
            "think": False,
            "format": _SCORE_SCHEMA,
            "options": _OPTIONS,
        }
        try:
            response = http.post(f"{settings.ollama_base_url}/api/chat", json=payload)
            response.raise_for_status()
            return parse_score(str(response.json()["message"]["content"]))
        except (httpx.HTTPError, KeyError, ValueError):
            return None

    return judge


def model_metadata(model: str, client: httpx.Client | None = None) -> dict[str, str]:
    """Digest of the local model and the Ollama version, for provenance."""
    http = client or httpx.Client(timeout=30.0)
    base = settings.ollama_base_url
    tags = http.get(f"{base}/api/tags").json()["models"]
    digest = next((m["digest"] for m in tags if m["name"] == model), "unknown")
    version = str(http.get(f"{base}/api/version").json().get("version", "unknown"))
    return {"model": model, "digest": digest, "ollama_version": version}


def run_judge(items: list[Item], judge: JudgeFn) -> dict[str, float | None]:
    """Score every item; a failed call is recorded as None, never guessed."""
    return {item.item_id: judge(item.answer, item.context) for item in items}


def load_frozen(path: Path = SCORES_PATH) -> dict[str, Any]:
    if not path.exists():
        return {"items": {}, "runs": {}}
    return dict(json.loads(path.read_text(encoding="utf-8")))


def freeze_run(
    frozen: dict[str, Any],
    items: list[Item],
    excluded: list[str],
    label: str,
    meta: dict[str, str],
    scores: dict[str, float | None],
) -> dict[str, Any]:
    """Return a new frozen document with ``label`` added (inputs are not mutated)."""
    runs = dict(frozen.get("runs", {}))
    runs[label] = {
        **meta,
        "options": dict(_OPTIONS),
        "think": False,
        "prompt_sha256": prompt_sha256(),
        "created": datetime.now(UTC).strftime("%Y-%m-%d"),
        "scores": scores,
    }
    return {
        "_comment": (
            "Frozen LLM-judge faithfulness scores (0..1, from a 0-4 rubric) for every "
            "answerable, non-abstaining held-out generation. Re-scored offline by "
            "scripts/calibrate_judge.py --check; see docs/eval-calibration.md."
        ),
        "generations": str(_GENERATIONS_PATH.relative_to(_ROOT)),
        "retrieval": {"provider": _PROVIDER, "mode": _RETRIEVAL_MODE, "k": _K},
        "judge_faithful_at": JUDGE_FAITHFUL,
        "excluded_abstentions": sorted(excluded),
        "items": {i.item_id: {"arm": i.arm, "case_id": i.case_id, "proxy": i.proxy} for i in items},
        "runs": runs,
    }


def _paired(frozen: dict[str, Any], label: str) -> tuple[list[str], list[float], list[float]]:
    """Clusters (question ids), proxy and judge scores for the items the judge scored."""
    scores = frozen["runs"][label]["scores"]
    clusters: list[str] = []
    proxy: list[float] = []
    judge: list[float] = []
    for item_id, item in frozen["items"].items():
        value = scores.get(item_id)
        if value is None:
            continue
        clusters.append(item["case_id"])
        proxy.append(float(item["proxy"]))
        judge.append(float(value))
    return clusters, proxy, judge


def _agreement(proxy: list[float], judge: list[float], cutoff: float) -> int:
    return sum((p >= cutoff) == (j >= JUDGE_FAITHFUL) for p, j in zip(proxy, judge, strict=True))


def _kappa_at(proxy: Sequence[float], judge: Sequence[float], cutoff: float) -> float:
    value = cohen_kappa([p >= cutoff for p in proxy], [j >= JUDGE_FAITHFUL for j in judge])
    return -1.0 if value != value else value


def best_cutoff(proxy: Sequence[float], judge: Sequence[float]) -> tuple[float, float]:
    """Proxy cutoff with the highest Cohen kappa against the judge verdict.

    Kappa, not raw agreement: when most answers are faithful, raw agreement is
    maximised by a cutoff that calls almost everything faithful, which says
    nothing. Candidates are the observed proxy values; ties go to the cutoff
    closest to the current CI floor so an indifferent sample does not move it.
    """
    floor = settings.faithfulness_threshold
    return max(
        ((c, _kappa_at(proxy, judge, c)) for c in sorted(set(proxy))),
        key=lambda pair: (pair[1], -abs(pair[0] - floor)),
    )


def cutoff_interval(
    proxy: list[float], judge: list[float], clusters: list[str], resamples: int = 2000
) -> tuple[float, float]:
    """Question-clustered bootstrap interval of the kappa-optimal cutoff.

    If the interval is wide, the "best" cutoff is an artefact of this sample.
    """

    def statistic(xs: Sequence[float], ys: Sequence[float]) -> float:
        return best_cutoff(xs, ys)[0]

    return cluster_bootstrap_interval(proxy, judge, clusters, statistic, resamples=resamples)


def per_arm(frozen: dict[str, Any], label: str) -> list[tuple[str, int, float, float]]:
    """``(arm, n, mean proxy, mean judge)``: does the proxy rank systems like the judge?"""
    scores = frozen["runs"][label]["scores"]
    arms: dict[str, list[tuple[float, float]]] = {}
    for item_id, item in frozen["items"].items():
        if scores.get(item_id) is not None:
            arms.setdefault(item["arm"], []).append((float(item["proxy"]), float(scores[item_id])))
    return [
        (
            arm,
            len(pairs),
            sum(p for p, _ in pairs) / len(pairs),
            sum(j for _, j in pairs) / len(pairs),
        )
        for arm, pairs in arms.items()
    ]


def proxy_vs_judge(frozen: dict[str, Any], label: str) -> dict[str, Any]:
    """Agreement of the lexical proxy with one frozen judge run."""
    clusters, proxy, judge = _paired(frozen, label)
    n = len(proxy)
    cutoff, kappa_best = best_cutoff(proxy, judge)
    report: dict[str, Any] = {
        "n": n,
        "questions": len(set(clusters)),
        "unscored": len(frozen["items"]) - n,
        "spearman": spearman(proxy, judge),
        "spearman_ci": cluster_bootstrap_interval(proxy, judge, clusters, spearman),
        "pearson": pearson(proxy, judge),
        "mae": sum(abs(p - j) for p, j in zip(proxy, judge, strict=True)) / n,
        "mean_proxy": sum(proxy) / n,
        "mean_judge": sum(judge) / n,
        "judge_faithful": sum(j >= JUDGE_FAITHFUL for j in judge),
        "best_cutoff": cutoff,
        "best_cutoff_ci": cutoff_interval(proxy, judge, clusters),
        "kappa_best": kappa_best,
        "agree_best": _agreement(proxy, judge, cutoff),
    }
    for c in _REPORTED_CUTOFFS:
        report[f"agree@{c}"] = _agreement(proxy, judge, c)
        report[f"kappa@{c}"] = _kappa_at(proxy, judge, c)
    return report


def judge_vs_judge(frozen: dict[str, Any], a: str, b: str) -> dict[str, Any]:
    """Agreement between two judge runs on the items both scored."""
    sa, sb = frozen["runs"][a]["scores"], frozen["runs"][b]["scores"]
    ids = [i for i in frozen["items"] if sa.get(i) is not None and sb.get(i) is not None]
    xs = [float(sa[i]) for i in ids]
    ys = [float(sb[i]) for i in ids]
    clusters = [frozen["items"][i]["case_id"] for i in ids]
    verdict_a = [x >= JUDGE_FAITHFUL for x in xs]
    verdict_b = [y >= JUDGE_FAITHFUL for y in ys]
    return {
        "n": len(ids),
        "exact": sum(x == y for x, y in zip(xs, ys, strict=True)),
        "verdict_agree": sum(x == y for x, y in zip(verdict_a, verdict_b, strict=True)),
        "kappa": cohen_kappa(verdict_a, verdict_b),
        "spearman": spearman(xs, ys),
        "spearman_ci": cluster_bootstrap_interval(xs, ys, clusters, spearman),
    }


def _rate(k: int, n: int) -> str:
    low, high = wilson_interval(k, n)
    return f"{k}/{n} = {k / n:.3f} [{low:.2f}, {high:.2f}]"


def _ci(value: float, interval: tuple[float, float]) -> str:
    return f"{value:.3f} [{interval[0]:.2f}, {interval[1]:.2f}]"


def report_lines(frozen: dict[str, Any]) -> list[str]:
    """Markdown tables for docs/eval-calibration.md, from the frozen scores only."""
    runs = list(frozen["runs"])
    lines = [
        "| Judge run | n (questions) | Judge says faithful | Spearman [95% CI] | MAE "
        "| Proxy>=0.5: agrees, kappa | Proxy>=0.7: agrees, kappa "
        "| Kappa-best cutoff [95% CI]: agrees, kappa |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for label in runs:
        r = proxy_vs_judge(frozen, label)
        lines.append(
            f"| {label} | {r['n']} ({r['questions']}) | {_rate(r['judge_faithful'], r['n'])} "
            f"| {_ci(r['spearman'], r['spearman_ci'])} | {r['mae']:.3f} "
            f"| {_rate(r['agree@0.5'], r['n'])}, {r['kappa@0.5']:.2f} "
            f"| {_rate(r['agree@0.7'], r['n'])}, {r['kappa@0.7']:.2f} "
            f"| {_ci(r['best_cutoff'], r['best_cutoff_ci'])}: "
            f"{_rate(r['agree_best'], r['n'])}, {r['kappa_best']:.2f} |"
        )
    pairs = [(a, b) for i, a in enumerate(runs) for b in runs[i + 1 :]]
    if pairs:
        lines += [
            "",
            "| Judge pair | n | Same score | Same verdict | Cohen kappa | Spearman [95% CI] |",
            "|---|---|---|---|---|---|",
        ]
    for a, b in pairs:
        r = judge_vs_judge(frozen, a, b)
        lines.append(
            f"| {a} vs {b} | {r['n']} | {_rate(r['exact'], r['n'])} "
            f"| {_rate(r['verdict_agree'], r['n'])} | {r['kappa']:.3f} "
            f"| {_ci(r['spearman'], r['spearman_ci'])} |"
        )
    if runs:
        means = {label: {arm: j for arm, _, _, j in per_arm(frozen, label)} for label in runs}
        header = " | ".join(f"Mean judge, {label}" for label in runs)
        lines += [
            "",
            f"| Arm | n | Mean proxy | {header} |",
            "|---|---|---|" + "---|" * len(runs),
        ]
        for arm, count, proxy_mean, _ in per_arm(frozen, runs[0]):
            cells = " | ".join(f"{means[label][arm]:.3f}" for label in runs)
            lines.append(f"| {arm} | {count} | {proxy_mean:.3f} | {cells} |")
        n = proxy_vs_judge(frozen, runs[0])["n"]
        questions = len({item["case_id"] for item in frozen["items"].values()})
        lines += [
            "",
            f"Smallest correlation detectable (alpha 0.05, power 0.80): "
            f"{min_detectable_correlation(questions):.2f} with {questions} independent "
            f"questions, {min_detectable_correlation(n):.2f} if all {n} generations "
            "were independent.",
        ]
    return lines


def summary_numbers(frozen: dict[str, Any]) -> dict[str, float]:
    """The flat numbers ``--check`` pins against ``_EXPECTED``."""
    out: dict[str, float] = {}
    runs = list(frozen["runs"])
    for label in runs:
        r = proxy_vs_judge(frozen, label)
        keys = ("n", "judge_faithful", "spearman", "mae", "agree@0.7", "kappa@0.7")
        for key in (*keys, "best_cutoff", "kappa_best"):
            out[f"{label}.{key}"] = float(r[key])
    for i, a in enumerate(runs):
        for b in runs[i + 1 :]:
            r = judge_vs_judge(frozen, a, b)
            out[f"{a}~{b}.kappa"] = float(r["kappa"])
            out[f"{a}~{b}.verdict_agree"] = float(r["verdict_agree"])
    return out


def check(frozen: dict[str, Any], items: list[Item], tolerance: float = 1e-3) -> list[str]:
    """Failures where the frozen file or its statistics drift from expectations."""
    failures: list[str] = []
    frozen_items = frozen["items"]
    if set(frozen_items) != {i.item_id for i in items}:
        failures.append("frozen item set differs from the answerable, non-abstaining generations")
    for item in items:
        stored = frozen_items.get(item.item_id, {}).get("proxy")
        if stored is not None and abs(stored - item.proxy) > tolerance:
            failures.append(f"{item.item_id}: proxy {stored} frozen, {item.proxy} now")
    for label, run in frozen["runs"].items():
        if run.get("prompt_sha256") != prompt_sha256():
            failures.append(f"{label}: judge prompt changed since the run was frozen")
    actual = summary_numbers(frozen)
    for key, want in _EXPECTED.items():
        got = actual.get(key, float("nan"))
        if got != got or abs(got - want) > tolerance:
            failures.append(f"{key}: expected {want:.3f}, got {got:.3f}")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", metavar="LABEL", help="judge with Ollama and freeze as LABEL")
    parser.add_argument("--model", default=settings.gen_model, help="Ollama judge model")
    parser.add_argument("--check", action="store_true", help="fail if frozen stats drift")
    args = parser.parse_args(argv)

    items, excluded = collect_items()
    frozen = load_frozen()

    if args.run:
        try:
            meta = model_metadata(args.model)
        except httpx.HTTPError:
            print(f"Ollama unreachable at {settings.ollama_base_url}; nothing judged.")
            return 1
        scores = run_judge(items, ollama_judge(args.model))
        frozen = freeze_run(frozen, items, excluded, args.run, meta, scores)
        SCORES_PATH.write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
        failed = sum(v is None for v in scores.values())
        print(f"Froze {len(scores) - failed}/{len(scores)} judge scores as '{args.run}'.")

    if not frozen["runs"]:
        print("No frozen judge runs yet; run with --run LABEL --model MODEL (needs Ollama).")
        return 0

    print("\n".join(report_lines(frozen)))
    if not args.check:
        return 0
    failures = check(frozen, items)
    if failures:
        print("\nCALIBRATION CHECK FAILED:")
        for line in failures:
            print(f"  - {line}")
        return 1
    print("\nOK: frozen judge scores reproduce docs/eval-calibration.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
