"""Re-score frozen held-out generations deterministically — no GPU, no network.

``evaluate_finetune.py`` needs a local Transformers model (and, for the LoRA
rows, a PEFT adapter) to *generate* answers, so its numbers cannot be reproduced
from a clean checkout. This script closes that gap: it reads the real generations
frozen in ``data/eval/holdout-generations.json`` and re-scores them through the
same :func:`evaluate_finetune.score_case` the GPU path uses. Retrieval is the
deterministic ``hash`` provider, so the aggregates reproduce
``docs/finetuning-results.md`` exactly and run in CI for free.

With ``--report`` it prints every number with its sample size and uncertainty:
``k/n`` plus the 95% Wilson interval for rates, a seeded bootstrap interval for
mean proxy scores, and the exact paired McNemar test for each two-arm comparison
that the docs make (both arms answer the same questions, so the test is paired).

With ``--check`` it compares each arm's re-scored metrics against the expected
values from the results doc and exits non-zero on any divergence beyond
``--tolerance`` — the honest regression gate.

Usage::

    uv run python scripts/score_generations.py            # print the table
    uv run python scripts/score_generations.py --check     # gate the numbers
    uv run python scripts/score_generations.py --report    # counts, CIs, McNemar
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_finetune import GenerationReport, GenerationScore, load_cases, score_case

from anchora.ingest import ingest_dir
from anchora.stats import format_mean, format_proportion, mcnemar_exact

_ROOT = Path(__file__).resolve().parents[1]
_CORPUS_DIR = _ROOT / "data" / "corpus"
_HOLDOUT_PATH = _ROOT / "data" / "golden" / "holdout.json"
_GENERATIONS_PATH = _ROOT / "data" / "eval" / "holdout-generations.json"
_PROVIDER = "hash"

# Expected held-out metrics per arm, from docs/finetuning-results.md (the honest,
# citation-correct + PT-aware-abstention numbers). These are the values the frozen
# generations must reproduce when re-scored; a mismatch is a real regression, not a
# number to be edited to fit.
_EXPECTED: dict[str, dict[str, float]] = {
    "base_fewshot": {"citation_accuracy": 0.500, "abstention_rate": 0.167},
    "lora0": {"citation_accuracy": 0.773, "faithfulness": 0.789, "reference_overlap": 0.519},
    "lora5": {
        "citation_accuracy": 0.818,
        "abstention_rate": 0.833,
        "faithfulness": 0.726,
        "reference_overlap": 0.457,
    },
    "lora10": {"citation_accuracy": 0.636, "abstention_rate": 0.833},
}

_COLUMNS = (
    ("citation_accuracy", "citation-correct"),
    ("abstention_rate", "abstention(PT)"),
    ("faithfulness", "faithfulness"),
    ("reference_overlap", "ref-overlap"),
)

# Two-arm comparisons the docs make, as (arm_a, arm_b, per-case field, answerable).
# ``answerable=True`` compares on the 22 in-corpus questions, ``False`` on the 6
# out-of-corpus ones.
_COMPARISONS: tuple[tuple[str, str, str, bool], ...] = (
    ("lora5", "base_fewshot", "citation_correct", True),
    ("lora5", "base_fewshot", "abstained", False),
    ("lora5", "lora0", "citation_correct", True),
    ("lora5", "lora0", "abstained", False),
    ("lora5", "lora10", "citation_correct", True),
    ("lora0", "base_fewshot", "citation_correct", True),
)


def score_cases(answers: dict[str, str], store: Any) -> list[GenerationScore]:
    """Re-score one arm's frozen generations case by case, in holdout order."""
    cases = load_cases(_HOLDOUT_PATH)
    return [score_case(answers[case["id"]], case, store) for case in cases if case["id"] in answers]


def score_arm(answers: dict[str, str], store: Any) -> dict[str, float]:
    """Re-score one arm's frozen generations and return its aggregate metrics."""
    scores = score_cases(answers, store)
    report = GenerationReport(
        name="frozen", base_model="frozen", adapter_path=None, few_shot=False, scores=scores
    )
    return {
        "citation_accuracy": report.citation_accuracy,
        "abstention_rate": report.abstention_rate,
        "faithfulness": report.mean_faithfulness,
        "reference_overlap": report.mean_reference_overlap,
    }


def score_all() -> dict[str, dict[str, float]]:
    """Re-score every arm in the frozen fixture."""
    fixture = json.loads(_GENERATIONS_PATH.read_text(encoding="utf-8"))
    store = ingest_dir(_CORPUS_DIR, provider=_PROVIDER)
    return {arm: score_arm(spec["generations"], store) for arm, spec in fixture["arms"].items()}


def score_all_cases() -> dict[str, list[GenerationScore]]:
    """Per-case scores for every arm, for counts and paired tests."""
    fixture = json.loads(_GENERATIONS_PATH.read_text(encoding="utf-8"))
    store = ingest_dir(_CORPUS_DIR, provider=_PROVIDER)
    return {arm: score_cases(spec["generations"], store) for arm, spec in fixture["arms"].items()}


def discordant_pairs(
    a: list[GenerationScore], b: list[GenerationScore], field: str, answerable: bool
) -> tuple[int, int]:
    """Count questions only ``a`` got right and only ``b`` got right on ``field``."""
    by_id = {s.case_id: s for s in b}
    only_a = only_b = 0
    for score_a in a:
        if score_a.answerable != answerable:
            continue
        hit_a = bool(getattr(score_a, field))
        hit_b = bool(getattr(by_id[score_a.case_id], field))
        only_a += hit_a and not hit_b
        only_b += hit_b and not hit_a
    return only_a, only_b


def _count(scores: list[GenerationScore], field: str, answerable: bool) -> tuple[int, int]:
    subset = [s for s in scores if s.answerable == answerable]
    return sum(bool(getattr(s, field)) for s in subset), len(subset)


def report_lines(per_case: dict[str, list[GenerationScore]]) -> list[str]:
    """Markdown report: every rate as k/n with a Wilson CI, plus paired McNemar."""
    lines = [
        "| Arm | Citation-correct | Abstention (PT-aware) | Grounded (cites or abstains) "
        "| Faithfulness (mean) | Ref. overlap (mean) |",
        "|---|---|---|---|---|---|",
    ]
    for arm, scores in per_case.items():
        answerable = [s for s in scores if s.answerable]
        cells = (
            format_proportion(*_count(scores, "citation_correct", True)),
            format_proportion(*_count(scores, "abstained", False)),
            format_proportion(*_count(scores, "grounded", True)),
            format_mean([s.faithfulness for s in answerable]),
            format_mean([s.reference_overlap for s in answerable]),
        )
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    m = len(_COMPARISONS)
    lines += [
        "",
        f"| Comparison | Metric | n | Only A right | Only B right | McNemar exact p "
        f"| Bonferroni p (x{m}) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for arm_a, arm_b, field, answerable_only in _COMPARISONS:
        only_a, only_b = discordant_pairs(per_case[arm_a], per_case[arm_b], field, answerable_only)
        n = _count(per_case[arm_a], field, answerable_only)[1]
        p = mcnemar_exact(only_a, only_b)
        lines.append(
            f"| {arm_a} vs {arm_b} | {field} | {n} | {only_a} | {only_b} | {p:.3f} "
            f"| {min(1.0, p * m):.3f} |"
        )
    return lines


def _fmt(value: float) -> str:
    return "  nan " if value != value else f"{value:6.3f}"


def print_table(results: dict[str, dict[str, float]]) -> None:
    header = f"{'arm':<16}" + "".join(f"{label:>16}" for _, label in _COLUMNS)
    print(header)
    print("-" * len(header))
    for arm, metrics_ in results.items():
        row = f"{arm:<16}" + "".join(f"{_fmt(metrics_[key]):>16}" for key, _ in _COLUMNS)
        print(row)


def check(results: dict[str, dict[str, float]], tolerance: float) -> list[str]:
    """Return human-readable failures where a re-scored metric drifts from expected."""
    failures: list[str] = []
    for arm, expected in _EXPECTED.items():
        actual = results.get(arm, {})
        for key, want in expected.items():
            got = actual.get(key, float("nan"))
            if got != got or abs(got - want) > tolerance:
                failures.append(f"{arm}.{key}: expected {want:.3f}, got {got:.3f}")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if metrics drift from expected")
    parser.add_argument("--tolerance", type=float, default=0.01)
    parser.add_argument(
        "--report", action="store_true", help="print k/n, 95%% CIs and paired McNemar tests"
    )
    args = parser.parse_args(argv)

    if args.report:
        print("\n".join(report_lines(score_all_cases())))
        return 0

    results = score_all()
    print_table(results)

    if not args.check:
        return 0

    failures = check(results, args.tolerance)
    if failures:
        print("\nHONEST-EVAL FAILED — frozen generations no longer reproduce the results doc:")
        for line in failures:
            print(f"  - {line}")
        return 1
    print(f"\nOK — all arms reproduce docs/finetuning-results.md within {args.tolerance}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
