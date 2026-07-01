"""Re-score frozen held-out generations deterministically — no GPU, no network.

``evaluate_finetune.py`` needs a local Transformers model (and, for the LoRA
rows, a PEFT adapter) to *generate* answers, so its numbers cannot be reproduced
from a clean checkout. This script closes that gap: it reads the real generations
frozen in ``data/eval/holdout-generations.json`` and re-scores them through the
same :func:`evaluate_finetune.score_case` the GPU path uses. Retrieval is the
deterministic ``hash`` provider, so the aggregates reproduce
``docs/finetuning-results.md`` exactly and run in CI for free.

With ``--check`` it compares each arm's re-scored metrics against the expected
values from the results doc and exits non-zero on any divergence beyond
``--tolerance`` — the honest regression gate.

Usage::

    uv run python scripts/score_generations.py            # print the table
    uv run python scripts/score_generations.py --check     # gate the numbers
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_finetune import GenerationReport, load_cases, score_case

from anchora.ingest import ingest_dir

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


def score_arm(answers: dict[str, str], store: Any) -> dict[str, float]:
    """Re-score one arm's frozen generations and return its aggregate metrics."""
    cases = load_cases(_HOLDOUT_PATH)
    scores = [
        score_case(answers[case["id"]], case, store) for case in cases if case["id"] in answers
    ]
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
    args = parser.parse_args(argv)

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
