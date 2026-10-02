"""Offline regression tests for the frozen held-out eval — no GPU, no network.

The generation numbers in ``docs/finetuning-results.md`` were produced on a local
GPU (Apple MPS). To make them reproducible from a clean checkout, the real decoded
outputs are frozen in ``data/eval/holdout-generations.json`` and re-scored here
through the deterministic scorer. These tests fail if the frozen generations stop
reproducing the documented numbers, or if the promotion gate stops making the
documented decision — a real regression, never a number to be edited to fit.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

import gate_promotion as gp  # noqa: E402
import score_generations as sg  # noqa: E402

from anchora.registry import ModelRegistry  # noqa: E402
from anchora.stats import mcnemar_exact  # noqa: E402

_TOL = 0.01


def test_frozen_generations_reproduce_results_doc() -> None:
    """Re-scoring the frozen answers reproduces docs/finetuning-results.md."""
    results = sg.score_all()
    failures = sg.check(results, _TOL)
    assert not failures, "\n".join(failures)


def test_every_documented_arm_is_present() -> None:
    results = sg.score_all()
    assert set(sg._EXPECTED) <= set(results)


def test_gate_promotes_lora5_and_rejects_lora10(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "registry.json")
    log = gp.run_gate(sg.score_all(), registry)

    assert log[0].startswith("Promoted anchora-qa:v0.3-lora0")
    assert log[1].startswith("Promoted anchora-qa:v0.3-lora5")
    assert log[2].startswith("REJECTED anchora-qa:v0.3-lora10")

    prod = registry.current("anchora-qa", "prod")
    assert prod is not None
    assert prod.version == "v0.3-lora5"


def test_lora10_regresses_citation_accuracy(tmp_path: Path) -> None:
    """The gate rejects lora10 specifically on citation_accuracy, not by accident."""
    results = sg.score_all()
    assert results["lora10"]["citation_accuracy"] < results["lora5"]["citation_accuracy"]


def test_lora5_beats_base_fewshot_on_citation_accuracy() -> None:
    results = sg.score_all()
    assert results["lora5"]["citation_accuracy"] > results["base_fewshot"]["citation_accuracy"]


def test_report_counts_match_documented_fractions() -> None:
    """The documented rates are these exact counts: 18/22, 5/6, 11/22, 1/6, 14/22."""
    per_case = sg.score_all_cases()
    lines = "\n".join(sg.report_lines(per_case))
    assert "| lora5 | 18/22 = 0.818 [0.61, 0.93] | 5/6 = 0.833 [0.44, 0.97]" in lines
    assert "| base_fewshot | 11/22 = 0.500 [0.31, 0.69] | 1/6 = 0.167 [0.03, 0.56]" in lines
    assert "| lora10 | 14/22 = 0.636 [0.43, 0.80]" in lines


def test_paired_comparisons_match_the_documented_read() -> None:
    """Pin the paired evidence the README states, so the prose cannot drift from it."""
    per_case = sg.score_all_cases()

    def p_value(a: str, b: str, field: str, answerable: bool) -> float:
        only_a, only_b = sg.discordant_pairs(per_case[a], per_case[b], field, answerable)
        return mcnemar_exact(only_a, only_b)

    # The only nominally significant difference: lora5 vs base+few-shot citation (7 vs 0).
    assert sg.discordant_pairs(
        per_case["lora5"], per_case["base_fewshot"], "citation_correct", True
    ) == (7, 0)
    assert p_value("lora5", "base_fewshot", "citation_correct", True) < 0.05
    # Abstention on 6 questions and the gate's lora10 rejection are within noise.
    assert p_value("lora5", "base_fewshot", "abstained", False) > 0.05
    assert p_value("lora5", "lora10", "citation_correct", True) > 0.05
