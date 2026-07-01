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
