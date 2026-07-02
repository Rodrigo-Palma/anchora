"""Offline tests for the judge-calibration math and harness.

The LLM judge itself needs Ollama, but the correlation statistics and the
harness plumbing must be correct regardless — so they are tested here with an
injected deterministic judge, no model or network.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

import calibrate_judge as cj  # noqa: E402


def test_pearson_perfect_positive() -> None:
    assert cj._pearson([0.0, 0.5, 1.0], [0.0, 0.5, 1.0]) == 1.0


def test_pearson_perfect_negative() -> None:
    assert cj._pearson([0.0, 0.5, 1.0], [1.0, 0.5, 0.0]) == -1.0


def test_spearman_handles_ties() -> None:
    # monotonic but non-linear → Spearman 1.0 even where Pearson is not
    assert cj._spearman([1.0, 2.0, 3.0, 4.0], [1.0, 4.0, 9.0, 16.0]) == 1.0


def test_correlate_reports_expected_fields() -> None:
    report = cj.correlate([0.2, 0.8, 0.9], [0.1, 0.7, 1.0])
    assert report["n"] == 3
    assert 0.0 <= report["binary_agreement@0.5"] <= 1.0
    assert report["mae"] >= 0.0


def test_correlate_empty() -> None:
    assert cj.correlate([], []) == {"n": 0}


def test_parse_score_extracts_and_clamps() -> None:
    assert cj._parse_score("0.8") == 0.8
    assert cj._parse_score("Score: 0.42 (mostly supported)") == 0.42
    assert cj._parse_score("1.5") == 1.0
    assert cj._parse_score("no number here") is None


def test_calibrate_with_stub_judge_runs_offline() -> None:
    """A deterministic stub judge exercises the full harness with no network."""
    report = cj.calibrate(lambda answer, context: 0.75)
    assert report["n"] > 0
    assert report["mean_judge"] == 0.75
    assert report["skipped"] == 0


def test_calibrate_skips_unscored_cases() -> None:
    """A judge returning None (outage) shrinks the sample, never poisons it."""
    report = cj.calibrate(lambda answer, context: None)
    assert report["n"] == 0
    assert report["skipped"] > 0
