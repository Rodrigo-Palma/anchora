"""Unit tests for the small-sample uncertainty helpers."""

from __future__ import annotations

import pytest

from anchora.stats import (
    bootstrap_mean_interval,
    format_mean,
    format_proportion,
    mcnemar_exact,
    wilson_interval,
)


@pytest.mark.parametrize(
    ("k", "n", "low", "high"),
    [
        # Reference values from statsmodels proportion_confint(method="wilson").
        (5, 6, 0.4365, 0.9699),
        (1, 6, 0.0301, 0.5635),
        (18, 22, 0.6148, 0.9269),
        (11, 22, 0.3072, 0.6928),
        (0, 10, 0.0, 0.2775),
        (44, 44, 0.9197, 1.0),
    ],
)
def test_wilson_matches_reference_values(k: int, n: int, low: float, high: float) -> None:
    got_low, got_high = wilson_interval(k, n)
    assert got_low == pytest.approx(low, abs=1e-4)
    assert got_high == pytest.approx(high, abs=1e-4)


def test_wilson_interval_contains_point_estimate() -> None:
    for n in range(1, 30):
        for k in range(n + 1):
            low, high = wilson_interval(k, n)
            assert 0.0 <= low <= k / n <= high <= 1.0


@pytest.mark.parametrize(("k", "n"), [(1, 0), (-1, 5), (6, 5)])
def test_wilson_rejects_invalid_counts(k: int, n: int) -> None:
    with pytest.raises(ValueError):
        wilson_interval(k, n)


@pytest.mark.parametrize(
    ("only_a", "only_b", "p"),
    [
        (7, 0, 0.015625),  # 2 * 0.5**7
        (4, 0, 0.125),  # 2 * 0.5**4
        (5, 1, 0.21875),  # 2 * (1 + 6) / 64
        (1, 0, 1.0),
        (0, 0, 1.0),
        (3, 3, 1.0),
    ],
)
def test_mcnemar_exact_known_values(only_a: int, only_b: int, p: float) -> None:
    assert mcnemar_exact(only_a, only_b) == pytest.approx(p)


def test_mcnemar_is_symmetric() -> None:
    assert mcnemar_exact(9, 2) == mcnemar_exact(2, 9)


def test_mcnemar_rejects_negative_counts() -> None:
    with pytest.raises(ValueError):
        mcnemar_exact(-1, 2)


def test_bootstrap_is_seeded_and_brackets_the_mean() -> None:
    values = [0.1, 0.9, 0.4, 0.7, 0.3, 0.8, 0.5]
    first = bootstrap_mean_interval(values, resamples=2000)
    assert first == bootstrap_mean_interval(values, resamples=2000)
    mean = sum(values) / len(values)
    assert first[0] <= mean <= first[1]


def test_bootstrap_of_constant_values_is_degenerate() -> None:
    assert bootstrap_mean_interval([0.5] * 8, resamples=500) == (0.5, 0.5)


@pytest.mark.parametrize("kwargs", [{"level": 1.0}, {"level": 0.0}, {"resamples": 0}])
def test_bootstrap_rejects_bad_arguments(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        bootstrap_mean_interval([0.1, 0.2], **kwargs)  # type: ignore[arg-type]


def test_bootstrap_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        bootstrap_mean_interval([])


def test_format_helpers() -> None:
    assert format_proportion(5, 6) == "5/6 = 0.833 [0.44, 0.97]"
    assert format_mean([0.5, 0.5]).startswith("0.500 [0.50, 0.50] (n=2)")
