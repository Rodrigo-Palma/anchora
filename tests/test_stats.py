"""Unit tests for the small-sample uncertainty helpers."""

from __future__ import annotations

import pytest

from anchora.stats import (
    bootstrap_mean_interval,
    cluster_bootstrap_interval,
    cohen_kappa,
    format_mean,
    format_proportion,
    mcnemar_exact,
    min_detectable_correlation,
    pearson,
    spearman,
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


# --- agreement statistics (judge calibration) --------------------------------


def test_pearson_perfect_and_degenerate() -> None:
    assert pearson([0.0, 0.5, 1.0], [0.0, 0.5, 1.0]) == pytest.approx(1.0)
    assert pearson([0.0, 0.5, 1.0], [1.0, 0.5, 0.0]) == pytest.approx(-1.0)
    constant = pearson([1.0, 1.0, 1.0], [0.0, 0.5, 1.0])
    assert constant != constant  # NaN: correlation is undefined for a constant vector


def test_spearman_is_rank_based_and_handles_ties() -> None:
    assert spearman([1.0, 2.0, 3.0, 4.0], [1.0, 4.0, 9.0, 16.0]) == pytest.approx(1.0)
    # scipy.stats.spearmanr([1, 2, 2, 3], [1, 3, 2, 4]).statistic == 0.9487
    assert spearman([1.0, 2.0, 2.0, 3.0], [1.0, 3.0, 2.0, 4.0]) == pytest.approx(0.9487, abs=1e-4)


def test_cohen_kappa_reference_values() -> None:
    assert cohen_kappa([True, False, True, False], [True, False, True, False]) == 1.0
    # 2x2 table a=20 b=5 c=10 d=15 -> po=0.70 pe=0.50 -> kappa=0.40
    a = [True] * 25 + [False] * 25
    b = [True] * 20 + [False] * 5 + [True] * 10 + [False] * 15
    assert cohen_kappa(a, b) == pytest.approx(0.4)


def test_cohen_kappa_is_nan_when_chance_agreement_is_total() -> None:
    value = cohen_kappa([True, True], [True, True])
    assert value != value


def test_min_detectable_correlation_shrinks_with_n() -> None:
    # Fisher-z approximation, alpha=0.05 two-sided, power=0.80.
    assert min_detectable_correlation(22) == pytest.approx(0.567, abs=1e-3)
    assert min_detectable_correlation(110) == pytest.approx(0.264, abs=1e-3)
    with pytest.raises(ValueError):
        min_detectable_correlation(3)


def test_cluster_bootstrap_is_seeded_and_brackets_the_estimate() -> None:
    clusters = ["a", "a", "b", "b", "c", "c", "d", "d", "e", "e"]
    xs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ys = [0.0, 0.3, 0.2, 0.5, 0.4, 0.7, 0.6, 0.9, 0.8, 1.0]
    first = cluster_bootstrap_interval(xs, ys, clusters, spearman, resamples=500)
    second = cluster_bootstrap_interval(xs, ys, clusters, spearman, resamples=500)
    assert first == second
    low, high = first
    assert low <= spearman(xs, ys) <= high <= 1.0


def test_cluster_bootstrap_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        cluster_bootstrap_interval([0.1], [0.2, 0.3], ["a"], spearman)
