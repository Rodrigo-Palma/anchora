"""Small-sample uncertainty for the eval reports, stdlib only.

The held-out sets here are small (22 answerable questions, 6 out-of-corpus), so
a bare rate such as ``0.833`` says little on its own. Every reported number goes
through one of these helpers:

* :func:`wilson_interval` for a single proportion ``k/n`` (Wilson score interval,
  well behaved at ``k=0`` and ``k=n`` where the Wald interval collapses).
* :func:`mcnemar_exact` for two models scored on the *same* questions: the exact
  two-sided binomial test on the discordant pairs. Comparing two independent
  Wilson intervals throws away the pairing and is conservative.
* :func:`bootstrap_mean_interval` for means of continuous per-question scores
  (faithfulness, precision@k, reciprocal rank), where no binomial model applies.
* :func:`spearman`, :func:`cohen_kappa` and :func:`cluster_bootstrap_interval`
  for agreement between two raters (lexical proxy vs LLM judge, judge vs judge),
  with :func:`min_detectable_correlation` to say what a null result can rule out.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence

Z_95 = 1.959963984540054
_DEFAULT_RESAMPLES = 10_000
_DEFAULT_SEED = 0


def wilson_interval(successes: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for ``successes`` out of ``n`` (95% by default)."""
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if not 0 <= successes <= n:
        raise ValueError(f"successes must be in [0, {n}], got {successes}")
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z2 / (4 * n * n)) / denom
    # Pin the exact bounds at the edges: the closed form gives 0 and 1 there, but
    # floating point can land a hair inside and exclude the point estimate.
    low = 0.0 if successes == 0 else max(0.0, centre - half)
    high = 1.0 if successes == n else min(1.0, centre + half)
    return (low, high)


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Exact two-sided McNemar p-value from the two discordant counts.

    ``only_a`` is the number of questions model A got right and B got wrong;
    ``only_b`` the reverse. Concordant pairs carry no information about the
    difference and are ignored. Under the null each discordant pair is a fair
    coin, so the p-value is ``2 * P(X <= min(only_a, only_b))`` for
    ``X ~ Binomial(only_a + only_b, 0.5)``, capped at 1.
    """
    if only_a < 0 or only_b < 0:
        raise ValueError(f"discordant counts must be >= 0, got {only_a}, {only_b}")
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2**n
    return min(1.0, 2.0 * tail)


def bootstrap_mean_interval(
    values: Sequence[float],
    *,
    level: float = 0.95,
    resamples: int = _DEFAULT_RESAMPLES,
    seed: int = _DEFAULT_SEED,
) -> tuple[float, float]:
    """Seeded percentile bootstrap interval for the mean of ``values``."""
    if not values:
        raise ValueError("values must be non-empty")
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")
    if resamples <= 0:
        raise ValueError(f"resamples must be positive, got {resamples}")
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(resamples))
    alpha = (1.0 - level) / 2.0
    low = means[math.floor(alpha * (resamples - 1))]
    high = means[math.ceil((1.0 - alpha) * (resamples - 1))]
    return (low, high)


def format_proportion(successes: int, n: int) -> str:
    """Render ``k/n = rate [low, high]`` with the 95% Wilson interval."""
    low, high = wilson_interval(successes, n)
    return f"{successes}/{n} = {successes / n:.3f} [{low:.2f}, {high:.2f}]"


def format_mean(values: Sequence[float]) -> str:
    """Render ``mean [low, high] (n)`` with the seeded 95% bootstrap interval."""
    low, high = bootstrap_mean_interval(values)
    return f"{sum(values) / len(values):.3f} [{low:.2f}, {high:.2f}] (n={len(values)})"


# --- agreement between two raters (proxy vs judge, judge vs judge) -----------

_Z_POWER_80 = 0.8416212335729143


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Pearson correlation; NaN when either vector is constant or ``n < 2``."""
    n = len(xs)
    if n != len(ys):
        raise ValueError(f"length mismatch: {n} vs {len(ys)}")
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    vx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    vy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if vx == 0.0 or vy == 0.0:
        return float("nan")
    return cov / (vx * vy)


def _ranks(values: Sequence[float]) -> list[float]:
    """Average (fractional) ranks, so ties do not distort Spearman."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Spearman rank correlation with average ranks for ties."""
    return pearson(_ranks(xs), _ranks(ys))


def cohen_kappa(a: Sequence[bool], b: Sequence[bool]) -> float:
    """Cohen's kappa for two binary raters; NaN when chance agreement is 1."""
    n = len(a)
    if n == 0 or n != len(b):
        raise ValueError(f"need two non-empty sequences of equal length, got {n} and {len(b)}")
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pa, pb = sum(a) / n, sum(b) / n
    expected = pa * pb + (1.0 - pa) * (1.0 - pb)
    if expected == 1.0:
        return float("nan")
    return (observed - expected) / (1.0 - expected)


def min_detectable_correlation(
    n: int, z_alpha: float = Z_95, z_power: float = _Z_POWER_80
) -> float:
    """Smallest true correlation detectable at alpha=0.05 (two-sided), power 0.80.

    Fisher-z approximation: ``tanh((z_alpha + z_power) / sqrt(n - 3))``. It says
    how large an effect the sample could have found, so a null result can be read
    for what it is.
    """
    if n < 4:
        raise ValueError(f"n must be at least 4, got {n}")
    return math.tanh((z_alpha + z_power) / math.sqrt(n - 3))


def cluster_bootstrap_interval(
    xs: Sequence[float],
    ys: Sequence[float],
    clusters: Sequence[str],
    statistic: Callable[[Sequence[float], Sequence[float]], float],
    *,
    level: float = 0.95,
    resamples: int = _DEFAULT_RESAMPLES,
    seed: int = _DEFAULT_SEED,
) -> tuple[float, float]:
    """Seeded percentile bootstrap that resamples whole clusters.

    Several generations answer the same question, so they are not independent;
    resampling questions (clusters) instead of rows keeps the interval honest.
    Resamples where the statistic is undefined (NaN) are dropped.
    """
    if not len(xs) == len(ys) == len(clusters) or not xs:
        raise ValueError("xs, ys and clusters must be non-empty and of equal length")
    groups: dict[str, list[int]] = {}
    for index, cluster in enumerate(clusters):
        groups.setdefault(cluster, []).append(index)
    keys = sorted(groups)
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(resamples):
        rows = [i for key in rng.choices(keys, k=len(keys)) for i in groups[key]]
        value = statistic([xs[i] for i in rows], [ys[i] for i in rows])
        if value == value:
            values.append(value)
    if not values:
        return (float("nan"), float("nan"))
    values.sort()
    alpha = (1.0 - level) / 2.0
    return (
        values[math.floor(alpha * (len(values) - 1))],
        values[math.ceil((1.0 - alpha) * (len(values) - 1))],
    )
