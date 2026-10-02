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
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

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
