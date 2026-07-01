"""Replay the promotion gate on the frozen held-out metrics — no GPU, no network.

This is the MLOps decision the headline 0.92 could never make. It re-scores the
frozen generations (``score_generations.score_all``), wraps each arm's honest
held-out metrics in a :class:`~anchora.registry.ModelCard`, and walks them through
the same multi-criteria gate ``scripts/register_finetune.py`` uses in production:
promote to prod only if the candidate does not regress on
``{citation_accuracy, abstention_rate}`` versus the incumbent
(``registry.regressions``).

Expected replay (see docs/finetuning-results.md):

* Promoted anchora-qa:v0.3-lora0  (no incumbent)
* Promoted anchora-qa:v0.3-lora5  (no regression on the gate metrics)
* REJECTED anchora-qa:v0.3-lora10 (regressed citation_accuracy 0.818 -> 0.636)

Usage::

    uv run python scripts/gate_promotion.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from score_generations import score_all

from anchora.registry import ModelCard, ModelRegistry, regressions

_NAME = "anchora-qa"
_GATE_METRICS = ("citation_accuracy", "abstention_rate")

# Candidates evaluated in registration order. lora0 lands with no incumbent, lora5
# must not regress vs lora0, lora10 must be rejected for regressing citation_accuracy.
_CANDIDATES = (
    ("v0.3-lora0", "lora0"),
    ("v0.3-lora5", "lora5"),
    ("v0.3-lora10", "lora10"),
)


def run_gate(results: dict[str, dict[str, float]], registry: ModelRegistry) -> list[str]:
    """Register each candidate and apply the gate; return the printed decision log."""
    log: list[str] = []
    for version, arm in _CANDIDATES:
        card = ModelCard(
            name=_NAME,
            version=version,
            base_model="Qwen/Qwen2.5-1.5B-Instruct",
            metrics=results[arm],
            notes="frozen held-out generation eval (28-case holdout)",
        )
        registry.register(card)
        incumbent = registry.current(_NAME, "prod")

        if incumbent is None:
            registry.promote(_NAME, version, "prod")
            log.append(f"Promoted {card.key} to prod (no incumbent).")
            continue

        lost = regressions(card, incumbent, _GATE_METRICS)
        if lost:
            details = ", ".join(
                f"{key} {incumbent.metrics.get(key, 0.0):.3f}->{card.metrics.get(key, 0.0):.3f}"
                for key in lost
            )
            log.append(
                f"REJECTED {card.key}: regressed on {details}; keeping {incumbent.key} in prod."
            )
            continue

        registry.promote(_NAME, version, "prod")
        log.append(f"Promoted {card.key} to prod (no regression on {list(_GATE_METRICS)}).")
    return log


def _expected_shape(log: list[str]) -> bool:
    return (
        len(log) == 3
        and log[0].startswith("Promoted anchora-qa:v0.3-lora0")
        and log[1].startswith("Promoted anchora-qa:v0.3-lora5")
        and log[2].startswith("REJECTED anchora-qa:v0.3-lora10")
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="registry file to write (default: in-memory temp, not persisted).",
    )
    args = parser.parse_args(argv)

    registry_path = args.registry or (Path(__file__).resolve().parent / ".gate-replay.json")
    if registry_path.exists():
        registry_path.unlink()
    registry = ModelRegistry(registry_path)

    log = run_gate(score_all(), registry)
    for line in log:
        print(line)

    if args.registry is None:
        registry_path.unlink(missing_ok=True)

    prod = registry.current(_NAME, "prod")
    prod_key = prod.key if prod is not None else "<none>"
    print(f"\nFinal prod: {prod_key}")

    if not _expected_shape(log):
        print("\nGATE REPLAY MISMATCH — decision log did not match the documented outcome.")
        return 1
    if prod_key != f"{_NAME}:v0.3-lora5":
        print(f"\nGATE REPLAY MISMATCH — expected prod {_NAME}:v0.3-lora5, got {prod_key}.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
