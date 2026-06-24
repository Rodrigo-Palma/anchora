"""Compare two model versions on the offline eval set and record the winner.

The MLOps loop in one script: run the eval harness, write the resulting
metrics into the file-backed model registry as a :class:`ModelCard`, and
(optionally) promote a version to ``prod`` only if it does not regress against
the model currently in production.

It is offline/deterministic by default (``hash`` retriever, extractive answers),
so it doubles as a CI gate. A real run after a LoRA fine-tune would pass
``--adapter`` and a generation-backed eval; the registry bookkeeping is the
same either way.

Usage::

    # record the current pipeline as a candidate
    uv run python scripts/compare_evals.py --name anchora-qa --version v1 \\
        --base qwen3:32b --created-at 2026-06-24T00:00:00

    # promote to prod only if it beats the incumbent on faithfulness
    uv run python scripts/compare_evals.py --name anchora-qa --version v2 \\
        --base qwen3:32b --created-at 2026-06-25T00:00:00 --promote
"""

from __future__ import annotations

import argparse
from pathlib import Path

from anchora.evals import run
from anchora.registry import ModelCard, ModelRegistry

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_REGISTRY = _ROOT / "artifacts" / "registry.json"


def evaluate_to_card(
    name: str,
    version: str,
    base_model: str,
    created_at: str,
    adapter_path: str | None = None,
) -> ModelCard:
    """Run the eval harness and pack the metrics into a ModelCard."""
    report = run()
    metrics = {
        "recall": report.mean_recall,
        "precision": report.mean_precision,
        "faithfulness": report.mean_faithfulness,
        "answer_relevance": report.mean_answer_relevance,
    }
    return ModelCard(
        name=name,
        version=version,
        base_model=base_model,
        metrics=metrics,
        adapter_path=adapter_path,
        created_at=created_at,
        notes="offline eval (hash retriever, extractive answers)",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="anchora-qa")
    parser.add_argument("--version", required=True)
    parser.add_argument("--base", required=True, dest="base_model")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--created-at", required=True, dest="created_at")
    parser.add_argument("--registry", type=Path, default=_DEFAULT_REGISTRY)
    parser.add_argument("--metric", default="faithfulness")
    parser.add_argument(
        "--promote",
        action="store_true",
        help="promote to prod only if it does not regress vs the current prod model",
    )
    args = parser.parse_args(argv)

    registry = ModelRegistry(args.registry)
    card = evaluate_to_card(
        name=args.name,
        version=args.version,
        base_model=args.base_model,
        created_at=args.created_at,
        adapter_path=args.adapter,
    )
    registry.register(card)
    print(f"Registered {card.key}: {card.metrics}")

    if not args.promote:
        return 0

    incumbent = registry.current(args.name, "prod")
    new_score = card.metrics.get(args.metric, 0.0)
    if incumbent is None:
        registry.promote(args.name, args.version, "prod")
        print(f"Promoted {card.key} to prod (no incumbent).")
        return 0

    old_score = incumbent.metrics.get(args.metric, 0.0)
    if new_score + 1e-9 >= old_score:
        registry.promote(args.name, args.version, "prod")
        print(f"Promoted {card.key} to prod ({args.metric}: {old_score:.4f} -> {new_score:.4f}).")
        return 0

    print(
        f"REJECTED {card.key}: {args.metric} regressed "
        f"({old_score:.4f} -> {new_score:.4f}); keeping {incumbent.key} in prod."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
