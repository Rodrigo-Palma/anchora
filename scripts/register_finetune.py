"""Register a fine-tuned adapter in the model registry using its HONEST metrics.

``scripts/compare_evals.py`` records the offline extractive eval (deterministic,
CI-friendly, but scored on the training golden set). This script closes the loop
with the *held-out* generation metrics instead — citation accuracy, abstention
rate, faithfulness, reference overlap — read from a ``holdout-comparison.json``
produced by ``evaluate_finetune.py``. Citation accuracy and Portuguese-aware
abstention are recomputed from the stored answers; retrieval is deterministic so
no model is needed here.

With ``--promote`` it applies a multi-criteria gate (``registry.regressions``):
promote to prod only if the candidate does not regress on the gate metrics versus
the current prod card. This rejects a model that trades, say, abstention for
citation accuracy.

Usage::

    uv run python scripts/register_finetune.py \\
        --comparison artifacts/holdout-comparison-abstention5.json \\
        --version v0.3-lora5 --base Qwen/Qwen2.5-1.5B-Instruct \\
        --adapter artifacts/lora-anchora-qwen15b-abstention5-lr1e4-e30 \\
        --created-at 2026-06-26T00:00:00 --promote
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from anchora import metrics
from anchora.guardrails import is_abstention
from anchora.ingest import ingest_dir
from anchora.rag import retrieve
from anchora.registry import ModelCard, ModelRegistry, regressions

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_REGISTRY = _ROOT / "artifacts" / "registry.json"
_HOLDOUT_PATH = _ROOT / "data" / "golden" / "holdout.json"
_CORPUS_DIR = _ROOT / "data" / "corpus"
_PROVIDER = "hash"
_GATE_METRICS = ("citation_accuracy", "abstention_rate")


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def honest_metrics(comparison_path: Path, report_name: str) -> dict[str, float]:
    """Recompute held-out generation metrics from a comparison report's answers."""
    store = ingest_dir(_CORPUS_DIR, provider=_PROVIDER)
    cases = json.loads(_HOLDOUT_PATH.read_text(encoding="utf-8"))["cases"]
    docs_by_id = {
        case["id"]: [c.doc_id for c in retrieve(store, case["question"], k=4, provider=_PROVIDER)]
        for case in cases
    }

    data = json.loads(comparison_path.read_text(encoding="utf-8"))
    report = next(r for r in data["reports"] if r["name"] == report_name)
    answerable = [s for s in report["scores"] if s["answerable"]]
    unanswerable = [s for s in report["scores"] if not s["answerable"]]

    return {
        "citation_accuracy": _mean(
            [
                metrics.citation_correct(s["answer"], docs_by_id[s["case_id"]], s["expected_doc"])
                for s in answerable
            ]
        ),
        "faithfulness": _mean([s["faithfulness"] for s in answerable]),
        "reference_overlap": _mean([s["reference_overlap"] for s in answerable]),
        "abstention_rate": _mean(
            [1.0 if is_abstention(s["answer"]) else 0.0 for s in unanswerable]
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--report", default="lora", help="which report row in the comparison")
    parser.add_argument("--name", default="anchora-qa")
    parser.add_argument("--version", required=True)
    parser.add_argument("--base", required=True, dest="base_model")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--created-at", required=True, dest="created_at")
    parser.add_argument("--registry", type=Path, default=_DEFAULT_REGISTRY)
    parser.add_argument("--notes", default="held-out generation eval (28-case holdout)")
    parser.add_argument(
        "--promote",
        action="store_true",
        help="promote to prod only if it does not regress on the gate metrics vs prod",
    )
    args = parser.parse_args(argv)

    card = ModelCard(
        name=args.name,
        version=args.version,
        base_model=args.base_model,
        metrics=honest_metrics(args.comparison, args.report),
        adapter_path=args.adapter,
        created_at=args.created_at,
        notes=args.notes,
    )
    registry = ModelRegistry(args.registry)
    registry.register(card)
    print(f"Registered {card.key}: {card.metrics}")

    if not args.promote:
        return 0

    incumbent = registry.current(args.name, "prod")
    if incumbent is None:
        registry.promote(args.name, args.version, "prod")
        print(f"Promoted {card.key} to prod (no incumbent).")
        return 0

    lost = regressions(card, incumbent, _GATE_METRICS)
    if lost:
        details = ", ".join(
            f"{key} {incumbent.metrics.get(key, 0.0):.3f}->{card.metrics.get(key, 0.0):.3f}"
            for key in lost
        )
        print(f"REJECTED {card.key}: regressed on {details}; keeping {incumbent.key} in prod.")
        return 1

    registry.promote(args.name, args.version, "prod")
    print(f"Promoted {card.key} to prod (no regression on {list(_GATE_METRICS)}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
