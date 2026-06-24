"""A small, dependency-free DAG runner for the anchora MLOps loop.

The end-to-end flow is four ordered stages::

    build-dataset  ->  finetune  ->  eval-and-register  ->  promote

Each stage is just a shell command (``uv run ...``) so the pipeline is the same
whether it runs on a laptop, in CI, or as the local mirror of the SageMaker
pipeline in :mod:`pipeline.sagemaker_pipeline`. ``finetune`` is opt-in (it needs
the heavy ``finetune`` extra and a GPU); the offline stages run anywhere.

The runner supports ``--dry-run`` (print the plan, run nothing) so the DAG is
inspectable and unit-testable without executing anything.

Usage::

    uv run python -m pipeline.ml_pipeline --dry-run
    uv run python -m pipeline.ml_pipeline --version v1 --created-at 2026-06-24T00:00:00
    uv run python -m pipeline.ml_pipeline --version v2 --created-at ... --train --promote
"""

from __future__ import annotations

import argparse
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass
class Stage:
    name: str
    command: list[str]
    enabled: bool = True


def build_pipeline(
    version: str,
    created_at: str,
    base_model: str = "qwen3:32b",
    train: bool = False,
    promote: bool = False,
) -> list[Stage]:
    """Construct the ordered list of stages for one pipeline run."""
    stages = [
        Stage("build-dataset", ["uv", "run", "python", "scripts/build_finetune_dataset.py"]),
        Stage(
            "finetune",
            ["uv", "run", "python", "scripts/finetune_lora.py", "--out", "artifacts/lora-anchora"],
            enabled=train,
        ),
    ]
    eval_cmd = [
        "uv",
        "run",
        "python",
        "scripts/compare_evals.py",
        "--version",
        version,
        "--base",
        base_model,
        "--created-at",
        created_at,
    ]
    if promote:
        eval_cmd.append("--promote")
    stages.append(Stage("eval-and-register", eval_cmd))
    return stages


def run_pipeline(stages: Sequence[Stage], dry_run: bool = False) -> int:
    """Execute (or print) each enabled stage in order. Returns an exit code."""
    for i, stage in enumerate(stages, start=1):
        status = "SKIP" if not stage.enabled else ("PLAN" if dry_run else "RUN ")
        print(f"[{i}/{len(stages)}] {status} {stage.name}: {' '.join(stage.command)}")
        if not stage.enabled or dry_run:
            continue
        result = subprocess.run(stage.command, check=False)
        if result.returncode != 0:
            print(f"Stage '{stage.name}' failed with code {result.returncode}; aborting.")
            return result.returncode
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="dev")
    parser.add_argument("--created-at", default="1970-01-01T00:00:00", dest="created_at")
    parser.add_argument("--base", default="qwen3:32b", dest="base_model")
    parser.add_argument("--train", action="store_true", help="run the LoRA fine-tune stage")
    parser.add_argument("--promote", action="store_true", help="promote to prod if no regression")
    parser.add_argument("--dry-run", action="store_true", help="print the plan, run nothing")
    args = parser.parse_args(argv)

    stages = build_pipeline(
        version=args.version,
        created_at=args.created_at,
        base_model=args.base_model,
        train=args.train,
        promote=args.promote,
    )
    return run_pipeline(stages, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
