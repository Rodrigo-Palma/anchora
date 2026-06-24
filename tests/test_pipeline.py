from __future__ import annotations

import pytest
from pipeline.ml_pipeline import build_pipeline, run_pipeline
from pipeline.sagemaker_pipeline import PipelineConfig, describe


def test_build_pipeline_default_skips_training() -> None:
    stages = build_pipeline(version="v1", created_at="2026-06-24T00:00:00")
    names = [s.name for s in stages]
    assert names == ["build-dataset", "finetune", "eval-and-register"]
    finetune = next(s for s in stages if s.name == "finetune")
    assert finetune.enabled is False


def test_build_pipeline_enables_training_and_promote() -> None:
    stages = build_pipeline(
        version="v2", created_at="2026-06-24T00:00:00", train=True, promote=True
    )
    finetune = next(s for s in stages if s.name == "finetune")
    assert finetune.enabled is True
    evald = next(s for s in stages if s.name == "eval-and-register")
    assert "--promote" in evald.command
    assert "v2" in evald.command


def test_run_pipeline_dry_run_executes_nothing(capsys: pytest.CaptureFixture[str]) -> None:
    stages = build_pipeline(version="v1", created_at="2026-06-24T00:00:00")
    code = run_pipeline(stages, dry_run=True)
    assert code == 0
    out = capsys.readouterr().out
    assert "PLAN build-dataset" in out
    assert "SKIP finetune" in out  # disabled by default


def test_sagemaker_describe_is_offline() -> None:
    info = describe()
    assert info["name"] == "anchora-qa-pipeline"
    assert "train" in info["stages"]


def test_sagemaker_config_defaults() -> None:
    cfg = PipelineConfig(role="arn:aws:iam::123:role/x")
    assert cfg.model_package_group == "anchora-qa"
    assert cfg.region == "us-east-1"
