"""AWS SageMaker Pipelines definition — the cloud mirror of the local DAG.

This is a *skeleton*: it builds a ``sagemaker.workflow.pipeline.Pipeline`` object
with the same logical stages as :mod:`pipeline.ml_pipeline`
(process -> train -> evaluate -> register), parameterised so it can be created
and upserted from CI. The ``sagemaker`` SDK is imported lazily so importing this
module (e.g. for docs or unit tests) needs no AWS dependency or credentials.

It intentionally does not run on the offline test path — it documents the
production-shaped flow and the IaC in ``infra/`` provisions what it needs.

Usage (requires ``pip install sagemaker`` and AWS credentials)::

    from pipeline.sagemaker_pipeline import build_pipeline
    pipe = build_pipeline(role="arn:aws:iam::...:role/anchora-sm")
    pipe.upsert(role_arn="arn:aws:iam::...:role/anchora-sm")
    pipe.start()
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PipelineConfig:
    role: str
    region: str = "us-east-1"
    bucket: str | None = None
    base_model: str = "Qwen/Qwen2.5-3B-Instruct"
    instance_type_train: str = "ml.g5.2xlarge"
    instance_type_process: str = "ml.m5.xlarge"
    model_package_group: str = "anchora-qa"


def build_pipeline(role: str, region: str = "us-east-1") -> object:
    """Assemble the SageMaker Pipeline object. Lazily imports the SDK."""
    # Lazy imports: keep AWS deps optional and this module import-safe.
    from sagemaker.workflow.parameters import (
        ParameterFloat,
        ParameterString,
    )
    from sagemaker.workflow.pipeline import Pipeline
    from sagemaker.workflow.pipeline_context import PipelineSession

    cfg = PipelineConfig(role=role, region=region)
    session = PipelineSession()

    param_base_model = ParameterString(name="BaseModel", default_value=cfg.base_model)
    param_lr = ParameterFloat(name="LearningRate", default_value=2e-4)
    param_faith_gate = ParameterFloat(name="FaithfulnessGate", default_value=0.70)

    # The concrete ProcessingStep / TrainingStep / ConditionStep / RegisterModel
    # wiring lives here in a full implementation; see ml_pipeline.py for the
    # equivalent local stages (build-dataset -> finetune -> eval -> register).
    steps: list[object] = []

    return Pipeline(
        name="anchora-qa-pipeline",
        parameters=[param_base_model, param_lr, param_faith_gate],
        steps=steps,
        sagemaker_session=session,
    )


def describe() -> dict[str, object]:
    """Return a plain-data summary of the pipeline (for docs / dry inspection)."""
    return {
        "name": "anchora-qa-pipeline",
        "stages": ["process", "train", "evaluate", "register"],
        "gate": "register only if faithfulness >= FaithfulnessGate and no regression",
        "model_package_group": "anchora-qa",
    }


if __name__ == "__main__":
    import json

    print(json.dumps(describe(), ensure_ascii=False, indent=2))
