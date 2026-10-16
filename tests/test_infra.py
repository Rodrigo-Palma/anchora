"""Static checks on infra/main.tf: the SageMaker role stays least-privilege."""

from __future__ import annotations

import re
from pathlib import Path

_MAIN_TF = (Path(__file__).resolve().parents[1] / "infra" / "main.tf").read_text()


def _block(header: str) -> str:
    start = _MAIN_TF.index(header)
    return _MAIN_TF[start : _MAIN_TF.index("\n}\n", start)]


def test_sagemaker_role_has_no_managed_policy() -> None:
    assert "arn:aws:iam::aws:policy/" not in _MAIN_TF
    assert "aws_iam_role_policy_attachment" not in _MAIN_TF


def test_role_policies_grant_no_wildcard_action() -> None:
    for name in ("sagemaker_exec", "sagemaker_artifacts"):
        block = _block(f'data "aws_iam_policy_document" "{name}"')
        assert "Deny" not in block
        assert not re.search(r'actions\s*=\s*\[[^\]]*"(\*|[a-z0-9]+:\*)"', block), name


def test_pass_role_is_limited_to_sagemaker() -> None:
    assert '"iam:PassRole"' in _MAIN_TF
    assert '"iam:PassedToService"' in _MAIN_TF


def test_artifacts_bucket_denies_plain_http() -> None:
    assert 'resource "aws_s3_bucket_policy" "artifacts"' in _MAIN_TF
    assert '"aws:SecureTransport"' in _MAIN_TF
