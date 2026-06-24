###############################################################################
# anchora — infrastructure skeleton (IaC, v0.4)
#
# Provisions the cloud footprint the project's MLOps loop needs:
#   * ECR repository   — container image for the FastAPI serving app;
#   * S3 bucket        — fine-tune artifacts, eval reports, model registry;
#   * SageMaker group   — model package group for registered QA models;
#   * IAM role          — execution role SageMaker assumes for train/eval jobs.
#
# This is a reviewed, valid skeleton (terraform validate passes). Apply only
# with real credentials and an intentional plan.
###############################################################################

locals {
  name_prefix = "${var.project}-${var.env}"
}

data "aws_caller_identity" "current" {}

# --- container registry for the serving image -------------------------------

resource "aws_ecr_repository" "api" {
  name                 = "${local.name_prefix}-api"
  image_tag_mutability = "IMMUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

# --- artifact storage (datasets, adapters, eval reports, registry) -----------

resource "aws_s3_bucket" "artifacts" {
  bucket        = "${local.name_prefix}-artifacts-${data.aws_caller_identity.current.account_id}"
  force_destroy = var.artifacts_force_destroy
}

resource "aws_s3_bucket_versioning" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_public_access_block" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# --- model registry (SageMaker model package group) -------------------------

resource "aws_sagemaker_model_package_group" "qa" {
  model_package_group_name        = var.model_package_group
  model_package_group_description = "anchora QA models (base + LoRA adapters) with eval metrics."
}

# --- SageMaker execution role -----------------------------------------------

data "aws_iam_policy_document" "sagemaker_assume" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["sagemaker.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "sagemaker_exec" {
  name               = "${local.name_prefix}-sagemaker-exec"
  assume_role_policy = data.aws_iam_policy_document.sagemaker_assume.json
}

data "aws_iam_policy_document" "sagemaker_artifacts" {
  statement {
    actions = ["s3:GetObject", "s3:PutObject", "s3:ListBucket"]
    resources = [
      aws_s3_bucket.artifacts.arn,
      "${aws_s3_bucket.artifacts.arn}/*",
    ]
  }
}

resource "aws_iam_role_policy" "sagemaker_artifacts" {
  name   = "${local.name_prefix}-artifacts-access"
  role   = aws_iam_role.sagemaker_exec.id
  policy = data.aws_iam_policy_document.sagemaker_artifacts.json
}

resource "aws_iam_role_policy_attachment" "sagemaker_full" {
  role       = aws_iam_role.sagemaker_exec.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}
