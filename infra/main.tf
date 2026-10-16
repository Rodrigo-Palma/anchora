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

# Refuse any request to the bucket that does not use TLS.
data "aws_iam_policy_document" "artifacts_tls_only" {
  statement {
    sid     = "DenyInsecureTransport"
    effect  = "Deny"
    actions = ["s3:*"]
    resources = [
      aws_s3_bucket.artifacts.arn,
      "${aws_s3_bucket.artifacts.arn}/*",
    ]

    principals {
      type        = "*"
      identifiers = ["*"]
    }

    condition {
      test     = "Bool"
      variable = "aws:SecureTransport"
      values   = ["false"]
    }
  }
}

resource "aws_s3_bucket_policy" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  policy = data.aws_iam_policy_document.artifacts_tls_only.json

  # Apply after the public access block so the policy is never evaluated
  # against a bucket that still allows public policies.
  depends_on = [aws_s3_bucket_public_access_block.artifacts]
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
    actions = ["s3:GetObject", "s3:PutObject", "s3:ListBucket", "s3:GetBucketLocation"]
    resources = [
      aws_s3_bucket.artifacts.arn,
      "${aws_s3_bucket.artifacts.arn}/*",
    ]
  }
}

# Least privilege for the pipeline (process -> train -> evaluate -> register).
# It replaces the AmazonSageMakerFullAccess managed policy, which granted every
# SageMaker action in the account, any "*sagemaker*" bucket and a broad
# iam:PassRole: a training job compromised through a dependency inherited all
# of it. If a step runs an AWS Deep Learning Container instead of the image in
# aws_ecr_repository.api, add that repository's ARN to the ECR statement.
locals {
  sm_arn_prefix   = "arn:aws:sagemaker:${var.region}:${data.aws_caller_identity.current.account_id}"
  logs_arn_prefix = "arn:aws:logs:${var.region}:${data.aws_caller_identity.current.account_id}"
}

data "aws_iam_policy_document" "sagemaker_exec" {
  statement {
    sid = "PipelineJobs"
    actions = [
      "sagemaker:CreateProcessingJob",
      "sagemaker:DescribeProcessingJob",
      "sagemaker:StopProcessingJob",
      "sagemaker:CreateTrainingJob",
      "sagemaker:DescribeTrainingJob",
      "sagemaker:StopTrainingJob",
      "sagemaker:AddTags",
      "sagemaker:ListTags",
    ]
    resources = [
      "${local.sm_arn_prefix}:processing-job/*",
      "${local.sm_arn_prefix}:training-job/*",
    ]
  }

  statement {
    sid = "ModelRegistry"
    actions = [
      "sagemaker:CreateModelPackage",
      "sagemaker:DescribeModelPackage",
      "sagemaker:UpdateModelPackage",
      "sagemaker:DescribeModelPackageGroup",
      "sagemaker:ListModelPackages",
    ]
    resources = [
      aws_sagemaker_model_package_group.qa.arn,
      "${local.sm_arn_prefix}:model-package/${aws_sagemaker_model_package_group.qa.model_package_group_name}/*",
    ]
  }

  statement {
    sid       = "PassOnlyThisRoleToSageMaker"
    actions   = ["iam:PassRole"]
    resources = [aws_iam_role.sagemaker_exec.arn]

    condition {
      test     = "StringEquals"
      variable = "iam:PassedToService"
      values   = ["sagemaker.amazonaws.com"]
    }
  }

  statement {
    sid = "JobLogs"
    actions = [
      "logs:CreateLogGroup",
      "logs:CreateLogStream",
      "logs:PutLogEvents",
      "logs:DescribeLogStreams",
    ]
    resources = [
      "${local.logs_arn_prefix}:log-group:/aws/sagemaker/*",
      "${local.logs_arn_prefix}:log-group:/aws/sagemaker/*:log-stream:*",
    ]
  }

  statement {
    sid       = "JobMetrics"
    actions   = ["cloudwatch:PutMetricData"]
    resources = ["*"] # PutMetricData has no resource ARN; scoped by namespace.

    condition {
      test     = "StringLike"
      variable = "cloudwatch:namespace"
      values   = ["/aws/sagemaker/*"]
    }
  }

  statement {
    sid       = "EcrAuth"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"] # account-level call; it grants no image access by itself.
  }

  statement {
    sid = "EcrPullServingImage"
    actions = [
      "ecr:BatchCheckLayerAvailability",
      "ecr:BatchGetImage",
      "ecr:GetDownloadUrlForLayer",
    ]
    resources = [aws_ecr_repository.api.arn]
  }
}

resource "aws_iam_role_policy" "sagemaker_artifacts" {
  name   = "${local.name_prefix}-artifacts-access"
  role   = aws_iam_role.sagemaker_exec.id
  policy = data.aws_iam_policy_document.sagemaker_artifacts.json
}

resource "aws_iam_role_policy" "sagemaker_exec" {
  name   = "${local.name_prefix}-pipeline-least-privilege"
  role   = aws_iam_role.sagemaker_exec.id
  policy = data.aws_iam_policy_document.sagemaker_exec.json
}
