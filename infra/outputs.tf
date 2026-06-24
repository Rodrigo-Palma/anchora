output "ecr_repository_url" {
  description = "Push the serving image here."
  value       = aws_ecr_repository.api.repository_url
}

output "artifacts_bucket" {
  description = "S3 bucket for datasets, adapters, eval reports and the registry."
  value       = aws_s3_bucket.artifacts.bucket
}

output "model_package_group" {
  description = "SageMaker model package group for registered QA models."
  value       = aws_sagemaker_model_package_group.qa.model_package_group_name
}

output "sagemaker_execution_role_arn" {
  description = "Role ARN to pass to the SageMaker pipeline."
  value       = aws_iam_role.sagemaker_exec.arn
}
