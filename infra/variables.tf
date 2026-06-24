variable "project" {
  description = "Project name, used as a prefix for resource names."
  type        = string
  default     = "anchora"
}

variable "env" {
  description = "Deployment environment (dev/staging/prod)."
  type        = string
  default     = "dev"
}

variable "region" {
  description = "AWS region."
  type        = string
  default     = "us-east-1"
}

variable "artifacts_force_destroy" {
  description = "Allow Terraform to delete the non-empty artifacts bucket on destroy."
  type        = bool
  default     = false
}

variable "model_package_group" {
  description = "SageMaker model package group name for the QA model."
  type        = string
  default     = "anchora-qa"
}
