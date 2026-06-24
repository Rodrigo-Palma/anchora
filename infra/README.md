# infra — anchora Terraform (IaC)

AWS infrastructure skeleton to package, train, and register the `anchora`
model. This is a **validatable skeleton**: `terraform validate` and `fmt -check`
run in CI, but `apply` requires an AWS account and **incurs costs** — project
development and tests stay 100% local and free.

## Resources

| Resource | Purpose |
|---|---|
| `aws_ecr_repository.api` | Serving image (FastAPI). Immutable tags + scan on push. |
| `aws_s3_bucket.artifacts` | Datasets, LoRA adapters, eval reports, and the registry. Versioned, encrypted (AES256), public access blocked. |
| `aws_sagemaker_model_package_group.qa` | Version group for the registered QA model. |
| `aws_iam_role.sagemaker_exec` | SageMaker pipeline execution role + bucket access policy. |

## Variables

| Variable | Default | Description |
|---|---|---|
| `project` | `anchora` | Resource name prefix. |
| `env` | `dev` | Environment (dev/staging/prod). |
| `region` | `us-east-1` | AWS region. |
| `model_package_group` | `anchora-qa` | Model package group name. |
| `artifacts_force_destroy` | `false` | Allow destroying a non-empty bucket. |

## Outputs

`ecr_repository_url`, `artifacts_bucket`, `model_package_group`,
`sagemaker_execution_role_arn` — consumed by the serving image and the
SageMaker pipeline (`pipeline/sagemaker_pipeline.py`).

## Usage

```bash
cd infra

# checks (what CI runs — no AWS credentials required)
terraform fmt -check
terraform init -backend=false
terraform validate

# to actually apply (requires AWS credentials and incurs costs):
terraform init
terraform plan
terraform apply
```

## State backend

For real use, enable the commented S3 backend in `versions.tf` (bucket
`anchora-tfstate`). In the skeleton, state is local so `validate` runs without
pre-existing infrastructure.

## Cost

Everything here is **opt-in**. The `anchora` development cycle (RAG, agent,
evals, local LoRA fine-tune) runs offline via Ollama and never touches AWS.
