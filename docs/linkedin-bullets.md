# anchora — material for LinkedIn / resume

Ready-to-use bullets for profile, resume, and posts. Adapt the tone to the channel.

## Headline / summary (1 line)

> Built a domain-specific RAG agent for Brazilian legal-administrative law, 100%
> local-first (Ollama), with tool use, production guardrails, automated evals
> in CI, LoRA fine-tuning, and IaC (Terraform/SageMaker).

## Resume bullets (result + technique)

- **RAG + tool-using agent** over a Brazilian legal corpus (LAI, 8.112,
  LGPD, CPC, Lei 14.133/9.784): **cited** answers `[n]` with **abstention**
  when there is no support in the context.
- **Deterministic production guardrails**: anti-injection/jailbreak, **PII
  detection and redaction** (CPF/email/phone), and mandatory output *grounding*.
- **Automated evaluation in CI** with an objective *gate* (retrieval recall = 1.0
  and faithfulness ≥ 0.70) using deterministic lexical proxies — **free and
  reproducible**, with an optional local LLM *judge* (DeepEval/RAGAS via Ollama).
- **LoRA/QLoRA fine-tuning** (PEFT/transformers) with a dataset generated from the golden set +
  corpus and baseline × tuned comparison promoted only when there is no regression.
- **MLOps**: file-backed *model registry* with dev/staging/prod promotion, a local
  ML pipeline (executable DAG), and a **SageMaker Pipelines skeleton**.
- **IaC with Terraform**: ECR (immutable + scan), versioned/encrypted S3,
  SageMaker Model Package Group, and an execution IAM role — `validate` green in CI.
- **Solid engineering**: `uv`, `ruff`, `mypy --strict`, `pytest` (102 tests,
  ~91% coverage), multi-stage non-root Docker, GitHub Actions, pre-commit.
- **Zero cost by design**: embeddings/generation via Ollama (`nomic-embed-text`,
  `qwen3:32b`) and a deterministic `hash` embedding provider for offline/CI.

## Skills for tags

`Python` · `RAG` · `LLM Agents` · `LangChain-free` · `Ollama` · `FastAPI` ·
`PEFT/LoRA` · `MLOps` · `SageMaker` · `Terraform` · `Docker` · `GitHub Actions` ·
`mypy` · `pytest` · `Evals/Guardrails`

## Why this impresses a technical recruiter

It is not "chat with a PDF". It shows the **full AI Engineering cycle**: retrieval
and a tool-using agent, security (guardrails + PII), **objective evaluation in
CI**, training (LoRA), packaging (Docker), delivery (API), and infrastructure
(Terraform/SageMaker) — with software rigor (types, tests, coverage) and
**zero cost**, proving the autonomy to run everything locally.
