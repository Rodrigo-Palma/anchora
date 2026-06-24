# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/)
and the project adheres to [Semantic Versioning](https://semver.org/lang/pt-BR/).

## [Unreleased]

### To do (v1.0)
- Recorded demo (asciinema/GIF) of the CLI + API flow.
- Technical post and eval methodology write-up published.

## [0.4.0] — 2026-06-24

### Added
- **File-backed model registry** (`registry.py`): `ModelCard` + `ModelRegistry`
  with `register`, `promote` (dev/staging/prod with demotion of the previous holder),
  `current`, `best` by metric, and idempotent JSON persistence.
- **Local ML pipeline** (`pipeline/ml_pipeline.py`): DAG `build-dataset →
  finetune (optional) → eval-and-register`, with `--dry-run`, `--train`, `--promote`.
- **SageMaker Pipelines skeleton** (`pipeline/sagemaker_pipeline.py`) with
  offline `describe()` (no AWS required to inspect the plan).
- **Terraform IaC** (`infra/`): ECR (immutable + scan), versioned/encrypted S3
  with public access block, SageMaker Model Package Group, and an IAM execution role.
- `scripts/compare_evals.py`: runs eval → packages into a `ModelCard` → registers →
  promotes to `prod` only if there is no regression on the target metric.
- CI: `terraform` job (`fmt -check`, `init -backend=false`, `validate`).

## [0.3.0] — 2026-06-24

### Added
- **LoRA/QLoRA fine-tuning** behind the optional `finetune` extra (PEFT/transformers/
  datasets/accelerate, with lazy imports and cuda/mps/cpu detection).
- `scripts/build_finetune_dataset.py`: generates instruction JSONL from the golden
  set + corpus using the deterministic offline retriever.
- `scripts/finetune_lora.py`: training entrypoint (`Qwen/Qwen2.5-3B-Instruct`).

## [0.2.0] — 2026-06-24

### Added
- **Deterministic guardrails** (`guardrails.py`): injection/jailbreak blocking
  on input, PII detection/redaction (CPF/email/phone), and output *grounding*
  (requires a `[n]` citation or abstains).
- **Evals in CI** (`evals.py` + `metrics.py`): deterministic lexical proxies for
  faithfulness/relevance/precision/recall and an objective *gate* (recall = 1.0 and
  faithfulness ≥ 0.70).

### Fixed
- Removal of pt-BR stopwords in the `hash` provider (`embeddings.py`), eliminating
  retrieval failures on the golden set.
- API-key guard moved to `dependencies=[Depends(...)]` on the decorator, fixing
  the 422s on `/ingest` and `/ask`.

## [0.1.0] — 2026-06-24

### Added
- RAG pipeline: `chunking`, `embeddings` (Ollama `nomic-embed-text` + `hash`
  fallback), `store` (in-memory cosine VectorStore), `ingest`, `rag`, `llm`.
- Agent with tools: `search_documents` (RAG) + `legal_deadline`.
- FastAPI API (`/health`, `/ingest`, `/ask`) and CLI (`ingest`/`ask`/`eval`/`serve`).
- Brazilian legal-administrative corpus + golden set of 24 questions.
- Engineering: `uv`, `ruff`, `mypy --strict`, `pytest` with coverage, README + diagram.
