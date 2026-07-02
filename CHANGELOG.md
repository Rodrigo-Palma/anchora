# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/)
and the project adheres to [Semantic Versioning](https://semver.org/lang/pt-BR/).

## [Unreleased]

### To do (v1.0)
- Recorded demo (asciinema/GIF) of the CLI + API flow.
- Public write-up of the eval methodology.

## [0.5.0] — 2026-07-01

Retrieval quality, adversarial robustness, observability and documentation —
each claim measured, each limitation documented rather than hidden.

### Added
- **Hybrid retrieval**: a pure-Python Okapi BM25 index (`lexical.py`) fused with
  dense cosine via Reciprocal Rank Fusion; `retrieval_mode` (`dense|bm25|hybrid`,
  default `hybrid`) in config. Backed by `scripts/ablation_retrieval.py`
  (`make ablation`) measuring recall/precision/MRR per mode.
- **Adversarial guardrail suite**: `data/adversarial/attacks.json` (44 attacks —
  injection, jailbreak, PII exfiltration, citation forgery, off-domain) replayed
  by `scripts/adversarial_suite.py` (`make adversarial`); gates CI.
- **Observability**: per-stage tracing (`observability.py`) on every
  `AgentResult`; `trace_id` + `timing_ms` on `/ask`; `x-request-id` on every
  response.
- **Latency benchmark**: `scripts/benchmark.py` (`make bench`) with a p95
  regression gate.
- **SSE streaming**: `POST /ask/stream` streams the answer then a terminal
  `done` event with sources, grounding and trace.
- **Judge calibration**: `scripts/calibrate_judge.py` measures proxy-vs-LLM-judge
  agreement (`docs/eval-calibration.md`).
- **Property-based tests** (`hypothesis`) for chunking and deadline invariants.
- **Docs**: ADRs (`docs/adr/0001-0005`), model card (`docs/model-card.md`),
  dataset datasheet (`data/README.md`).

### Changed
- `validate_output` now verifies every `[n]` resolves to a retrieved chunk;
  forged indices abstain instead of passing as grounded.
- Out-of-domain floor: questions with no lexical overlap with the corpus abstain
  instead of quoting the nearest-by-cosine chunk.
- Hardened injection/jailbreak patterns (forget/override/print-prompt/pretend/
  `SYSTEM:`), closing regressions the adversarial suite exposed.
- Fine-tune replays pin `dense` retrieval so frozen generations are re-scored
  under the mode they were produced in.

### Fixed
- README roadmap corrected: the 5-abstention adapter **was** promoted via the
  gate (10-abstention variant auto-rejected); the LLM-judge pointer now
  references the calibration script instead of the registry tool.

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
