# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/)
and the project adheres to [Semantic Versioning](https://semver.org/lang/pt-BR/).

## [0.6.0] - 2026-10-09

Measure what was previously assumed: the faithfulness proxy against LLM judges,
the guardrails' cost on legitimate questions and their recall on external
attacks, and retrieval with the production embedder. Several results are
negative and are reported as such.

### Added
- **Judge calibration, measured** (`scripts/calibrate_judge.py`,
  `data/eval/judge-scores.json`, `make calibration`): 100 answerable held-out
  generations scored by `qwen3:32b` (twice) and `gemma4:31b` at temperature 0,
  frozen with model digests and the prompt hash; CI recomputes every published
  statistic offline. See `docs/eval-calibration.md` and
  [ADR 7](docs/adr/0007-faithfulness-threshold.md).
- **Agreement statistics** in `anchora.stats`: Spearman, Cohen's kappa, a
  question-clustered bootstrap and the smallest detectable correlation.
- **False-positive rate of the guardrails** (`data/adversarial/benign.json`):
  46 in-domain and 30 attack-looking legitimate questions, gated on ceilings
  declared before the first run.
- **External attacks, not gated** (`make adversarial-external`): the
  `deepset/prompt-injections` test split (Apache-2.0), frozen with revision and
  file hash.
- **Production-embedder ablation** (`--provider ollama --freeze`, `--compare`):
  retrieval with `nomic-embed-text`, frozen with the model digest and compared
  with `hash` by paired McNemar.
- **Real-token SSE**: `/ask/stream` forwards Ollama tokens as they decode and
  emits `retracted` when the complete answer fails the grounding check
  ([ADR 8](docs/adr/0008-streaming-with-output-validation.md)).
- `py.typed`, so the package's annotations reach consumers (PEP 561).
- `docs/methodology.md`: the evaluation write-up (replaces the draft post).
- Wilson intervals, paired McNemar and bootstrap intervals in every eval report
  (holdout, ablation, adversarial), and the out-of-domain floor of
  [ADR 6](docs/adr/0006-out-of-domain-floor.md).

### Changed
- `Agent.run` is split into `prepare` and `finalize`, so a streaming caller
  stops before the model on refusals and out-of-domain questions.
- The test suite points Ollama at a closed port, so no test can pass by
  reaching a local model.

### Fixed
- `ood-008` (one incidental token defeated the old zero-overlap floor) is
  handled; the adversarial suite has two documented gaps (`inj-012`, `jb-008`).

### Found by the new measurements
- The input guardrail refuses 1/60 external injections; the pipeline declined
  58/60 only because the out-of-domain floor rejected them.
- The out-of-domain floor abstains on 4/22 answerable holdout questions; they
  are pinned as known over-blocks and the trade-off is in the ADR 6 addendum.
- The lexical proxy scores the base model's English answers about 0.2 while the
  judge scores them about 0.5 to 0.6: part of what the proxy calls unfaithful is
  language mismatch.

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
