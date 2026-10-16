# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/)
and the project adheres to [Semantic Versioning](https://semver.org/lang/pt-BR/).

## [Unreleased]

### Added
- `citations` on `/ask`, on the stream's `done` event and in the CLI
  (`[n] Title`): one `{n, title, doc_id}` per retrieved excerpt, numbered as in
  the prompt, so every inline `[n]` resolves to its law. Repeated documents keep
  one entry per excerpt.

### Changed
- `sources` is unchanged and kept for compatibility, now documented as the
  distinct titles in alphabetical order: it does not follow the `[n]`
  numbering (an answer citing `[1]` could be read against the wrong law).

## [0.8.0] - 2026-10-16

Close two roadmap items with measurements: the out-of-domain floor on the
production embedder and the demo with the local models. The container image is
now built and smoke-tested in CI, and the audit covers the finetune extra.

### Added
- **Demo with the local models** (`docs/demo.md#with-local-models`): an
  asciinema recording (`docs/demo-local.cast`) of real SSE tokens on
  `qwen3:32b` with `nomic-embed-text`, both pinned by digest, and the same run
  frozen in `data/demo/stream-transcript.json`. The golden `lai-prazo` question
  streams 40 tokens and ends cited; one answer is retracted (ADR 8). It is an
  observed example, not guaranteed behaviour.
- `scripts/record_stream_demo.py`: the recorder. Its candidate questions and a
  budget of 10 attempts were committed before any recording run; the
  retraction appeared on attempt 2 of 10. `tests/test_demo_transcript.py`
  checks the transcript's event order and schema offline (no model needed).
- **Frozen out-of-domain similarities** (`scripts/calibrate_ood_floor.py`,
  `data/eval/ood-similarities.json`): distinct corpus-token overlap and top
  `nomic-embed-text` cosine of the labelled questions, frozen with the model
  digest. `make ood-floor` rebuilds the ADR 6 tables offline and is part of
  `make check` and CI.
- **Docker smoke test in CI** (`docker-smoke` job, `scripts/docker_smoke.sh`,
  `make docker-smoke`): builds the image and checks `/health`, a `401` on
  `/ask` without the key, a `200` on `/ask` with at least one source (lazy
  corpus ingestion inside the image) and a `healthy` `HEALTHCHECK`. Image size
  is documented in `docs/architecture.md` (56.3 MB compressed on linux/arm64).

### Changed
- Out-of-domain floor on the Ollama embedder: a question sharing exactly one
  distinct corpus token is now in-domain if its top `nomic-embed-text` cosine
  is >= `ANCHORA_OOD_OVERLAP1_SIMILARITY_THRESHOLD` (default 0.56, 0.0 turns it
  off). The rule and its acceptance criterion were registered in ADR 6 before
  the cosines were measured; on the frozen similarities it recovers 1 holdout
  question (18/22 -> 19/22 answered) while 0/6 out-of-corpus questions are
  answered and 12/12 attacks still abstain. The holdout is spent for this
  decision. The `hash` path (CI, adversarial suite) is unchanged; `make
  ood-floor` is the gate for this path. If Ollama is unreachable, the question
  abstains.
- The dense similarity floor as wired only adds abstentions, all on real
  questions, so it stays off (ADR 6 addendum).
- Container `HEALTHCHECK`: `--start-period=10s --start-interval=1s`, so the
  container no longer waits for the 30 s interval to report `healthy` after
  start. The steady-state probe is unchanged.

### Security
- `make audit` and the CI audit step now cover the `finetune` extra too
  (runtime + dev + finetune, from `uv.lock`, without installing it). Before,
  only runtime + dev were audited, and the finetune set carried 16 `pip-audit`
  findings in 8 packages that no gate saw. CI calls `make audit`, so the two
  cannot drift.
- Finetune extra refreshed in the lockfile: `torch` 2.14.1, `transformers`
  5.20.0, `datasets` 5.1.0, `accelerate` 1.15.0, `aiohttp` 3.14.5 and the
  flagged transitives. `pyproject.toml` floors are unchanged.

### Documentation
- README roadmap: the local-model demo and the production-embedder
  out-of-domain floor are closed, each stated as what was measured.
- `docs/architecture.md`: a Container image section (build stages, non-root
  user, image size, what the smoke test checks).

## [0.7.0] - 2026-10-09

Close the gaps a review found between what the docs promise and what the code
does: the API now fails closed, the exceeded false-positive ceiling is replaced
by a gate that holds, and CI checks the lockfile and known advisories.

### Security
- `/ingest` no longer reads arbitrary server paths. `directory` must resolve
  inside `ANCHORA_CORPUS_ROOT` (default `data/corpus`, else `400`), symlinks
  that leave it are skipped, and file count and size are capped (`413`).
  Before this, a caller could index any `.md`/`.txt` the process could read
  and read it back through `/ask`.
- With no `ANCHORA_API_KEY` the API serves loopback peers only and answers
  `503` to anyone else, so the Docker image (bound to `0.0.0.0`) no longer
  starts open. The key is compared with `secrets.compare_digest`. **Breaking**
  for anyone calling a keyless server from another host: set a key.
- PII in the answer is redacted on `/ask`, and retracted then redacted on
  `/ask/stream` (ADR 8). Only the question was redacted before.
- `anyio` and `virtualenv` bumped past 7 advisories found by `pip-audit`;
  `fastapi` 0.143.0 and `mypy` 2.4.0.

### Changed
- The in-domain false-positive ceiling (0.05) is recorded as revoked: it was
  exceeded on its first measurement (4/46). `make adversarial` now fails if the
  blocked in-domain questions are not exactly the pinned four, in either
  direction (ADR 6 addendum). The hard look-alike ceiling (0.25) still gates,
  at 7/30 with no slack.
- `/ask/stream`: `done` carries `truncated: true` when the model stream breaks
  after the first token, and `output_pii_redacted`.
- CI: `uv sync --locked`, `permissions: contents: read`, actions pinned by SHA
  (node24 releases), a `pip-audit` step (`make audit` locally), and a
  security-only `dependabot.yml`. The Docker image pins `uv` 0.10.10.

### Documentation
- README: a Key results table under the demo; the faithfulness gate is stated
  as scored on extractive answers (a regression floor on retrieval plus
  extraction, not on model output); the promoted LoRA adapter is stated as a
  separate study, not the serving path (`qwen3:32b`, no adapter).
- The ceilings in `benign.json` and the rule in ADR 7 were declared before
  their measurements but committed with them; 0.6.0 said "declared before the
  first run" without that caveat. Future rules get their own commit first.
- Em dashes removed from the changelog, datasheet and fine-tuning notes.

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

## [0.5.0] - 2026-07-01

Retrieval quality, adversarial robustness, observability and documentation:
each claim measured, each limitation documented rather than hidden.

### Added
- **Hybrid retrieval**: a pure-Python Okapi BM25 index (`lexical.py`) fused with
  dense cosine via Reciprocal Rank Fusion; `retrieval_mode` (`dense|bm25|hybrid`,
  default `hybrid`) in config. Backed by `scripts/ablation_retrieval.py`
  (`make ablation`) measuring recall/precision/MRR per mode.
- **Adversarial guardrail suite**: `data/adversarial/attacks.json` (44 attacks:
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

## [0.4.0] - 2026-06-24

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

## [0.3.0] - 2026-06-24

### Added
- **LoRA/QLoRA fine-tuning** behind the optional `finetune` extra (PEFT/transformers/
  datasets/accelerate, with lazy imports and cuda/mps/cpu detection).
- `scripts/build_finetune_dataset.py`: generates instruction JSONL from the golden
  set + corpus using the deterministic offline retriever.
- `scripts/finetune_lora.py`: training entrypoint (`Qwen/Qwen2.5-3B-Instruct`).

## [0.2.0] - 2026-06-24

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

## [0.1.0] - 2026-06-24

### Added
- RAG pipeline: `chunking`, `embeddings` (Ollama `nomic-embed-text` + `hash`
  fallback), `store` (in-memory cosine VectorStore), `ingest`, `rag`, `llm`.
- Agent with tools: `search_documents` (RAG) + `legal_deadline`.
- FastAPI API (`/health`, `/ingest`, `/ask`) and CLI (`ingest`/`ask`/`eval`/`serve`).
- Brazilian legal-administrative corpus + golden set of 24 questions.
- Engineering: `uv`, `ruff`, `mypy --strict`, `pytest` with coverage, README + diagram.
