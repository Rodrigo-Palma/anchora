# anchora Architecture

Technical reference document. For an overview and usage, see the [README](../README.md).

## High-level view

`anchora` is a domain RAG agent with tool use and deterministic guardrails.
Every answer is **anchored** in the corpus: it either cites the source `[n]`,
or it abstains. The system is **local-first** — embeddings and generation via
Ollama, with a deterministic `hash` embedding provider so that tests and CI run
offline and reproducibly.

## Flow of a question

```
question
  → input guardrail (injection / jailbreak)        ── blocked → safe refusal
  → PII redaction (CPF / email / phone)
  → agent planner
       ├─ date + N days?  → legal_deadline tool (business / calendar days)
       └─ always          → search_documents tool → RAG (embed → cosine top-k)
  → generation (qwen3:32b + extractive fallback)
  → output guardrail (citation [n] or abstention)   ── ungrounded → abstention
  → cited answer + sources
```

## Layers

| Module (`src/anchora/`) | Responsibility | Key decisions |
|---|---|---|
| `config` | Settings via pydantic-settings | `env_prefix="ANCHORA_"`, local-first defaults |
| `chunking` | word windows with overlap | `size=180`, `overlap=40` |
| `embeddings` | Ollama `nomic-embed-text` + `hash` fallback | hash with signed-hashing, accent-fold, pt stopwords, unit-norm |
| `store` | in-memory `VectorStore` | cosine, JSON persistence |
| `ingest` | corpus → chunk → embed → store | reads `title:` front-matter |
| `rag` | `retrieve(store, query, k)` | top-k (`k=4`) |
| `llm` | generation with citations | `None`/extractive when offline |
| `tools` | `search_documents` + `legal_deadline` | deterministic tools |
| `agent` | orchestrates the full flow | rule-based planner |
| `guardrails` | input / PII / output | 3 deterministic layers |
| `metrics` | lexical proxies | faithfulness/relevance/precision/recall |
| `evals` | harness over golden set | CI *gate* |
| `api` | FastAPI | `/health`, `/ingest`, `/ask` |
| `cli` | command line | `ingest`/`ask`/`eval`/`serve` |

## Design decisions

### Why a `hash` embedding provider?

An LLM-based *judge*/embedding is non-deterministic and, on hosted services,
costs money. The `hash` provider projects tokens (accent-folded, without pt-BR
stopwords) into a unit-norm vector via signed hashing. It is not semantic, but it
is **stable, free, and sufficient** to validate the retrieval mechanics in CI.
The production path uses `nomic-embed-text` via Ollama; just omit `--provider
hash`.

### Why lexical proxies in the evals?

The *gate* must be objective and reproducible on every build. Lexical proxies for
faithfulness/recall provide a reliable floor with no cost or variance. LLM
*judges* (DeepEval/RAGAS via Ollama) remain available locally for richer analysis
— see `scripts/compare_evals.py`.

### Deterministic guardrails

Three layers, all LLM-free (therefore testable and free of cost):
1. **Input**: regex against injection/jailbreak patterns.
2. **PII**: detection and redaction of CPF/e-mail/phone → `[REDACTED_<KIND>]`.
3. **Output**: requires a `[n]` citation; otherwise, forces an explicit abstention.

### Dependency injection in FastAPI

The API-key guard uses `dependencies=[Depends(require_api_key)]` on the route
decorator (not as a function parameter — that would turn it into a query param
and cause a 422).

## MLOps (v0.3 / v0.4)

```
build-dataset (golden + corpus → JSONL)
  → LoRA finetune (optional, behind the `finetune` extra)
  → eval-and-register (ModelCard → ModelRegistry → promote if no regression)
```

- **Model registry** file-backed: `ModelCard` (metrics, stage, adapter) and
  `ModelRegistry` with dev/staging/prod promotion and `best`/`current` selection.
- **Local pipeline** (`pipeline/ml_pipeline.py`) as an executable DAG with
  `--dry-run`; **SageMaker skeleton** (`pipeline/sagemaker_pipeline.py`) with
  offline `describe()`.
- **Terraform** (`infra/`): ECR, S3, SageMaker Model Package Group, IAM. See
  [infra/README.md](../infra/README.md).

## Offline vs. production path

| | Offline / CI | Local production |
|---|---|---|
| Embeddings | `hash` (deterministic) | Ollama `nomic-embed-text` |
| Generation | extractive fallback | Ollama `qwen3:32b` |
| Network | none | localhost (Ollama) |
| Cost | zero | zero |
