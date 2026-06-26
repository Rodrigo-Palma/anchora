# anchora

> Domain RAG agent for **Brazilian legal-administrative texts** — answers with citations, tool use, *evals* in CI, guardrails, and LoRA *fine-tuning*. **100% local-first** (Ollama), with no paid APIs.

`anchora` ("anchor") is an agent that **anchors** every answer in the source documents: it retrieves passages from the corpus, answers by citing the source `[n]`, and **refuses to make things up** — if the answer is not in the documents, it abstains. It goes beyond a simple RAG by **using tools** (search + legal deadline calculation) and by validating input and output with **deterministic guardrails**.

The domain is Brazilian public law: LAI, Lei 8.112, Defensoria Pública, LGPD, CPC deadlines, Lei 14.133 (procurement), Lei 9.784 (administrative procedure), and free legal aid.

---

## Why this project

It is an end-to-end **AI/ML Engineering** *engineering project*:

- **RAG + agent with tools** (not just "chat over a PDF");
- **production guardrails** (anti-injection, PII redaction, mandatory *grounding*);
- **automated evaluation in CI** with an objective *gate* (retrieval recall + faithfulness);
- **LoRA/QLoRA fine-tuning** with a baseline vs. tuned comparison;
- **ML pipeline** (process → train → evaluate → register) + SageMaker and Terraform scaffolding;
- **engineering**: `uv`, `ruff`, `mypy --strict`, `pytest` with coverage, Docker, GitHub Actions.

Everything runs **offline and for free**: embeddings and generation via Ollama, plus a deterministic `hash` embedding provider so that **tests and CI are reproducible without a model or network**.

---

## Architecture

```mermaid
flowchart TD
    Q[User question] --> G1{Input guardrail<br/>injection / jailbreak}
    G1 -- blocked --> R[Safe refusal]
    G1 -- ok --> PII[PII redaction<br/>CPF / email / phone]
    PII --> PLAN[Agent planner]

    PLAN -->|date + N days| T2[Tool: legal_deadline<br/>business / calendar days]
    PLAN --> T1[Tool: search_documents]

    subgraph RAG
        T1 --> EMB[embed_texts<br/>Ollama nomic-embed-text<br/>or deterministic hash]
        EMB --> VS[(VectorStore<br/>cosine top-k)]
        VS --> CTX[Numbered context]
    end

    CTX --> GEN[Generation<br/>Ollama qwen3:32b<br/>+ extractive fallback]
    T2 --> GEN
    GEN --> G2{Output guardrail<br/>citation or abstention}
    G2 -- ungrounded --> ABS[Explicit abstention]
    G2 -- ok --> A[Cited answer + sources]

    subgraph Offline
        CORP[legal corpus] --> ING[ingest: chunk + embed]
        ING --> VS
        GOLD[golden set] --> EVAL[evals: recall / faithfulness<br/>CI gate]
        VS --> EVAL
    end
```

Layers (`src/anchora/`):

| Module | Responsibility |
|---|---|
| `chunking` | splits text into overlapping word windows |
| `embeddings` | Ollama `nomic-embed-text` + deterministic `hash` fallback (unit-norm, *accent-folded*) |
| `store` | in-memory `VectorStore`, cosine search, JSON persistence |
| `ingest` | reads the corpus (`title:` front-matter), *chunk* → *embed* → store |
| `rag` | `retrieve(store, query, k)` |
| `llm` | generation with citations via Ollama; `None` when offline |
| `tools` | `search_documents` (RAG) + `legal_deadline` (business/calendar days) |
| `agent` | orchestrates guardrails → planner → tools → answer → validation |
| `guardrails` | anti-injection, PII detection/redaction, output *grounding* |
| `metrics` | deterministic lexical proxies for faithfulness / relevance / precision / recall |
| `evals` | offline harness over the *golden set* + CI *gate* |
| `api` | FastAPI: `/health`, `/ingest`, `/ask` (API-key + PII redaction) |
| `cli` | `ingest` / `ask` / `eval` / `serve` |

---

## Installation

Prerequisites: Python ≥ 3.12 and [`uv`](https://docs.astral.sh/uv/). To use the local models, [Ollama](https://ollama.com) with:

```bash
ollama pull nomic-embed-text
ollama pull qwen3:32b
```

```bash
uv sync --extra dev
```

---

## Usage

### CLI (offline, no model — `hash` provider)

```bash
# Ask (deterministic extractive fallback, no LLM)
uv run anchora ask "What are the bidding modalities?" --provider hash --no-llm

# Deadline calculation (the agent detects a date + N days and calls the tool)
uv run anchora ask "Deadline of 15 business days from 2026-06-24?" --provider hash --no-llm

# Index a corpus and save the index
uv run anchora ingest --corpus data/corpus --out store.json --provider hash

# Run the evaluation gate
uv run anchora eval
```

### With the local models (Ollama)

```bash
# omit --provider/--no-llm to use nomic-embed-text + qwen3:32b
uv run anchora ask "What is the appeal deadline under the LAI?"
```

### API

```bash
uv run anchora serve              # http://127.0.0.1:8000  (/docs for Swagger)
```

```bash
curl -s localhost:8000/health
curl -s -X POST localhost:8000/ingest -H 'content-type: application/json' \
  -d '{"provider":"hash"}'
curl -s -X POST localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question":"What are the bidding modalities?","use_llm":false,"provider":"hash"}'
```

Set `ANCHORA_API_KEY` (or `api_key` in `.env`) to require the `x-api-key` header.

---

## Evaluation

`anchora` is evaluated against a *golden set* of 24 questions (`data/golden/golden.json`) covering the 8 documents in the corpus. The metrics are **deterministic lexical proxies** of DeepEval/RAGAS — honest and reproducible, suitable for a CI *gate* at no cost:

| Metric | What it measures |
|---|---|
| `context_recall` | did the expected document appear in the top-k? |
| `context_precision` | fraction of retrieved passages that came from the expected doc |
| `faithfulness` | how much of the answer is supported by the retrieved context |
| `answer_relevance` | how much of the question's intent the answer covers |

The *gate* (`uv run anchora eval`) fails the build if **retrieval recall < 1.0** or if **average faithfulness < 0.70** (`faithfulness_threshold`). The **LLM-judge** versions (DeepEval/RAGAS via Ollama) can be run locally — see `scripts/compare_evals.py`.

> Why lexical proxies in CI? An LLM *judge* is non-deterministic and (for hosted judges) costs money. The proxies provide an objective, free floor; the local *judge* remains available for richer analysis.

### Fine-tuning comparison

LoRA fine-tuning is wired and measured with `scripts/finetune_lora.py` and
`scripts/evaluate_finetune.py`. Local benchmarks were run on Apple Silicon MPS
with `Qwen/Qwen2.5-0.5B-Instruct` and `Qwen/Qwen2.5-1.5B-Instruct`, comparing
each base model against LoRA adapters on the same 24-case golden set.

Best stable results so far:

| Base model | Variant | Grounded rate | Faithfulness | Answer relevance | Reference overlap |
|---|---|---:|---:|---:|---:|
| `Qwen2.5-1.5B` | Base | 0.1667 | 0.2668 | 0.8376 | 0.1668 |
| `Qwen2.5-1.5B` | LoRA + early stop | 0.9167 | 0.9208 | 0.0458 | 0.7338 |

Decision: **promote only as an experimental adapter**. The early-stopped LoRA
run substantially improved grounded/cited outputs, faithfulness and overlap with
the reference answers. The low `answer_relevance` is expected for this lexical
proxy because the questions are English and the tuned answers are concise
Portuguese legal answers. See `docs/finetuning-results.md` for commands, failed
runs, and next iteration.

---

## Roadmap

| Version | Deliverable |
|---|---|
| **v0.1** | RAG + agent with tools + FastAPI + README/diagram ✅ |
| **v0.2** | *evals* in CI + guardrails ✅ |
| **v0.3** | LoRA *fine-tune* + baseline vs. tuned comparison measured; adapter not promoted yet ⚠️ |
| **v0.4** | managed ML pipeline (SageMaker scaffolding) + *model registry* + Terraform ✅ |
| **v1.0** | demo + technical post + eval methodology write-up |

---

## Development

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
```

Or all at once: `make check`.

## License

MIT — see [LICENSE](LICENSE).
