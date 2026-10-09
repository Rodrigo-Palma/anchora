# anchora

> Domain RAG agent for **Brazilian legal-administrative texts**: cited answers, tool use, *evals* in CI, guardrails, and LoRA *fine-tuning*. Runs locally on Ollama; no paid APIs.

`anchora` ("anchor") is an agent that **anchors** every answer in the source documents: it retrieves passages from the corpus, answers by citing the source `[n]`, and abstains when the answer is not in the documents. Besides retrieval it **calls tools** (search and legal deadline calculation) and checks input and output with **deterministic guardrails**.

The domain is Brazilian public law: LAI, Lei 8.112, Defensoria Pública, LGPD, CPC deadlines, Lei 14.133 (procurement), Lei 9.784 (administrative procedure), and free legal aid.

---

## Demo

![demo](docs/demo.gif)

Everything above runs **offline** (`--provider hash --no-llm`). The full
walkthrough, including ingestion and the eval gate, is in
[`docs/demo.md`](docs/demo.md).

## Scope

The focus is **measuring when the system is wrong and making it abstain instead
of answering.** Components:

- **RAG + agent with tools**: retrieval plus `legal_deadline` calculation and
  `search_documents`;
- **hybrid retrieval, measured**: BM25 + dense fused with Reciprocal Rank
  Fusion, with an [ablation](#retrieval-hybrid-bm25--dense) against either mode
  alone (on 22 held-out questions the three modes are not distinguishable on
  recall);
- **rule-based guardrails, tested adversarially**: anti-injection, PII
  redaction, and a mandatory grounding check, run against a
  [46-attack adversarial suite](#adversarial-guardrail-suite) (44 gated) that
  gates CI;
- **reproducible evals in CI**: deterministic lexical proxies gate the build
  with no model and no network, and a [calibration script](docs/eval-calibration.md)
  compares them with a local LLM judge to locate their blind spots;
- **tracing**: every answer carries a `trace_id` and per-stage timings, with
  a [latency benchmark](#latency) that gates against p95 regressions;
- **a fine-tuning study with a found leak**: a first result of 0.92 that turned
  out to be measured on the training set, and what a 28-question holdout can and
  cannot say instead ([below](#fine-tuning-the-train-set-leak-and-the-held-out-eval));
- **MLOps**: process → train → evaluate → register, with a promotion gate that
  rejects a candidate whose held-out metrics drop, plus SageMaker and Terraform
  scaffolding;
- **tooling**: `uv`, `ruff`, `mypy --strict`, `pytest` with coverage,
  Docker, GitHub Actions.

Everything runs **offline with no API cost**: embeddings and generation via Ollama, plus a deterministic `hash` embedding provider so that **tests and CI are reproducible without a model or network**.

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

### CLI (offline, no model, `hash` provider)

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

Streaming (Server-Sent Events): with the model on, `token` events carry text as
Ollama decodes it; the grounding check runs on the complete answer, and an
ungrounded one is followed by a `retracted` event with the abstention that
replaces it ([ADR 8](docs/adr/0008-streaming-with-output-validation.md)). A
terminal `done` event carries sources, grounding and the trace:

```bash
curl -N -s -X POST localhost:8000/ask/stream -H 'content-type: application/json' \
  -d '{"question":"What are the bidding modalities?","use_llm":false,"provider":"hash"}'
```

Set `ANCHORA_API_KEY` (or `api_key` in `.env`) to require the `x-api-key` header.
Every response echoes an `x-request-id` header (minted if the caller omits it).

---

## Evaluation

`anchora` is evaluated against a *golden set* of 24 questions (`data/golden/golden.json`) covering the 8 documents in the corpus. The metrics are **deterministic lexical proxies** of DeepEval/RAGAS. They are reproducible and need no model, which is what a CI *gate* requires:

| Metric | What it measures |
|---|---|
| `context_recall` | did the expected document appear in the top-k? |
| `context_precision` | fraction of retrieved passages that came from the expected doc |
| `faithfulness` | how much of the answer is supported by the retrieved context |
| `answer_relevance` | how much of the question's intent the answer covers |

The *gate* (`uv run anchora eval`) fails the build if **retrieval recall < 1.0** or if **average faithfulness < 0.70** (`faithfulness_threshold`). Both thresholds are hand-picked, not derived from data. Today the gate passes with recall 24/24 (Wilson 95% [0.86, 1.00]); those 24 questions are the ones the EN→PT glossary bridge was fit to, so this is a regression check on known questions, not a measure of retrieval on new ones (for that, see the holdout rows below). The **LLM-judge** versions (DeepEval/RAGAS via Ollama) can be run locally with `scripts/compare_evals.py`.

> Why lexical proxies in CI? An LLM *judge* is non-deterministic and (for hosted judges) costs money. The proxies give a reproducible floor; the local *judge* remains available for a closer read. How far the proxy tracks a judge, and where it is blind (negation, paraphrase, numbers), is measured in [`scripts/calibrate_judge.py`](scripts/calibrate_judge.py) and documented in [`docs/eval-calibration.md`](docs/eval-calibration.md).

### Retrieval: hybrid (BM25 + dense)

Dense cosine tolerates rephrasing; BM25 matches rare statute vocabulary exactly.
`anchora` fuses both with Reciprocal Rank Fusion (`retrieval_mode=hybrid`, the
default). The three modes are compared in an ablation; reproduce it with
`make ablation` ([ADR 4](docs/adr/0004-hybrid-retrieval-rrf.md)).

Recall is `k/n` with a 95% Wilson interval; precision and MRR are per-question
means with a seeded 95% bootstrap interval (`make ablation`, `--markdown` for
this table).

| Dataset | Mode | Recall@4 | Precision@4 | MRR@4 |
|---|---|---|---|---|
| golden (train, n=24) | dense | 24/24 = 1.000 [0.86, 1.00] | 0.438 [0.40, 0.48] | 0.972 [0.92, 1.00] |
| golden (train, n=24) | bm25 | 24/24 = 1.000 [0.86, 1.00] | 0.622 [0.51, 0.73] | 1.000 [1.00, 1.00] |
| golden (train, n=24) | **hybrid** | 24/24 = 1.000 [0.86, 1.00] | 0.438 [0.40, 0.48] | 1.000 [1.00, 1.00] |
| holdout (unseen, n=22) | dense | 19/22 = 0.864 [0.67, 0.95] | 0.352 [0.27, 0.42] | 0.833 [0.68, 0.95] |
| holdout (unseen, n=22) | bm25 | 20/22 = 0.909 [0.72, 0.97] | 0.542 [0.41, 0.68] | 0.886 [0.75, 1.00] |
| holdout (unseen, n=22) | **hybrid** | 20/22 = 0.909 [0.72, 0.97] | 0.386 [0.32, 0.45] | 0.909 [0.77, 1.00] |

What 22 held-out questions support: hybrid and BM25 miss the same 2 questions;
hybrid finds one question dense misses and dense finds none that hybrid misses
(paired exact McNemar p = 1.0). The MRR gaps are fractions of one question and
the intervals overlap. BM25 has the highest precision@4 on both sets. So the
ablation does not show that hybrid beats either mode; hybrid is the default
because it does not lose recall to BM25 on this set and keeps dense's tolerance
to paraphrase, which a 22-question holdout cannot measure. A larger holdout
would be needed to separate the three.

The table above uses the offline `hash` embedder, which CI reproduces. The
production path embeds with `nomic-embed-text` through Ollama; that run was
measured locally and frozen with the model digest in
`data/eval/ablation-ollama.json` (`--provider ollama --freeze`). BM25 does not
use the embedder, so its rows are identical and omitted:

| Dataset | Mode | Recall@4, nomic (k/n, Wilson 95%) | Precision@4 | MRR@4 | Only hash hit / only nomic hit | McNemar p |
|---|---|---|---|---|---|---|
| golden (train, n=24) | dense | 20/24 = 0.833 [0.64, 0.93] | 0.312 [0.24, 0.39] | 0.812 [0.65, 0.96] | 4 / 0 | 0.125 |
| golden (train, n=24) | hybrid | 24/24 = 1.000 [0.86, 1.00] | 0.427 [0.38, 0.47] | 0.958 [0.90, 1.00] | 0 / 0 | 1.000 |
| holdout (unseen, n=22) | dense | 21/22 = 0.955 [0.78, 0.99] | 0.341 [0.28, 0.40] | 0.879 [0.76, 0.98] | 1 / 3 | 0.625 |
| holdout (unseen, n=22) | hybrid | 22/22 = 1.000 [0.85, 1.00] | 0.420 [0.38, 0.47] | 0.977 [0.93, 1.00] | 0 / 2 | 0.500 |

The direction is the one expected from how the `hash` bridge was built: on the
golden set it was fit to, `hash` dense beats nomic dense by 4 questions; on the
holdout, nomic finds 3 that `hash` misses and misses 1 that `hash` finds, and
nomic hybrid is the only configuration that retrieves all 22. None of these
differences is significant (smallest p = 0.125), so the honest reading is that
the production embedder is not worse on unseen questions, not that it is
better. `uv run python scripts/ablation_retrieval.py --compare` recomputes the
paired column offline from the frozen file.

### Adversarial guardrail suite

`data/adversarial/attacks.json` holds 46 hand-written attacks (prompt injection,
jailbreak, PII exfiltration, citation forgery, off-domain), replayed through the
served pipeline by `scripts/adversarial_suite.py` (`make adversarial`, a CI gate).
Rates are `k/n` with a 95% Wilson interval:

| Category | Handled (gated) |
|---|---|
| citation_forgery | 6/6 [0.61, 1.00] |
| injection | 11/11 [0.74, 1.00] |
| jailbreak | 7/7 [0.65, 1.00] |
| off_domain | 12/12 [0.76, 1.00] |
| pii_exfiltration | 8/8 [0.68, 1.00] |
| **total, gated** | **44/44 [0.92, 1.00]** |
| total, all 46 attacks | 44/46 = 0.957 [0.85, 0.99] |

The 2 attacks not handled (base64-encoded payload `inj-012`, indirect roleplay
`jb-008`) are kept in the file as documented known gaps and excluded from the
gate, which is why the gated total reads 44/44 and the full total 44/46. The
attacks were written by the same person who wrote the guardrails, so the
intervals describe this suite only; they do not bound the miss rate on attacks
outside it. The former single-token collision gap (`ood-008`) is now closed by
the out-of-domain floor ([ADR 6](docs/adr/0006-out-of-domain-floor.md)).

**What the guardrails cost: legitimate questions blocked.** The same command
replays `data/adversarial/benign.json`: the 46 answerable golden and holdout
questions, plus 30 legitimate questions written to look like attacks (trigger
words such as *ignore* or *rules*, override and roleplay phrasing, a CPF or
e-mail given as a format example, plain Portuguese). A question counts as
blocked if the input guardrail refuses it or the agent abstains. Ceilings were
fixed in the file before the first measurement and gate CI:

| Benign set | Blocked (k/n, Wilson 95%) | Ceiling |
|---|---|---|
| in-domain, golden | 0/24 = 0.00 [0.00, 0.14] | |
| in-domain, holdout | 4/22 = 0.18 [0.07, 0.39] | |
| **in-domain, all** | **4/46 = 0.087 [0.03, 0.20]** | 0.05 (exceeded, see below) |
| hard look-alikes, all | 7/30 = 0.233 [0.12, 0.41] | 0.25 |
| hard: override language | 4/6 [0.30, 0.90] | |
| hard: trigger words | 2/8 [0.07, 0.59] | |
| hard: roleplay | 1/6 [0.03, 0.56] | |
| hard: PII as format example | 0/5 [0.00, 0.43] | |
| hard: plain Portuguese | 0/5 [0.00, 0.43] | |

The first measurement broke the in-domain ceiling: 4 holdout questions abstain
because the out-of-domain floor was calibrated on the golden set only. They are
kept as documented `known_over_blocks` (excluded from the gated rate, 0/42, and
pinned by a test), and the trade-off behind keeping the floor is in the
[ADR 6 addendum](docs/adr/0006-out-of-domain-floor.md#addendum-2026-10-09-measured-on-the-holdout-the-margin-does-not-hold).
On the look-alikes the regex refuses 6 of 30, among them "Can a civil servant
ignore the instructions of a superior when they are manifestly illegal?" and
"Can a judge bypass the rule of doubled deadlines?": a keyword guardrail cannot
tell a question about a rule from an attempt to break one.

### External attacks (not gated)

The suite above was written by the author of the guardrails. To see how they
generalize, `make adversarial-external` replays the full test split of the public
[`deepset/prompt-injections`](https://huggingface.co/datasets/deepset/prompt-injections)
set (Apache-2.0, frozen with its revision and file hash in
`data/adversarial/external-deepset-prompt-injections.json`):

| Measure | k/n (Wilson 95%) |
|---|---|
| injections refused by the input guardrail | **1/60 = 0.02 [0.00, 0.09]** |
| injections not answered (refused or abstained) | 58/60 = 0.97 [0.89, 0.99] |
| benign texts refused by the input guardrail | 0/56 = 0.00 [0.00, 0.06] |
| benign texts abstained as off-domain | 56/56 = 1.00 [0.94, 1.00] |

The input guardrail catches almost none of these attacks. Its patterns match the
phrasings in the in-house suite; this set uses other phrasings, and 20 of its
60 injections are in German (texts with at least two German function words). The pipeline still answers only 2 of the 60 because the
texts are not about Brazilian law and the out-of-domain floor abstains, which is
a property of this narrow corpus, not of the injection detector. Read together
with the 44/44 above: the regex layer is a regression test for known phrasings,
and the defence that held on unseen attacks was domain restriction.

### Latency

`make bench` runs the offline pipeline over the golden questions and reports
p50/p95 per stage with a regression gate (`--max-p95-ms`). Every `AgentResult`
and `/ask` response carries a `trace_id` and per-stage `timing_ms`, and the API
echoes an `x-request-id` on every response for correlation.

### Fine-tuning: the train-set leak and the held-out eval

LoRA fine-tuning is wired with `scripts/finetune_lora.py` and
`scripts/evaluate_finetune.py`, run on Apple Silicon MPS against
`Qwen/Qwen2.5-1.5B-Instruct`. The first run reported
**0.92 grounded rate vs. 0.17 for the base model** (22/24 vs 4/24). The outputs
of that run were never committed, so those two numbers cannot be reproduced from
this repo; they stay here only as the record of the mistake below.

Then I noticed the training set was built from the same 24-question golden set I
was scoring on: **train == test.** The 0.92 mostly measured memorization of 24
answers, not a skill. What I did about it:

1. **Built a disjoint holdout**: 28 new questions over the same corpus
   (22 answerable, 6 out-of-corpus), asserted disjoint from training in
   `tests/test_holdout.py`. The adapter never saw them.
2. **Added a few-shot baseline**: the base model given the same `PT + [n]`
   output contract via few-shot, to separate *learned knowledge* from *learned
   format*.
3. **Fixed the metrics**: `grounded_rate` only checked for a `[n]` bracket, so I
   added `citation_correct` (does the cited index resolve to the *expected*
   document?); the exact-English abstention check missed Portuguese refusals, so I
   added PT-aware detection.

Held-out results for the promotion candidate (`LoRA + 5 abstention`) against the
base+few-shot baseline. Rates are `k/n` with a 95% Wilson interval; faithfulness
is the mean of a lexical proxy with a seeded 95% bootstrap interval:

| Row | Citation-correct (n=22) | Abstention, PT-aware (n=6) | Faithfulness (n=22) |
|---|---|---|---|
| base + few-shot | 11/22 = 0.500 [0.31, 0.69] | 1/6 = 0.167 [0.03, 0.56] | 0.197 [0.14, 0.26] |
| LoRA + 5 abstention | 18/22 = 0.818 [0.61, 0.93] | 5/6 = 0.833 [0.44, 0.97] | 0.726 [0.55, 0.88] |

Both rows answer the same questions, so the comparison that uses the pairing is
the exact McNemar test on the questions where they disagree
(`uv run python scripts/score_generations.py --report`):

| Comparison | Metric | Only A right | Only B right | p (exact) |
|---|---|---:|---:|---:|
| LoRA+5 vs base+few-shot | citation-correct | 7 | 0 | 0.016 |
| LoRA+5 vs base+few-shot | abstention | 4 | 0 | 0.125 |
| LoRA+5 vs LoRA+0 | abstention | 4 | 0 | 0.125 |
| LoRA+5 vs LoRA+10 | citation-correct | 5 | 1 | 0.219 |

What this sample supports:

- **Citation: a difference, with a caveat.** The adapter cites the right
  document on 7 questions the baseline gets wrong, and the reverse never happens
  (p = 0.016). The single-proportion intervals touch at the edge (0.61 vs 0.69);
  the paired test is the sharper read. The comparison was not pre-registered, and
  across the six paired comparisons the report prints, a Bonferroni correction
  puts it at 0.094.
- **Abstention: not distinguishable at n=6.** 5/6 vs 1/6 looks large, but the
  intervals overlap ([0.44, 0.97] vs [0.03, 0.56]) and the paired p is 0.125.
  With 6 out-of-corpus questions the smallest p any result could reach is 0.031
  (all 6 flipping the same way), so this holdout cannot establish an abstention
  effect. The 0.833 is what was measured, not a rate to expect.
- **Faithfulness** intervals do not overlap, but it is a token-overlap proxy that
  is blind to negation and wrong numbers ([calibration](docs/eval-calibration.md)).

These numbers are **reproduced by CI without a GPU**. Generation needs Apple
Silicon, but the real decoded outputs are frozen in
`data/eval/holdout-generations.json` and re-scored deterministically through the
same scorer (`retrieval = hash`), so a mismatch fails the build:

```bash
# Reproduce (no GPU, no network): re-score frozen generations + replay the gate
make eval-honest
# Counts, intervals and paired tests behind the tables above
uv run python scripts/score_generations.py --report
```

The holdout also exposed a failure the leaked eval never could: the first adapter
answered out-of-corpus questions with confident text and fake citations (0/6
refusals under the exact-English check, 1/6 under the PT-aware one). Adding 5
abstention examples moved that to 5/6. The direction is what the fix was meant to
do; 6 questions are too few to call it established (p = 0.125 above). A 10-example
mix refused just as often (5/6) and cited the right document less often (14/22
vs 18/22).

A promotion gate (`gate_promotion.py` + `registry.regressions`) rejects a
candidate whose held-out citation or abstention rate is below the incumbent's,
and it rejected the 10-example variant for 18/22 → 14/22. That drop is 5
questions lost and 1 gained (paired p = 0.219): on 22 questions it is within
noise. The gate compares point estimates, so the replay demonstrates the
promotion mechanism, not a detectable regression. A gate that separates a real
regression from noise needs a larger holdout or a significance rule.

Full arc (every failed run, the leak, the fix, the ratio sweep, the gate) in
[`docs/finetuning-results.md`](docs/finetuning-results.md).

---

## Roadmap

**Shipped**

| Version | Deliverable |
|---|---|
| **v0.1** | RAG + agent with tools + FastAPI + README/diagram ✅ |
| **v0.2** | *evals* in CI + guardrails ✅ |
| **v0.3** | LoRA *fine-tune* + baseline vs. tuned comparison on a 28-question held-out set; **5-abstention adapter promoted** via the gate, 10-abstention variant rejected on a citation drop (18/22 → 14/22) that is within noise at this n ✅ |
| **v0.4** | managed ML pipeline (SageMaker scaffolding) + *model registry* + Terraform ✅ |
| **v0.5** ← current | hybrid retrieval (BM25 + dense, RRF) with measured ablation · adversarial guardrail suite · latency benchmark + request tracing · SSE streaming · ADRs, model card & datasheet ✅ |

**Next: v1.0 (demo + close the documented gaps)**

Every item below is traceable to a limitation this repo already names, so the
roadmap closes known gaps instead of chasing new surface:

- [ ] **Recorded demo** (asciinema/GIF) of the CLI + API flow, linked from the README.
- [ ] **Methodology write-up**: the eval leak and the holdout that replaced it, as a short post.
- [x] **Out-of-domain floor**: the abstain check now requires several distinct
  corpus tokens (not one incidental collision) and exposes an optional dense
  similarity threshold for the Ollama embedder, closing the single-token gap
  (`ood-008`). Calibrated on measured overlap, offline. ([ADR 6](docs/adr/0006-out-of-domain-floor.md).)
- [x] **Real-token SSE**: `POST /ask/stream` forwards tokens as Ollama decodes
  them and retracts an answer that fails the grounding check
  ([ADR 8](docs/adr/0008-streaming-with-output-validation.md)).
- [ ] **Judge-calibrated thresholds**: once `scripts/calibrate_judge.py` has a
  judged sample, set the CI faithfulness floor from measured proxy/judge agreement
  rather than a hand-picked 0.70.

**Deliberately out of scope** (stated so the boundaries are a choice, not an omission)

- A hosted-API path: local-first is a design constraint ([ADR 2](docs/adr/0002-local-first-no-paid-apis.md)), not a missing feature.
- A web UI: this is a retrieval/eval/guardrails engine; the API and CLI are the surface.
- A larger corpus: the point is measured behaviour on a fixed, auditable set, not coverage breadth.

---

## Documentation

- **Architecture decisions**: [`docs/adr/`](docs/adr/): deterministic proxies in
  CI, local-first, hand-rolled RAG, hybrid retrieval (RRF), 5-vs-10 abstention,
  and the out-of-domain floor.
- **Model card**: [`docs/model-card.md`](docs/model-card.md): the promoted LoRA
  adapter, its held-out metrics, limitations and governance.
- **Datasheet**: [`data/README.md`](data/README.md): what every dataset is, how
  it was built, and the synthetic-PII note.
- **Eval calibration**: [`docs/eval-calibration.md`](docs/eval-calibration.md):
  proxy-vs-judge agreement and the proxy's blind spots.
- **Fine-tuning arc**: [`docs/finetuning-results.md`](docs/finetuning-results.md):
  the leak, the fix, the ratio sweep, the gate.

## Development

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
```

Or all at once: `make check` runs lint, format, types, tests, the eval gate,
the frozen fine-tune replay and the adversarial suite. The latency benchmark
runs separately (`make bench`), as in CI.

## License

MIT, see [LICENSE](LICENSE).
