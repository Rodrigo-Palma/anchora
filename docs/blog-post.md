# Building anchora: a legal RAG agent, local-first and with evals in CI

> Technical post draft. Tone: first person, direct, for an audience of
> software/ML engineers.

## The problem

Most portfolio RAG projects stop at "chat over a PDF": they retrieve
snippets, drop them into the prompt, and answer. They lack what makes an AI
system trustworthy in production — it **hallucinates** when it doesn't know,
has no **guardrails**, and there's no way to tell whether a change **improved
or degraded** quality.

I wanted to build the opposite of that, and with two self-imposed constraints:

1. **Zero cost.** No paid APIs. Everything runs locally.
2. **Reproducible CI.** Tests and evaluation have to run offline, without variance.

The result is `anchora` — a RAG agent over Brazilian public law (LAI,
Lei 8.112, LGPD, CPC, procurement/licitações) that **anchors** every answer
to its source: it cites `[n]` or abstains.

## Decision 1: local-first with a deterministic fallback

In local production, embeddings and generation come from Ollama (`nomic-embed-text` and
`qwen3:32b`). But LLMs are non-deterministic and Ollama doesn't fit in a free
CI runner. The solution was a **`hash` embedding provider**: it projects tokens
(accent-folded, pt-BR stopwords removed) into a unit-norm vector via signed hashing.

It isn't semantic — and it doesn't need to be. It validates the **mechanics** of
retrieval in a stable, free way. Switching to production is just a matter of
omitting `--provider hash`.

The first lesson came here: my initial version of `hash` got questions wrong
because common stopwords ("de", "a", "que") dominated the vectors. Adding
pt-BR stopword removal eliminated the retrieval failures.

## Decision 2: the agent uses tools, it doesn't just retrieve

Beyond `search_documents` (the RAG), there's `legal_deadline`: ask "a deadline of 15
business days starting from 2026-06-24?" and a rule-based planner detects a date +
number of days and calls the tool. Deterministic, testable, free.

## Decision 3: guardrails as code, not as prompt

Three layers, all without an LLM:

- **Input**: blocks injection/jailbreak ("ignore all previous instructions").
- **PII**: detects and **redacts** CPF/email/phone → `[REDACTED_CPF]`.
- **Output**: requires a `[n]` citation; without it, forces explicit abstention.

Because they're deterministic, they're **testable** and go into CI at no cost.

## Decision 4: evals that fail the build

This is the heart of the project. On every PR, `anchora eval` measures retrieval recall,
context precision, faithfulness, and answer relevance against a golden set of 24
questions, using **deterministic lexical proxies**. The *gate* fails the build if
recall < 1.0 or faithfulness < 0.70.

Why proxies and not an LLM *judge* in CI? Because a judge is non-deterministic
and (when hosted) costs money. The proxies provide an **objective, free floor**; the
LLM judge (DeepEval/RAGAS via Ollama) is reserved for richer local analysis.

## Decision 5: closing the MLOps loop

- **LoRA fine-tuning** (PEFT) with a dataset generated from the golden set itself + corpus.
- **File-backed model registry** with dev/staging/prod promotion — and promotion to
  `prod` only happens **if there's no regression** in the target metric.
- **Local ML pipeline** as an executable DAG + SageMaker skeleton.
- **Terraform** for ECR/S3/SageMaker/IAM, with `validate` in CI.

## What I'd take to real production

- Semantic embeddings in the index (already supported via Ollama) + a persistent
  vector store (pgvector/Qdrant) instead of the in-memory one.
- An LLM judge in the nightly evals, keeping the proxies as a fast PR gate.
- Observability (retrieval traces, latency, abstention rate).

## Conclusion

`anchora` demonstrates the full AI Engineering loop — retrieval, agent,
safety, **objective evaluation**, training, packaging, and infra — with software
rigor and **zero cost**. The piece I'm proudest of isn't the RAG: it's the **eval
gate in CI**. That's what separates "a demo that works on my machine" from a system
you can trust on every change.

---

*Code: [anchora](../README.md). Runs offline in one line: `make check`.*
