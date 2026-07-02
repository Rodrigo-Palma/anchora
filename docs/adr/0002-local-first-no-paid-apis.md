# 2. Local-first: Ollama for embeddings and generation, no paid APIs

Date: 2026-06-23 · Status: Accepted

## Context

The domain is Brazilian public-law documents handled inside a public
institution, where sending data to a third-party API can be a compliance and
privacy problem. The project also has to be runnable and auditable by anyone
with a clone, at no cost.

## Decision

All inference is **local-first via Ollama** (`nomic-embed-text` for embeddings,
`qwen3:32b` for generation). No hosted LLM or embedding API is used anywhere.
For tests and CI a deterministic **`hash` embedding provider** and an extractive
answer fallback stand in for the models, so the whole pipeline runs offline.

## Consequences

- Sensitive documents never leave the machine; no per-token cost.
- CI needs no secrets and no network — see [ADR 1](0001-deterministic-lexical-proxies-in-ci.md).
- The production embedding path is multilingual; the offline `hash` provider
  needs an explicit EN→PT glossary bridge to keep retrieval honest without a
  model (see [ADR 3](0003-hand-rolled-rag-over-a-framework.md)).
