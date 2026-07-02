# 3. Hand-rolled RAG core instead of a framework

Date: 2026-06-23 · Status: Accepted

## Context

LangChain/LlamaIndex accelerate a RAG prototype but hide the retrieval, chunking
and grounding logic behind abstractions. For a portfolio project whose whole
point is *measuring* retrieval and grounding, that opacity is a liability, and
the heavy dependency tree works against the offline, deterministic-CI goal.

## Decision

Implement the core in plain Python: chunking, an in-memory `VectorStore` with
cosine search, a pure-Python BM25 index, RRF fusion, and rule-based guardrails.
Ollama is the only external moving part, behind a thin client.

## Consequences

- Every ranking and grounding decision is inspectable and unit-testable.
- No framework version churn; the dependency surface stays small.
- We own code a framework would provide (e.g. the BM25 index) — accepted,
  because that code is exactly the part being demonstrated and measured.
