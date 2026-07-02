# 1. Deterministic lexical proxies gate CI; the LLM judge stays local

Date: 2026-06-24 · Status: Accepted

## Context

RAG quality (faithfulness, answer relevance, context precision/recall) is
usually scored with an LLM judge (DeepEval/RAGAS). An LLM judge is
non-deterministic and, for hosted judges, costs money per run. A CI gate must
be reproducible and free, or it cannot block a merge honestly.

## Decision

CI gates on **deterministic lexical proxies** (`anchora.metrics`): accent-folded
token-overlap stand-ins for each metric, computed with the offline `hash`
retriever. The richer **LLM judge remains available locally** and is *calibrated*
against the proxy (`scripts/calibrate_judge.py`) so we know how far the cheap
signal tracks the expensive one instead of assuming it.

## Consequences

- CI is free, deterministic, and re-runs identically on any machine.
- The proxy has known blind spots (negation, paraphrase, numeric correctness) —
  documented in [`eval-calibration.md`](../eval-calibration.md), not hidden.
- A metric regression fails the build objectively; the judge adds depth on
  demand without ever being on the critical path.
