# 4. Hybrid retrieval (dense + BM25) fused with Reciprocal Rank Fusion

Date: 2026-07-01 · Status: Accepted

## Context

Dense cosine retrieval generalizes across phrasing but blurs rare, exact statute
vocabulary ("ultrassecreta", "estágio probatório"); BM25 nails those exact terms
but misses paraphrase. Choosing one leaves recall on the table. Combining raw
scores would require calibrating two incomparable scales.

## Decision

Retrieve with both and fuse by **Reciprocal Rank Fusion**: `score = Σ 1/(k+rank)`
over each list, `k=60`. RRF uses only ranks, so no score calibration is needed
and the fusion stays deterministic. `retrieval_mode` (`dense|bm25|hybrid`,
default `hybrid`) is configurable; frozen fine-tune replays pin `dense` so past
experiments are re-scored as they ran.

## Consequences

- Measured in `scripts/ablation_retrieval.py`: on the unseen holdout, hybrid
  reaches BM25-level recall (0.909) while beating dense on MRR (0.909 vs 0.833).
- The default is backed by an ablation table, not intuition — re-run it before
  changing the default.
- Two rankings per query; negligible at this corpus size, revisit at scale.
