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

- Measured in `scripts/ablation_retrieval.py` on the 22 answerable holdout
  questions: recall@4 is 20/22 for hybrid and BM25 (Wilson 95% [0.72, 0.97]) and
  19/22 for dense ([0.67, 0.95]). Hybrid and BM25 miss the same 2 questions;
  hybrid vs dense differs on 1 question (paired McNemar p = 1.0). MRR is 0.909
  [0.77, 1.00] for hybrid vs 0.833 [0.68, 0.95] for dense, intervals overlapping.
  BM25 has the highest precision@4 (0.542 [0.41, 0.68] vs 0.386 [0.32, 0.45]).
- So the ablation does not show hybrid beating either mode at this n. The
  default rests on the design argument in Context (rank fusion keeps BM25's exact
  terms and dense's paraphrase tolerance) plus "no recall lost vs BM25 here".
  Re-run the ablation, ideally on a larger holdout, before changing the default.
- Two rankings per query; negligible at this corpus size, revisit at scale.
