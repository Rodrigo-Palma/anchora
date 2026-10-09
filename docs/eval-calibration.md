# Eval calibration: how far the lexical proxy tracks an LLM judge

The CI gate scores faithfulness with a **deterministic lexical proxy**
(`anchora.metrics.faithfulness`: the share of the answer's content tokens that
occur in the retrieved context) rather than an LLM judge, because a gate must be
reproducible and free ([ADR 1](adr/0001-deterministic-lexical-proxies-in-ci.md)).
That trade is acceptable only if we know how far the proxy and a judge agree.
This page reports that agreement, measured.

## Setup

- **Items.** Every answerable held-out question (22) answered by each of the
  five frozen fine-tune arms (`data/eval/holdout-generations.json`), minus the
  10 generations that abstain (faithfulness is undefined for a refusal):
  **100 generations over 22 questions**. Context is the same retrieval the
  generations were produced under (`hash`, dense, k=4).
- **Judges.** `qwen3:32b` and `gemma4:31b-it-qat` through Ollama 0.35.1, at
  temperature 0 with a fixed seed and thinking off, constrained to a JSON reply
  on a 0-4 rubric (every claim supported ... no claim supported), mapped to
  0..1. A score of 3 or 4 counts as "faithful". `qwen3:32b` was run twice to
  check determinism.
- **Frozen.** Per-item proxy and judge scores, the model digests and the SHA-256
  of the judge prompt are in `data/eval/judge-scores.json`.
  `make calibration` recomputes every number below from that file with no model
  and fails if any of them drifts, if the proxy changes, or if the prompt no
  longer matches the hash the scores were produced with.
- **Uncertainty.** Several generations answer the same question, so they are not
  independent: Spearman intervals come from a bootstrap that resamples the 22
  questions, not the 100 rows. Rates are `k/n` with a Wilson interval, which
  treats the 100 generations as independent and is therefore optimistic.

```bash
make calibration                                                     # offline
uv run python scripts/calibrate_judge.py --run qwen3-32b --model qwen3:32b   # needs Ollama
```

## Results

Proxy against each judge:

| Judge | n (questions) | Judge says faithful | Spearman [95% CI] | MAE | Proxy >= 0.5: agrees, kappa | Proxy >= 0.7: agrees, kappa | Kappa-best cutoff [95% CI]: agrees, kappa |
|---|---|---|---|---|---|---|---|
| qwen3:32b | 100 (22) | 74/100 = 0.740 [0.65, 0.82] | 0.498 [0.29, 0.68] | 0.290 | 68/100 [0.58, 0.76], 0.37 | 66/100 [0.56, 0.75], 0.36 | 0.158 [0.06, 0.60]: 78/100 [0.69, 0.85], 0.43 |
| gemma4:31b | 100 (22) | 65/100 = 0.650 [0.55, 0.74] | 0.490 [0.25, 0.72] | 0.280 | 69/100 [0.59, 0.77], 0.39 | 71/100 [0.61, 0.79], 0.44 | 1.000 [0.29, 1.00]: 73/100 [0.64, 0.81], 0.49 |

Judge against judge, the ceiling any proxy could reach against "the judge":

| Pair | n | Same score (0-4) | Same verdict | Cohen kappa | Spearman [95% CI] |
|---|---|---|---|---|---|
| qwen3:32b vs gemma4:31b | 100 | 75/100 [0.66, 0.82] | 83/100 [0.74, 0.89] | 0.603 | 0.778 [0.58, 0.92] |
| qwen3:32b vs qwen3:32b (repeat) | 100 | 100/100 [0.96, 1.00] | 100/100 [0.96, 1.00] | 1.000 | 1.000 |

Mean score per system (arm):

| Arm | n | Mean proxy | Mean judge, qwen3:32b | Mean judge, gemma4:31b |
|---|---|---|---|---|
| base, zero-shot | 22 | 0.204 | 0.636 | 0.557 |
| base, few-shot | 21 | 0.205 | 0.524 | 0.429 |
| LoRA, 0 abstention | 20 | 0.848 | 0.825 | 0.850 |
| LoRA, 5 abstention | 20 | 0.785 | 0.887 | 0.925 |
| LoRA, 10 abstention | 17 | 0.805 | 0.809 | 0.912 |

Power: with 22 independent questions the smallest correlation this design
detects (alpha 0.05, power 0.80) is 0.57; if all 100 generations were
independent it would be 0.28. The truth is in between.

## Reading

- **The proxy tracks the judges moderately, and about equally for both.**
  Spearman is about 0.49 against either judge, with intervals from 0.25 to 0.72.
  The two judges agree with each other more (Spearman 0.78, kappa 0.60), so the
  proxy reaches roughly two thirds of the judge-to-judge agreement.
- **The judges are not interchangeable.** They give the same 0-4 score on 75 of
  100 items and the same verdict on 83. A single judge is a noisy reference,
  not ground truth; no human labels exist here.
- **The repeat run measures determinism, not reliability.** At temperature 0
  with a fixed seed, `qwen3:32b` reproduced every score. That makes the frozen
  file reproducible; it says nothing about whether the scores are right.
- **There is no stable per-item cutoff.** The cutoff that maximizes kappa is
  0.158 against `qwen3:32b` and 1.000 against `gemma4:31b`, and each bootstrap
  interval is wide ([0.06, 0.60] and [0.29, 1.00]). The kappa curve is flat:
  against `qwen3:32b` every observed cutoff from 0.037 to 1.000 gives kappa
  between 0.22 and 0.43. Kappa is used instead of raw agreement because 74% of items are
  faithful, and raw agreement is maximized by calling everything faithful.
- **The proxy is much harsher than the judges on the base model.** The base
  model answers in English over a Portuguese context; the proxy scores those
  answers about 0.20 while the judges score them 0.43 to 0.64. A likely cause is
  vocabulary mismatch that the glossary bridge does not cover; it was not
  isolated by an experiment. On the LoRA arms the two scales roughly agree.
  The 0.197 vs 0.726
  proxy faithfulness gap between the few-shot baseline and the adapter in the
  README (all 22 answerable questions) is therefore larger than either judge
  sees on the non-abstaining generations (0.52 vs 0.89 for `qwen3:32b`, 0.43
  vs 0.93 for `gemma4:31b`). The direction is the same on all three scales.
- **At the level of systems, the proxy ranks like the judges on this sample.**
  Both judges put every LoRA arm above both base arms, and so does the proxy.
  With five systems this is weak evidence, but it is the level at which the CI
  floor operates.

What this means for the CI floor of 0.70 is decided in
[ADR 7](adr/0007-faithfulness-threshold.md): the value stays, with the measured
reasons.

## Known blind spots of the lexical proxy

The proxy is token-overlap based, so by construction it cannot see:

- **Negation**: "the deadline is *not* 10 days" overlaps the context as much as
  the correct claim.
- **Paraphrase and language**: a correct answer worded differently, or in
  English over a Portuguese context, scores lower than it should (measured
  above). The EN→PT glossary bridge softens this but does not remove it.
- **Numeric correctness**: "20 days" vs "30 days" are one token apart.

The judges are not checked for these either: this calibration measures
agreement, and no item set targets negation or numbers specifically.
