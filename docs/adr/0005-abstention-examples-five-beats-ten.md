# 5. Five abstention examples, not ten; a promotion gate enforces it

Date: 2026-06-25 · Status: Accepted

## Context

The first LoRA adapter rarely abstained: on out-of-corpus questions it fabricated
confident answers with fake citations (0/6 refusals under the exact-English check,
1/6 under the later PT-aware one; 6 held-out out-of-corpus questions in total). Adding abstention
examples to the training set fixes that — but too many teach the model to refuse
answerable questions, trading one failure for another.

## Decision

Train with **5 abstention examples**. In the held-out sweep, 5 and 10 examples
refused equally often (5/6 each) and 5 cited the right document more often
(18/22 vs 14/22, Wilson 95% [0.61, 0.93] vs [0.43, 0.80]). That citation gap is
5 questions lost and 1 gained, paired McNemar p = 0.219: not distinguishable on
22 questions. With no measured advantage for 10, the smaller dose is kept.
A **promotion gate** (`scripts/gate_promotion.py` + `registry.regressions`) wired
to the held-out metrics rejects any candidate whose point estimate drops, so the
choice is enforced by code; at this n the gate cannot tell a real drop from
noise.

## Consequences

- Abstention on out-of-corpus questions: 1/6 → 5/6 (PT-aware), Wilson 95%
  [0.03, 0.56] → [0.44, 0.97], paired p = 0.125. The direction matches the
  intent; 6 questions cannot establish the effect.
- Promotion is mechanical and re-runs in CI without a GPU via frozen generations
  (`make eval-honest`). Full arc in [`finetuning-results.md`](../finetuning-results.md).
