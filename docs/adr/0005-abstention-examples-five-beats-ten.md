# 5. Five abstention examples beat ten; a promotion gate enforces it

Date: 2026-06-25 · Status: Accepted

## Context

The first LoRA adapter never abstained: on out-of-corpus questions it fabricated
confident answers with fake citations (abstention rate 0.00). Adding abstention
examples to the training set fixes that — but too many teach the model to refuse
answerable questions, trading one failure for another.

## Decision

Train with **5 abstention examples**. A held-out sweep showed 5 dominates 10 on
every axis: the 10-example variant regressed citation accuracy 0.818 → 0.636.
A **promotion gate** (`scripts/gate_promotion.py` + `registry.regressions`) wired
to the honest held-out metrics **auto-rejects** any candidate that regresses,
so this decision is enforced by code, not by memory.

## Consequences

- Abstention on out-of-corpus questions: 0.00 → 0.833, at a small measured cost
  to answer precision — a trade made on evidence, not vibes.
- Promotion is mechanical and re-runs in CI without a GPU via frozen generations
  (`make eval-honest`). Full arc in [`finetuning-results.md`](../finetuning-results.md).
