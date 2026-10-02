# Model card — anchora-qa LoRA adapter

A small model card for the fine-tuned adapter that the promotion gate keeps in
`prod`. Numbers are the honest held-out figures reproduced in CI without a GPU
(`make eval-honest`); the full experimental arc is in
[`finetuning-results.md`](finetuning-results.md).

## Overview

| Field | Value |
|---|---|
| Model | `anchora-qa` LoRA adapter (`v0.3-lora5`) |
| Base model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Method | LoRA (PEFT), completion-only loss (`DataCollatorForCompletionOnlyLM`) |
| Hyperparameters | r=16, α=32, 3 epochs; see finetuning-results.md Exp. F for the promoted run's LR / early-stopping |
| Training data | 24-question golden set + 5 abstention examples ([datasheet](../data/README.md)) |
| Intended use | Cited Q&A over the bundled Brazilian public-law corpus, local-first |
| Out of scope | Legal advice; any domain outside the ingested corpus |

## Evaluation (held-out, 28 unseen questions)

Rates are `k/n` with a 95% Wilson interval; faithfulness is a per-question mean
with a seeded 95% bootstrap interval; p is the exact paired McNemar test
(`uv run python scripts/score_generations.py --report`).

| Metric | base + few-shot | **LoRA + 5 abstention (prod)** | Paired p |
|---|---|---|---:|
| Citation-correct ↑ (n=22) | 11/22 = 0.500 [0.31, 0.69] | **18/22 = 0.818 [0.61, 0.93]** | 0.016 |
| Abstention, PT-aware ↑ (n=6) | 1/6 = 0.167 [0.03, 0.56] | **5/6 = 0.833 [0.44, 0.97]** | 0.125 |
| Faithfulness ↑ (n=22) | 0.197 [0.14, 0.26] | **0.726 [0.55, 0.88]** | n/a |

Measured on a holdout **disjoint from training** (`tests/test_holdout.py` asserts
the disjointness). Read with its size in mind: the citation difference is the one
the paired test resolves (7 questions only the adapter gets right, 0 the other
way; not pre-registered, 0.094 after a Bonferroni correction over the six
comparisons in the report). The abstention difference is not distinguishable on
6 questions, and 0.833 should not be read as an expected refusal rate.
Faithfulness is a lexical proxy. The headline was once 0.92 (22/24) measured on
the training set, a leak that was found and fixed; see finetuning-results.md.

## Limitations & ethical considerations

- **Not legal advice.** Deadlines are planning aids; holidays are not modelled.
- **Corpus-bound.** Out-of-corpus questions must abstain; the agent enforces an
  out-of-domain floor, but coverage is limited to the ingested statutes.
- **Small base model.** 1.5B params: fluent Portuguese legal phrasing is not
  guaranteed; grounding and abstention are prioritized over eloquence.
- **PII.** Inputs are redacted (CPF/email/phone) before the model sees them;
  redaction is regex-based and not exhaustive.

## Governance

Promotion to `prod` is gated (`scripts/gate_promotion.py`): a candidate that
regresses citation accuracy or abstention against the incumbent is auto-rejected.
A 10-abstention variant was rejected for citation 18/22 → 14/22; that drop has a
paired McNemar p of 0.219, so on this holdout it is within noise. The gate
compares point estimates: it shows the promotion mechanism works, not that it
can detect a real regression at n=22.
See [ADR 5](adr/0005-abstention-examples-five-beats-ten.md).
