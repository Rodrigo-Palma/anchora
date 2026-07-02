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

| Metric | base + few-shot | **LoRA + 5 abstention (prod)** |
|---|---:|---:|
| Citation-correct ↑ | 0.500 | **0.818** |
| Abstention (PT-aware) ↑ | 0.167 | **0.833** |
| Faithfulness ↑ | 0.197 | **0.726** |

Measured on a holdout **disjoint from training** (`tests/test_holdout.py` asserts
the disjointness). The headline was once 0.92 measured on the training set — a
leak that was found and fixed; see finetuning-results.md.

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
regresses citation accuracy or abstention against the incumbent is auto-rejected
(a 10-abstention variant was rejected for regressing citation 0.818 → 0.636).
See [ADR 5](adr/0005-abstention-examples-five-beats-ten.md).
