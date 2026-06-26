# Fine-Tuning Results

This page records the first local LoRA fine-tuning experiments for `anchora`.
The goal was not to claim a production-quality adapter yet; it was to make the
v0.3 claim measurable: **base model vs. tuned adapter on the same golden set**.

## Setup

| Item | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-0.5B-Instruct` |
| Hardware | Apple Silicon MPS |
| Dataset | `data/finetune/instructions.jsonl` |
| Golden set | 24 questions over 8 Brazilian legal/administrative documents |
| Retrieval | deterministic `hash` provider, `k=4` |
| Metrics | grounded rate, faithfulness, answer relevance |
| APIs | none |

The dataset is generated from the golden set and retrieved context:

```bash
uv run python scripts/build_finetune_dataset.py
```

Each completion includes the reference answer plus the citation marker for the
first retrieved chunk from the expected document, for example:

```text
Até 20 dias, prorrogável por mais 10 dias mediante justificativa. [1]
```

## Experiments

### Experiment A — naive prompt+completion loss

Command:

```bash
uv run python scripts/finetune_lora.py \
  --base Qwen/Qwen2.5-0.5B-Instruct \
  --data data/finetune/instructions.jsonl \
  --out artifacts/lora-anchora-qwen05b \
  --epochs 6
```

Result:

| Model | Grounded rate | Faithfulness | Answer relevance |
|---|---:|---:|---:|
| Base | 0.3750 | 0.2866 | 0.2036 |
| LoRA | 0.0833 | 0.3185 | 0.3524 |

Interpretation: the adapter improved content overlap, but it failed the main
RAG requirement: cite or abstain. It should **not** be promoted.

### Experiment B — completion-only loss, too aggressive

Command:

```bash
uv run python scripts/finetune_lora.py \
  --base Qwen/Qwen2.5-0.5B-Instruct \
  --data data/finetune/instructions.jsonl \
  --out artifacts/lora-anchora-qwen05b-completion \
  --epochs 10
```

Result:

| Model | Grounded rate | Faithfulness | Answer relevance |
|---|---:|---:|---:|
| Base | 0.3750 | 0.2866 | 0.2036 |
| LoRA | 0.0000 | 0.0000 | 0.0000 |

Interpretation: this run became unstable after epoch 6 (`grad_norm=nan`) and
collapsed at evaluation time. It is a useful failed experiment, not a candidate.

### Experiment C — completion-only loss, lower LR

Command:

```bash
uv run python scripts/finetune_lora.py \
  --base Qwen/Qwen2.5-0.5B-Instruct \
  --data data/finetune/instructions.jsonl \
  --out artifacts/lora-anchora-qwen05b-completion-lr1e4-e5 \
  --epochs 5 \
  --lr 1e-4
```

Result:

| Model | Grounded rate | Faithfulness | Answer relevance |
|---|---:|---:|---:|
| Base | 0.3750 | 0.2866 | 0.2036 |
| LoRA | 0.2083 | 0.3155 | 0.2083 |

Interpretation: this was stable and improved faithfulness slightly, but still
reduced grounded/cited outputs. It should **not** be promoted yet.

## Evaluation Command

The comparison was generated with:

```bash
uv run python scripts/evaluate_finetune.py \
  --base Qwen/Qwen2.5-0.5B-Instruct \
  --adapter artifacts/lora-anchora-qwen05b-completion-lr1e4-e5 \
  --out artifacts/finetune-comparison-qwen05b-completion-lr1e4-e5.json \
  --max-new-tokens 96
```

The raw JSON outputs live under `artifacts/` and are intentionally not tracked
by Git.

## Decision

Do **not** promote any adapter from these first experiments. The honest v0.3
result is:

> LoRA is wired end-to-end and measured. On a tiny 0.5B local model with only 24
> training examples, it improves faithfulness slightly but does not yet improve
> the production-critical grounding/citation behavior.

## Next Iteration

The next run should use:

1. A larger local model (`Qwen2.5-1.5B` or `Qwen2.5-3B`) once cached locally.
2. More training examples (at least 200-500 synthetic, source-grounded records).
3. A stricter response template: short answer first, mandatory citation last.
4. A validation split so the adapter is not judged only on memorized golden cases.
5. A promotion rule: promote only if grounded rate and faithfulness both improve.
