# Fine-Tuning Results

This page records the first local LoRA fine-tuning experiments for `anchora`.
The goal was not to claim a production-quality adapter yet; it was to make the
v0.3 claim measurable: **base model vs. tuned adapter on the same golden set**.

## Setup

| Item | Value |
|---|---|
| Base models | `Qwen/Qwen2.5-0.5B-Instruct`, `Qwen/Qwen2.5-1.5B-Instruct` |
| Hardware | Apple Silicon MPS |
| Dataset | `data/finetune/instructions.jsonl` |
| Golden set | 24 questions over 8 Brazilian legal/administrative documents |
| Retrieval | deterministic `hash` provider, `k=4` |
| Metrics | grounded rate, faithfulness, answer relevance, reference overlap |
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

### Experiment D — larger base model, completion-only loss

Command:

```bash
uv run python scripts/finetune_lora.py \
  --base Qwen/Qwen2.5-1.5B-Instruct \
  --data data/finetune/instructions.jsonl \
  --out artifacts/lora-anchora-qwen15b-completion-lr1e4-e5 \
  --epochs 5 \
  --lr 1e-4
```

Result:

| Model | Grounded rate | Faithfulness | Answer relevance |
|---|---:|---:|---:|
| Base | 0.2083 | 0.3121 | 0.4647 |
| LoRA | 0.2917 | 0.2844 | 0.4552 |

Interpretation: the larger base model is a better benchmark than the `0.5B`
smoke test. The LoRA adapter improved the production-critical citation/grounding
rate, but it reduced faithfulness. This is useful evidence, but it is still not
a promotion candidate.

### Experiment E — larger base model, lower LR

Command:

```bash
uv run python scripts/finetune_lora.py \
  --base Qwen/Qwen2.5-1.5B-Instruct \
  --data data/finetune/instructions.jsonl \
  --out artifacts/lora-anchora-qwen15b-completion-lr5e5-e5 \
  --epochs 5 \
  --lr 5e-5
```

Result:

| Model | Grounded rate | Faithfulness | Answer relevance |
|---|---:|---:|---:|
| Base | 0.2083 | 0.3121 | 0.4647 |
| LoRA | 0.2083 | 0.2726 | 0.4364 |

Interpretation: the lower learning rate was stable, but it did not improve
grounding and reduced both faithfulness and answer relevance. It should not be
promoted.

### Experiment F — larger base model, early stopping and fixed truncation

The first long-running early-stopping attempts exposed a real bug in the SFT
pipeline: long prompts could consume the full sequence length, leaving no answer
tokens to train on. This produced `loss=0` and `grad_norm=nan` around epoch 5.
The fix reserves sequence budget for the completion and left-truncates only the
prompt. The generative evaluator also left-truncates prompts so the question and
`Answer:` suffix are preserved.

Command:

```bash
uv run python scripts/finetune_lora.py \
  --base Qwen/Qwen2.5-1.5B-Instruct \
  --data data/finetune/instructions.jsonl \
  --out artifacts/lora-anchora-qwen15b-earlystop-fixed-lr1e4-e30 \
  --epochs 30 \
  --lr 1e-4 \
  --validation-ratio 0.2 \
  --early-stopping-patience 5
```

Result (`max_new_tokens=48`):

| Model | Grounded rate | Faithfulness | Answer relevance | Reference overlap |
|---|---:|---:|---:|---:|
| Base | 0.1667 | 0.2668 | 0.8376 | 0.1668 |
| LoRA | 0.9167 | 0.9208 | 0.0458 | 0.7338 |

Interpretation: this is the first strong LoRA result. The tuned adapter produces
short Portuguese legal answers with citations and much higher faithfulness. The
low `answer_relevance` is a known limitation of this lexical proxy when comparing
English questions against concise Portuguese answers; `reference_overlap` is the
more appropriate supervised fine-tuning metric here.

## Evaluation Command

The comparison was generated with:

```bash
uv run python scripts/evaluate_finetune.py \
  --base Qwen/Qwen2.5-1.5B-Instruct \
  --adapter artifacts/lora-anchora-qwen15b-earlystop-fixed-lr1e4-e30 \
  --out artifacts/finetune-comparison-qwen15b-earlystop-fixed-lr1e4-e30-final.json \
  --max-new-tokens 48
```

The raw JSON outputs live under `artifacts/` and are intentionally not tracked
by Git.

## Decision

Promote only the early-stopped `1.5B` adapter as **experimental**. The honest
v0.3 result is:

> LoRA is wired end-to-end and measured. The initial 5-epoch runs were too small
> and the first long run exposed a sequence-truncation bug. After fixing
> completion preservation and adding early stopping, the `1.5B` adapter improves
> grounded rate, faithfulness and reference overlap on the 24-case benchmark.

## Next Iteration

The next run should use:

1. A larger local model (`Qwen2.5-3B`) once cached locally.
2. More training examples (at least 200-500 synthetic, source-grounded records).
3. A separate validation set, not sampled from the same 24-case golden set.
4. A stricter response template: short answer first, mandatory citation last.
5. A promotion rule: promote only if grounded rate, faithfulness and reference
   overlap improve without obvious answer drift.
