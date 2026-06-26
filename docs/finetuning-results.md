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

## Methodology Fix — the headline number was measured on the training set

The most important thing I learned here is not the 0.92. It is that **the 0.92
is not trustworthy as stated**. `build_finetune_dataset.py` builds the training
data from the golden set, and the Experiment F table above was scored on *that
same golden set*. Train == test. So the jump from 0.17 to 0.92 largely measures
**memorization of 24 answers**, not generalization. Two related caveats compound
it: `reference_overlap` compares the answer against the very gold string the
model was trained to reproduce (near-tautological on the training split), and
`grounded_rate` only checks that a `[n]` bracket is present, not that the cited
index is the right document — so an adapter that learns to always append a
bracket scores high "grounding" for free.

This is a valid **smoke test** (the SFT pipeline is correctly wired and the model
can learn the target format). It is not evidence that the adapter is *better*.
Calling it that would be the kind of inflated claim a sharp interviewer catches
in sixty seconds.

### What changed (v0.3.1)

1. **A held-out evaluation set** — `data/golden/holdout.json`: 28 brand-new
   questions over the *same* corpus, fully disjoint from the training golden set
   (asserted by `tests/test_holdout.py`). 22 are answerable; 6 are out-of-corpus
   cases whose only correct behavior is to abstain with the exact refusal
   sentence. The adapter never saw any of these.

2. **A fair few-shot baseline** — `evaluate_finetune.py --few-shot` adds a third
   row: the *base* model prompted with a few worked examples in the same
   `PT + [n]` output contract (exemplars taken only from the training golden set,
   never the holdout). This isolates the real question: did fine-tuning teach
   *knowledge*, or just the *output format* that few-shot prompting gives the base
   model for free?

3. **Abstention-aware scoring** — answerable cases are scored with the lexical
   proxies; out-of-corpus cases are scored by whether the model correctly
   abstained (`abstention_rate`), not by answer overlap.

### First honest signal (no model required)

The deterministic `hash` retriever already exposes a generalization gap before a
single token is generated:

| Split | Retrieval recall | Notes |
|---|---:|---|
| Golden (train, 24 q) | **1.000** | the EN→PT glossary bridge is hand-fit to these questions |
| Holdout (new, 22 q)  | **0.864** | 3 misses where the bridge does not generalize |

That 1.000 → 0.864 drop is the honest cost the perfect-recall CI gate was hiding.
It is itself a finding worth reporting.

### Run the honest comparison

```bash
# three fair rows — base zero-shot, base few-shot, LoRA — on UNSEEN questions
uv run python scripts/evaluate_finetune.py \
  --base Qwen/Qwen2.5-1.5B-Instruct \
  --adapter artifacts/lora-anchora-qwen15b-earlystop-fixed-lr1e4-e30 \
  --golden data/golden/holdout.json \
  --few-shot \
  --out artifacts/holdout-comparison.json \
  --max-new-tokens 48
```

Both outcomes are good outcomes: if the LoRA still beats base+few-shot on the
holdout, the gain is real and defensible; if it does not, the honest finding is
*"for this task, few-shot matched fine-tuning — the adapter did not pay for
itself."* Either is a stronger signal than a memorized 0.92.

### Results on the holdout (actual, `Qwen2.5-1.5B`, `max_new_tokens=48`)

Scored on the 22 answerable + 6 out-of-corpus held-out questions:

| Row | Grounded ↑ | Faithfulness ↑ | Ref. overlap ↑ | Abstention ↑ |
|---|---:|---:|---:|---:|
| base (zero-shot)   | 0.227 | 0.204 | 0.138 | 0.000 |
| base + few-shot    | 0.636 | 0.197 | 0.237 | 0.167 |
| **LoRA**           | **0.864** | **0.789** | **0.519** | **0.000** |

*(`answer_relevance` omitted — same EN-question-vs-PT-answer proxy artifact as
before: base 0.84 → LoRA 0.09.)*

**The win survives the fair test — with one important asterisk.**

1. **Fine-tuning taught real behavior, not just format.** Few-shot prompting
   closes much of the *grounding* gap on its own (0.23 → 0.64), confirming part
   of the earlier headline was format conformance. But on **faithfulness** it
   does nothing (0.20 → 0.20), while the LoRA jumps to **0.79**. Few-shot can't
   buy that — the adapter genuinely learned to answer from context.

2. **It generalizes.** These reference answers were never trained on, yet the
   LoRA's reference_overlap (0.52) is far above base+few-shot (0.24). The earlier
   0.73 was inflated by train==test, but the real, unseen-data number still wins
   clearly. The adapter learned a skill, it did not just memorize 24 answers.

3. **The holdout exposed a real failure mode the leaked eval could never show:
   the LoRA never abstains** (abstention 0.000). On out-of-corpus questions it
   confidently fabricates answers *with fake citations* — e.g. *"Três anos. [1]"*
   for the homicide statute of limitations, *"6 pontos. [1]"* for license points.
   The base+few-shot model at least refuses sometimes (0.167). Cause is obvious in
   hindsight: the training set contains **zero abstention examples**, so the
   adapter learned the rule *"always answer and append a bracket."* It optimized
   exactly what it was shown.

**Net:** the adapter is a real, measurable improvement on in-corpus grounding and
faithfulness — and a measurable *regression* on knowing when to shut up. That
second sentence is the one the first eval was structurally incapable of telling
me, and it sets the next iteration's top priority: put abstention cases in the
training data.

### Closing the loop — re-training with abstention (v0.3.2)

I added 10 out-of-corpus questions to the training set (`data/finetune/
abstention_train.json`, completion = the refusal sentence, no citation; disjoint
from the 6 held-out abstention cases) and re-ran the same recipe. Same holdout,
same fair baselines:

| Row | Grounded ↑ | Abstention ↑ | Faithfulness ↑ | Ref. overlap ↑ |
|---|---:|---:|---:|---:|
| base + few-shot         | 0.636 | 0.167 | 0.197 | 0.237 |
| LoRA (answerable-only)  | 0.864 | 0.000 | **0.789** | **0.519** |
| **LoRA + abstention**   | 0.864 | **0.500** | 0.658 | 0.394 |

**The fix works, and it costs something — both worth stating plainly.**

* **Abstention 0.000 → 0.500** on unseen out-of-corpus questions, with grounding
  on answerable cases **unchanged (0.864)**. 10 refusal examples were enough to
  teach "say no" without unlearning "cite." The confident fake citations are
  gone: *"Três anos. [1]"* for the homicide statute became *"I could not find
  this information in the provided documents."*
* **The measured 0.500 understates the real behavior change.** The metric demands
  the exact English refusal sentence, but the adapter now also refuses *in
  Portuguese* — *"Não há uma data específica…"*, *"Nenhum dado foi fornecido"* —
  which scores as a miss. So the true rate of "stopped fabricating" is higher than
  0.500. (This is exactly why rec #3, a citation-correctness metric, is next.)
* **The cost is real: faithfulness 0.79 → 0.66, reference overlap 0.52 → 0.39.**
  A model taught to sometimes refuse gets more hedged on answerable questions too
  — a classic precision/abstention tradeoff. At a 10/34 (~29%) abstention ratio
  the trade leans a bit far toward caution; the next knob to turn is that ratio,
  not the model size.

Honest one-liner for the v0.3 story: *"My first fine-tune scored 0.92 — on its own
training set. I built a held-out eval, controlled for prompt format with a
few-shot baseline, and found the real gain was smaller but genuine — and that the
adapter never abstained. I fixed that with abstention data; it now refuses half
the out-of-corpus questions, at a measured cost to answer precision I can show you
in a table."*

## Next Iteration

Reordered — a defensible eval comes before a bigger model. Item 1 is done (see
the holdout results above); the failure it surfaced sets the new top priority:

1. ✅ **Run the holdout + few-shot comparison** — done. LoRA wins on grounding and
   faithfulness over unseen data; never abstains.
2. ✅ **Teach abstention** — done. Abstention 0.000 → 0.500 with grounding held,
   at a measured cost to faithfulness/overlap. Next sub-step: **tune the abstention
   ratio** (try ~5/29 instead of 10/34) to recover answer precision.
3. **Add a citation-correctness metric** (cited index resolves to the expected
   document) and accept Portuguese refusals, replacing presence-of-bracket and
   exact-English-string as the signals — both currently mis-score real behavior.
4. Only then scale training data (200–500 synthetic, source-grounded records),
   keeping the holdout strictly separate.
5. A promotion rule (already encoded in `registry.py`): promote only if the
   holdout metrics improve without answer drift **and** abstention does not
   regress.

Deliberately **not** on the list: a larger base model (`Qwen2.5-3B`). A bigger
model on a leaked eval is the same problem with more GPU. Fix the methodology
first.
