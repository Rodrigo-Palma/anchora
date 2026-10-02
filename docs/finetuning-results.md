# Fine-Tuning Results

This page records the first local LoRA fine-tuning experiments for `anchora`.
The goal was not to claim a production-quality adapter yet; it was to make the
v0.3 claim measurable: **base model vs. tuned adapter on the same golden set**.

## Reading the numbers

- **Sample sizes are small.** Experiments A to F are scored on the 24 training
  golden questions. The holdout sections use 22 answerable and 6 out-of-corpus
  questions. Every rate below is written as `k/n` with a 95% Wilson interval;
  per-question means carry a seeded 95% bootstrap interval; two models on the
  same questions are compared with the exact paired McNemar test
  (`uv run python scripts/score_generations.py --report`, `anchora.stats`).
- **Experiments A to F are not reproducible from this repo.** Their outputs
  live under the untracked `artifacts/`. The grounded-rate counts are recovered
  from the rates (`k/24`); the faithfulness, relevance and overlap means have no
  per-question data in the repo, so they carry no interval.
- **The holdout sections are reproducible.** The decoded generations are frozen
  in `data/eval/holdout-generations.json` and re-scored in CI. Two historical
  columns (bracket-presence "grounded" and exact-English abstention) come from
  an earlier version of the scorer; recounting the frozen strings with those
  definitions gives the counts shown, but the current scorer does not compute
  them.
- **What n=22 and n=6 support**, in one line: the citation gain of the promoted
  adapter over the few-shot baseline is the one paired difference this holdout
  resolves (p = 0.016, not pre-registered, 0.094 after Bonferroni over six
  comparisons); every abstention difference on 6 questions, and the 5-vs-10
  citation gap behind the gate's rejection, are within noise.

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

### Experiment A: naive prompt+completion loss

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
| Base | 9/24 = 0.375 [0.21, 0.57] | 0.2866 | 0.2036 |
| LoRA | 2/24 = 0.083 [0.02, 0.26] | 0.3185 | 0.3524 |

Interpretation: the adapter improved content overlap, but it failed the main
RAG requirement: cite or abstain. It should **not** be promoted.

### Experiment B: completion-only loss, too aggressive

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
| Base | 9/24 = 0.375 [0.21, 0.57] | 0.2866 | 0.2036 |
| LoRA | 0/24 = 0.000 [0.00, 0.14] | 0.0000 | 0.0000 |

Interpretation: this run became unstable after epoch 6 (`grad_norm=nan`) and
collapsed at evaluation time. It is a useful failed experiment, not a candidate.

### Experiment C: completion-only loss, lower LR

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
| Base | 9/24 = 0.375 [0.21, 0.57] | 0.2866 | 0.2036 |
| LoRA | 5/24 = 0.208 [0.09, 0.40] | 0.3155 | 0.2083 |

Interpretation: this was stable and improved faithfulness slightly, but still
reduced grounded/cited outputs. It should **not** be promoted yet.

### Experiment D: larger base model, completion-only loss

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
| Base | 5/24 = 0.208 [0.09, 0.40] | 0.3121 | 0.4647 |
| LoRA | 7/24 = 0.292 [0.15, 0.49] | 0.2844 | 0.4552 |

Interpretation: the larger base model is a better benchmark than the `0.5B`
smoke test. The LoRA grounded rate went from 5/24 to 7/24, a 2-question change
with overlapping intervals, so it is not evidence of improvement, and
faithfulness went down. This is useful evidence, but it is still not
a promotion candidate.

### Experiment E: larger base model, lower LR

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
| Base | 5/24 = 0.208 [0.09, 0.40] | 0.3121 | 0.4647 |
| LoRA | 5/24 = 0.208 [0.09, 0.40] | 0.2726 | 0.4364 |

Interpretation: the lower learning rate was stable, but it did not improve
grounding and reduced both faithfulness and answer relevance. It should not be
promoted.

### Experiment F: larger base model, early stopping and fixed truncation

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
| Base | 4/24 = 0.167 [0.07, 0.36] | 0.2668 | 0.8376 | 0.1668 |
| LoRA | 22/24 = 0.917 [0.74, 0.98] | 0.9208 | 0.0458 | 0.7338 |

Interpretation at the time: the tuned adapter produces short Portuguese legal
answers with citations and much higher faithfulness. This was scored on the
training questions, so it measures fit, not generalization (see the methodology
fix below). The
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

Promote only the early-stopped `1.5B` adapter as **experimental**. The v0.3
result as written at the time (superseded by the methodology fix below):

> LoRA is wired end-to-end and measured. The initial 5-epoch runs were too small
> and the first long run exposed a sequence-truncation bug. After fixing
> completion preservation and adding early stopping, the `1.5B` adapter improves
> grounded rate, faithfulness and reference overlap on the 24-case benchmark.

## Methodology Fix: the headline number was measured on the training set

The most important thing I learned here is not the 0.92. It is that **the 0.92
is not trustworthy as stated**. `build_finetune_dataset.py` builds the training
data from the golden set, and the Experiment F table above was scored on *that
same golden set*. Train == test. So the jump from 0.17 to 0.92 (4/24 to 22/24) largely measures
**memorization of 24 answers**, not generalization. Two related caveats compound
it: `reference_overlap` compares the answer against the very gold string the
model was trained to reproduce (near-tautological on the training split), and
`grounded_rate` only checks that a `[n]` bracket is present, not that the cited
index is the right document, so an adapter that learns to always append a
bracket scores high "grounding" for free.

This is a valid **smoke test** (the SFT pipeline is correctly wired and the model
can learn the target format). It is not evidence that the adapter is *better*,
and reporting it as such would be an unsupported claim.

### What changed (v0.3.1)

1. **A held-out evaluation set**, `data/golden/holdout.json`: 28 new
   questions over the *same* corpus, fully disjoint from the training golden set
   (asserted by `tests/test_holdout.py`). 22 are answerable; 6 are out-of-corpus
   cases whose only correct behavior is to abstain with the exact refusal
   sentence. The adapter never saw any of these.

2. **A few-shot baseline**: `evaluate_finetune.py --few-shot` adds a third
   row: the *base* model prompted with a few worked examples in the same
   `PT + [n]` output contract (exemplars taken only from the training golden set,
   never the holdout). This isolates the real question: did fine-tuning teach
   *knowledge*, or just the *output format* that few-shot prompting gives the base
   model for free?

3. **Abstention-aware scoring**: answerable cases are scored with the lexical
   proxies; out-of-corpus cases are scored by whether the model correctly
   abstained (`abstention_rate`), not by answer overlap.

### First held-out signal (no model required)

The deterministic `hash` retriever already exposes a generalization gap before a
single token is generated:

| Split | Retrieval recall | Notes |
|---|---:|---|
| Golden (train, 24 q) | **24/24 = 1.000** [0.86, 1.00] | the EN→PT glossary bridge is hand-fit to these questions |
| Holdout (new, 22 q)  | **19/22 = 0.864** [0.67, 0.95] | 3 misses where the bridge does not generalize |

The intervals overlap, so 22 questions do not establish a recall drop; what
they do show is 3 concrete misses the perfect-recall CI gate could never
surface, because it only asks the questions the glossary was fit to.

### Run the held-out comparison

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

Either outcome is informative: if the LoRA beats base+few-shot on the holdout,
the gain is not explained by output format alone; if it does not, the finding is
*"for this task, few-shot matched fine-tuning; the adapter did not pay for
itself."* Either says more than a memorized 0.92.

### Results on the holdout (actual, `Qwen2.5-1.5B`, `max_new_tokens=48`)

Scored on the 22 answerable + 6 out-of-corpus held-out questions. Grounded and
abstention here use the original definitions (a `[n]` bracket or the exact
English refusal counts as grounded; only the exact English refusal counts as
abstention):

| Row | Grounded ↑ (n=22) | Faithfulness ↑ (n=22) | Ref. overlap ↑ (n=22) | Abstention ↑ (n=6) |
|---|---|---|---|---|
| base (zero-shot)   | 5/22 = 0.227 [0.10, 0.43] | 0.204 [0.15, 0.26] | 0.138 [0.06, 0.24] | 0/6 [0.00, 0.39] |
| base + few-shot    | 14/22 = 0.636 [0.43, 0.80] | 0.197 [0.14, 0.26] | 0.237 [0.12, 0.38] | 1/6 [0.03, 0.56] |
| **LoRA**           | **19/22 = 0.864 [0.67, 0.95]** | **0.789 [0.64, 0.92]** | **0.519 [0.36, 0.68]** | **0/6 [0.00, 0.39]** |

*(`answer_relevance` omitted: same EN-question-vs-PT-answer proxy artifact as
before, base 0.84 → LoRA 0.09.)*

What 22 and 6 questions support:

1. **Few-shot closes part of the grounding gap on its own** (5/22 → 14/22, with
   intervals that barely overlap), so part of the earlier headline was format
   conformance. On faithfulness few-shot does nothing (0.20 → 0.20) while the
   LoRA reaches 0.79, and those bootstrap intervals do not overlap. Faithfulness
   is a token-overlap proxy, so this says the adapter's answers reuse context
   tokens, not that they are correct.
2. **Reference overlap** on unseen answers is 0.52 [0.36, 0.68] for the LoRA vs
   0.24 [0.12, 0.38] for few-shot: higher, with intervals touching. The earlier
   0.73 was inflated by train == test.
3. **The holdout exposed a failure mode the leaked eval could not show:** the
   LoRA refused 0 of the 6 out-of-corpus questions. It fabricated answers *with
   fake citations*, e.g. *"Três anos. [1]"* for the homicide statute of
   limitations, *"6 pontos. [1]"* for license points. Few-shot refused 1 of 6;
   on 6 questions that difference means nothing, but 0/6 fabrications with
   citations are concrete failures. The cause is visible in the data: the
   training set contained **zero abstention examples**, so the adapter learned
   "always answer and append a bracket."

**Net:** a measured gain on in-corpus faithfulness (proxy) and a visible failure
on out-of-corpus questions. The second is what the first eval could not report,
and it set the next step: put abstention cases in the training data.

### Closing the loop: re-training with abstention (v0.3.2)

I added 10 out-of-corpus questions to the training set (`data/finetune/
abstention_train.json`, completion = the refusal sentence, no citation; disjoint
from the 6 held-out abstention cases) and re-ran the same recipe. Same holdout,
same baselines, same original definitions:

| Row | Grounded ↑ (n=22) | Abstention ↑ (n=6) | Faithfulness ↑ (n=22) | Ref. overlap ↑ (n=22) |
|---|---|---|---|---|
| base + few-shot         | 14/22 = 0.636 [0.43, 0.80] | 1/6 [0.03, 0.56] | 0.197 [0.14, 0.26] | 0.237 [0.12, 0.38] |
| LoRA (answerable-only)  | 19/22 = 0.864 [0.67, 0.95] | 0/6 [0.00, 0.39] | 0.789 [0.64, 0.92] | 0.519 [0.36, 0.68] |
| **LoRA + abstention**   | 19/22 = 0.864 [0.67, 0.95] | 3/6 = 0.500 [0.19, 0.81] | 0.658 [0.47, 0.83] | 0.394 [0.24, 0.56] |

* **Exact-English refusals went from 0/6 to 3/6.** The confident fake citations
  on those questions are gone: *"Três anos. [1]"* for the homicide statute became
  *"I could not find this information in the provided documents."* With 6
  questions the intervals overlap; this is a change in observed behavior, not a
  measured rate.
* **The exact-English check undercounts.** The adapter also refuses *in
  Portuguese* (*"Não há uma data específica…"*, *"Nenhum dado foi fornecido"*),
  which this check scores as a miss. That motivated the PT-aware detection below.
* **A possible cost on answerable questions:** faithfulness 0.79 → 0.66 and
  reference overlap 0.52 → 0.39. Both pairs of intervals overlap, so 22
  questions do not establish the cost; it is the expected direction for a model
  taught to hedge, and the reason the abstention ratio was swept next.

### Measuring the right thing (rec #3): two metrics that move in opposite directions

`grounded_rate` only asks "is a `[n]` present?" and the exact-English abstention
check only matches one sentence. Both mis-measure real behavior. I added
`metrics.citation_correct` (does the cited index resolve to the *expected*
document?) and `guardrails.is_abstention` (recognize Portuguese refusals too), and
re-scored the existing run outputs. No re-generation was needed; retrieval is
deterministic.

| Adapter | Grounded, bracket or exact refusal (n=22) | **Citation-correct** (n=22) | Abstention, exact EN (n=6) | **Abstention, PT-aware** (n=6) |
|---|---|---|---|---|
| base + few-shot          | 14/22 [0.43, 0.80] | 11/22 = 0.500 [0.31, 0.69] | 1/6 [0.03, 0.56] | 1/6 = 0.167 [0.03, 0.56] |
| LoRA (answerable-only)   | 19/22 [0.67, 0.95] | 17/22 = 0.773 [0.57, 0.90] | 0/6 [0.00, 0.39] | 1/6 = 0.167 [0.03, 0.56] |
| LoRA + abstention        | 19/22 [0.67, 0.95] | **14/22 = 0.636** [0.43, 0.80] | 3/6 [0.19, 0.81] | **5/6 = 0.833** [0.44, 0.97] |

Two corrections, pointing opposite ways:

* **Bracket presence was optimistic.** Part of the 19/22 "grounded" was a
  bracket pointing at the *wrong* document; citation-correct is 14/22 to 17/22
  for the adapters. Against base+few-shot (11/22) the answerable-only adapter is
  right on 7 questions the baseline misses and the reverse happens once
  (paired p = 0.070): suggestive, not established at n=22.
* **Exact-English abstention was pessimistic.** The abstention-trained adapter
  refuses 5 of 6 out-of-corpus questions once Portuguese refusals count. The
  intervals for 5/6 and 1/6 overlap, and on 6 questions no paired test can go
  below p = 0.031.

### Tuning the abstention ratio: 5 vs 10

The 10/34 (~29%) abstention mix looked like it cost answer quality. So I swept
the ratio: same recipe, 5 abstention examples (`--max-abstention 5`, 17%)
instead of 10. Scored on the holdout with the current metrics:

| Adapter | Citation-correct ↑ (n=22) | Faithfulness ↑ (n=22) | Ref. overlap ↑ (n=22) | Abstention, PT ↑ (n=6) |
|---|---|---|---|---|
| LoRA (0 abstention)     | 17/22 = 0.773 [0.57, 0.90] | **0.789** [0.64, 0.92] | **0.519** [0.36, 0.68] | 1/6 = 0.167 [0.03, 0.56] |
| **LoRA + 5 abstention** | **18/22 = 0.818** [0.61, 0.93] | 0.726 [0.55, 0.88] | 0.457 [0.29, 0.62] | **5/6 = 0.833** [0.44, 0.97] |
| LoRA + 10 abstention    | 14/22 = 0.636 [0.43, 0.80] | 0.658 [0.47, 0.83] | 0.394 [0.24, 0.56] | 5/6 = 0.833 [0.44, 0.97] |

Paired exact McNemar on the same questions:

| Comparison | Metric | Only A right | Only B right | p |
|---|---|---:|---:|---:|
| LoRA+5 vs LoRA+0  | citation-correct | 1 | 0 | 1.000 |
| LoRA+5 vs LoRA+0  | abstention (PT) | 4 | 0 | 0.125 |
| LoRA+5 vs LoRA+10 | citation-correct | 5 | 1 | 0.219 |
| LoRA+5 vs base+few-shot | citation-correct | 7 | 0 | 0.016 |
| LoRA+5 vs base+few-shot | abstention (PT) | 4 | 0 | 0.125 |

**What the sweep shows at this n.** Five and ten examples refuse equally often
(5/6). Five cites the right document on 18/22 and ten on 14/22, a gap of 5
questions lost and 1 gained (p = 0.219), so the sweep does not show five beating
ten; it shows no advantage for ten. Against the answerable-only adapter, five
costs nothing measurable on citation (1 question) and moves abstention from 1/6
to 5/6 (p = 0.125). The faithfulness and overlap differences between the three
adapters all have overlapping intervals.

**`LoRA + 5 abstention` is the promotion candidate** because it is never worse on
a point estimate and is the smaller dose:
`data/finetune/instructions-abstention5.jsonl`,
`artifacts/lora-anchora-qwen15b-abstention5-lr1e4-e30`. Against the few-shot
baseline its citation gain is the one difference this holdout resolves
(7 vs 0 discordant, p = 0.016; not pre-registered, 0.094 after Bonferroni over
the six comparisons `--report` prints).

Lesson worth keeping: the fix (teach abstention) and its dosage (how much) are two
separate decisions. The first needs a held-out eval to even see; the second needs
a sweep, and a sweep needs enough questions to separate its arms. Six
out-of-corpus questions are not enough.

### Closing the MLOps loop: promote on held-out metrics, with a gate

`scripts/register_finetune.py` writes each adapter's *held-out* metrics into the
model registry and promotes to `prod` only if it does not regress on a gate of
`{citation_accuracy, abstention_rate}` (`registry.regressions`). Registering the
three candidates in turn:

```
Promoted anchora-qa:v0.3-lora0  to prod (no incumbent).
Promoted anchora-qa:v0.3-lora5  to prod (no regression on citation/abstention).
REJECTED anchora-qa:v0.3-lora10: regressed on citation_accuracy 0.818->0.636;
         keeping anchora-qa:v0.3-lora5 in prod.
```

The gate compares point estimates. The rejection of `lora10` is 18/22 → 14/22,
5 questions lost and 1 gained (paired p = 0.219), which is within noise on 22
questions. So the replay shows the mechanism (a regression rule wired to
held-out metrics, enforced by code and replayed in CI); it does not show the gate
detecting a real regression. Making it able to do that needs a larger holdout or
a rule that requires the drop to be significant. Final prod: `v0.3-lora5`. (The
registry file lives under `artifacts/` and is not tracked; the capability is the
code and the gate, not the JSON.)

### Frozen so it runs in CI: no GPU, no network

The generation runs above need a GPU (Apple MPS), and the raw comparison JSONs
under `artifacts/` are not tracked, so from a clean checkout none of these
numbers could be reproduced. To close that gap without re-generating, the real
decoded outputs are frozen per arm in `data/eval/holdout-generations.json` (actual
model outputs, not invented numbers) and re-scored deterministically:

* `scripts/evaluate_finetune.py` was refactored so its GPU generation loop and the
  offline re-scoring share one scorer, `score_case(answer, case, store)`. Same
  code scores a freshly generated answer and a frozen one.
* `scripts/score_generations.py --check` re-scores every frozen arm through
  `score_case` (retrieval = deterministic `hash`) and fails if any metric drifts
  from the numbers above beyond a small tolerance.
* `scripts/gate_promotion.py` replays the promotion gate on those re-scored
  metrics and reproduces the decision: promote `lora0`, promote `lora5`, reject
  `lora10`.

```bash
make eval-honest   # score_generations.py --check && gate_promotion.py
```

`scripts/score_generations.py --report` prints the counts, intervals and
paired tests used in this document. `tests/test_frozen_eval.py` runs the same
checks in the suite (including the paired read), and CI runs
`make eval-honest`, so the fine-tuning table is a build-time invariant rather than
a claim about a run that happened once on my laptop.

## Next Iteration

A defensible eval comes before a bigger model. Status, with what each step
showed at its sample size:

1. ✅ **Holdout + few-shot comparison.** The answerable-only LoRA cited a bracket
   on 19/22 vs 14/22 for few-shot and refused 0/6 out-of-corpus questions.
2. ✅ **Teach abstention.** Exact-English refusals 0/6 → 3/6 with grounding held
   at 19/22; the faithfulness and overlap drops have overlapping intervals.
3. ✅ **Citation-correctness metric + Portuguese refusal detection.**
   Citation-correct is 14/22 to 18/22 across adapters (not the 19/22
   bracket-presence count); PT-aware abstention after training is 5/6.
4. ✅ **Abstention ratio.** 5 and 10 examples both refuse 5/6; citation 18/22 vs
   14/22 (p = 0.219). No advantage for 10, so 5 is the candidate.
5. **Grow the holdout before drawing finer conclusions.** With 6 out-of-corpus
   questions no paired test can go below p = 0.031, and on 22 answerable
   questions a 4-question gap is within noise. Both sets need to be several times
   larger for the abstention effect or the gate to mean anything statistically.
6. Then scale training data (200 to 500 synthetic, source-grounded records),
   keeping the holdout strictly separate, and re-sweep the abstention ratio at the
   new scale.
7. ✅ A promotion rule wired to the held-out metrics: `register_finetune.py` +
   `registry.regressions` promote to prod only if neither citation accuracy nor
   abstention drops. It rejected the 10-abstention adapter on a point estimate;
   prod is `v0.3-lora5`. A significance-aware rule is open.

Deliberately **not** on the list: a larger base model (`Qwen2.5-3B`). A bigger
model on a leaked eval is the same problem with more GPU. Fix the methodology
first.
