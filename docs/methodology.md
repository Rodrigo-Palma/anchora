# Evaluation methodology: what anchora measures and what it cannot

`anchora` answers questions over eight Brazilian public-law texts, cites the
passage it used, and abstains when the corpus does not hold the answer. This
document describes how each claim in the README is measured, the mistakes the
measurements exposed, and the limits that remain. Every number here is
reproducible from the repo; the commands are listed next to each one.

## Constraints

1. **No paid APIs.** Embeddings and generation run on Ollama
   (`nomic-embed-text`, `qwen3:32b`); the fine-tune runs on Apple Silicon.
2. **CI must be deterministic and model-free.** A free GitHub runner has no GPU
   and no Ollama, so everything CI checks has to run offline and give the same
   answer twice. This is what shapes most of the design below.

## 1. Offline reproducibility: the `hash` provider and frozen outputs

CI embeds with a deterministic `hash` provider (signed hashing of
accent-folded, stopword-free tokens, plus a small EN→PT glossary so English
questions can match the Portuguese corpus). It is a bag of words, not a
semantic model. Two consequences are measured, not assumed:

- The glossary was written while looking at the 24 golden questions, so recall
  on those questions (24/24) is a regression check, not an estimate. Retrieval
  on unseen questions is measured on a separate 28-question holdout.
- The production embedder was measured on the same sets and frozen
  (`data/eval/ablation-ollama.json`, `make ablation`, `--compare`). On the
  golden set `hash` dense beats nomic dense 24/24 to 20/24, which is what a
  glossary fit to those questions predicts; on the holdout nomic hybrid
  retrieves 22/22 against 20/22 for `hash` hybrid. No difference between
  embedders reaches significance (smallest paired McNemar p = 0.125).

Anything that needs a model is run once locally and its outputs are frozen in
`data/eval/` with the model name and digest. CI re-scores the frozen outputs
and fails if the numbers drift: `make eval-honest` for the fine-tune
generations, `make calibration` for the judge scores.

## 2. The leak, and the holdout that replaced it

The first fine-tune result was a grounded rate of 0.92 for the LoRA adapter
against 0.17 for the base model (22/24 vs 4/24). The instruction set had been
built from the same 24 golden questions the result was scored on: train and
test were the same set, and the number mostly measured memorization.

What replaced it:

- **A disjoint holdout** of 28 questions (22 answerable, 6 out of corpus),
  checked for disjointness from training in `tests/test_holdout.py`.
- **A fair baseline**: the base model with few-shot examples in the same output
  contract (Portuguese answer plus `[n]`), to separate learned knowledge from
  learned format.
- **Metrics that cannot be satisfied by format alone**: `citation_correct`
  resolves each `[n]` to the retrieved chunk and checks it is the expected
  document (a bracket alone used to count as grounded), and abstention is
  detected in Portuguese as well as in the canonical English sentence.

On the holdout the promoted adapter cites the right document on 18/22
questions against 11/22 for the few-shot baseline, paired McNemar p = 0.016
(7 questions only the adapter gets right, none the other way). The abstention
difference (5/6 vs 1/6) is not significant: with 6 out-of-corpus questions the
smallest p any outcome could reach is 0.031. Details:
[`finetuning-results.md`](finetuning-results.md).

## 3. Reporting uncertainty

Every rate is printed as `k/n` with a 95% Wilson interval (`anchora.stats`),
every mean of a per-question score with a seeded bootstrap interval, and two
systems scored on the same questions are compared with the exact McNemar test
on the discordant pairs. When several generations answer the same question
(the judge calibration below), intervals resample whole questions, not rows.
Each analysis states the smallest effect it could have detected, so a null
result reads as "not resolvable at this n" rather than "no difference".

## 4. The lexical proxy against an LLM judge

CI scores faithfulness with a lexical proxy: the share of the answer's content
tokens that occur in the retrieved context. It is free and deterministic, and it
cannot see negation, paraphrase or wrong numbers. To know how far it can be
trusted, 100 held-out generations (22 questions, five fine-tune arms,
abstentions excluded) were scored by the proxy and by two local judges,
`qwen3:32b` and `gemma4:31b`, at temperature 0 on a 0-4 rubric. The scores, the
model digests and the prompt hash are frozen; `make calibration` recomputes the
statistics offline.

| | Spearman [95% CI, by question] | Same verdict | Cohen kappa |
|---|---|---|---|
| proxy vs qwen3:32b | 0.498 [0.29, 0.68] | 66/100 at 0.70 | 0.36 |
| proxy vs gemma4:31b | 0.490 [0.25, 0.72] | 71/100 at 0.70 | 0.44 |
| qwen3:32b vs gemma4:31b | 0.778 [0.58, 0.92] | 83/100 | 0.60 |

Three readings. The proxy reaches about two thirds of the agreement the two
judges have with each other. The judges themselves disagree on 17 of 100
verdicts, so "the judge" is a noisy reference, not ground truth. And the cutoff
that best reproduces each judge's verdict is unstable (0.16 for one judge, 1.0
for the other), so the CI floor of 0.70 was kept rather than replaced by a
number fitted to this sample ([ADR 7](adr/0007-faithfulness-threshold.md)).
Re-running `qwen3:32b` reproduced every score, which makes the frozen file
reproducible but says nothing about correctness.

The calibration also corrected a reading of the fine-tune results. On the
proxy, the few-shot baseline scores 0.197 against 0.726 for the adapter. Both
judges agree on the direction but see a smaller gap (0.52 vs 0.89 and 0.43 vs
0.93): the proxy is much harsher on the base model's English answers over a
Portuguese context. Details: [`eval-calibration.md`](eval-calibration.md).

## 5. Guardrails: block rate, cost, and generalization

The guardrails are deterministic: a regex input filter for injection and
jailbreak phrasing, PII redaction (CPF, e-mail, phone), an out-of-domain floor
(the question must share at least two distinct tokens with the corpus), and a
grounding check on the output (every `[n]` must resolve to a retrieved chunk,
or the answer must abstain). Three measurements, each with a different author
relationship to the data:

| Measurement | Data | Result |
|---|---|---|
| attacks handled (gated) | 44 attacks written with the guardrails | 44/44 [0.92, 1.00] |
| legitimate questions blocked | 46 golden and holdout questions | 4/46 = 0.087 [0.03, 0.20] |
| attack-looking legitimate questions blocked | 30 written to look like attacks | 7/30 = 0.233 [0.12, 0.41] |
| external injections refused by the input filter | `deepset/prompt-injections` test split | 1/60 = 0.02 [0.00, 0.09] |

The first row says the guardrails catch the phrasings they were written
against. The external row says the input filter does not generalize to other
phrasings; the pipeline still declined 58 of those 60 only because the
out-of-domain floor rejects text unrelated to Brazilian law. The middle rows
are the cost: 4 holdout questions abstain because the out-of-domain floor was
calibrated on the golden set (ADR 6 addendum), and the regex refuses questions
such as "Can a judge bypass the rule of doubled deadlines?".

## 6. Streaming without losing the output check

The output guardrail needs the complete answer, because the citation usually
comes last. `POST /ask/stream` forwards tokens as Ollama decodes them, runs the
check when the model finishes, and emits a `retracted` event if the answer is
not grounded ([ADR 8](adr/0008-streaming-with-output-validation.md)). The
guarantee changes from "ungrounded text is never shown" to "ungrounded text is
always withdrawn before the stream ends"; `POST /ask` keeps the stricter
behaviour for clients that cannot handle a retraction.

## What remains open

- The holdout is small. On the exact McNemar test, the smallest paired result
  that reaches p < 0.05 is 6 discordant questions all in the same direction
  (p = 0.031); 5 to 0 gives p = 0.0625.
- The guardrails are keywords. The external measurement shows they are a
  regression test for known phrasings; a learned classifier evaluated on
  external data would be the next step.
- The out-of-domain floor costs 4 of 22 legitimate holdout questions on the
  offline embedder. The dense similarity floor on the production embedder is
  wired but not calibrated.
- The faithfulness proxy is the CI signal; the judge is the reference. Both are
  automatic: no human labels exist in this repo.
