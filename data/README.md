# Datasheet — anchora datasets

Following the spirit of *Datasheets for Datasets* (Gebru et al.), this documents
what lives under `data/`, how it was built, and how it is used. Everything here
is small, hand-authored, and versioned so the whole pipeline is reproducible
offline.

## `corpus/` — the knowledge base

- **What:** excerpts of Brazilian public-law statutes (LAI, Lei 8.112, LC 80 /
  Defensoria Pública, LGPD, CPC deadlines, Lei 14.133 procurement, Lei 9.784
  administrative procedure, free legal aid).
- **Format:** `.md`/`.txt`, each with an optional `title:` front-matter line used
  as the human-readable citation source.
- **Provenance:** public-domain Brazilian legislation, condensed to the passages
  the golden/holdout questions probe. Not the full statutes.
- **Use:** ingested (chunked + embedded) into the vector store for retrieval.

## `golden/golden.json` — training / dev set (24 questions)

- **What:** 24 questions with `expected_doc` and a `reference_answer`, covering
  the 8 corpus documents.
- **Use:** the CI eval gate (retrieval recall + faithfulness floor) and the
  fine-tune training signal. Few-shot exemplars are drawn **only** from here.
- **Language:** questions in English, corpus in Portuguese — an intentional
  cross-lingual setup exercised by the EN→PT glossary bridge.

## `golden/holdout.json` — held-out test set (28 questions)

- **What:** 28 brand-new questions (22 answerable, 6 out-of-corpus) the adapter
  never trained on. Abstention cases carry the exact refusal sentence and
  `expected_doc: "NONE"`.
- **Why:** the honest generalization test. Disjointness from training is
  asserted in `tests/test_holdout.py` (no shared id or question text).
- **Use:** the held-out fine-tune metrics (citation-correct, PT-aware abstention,
  faithfulness) and the judge-calibration sample.

## `eval/holdout-generations.json` — frozen model outputs

- **What:** the real decoded generations for each arm (base+few-shot, LoRA-0/5/10)
  on the holdout, frozen so they can be **re-scored deterministically without a
  GPU** (`make eval-honest`, `scripts/score_generations.py`).
- **Why:** generation needs Apple Silicon/GPU; re-scoring must not. Freezing the
  outputs makes the reported numbers reproducible in CI. A drift from the values
  in `finetuning-results.md` fails the build.

## `finetune/` — instruction dataset

- **What:** the built instruction/completion pairs for LoRA fine-tuning
  (`scripts/build_finetune_dataset.py`), including the 5 abstention examples.
- **Use:** training input for `scripts/finetune_lora.py` (completion-only loss).

## `adversarial/attacks.json` — guardrail attack suite

- **What:** 46 attacks across injection, jailbreak, PII exfiltration, citation
  forgery and off-domain, each with an `expected` contract and a `known_gap`
  flag for documented limitations.
- **Use:** `scripts/adversarial_suite.py` / `make adversarial` — a CI gate on the
  guardrail block rate. See the file's own `description` field for the contract.

## `adversarial/benign.json`: legitimate questions (false-positive rate)

- **What:** 30 hand-written legitimate questions that look like attacks
  (trigger words, override and roleplay phrasing, PII as a format example,
  plain Portuguese). The answerable golden and holdout questions are loaded by
  reference, not copied. The file also declares the false-positive ceilings and
  the `known_over_blocks` (holdout questions the out-of-domain floor rejects).
- **Provenance:** written by the same author as the guardrails, before the
  false-positive rate was first measured.
- **Use:** the benign side of `make adversarial`, gated on the declared ceilings.

## `adversarial/external-deepset-prompt-injections.json`: external attacks

- **What:** a verbatim copy of the test split (116 rows: 60 injections, 56
  benign; mostly English, some German) of
  [`deepset/prompt-injections`](https://huggingface.co/datasets/deepset/prompt-injections).
- **Provenance and license:** Apache-2.0, dataset revision
  `4f61ecb038e9c3fb77e21034b22511b523772cdd`, source file
  `data/test-00000-of-00001-701d16158af87368.parquet` with SHA-256
  `39ac797cabc157eeed58435a08593b2952bb6cb16fc394a2d383f447cc7b246e` (both recorded
  in the file). Labels are the dataset's own (1 = injection, 0 = benign).
- **Use:** `make adversarial-external`, an ungated generalization measurement.
  It is not used to write or tune any guardrail pattern.

## PII note

No real personal data appears in any dataset. PII-shaped strings in the
adversarial suite (CPF/email/phone) are synthetic and exist only to prove the
redaction path never echoes them.
