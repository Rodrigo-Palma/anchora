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

- **What:** 44 attacks across injection, jailbreak, PII exfiltration, citation
  forgery and off-domain, each with an `expected` contract and a `known_gap`
  flag for documented limitations.
- **Use:** `scripts/adversarial_suite.py` / `make adversarial` — a CI gate on the
  guardrail block rate. See the file's own `description` field for the contract.

## PII note

No real personal data appears in any dataset. PII-shaped strings in the
adversarial suite (CPF/email/phone) are synthetic and exist only to prove the
redaction path never echoes them.
