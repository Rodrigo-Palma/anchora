# Eval calibration — how far the cheap proxy tracks a real judge

The CI gate scores RAG quality with **deterministic lexical proxies**
(`anchora.metrics`) rather than an LLM judge, because a gate must be
reproducible and free ([ADR 1](adr/0001-deterministic-lexical-proxies-in-ci.md)).
That is only honest if we know where the proxy and a real judge disagree — so we
measure it instead of assuming it.

## How to run

```bash
uv run python scripts/calibrate_judge.py          # needs a local Ollama
uv run python scripts/calibrate_judge.py --json
```

The script scores every frozen held-out generation with both the lexical proxy
and an Ollama LLM judge, then reports Pearson, Spearman, mean absolute error and
binary agreement at the 0.5 threshold. It is a **local analysis tool, not a CI
gate**: with no Ollama reachable it says so and exits 0. The correlation math is
unit-tested offline with an injected deterministic judge
(`tests/test_calibration.py`).

## Known blind spots of the lexical proxy

The proxy is token-overlap based, so by construction it cannot see:

- **Negation** — "the deadline is *not* 10 days" overlaps the context as much as
  the correct claim; the judge catches the flipped meaning, the proxy does not.
- **Paraphrase without shared tokens** — a correct answer worded with synonyms
  scores lower than it should. The EN→PT glossary bridge softens this but does
  not remove it.
- **Numeric correctness** — "20 days" vs "30 days" are one token apart; the proxy
  treats them as near-identical.

## How the design contains those blind spots

- The proxy gates **retrieval recall** (near-binary and reliable) and provides a
  faithfulness *floor*, not a precise faithfulness score.
- The honest fine-tune numbers use `citation_correct` (does `[n]` resolve to the
  expected document?) and PT-aware abstention — signals overlap alone cannot fake.
- The LLM judge is available for the richer read whenever depth is needed.

The proxy is a cheap, reproducible floor. Calibration is what keeps calling it a
floor honest.
