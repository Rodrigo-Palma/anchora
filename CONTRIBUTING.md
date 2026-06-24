# Contributing to anchora

Thanks for your interest. This guide describes the development workflow.

## Principles

- **Local-first, zero cost.** No paid APIs. Embeddings/generation via Ollama;
  tests and CI run with the `hash` provider (deterministic, offline).
- **Determinism in CI.** Anything on the test/eval path must be
  reproducible without network or model.
- **Typing and quality.** `mypy --strict`, `ruff`, and test coverage are
  requirements, not optional.

## Setup

Prerequisites: Python ≥ 3.12 and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev
uv run pre-commit install   # optional, but recommended
```

To run with the local models:

```bash
ollama pull nomic-embed-text
ollama pull qwen3:32b
```

## Before opening a PR

Run the full suite — the same set of checks as CI:

```bash
make check
# equivalent to:
#   uv run ruff check .
#   uv run ruff format --check .
#   uv run mypy
#   uv run pytest
#   uv run anchora eval
```

Everything must pass green. The eval *gate* fails if retrieval recall drops
below 1.0 or the average faithfulness falls below 0.70.

## Code standards

- **Style**: `ruff format` (line 100). Do not format by hand.
- **Lint**: rules `E,F,I,N,UP,B,C4,SIM,RUF`. Fix, don't silence — `noqa`
  only with justification.
- **Types**: `mypy --strict` covers `src`, `pipeline`, and `scripts`. No loose `Any`.
- **Lazy imports**: heavy/optional dependencies (torch, peft, sagemaker)
  are imported only inside the functions that use them, behind optional extras.
- **Tests**: every new behavior comes with a test. Prefer the offline path
  (`provider="hash"`, `use_llm=False`).

## Commits

- Concise messages, in English, imperative mood (e.g., `add model registry`).
- Atomic commits with a clear scope.

## Structure

```
src/anchora/   application code (RAG, agent, guardrails, evals, api, cli)
pipeline/      local ML pipeline + SageMaker skeleton
scripts/       fine-tune dataset, LoRA training, eval comparison
infra/         Terraform (ECR, S3, SageMaker, IAM)
tests/         pytest
data/          corpus, golden set, fine-tune dataset
docs/          architecture, demo, outreach material
```
