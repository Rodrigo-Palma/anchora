.DEFAULT_GOAL := help
.PHONY: help install lint fmt fmt-check type test eval eval-honest ablation adversarial check api dataset pipeline clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Sync the dev environment
	uv sync --extra dev

lint: ## Ruff lint
	uv run ruff check .

fmt: ## Ruff auto-format
	uv run ruff format .

fmt-check: ## Ruff format check (CI)
	uv run ruff format --check .

type: ## mypy type check
	uv run mypy

test: ## Run the test suite with coverage
	uv run pytest

eval: ## Run the offline eval gate
	uv run anchora eval

eval-honest: ## Re-score frozen held-out generations + replay the promotion gate (no GPU)
	uv run python scripts/score_generations.py --check
	uv run python scripts/gate_promotion.py

ablation: ## Measure retrieval modes (dense vs bm25 vs hybrid) on golden + holdout
	uv run python scripts/ablation_retrieval.py

adversarial: ## Replay the adversarial guardrail suite and gate on the block rate
	uv run python scripts/adversarial_suite.py --check

check: lint fmt-check type test eval eval-honest adversarial ## Run the full local CI gate

dataset: ## Build the fine-tune instruction dataset
	uv run python scripts/build_finetune_dataset.py

pipeline: ## Print the MLOps pipeline plan (dry run)
	uv run python -m pipeline.ml_pipeline --dry-run

api: ## Serve the API locally
	uv run uvicorn anchora.api.main:app --reload

clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage dist build
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
