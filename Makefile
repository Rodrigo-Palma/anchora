.DEFAULT_GOAL := help
.PHONY: help install audit lint fmt fmt-check type test eval eval-honest calibration ablation adversarial adversarial-external bench check api dataset pipeline clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Sync the dev environment
	uv sync --locked --extra dev

audit: ## Audit locked runtime + dev dependencies for known vulnerabilities
	uv export --frozen --extra dev --no-emit-project --format requirements-txt -o .audit-requirements.txt
	uvx pip-audit==2.10.1 -r .audit-requirements.txt --disable-pip --require-hashes; \
		status=$$?; rm -f .audit-requirements.txt; exit $$status

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

calibration: ## Recompute proxy-vs-judge agreement from frozen judge scores (no model)
	uv run python scripts/calibrate_judge.py --check

ablation: ## Measure retrieval modes (dense vs bm25 vs hybrid) on golden + holdout
	uv run python scripts/ablation_retrieval.py

adversarial: ## Replay attacks + benign questions; gate on block rate and false-positive rate
	uv run python scripts/adversarial_suite.py --check

adversarial-external: ## Replay a public prompt-injection set (deepset, frozen) - not gated
	uv run python scripts/adversarial_suite.py --external

bench: ## Benchmark offline pipeline latency (p50/p95 per stage) with a regression gate
	uv run python scripts/benchmark.py --max-p95-ms 250

check: lint fmt-check type test eval eval-honest calibration adversarial ## Run the full local CI gate

dataset: ## Build the fine-tune instruction dataset
	uv run python scripts/build_finetune_dataset.py

pipeline: ## Print the MLOps pipeline plan (dry run)
	uv run python -m pipeline.ml_pipeline --dry-run

api: ## Serve the API locally
	uv run uvicorn anchora.api.main:app --reload

clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage dist build
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
