.PHONY: help install install-dev test lint format type-check pre-commit-install clean

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install production dependencies
	uv sync

install-dev:  ## Install development dependencies
	uv sync --group dev

test:  ## Run tests with pytest
	uv run pytest tests/ -v --cov=tabula_rasa --cov-report=term-missing

lint:  ## Run linting with ruff
	uv run ruff check .

format:  ## Format code with ruff
	uv run ruff check --fix .
	uv run ruff format .


pre-commit-install:  ## Install pre-commit hooks
	uv run pre-commit install

pre-commit-run:  ## Run pre-commit hooks on all files
	uv run pre-commit run --all-files

clean:  ## Clean up build artifacts and caches
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf build/ dist/ htmlcov/ .coverage coverage.xml

# GitHub Actions local testing with act
act-test:  ## Run GitHub Actions CI workflow locally with act
	@echo "Running CI workflow locally..."
	@if command -v act >/dev/null 2>&1; then \
		act -j test; \
	else \
		echo "Error: 'act' is not installed. Install it with:"; \
		echo "  macOS:   brew install act"; \
		echo "  Linux:   curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash"; \
		echo "  Or visit: https://github.com/nektos/act"; \
	fi

act-lint:  ## Run GitHub Actions lint workflow locally with act
	@echo "Running lint workflow locally..."
	@if command -v act >/dev/null 2>&1; then \
		act -j lint; \
	else \
		echo "Error: 'act' is not installed. See 'make act-test' for installation instructions."; \
	fi
