# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Tabula Rasa is a production Table Knowledge LLM that teaches language models to accurately answer questions about tabular data. It solves the challenge of querying large tables that don't fit in an LLM's context window by:

1. **Statistical Sketch Extraction**: Compressing tables 50-200x while preserving statistical structure using Gaussian copulas and distribution modeling
2. **Execution Grounding**: Training neural models on actual query results to eliminate hallucination
3. **Multi-Modal Architecture**: Combining T5 text understanding with statistical encoders for robust table reasoning

Key architectural components:
- `tabula_rasa/core/sketch.py`: AdvancedStatSketch class for table compression using Gaussian copulas
- `tabula_rasa/models/qa_model.py`: ProductionTableQA model combining T5 with statistical encoders
- `tabula_rasa/training/trainer.py`: ProductionTrainer with execution grounding methodology
- `tabula_rasa/core/executor.py`: Query execution engine for generating ground truth training data

## Development Commands

### Setup and Installation
```bash
# Install development dependencies
make install-dev
# or
uv sync --group dev

# Install pre-commit hooks
make pre-commit-install
# or
uv run pre-commit install
```

### Testing
```bash
# Run full test suite with coverage
make test
# or
pytest tests/ -v --cov=tabula_rasa --cov-report=term-missing

# Run specific test files
pytest tests/test_models.py -v
pytest tests/test_integration.py -v
```

### Code Quality
```bash
# Format code (ruff)
make format

# Run linting
make lint
# or
uv run ruff check .


# Run all pre-commit hooks
make pre-commit-run
```

### Local CI Testing
```bash
# Test using act (if installed)
make act-test
make act-lint
```

### CLI Usage
```bash
# Analyze a CSV file
tabula-rasa analyze data.csv

# Train a model
tabula-rasa train data.csv --epochs 10 --output model.pt

# Run inference
tabula-rasa inference model.pt data.csv "What is the average price?"
```

## Code Architecture

### Core Components

**Statistical Sketching (`tabula_rasa/core/`)**:
- `sketch.py`: AdvancedStatSketch extracts 15+ statistical features per column, correlations, and Gaussian copula parameters
- `executor.py`: AdvancedQueryExecutor handles query parsing and execution for training data generation

**Neural Models (`tabula_rasa/models/`)**:
- `qa_model.py`: ProductionTableQA combines T5 encoder with StatisticalEncoder
- `encoders.py`: StatisticalEncoder processes table sketches into 768-dim representations

**Training Pipeline (`tabula_rasa/training/`)**:
- `trainer.py`: ProductionTrainer implements execution grounding methodology
- `dataset.py`: TableQADataset manages query-answer pairs for training

### Key Technical Details

**Gaussian Copulas**: Separates marginal distributions from dependency structure, enabling conditional query answering by modeling P(X|Y) through rank-based transformations and Ledoit-Wolf covariance estimation.

**Execution Grounding**: Auto-generates 500-1000 training queries, executes them on real data with pandas/numpy, then trains neural model to predict actual results using only statistical sketch.

**Multi-task Learning**: Model has three heads:
- Answer prediction (numerical output)
- Confidence estimation (0-100%)
- Query type classification (aggregate/filter/conditional)

## Development Workflow

1. **Feature Development**: Create feature branches, ensure comprehensive tests
2. **Code Quality**: All code must pass ruff linting and ruff formatting
3. **Testing Requirements**: Maintain >80% code coverage, test edge cases and error conditions
4. **Documentation**: Update docstrings for API changes, add examples for new features

## Supported Query Types

The system handles:
- **Aggregate queries**: "What is the average price?", "What is the maximum revenue?"
- **Conditional queries**: "What is the average sales when region is West?"
- **Filter queries**: "How many rows have price > 1000?"
- **Percentile queries**: "What is the median house price?"

Current limitations: Numeric queries only, single table, no categorical aggregations or multi-table joins.

## Configuration

**Code Style**:
- Ruff formatting and linting with 100-character line length
- Comprehensive ruff rule set including import sorting

**Dependencies**:
- Python 3.11+ required
- Core: numpy, pandas, scipy, scikit-learn, torch, transformers
- Development: pytest, ruff, pre-commit, uv

**CI/CD**: GitHub Actions runs tests on Ubuntu/macOS/Windows with Python 3.11-3.13, includes lint checking and coverage reporting.
