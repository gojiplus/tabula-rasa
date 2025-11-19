# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-11-19

### Added
- Initial release of tabula-rasa
- Core statistical sketch extraction (`AdvancedStatSketch`)
- Query execution engine (`AdvancedQueryExecutor`)
- Production Table QA model with T5 backbone (`ProductionTableQA`)
- Training pipeline with execution grounding (`ProductionTrainer`)
- CLI interface for training, inference, and analysis
- Comprehensive test suite with pytest
- Example scripts and Jupyter notebooks
- Full documentation and README

### Features
- Statistical sketching for 50-200x table compression
- Gaussian copula-based conditional reasoning
- Execution grounding to prevent hallucination
- Multi-task learning (answer, confidence, query type)
- Support for aggregate, conditional, and filter queries
- Automatic distribution detection
- Robust correlation estimation
- Mutual information computation

### Developer Experience
- Modern Python packaging with pyproject.toml
- Pre-commit hooks for code quality
- GitHub Actions CI/CD
- Type hints throughout
- Black formatting, Ruff linting
- 80%+ test coverage

[0.1.0]: https://github.com/gojiplus/tabula-rasa/releases/tag/v0.1.0
