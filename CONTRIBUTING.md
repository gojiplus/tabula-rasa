# Contributing to Tabula Rasa

Thank you for your interest in contributing to Tabula Rasa! This document provides guidelines and instructions for contributing.

## Development Setup

1. **Fork and clone the repository**

```bash
git clone https://github.com/YOUR_USERNAME/tabula-rasa.git
cd tabula-rasa
```

2. **Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install in development mode**

```bash
pip install -e ".[dev]"
```

4. **Install pre-commit hooks**

```bash
pre-commit install
```

## Development Workflow

1. **Create a feature branch**

```bash
git checkout -b feature/your-feature-name
```

2. **Make your changes**

Follow the code style guidelines below.

3. **Run tests**

```bash
pytest tests/ -v
```

4. **Run linters and formatters**

```bash
# Format code
black tabula_rasa/ tests/
isort tabula_rasa/ tests/

# Lint code
ruff check tabula_rasa/ tests/

# Type check
mypy tabula_rasa
```

5. **Commit your changes**

```bash
git add .
git commit -m "feat: add your feature description"
```

Follow [Conventional Commits](https://www.conventionalcommits.org/) format:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `test:` for test additions/changes
- `refactor:` for code refactoring
- `chore:` for maintenance tasks

6. **Push and create a pull request**

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub.

## Code Style Guidelines

- **Python version**: Support Python 3.8+
- **Formatting**: Use Black with 100 character line length
- **Imports**: Use isort with Black profile
- **Linting**: Follow Ruff rules
- **Type hints**: Add type hints to all public functions
- **Docstrings**: Use Google or NumPy style docstrings

## Testing Guidelines

- Write tests for all new features
- Maintain >80% code coverage
- Use pytest fixtures for common test data
- Test edge cases and error conditions
- Run tests locally before pushing:

```bash
pytest tests/ -v --cov=tabula_rasa
```

## Documentation

- Update README.md for user-facing changes
- Update docstrings for API changes
- Add examples for new features
- Update CHANGELOG.md

## Pull Request Process

1. Ensure all tests pass
2. Update documentation
3. Add entry to CHANGELOG.md
4. Request review from maintainers
5. Address review feedback
6. Maintainers will merge when approved

## Code Review

We value:
- Clear, readable code
- Comprehensive tests
- Good documentation
- Performance considerations
- Backward compatibility

## Questions?

- Open an issue for bugs or feature requests
- Start a discussion for questions
- Contact maintainers for security issues

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
