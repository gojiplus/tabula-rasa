# Tabula Rasa Documentation

**Production Table Knowledge LLM: Teaching LLMs to accurately answer questions about tabular data**

Tabula Rasa is a machine learning framework designed to help large language models (LLMs) accurately understand and answer questions about tabular data. Built on modern transformer architectures, it provides tools for training, evaluation, and deployment of table-aware language models.

## Features

- 🎯 **Table-Aware LLMs**: Specialized models for understanding tabular data
- 🚀 **Modern Architecture**: Built on PyTorch and Transformers
- 📊 **Comprehensive Evaluation**: Tools for assessing model performance on table QA tasks
- 🔧 **Easy Integration**: Simple API for training and inference
- 📈 **Production Ready**: Optimized for deployment in production environments

## Quick Start

### Installation

```bash
pip install tabula-rasa
```

For development installation:

```bash
git clone https://github.com/gojiplus/tabula-rasa.git
cd tabula-rasa
pip install -e ".[dev]"
```

### Basic Usage

```python
from tabula_rasa import TabulaRasa

# Initialize the model
model = TabulaRasa()

# Process tabular data
table = {
    "columns": ["Name", "Age", "City"],
    "rows": [
        ["Alice", 30, "New York"],
        ["Bob", 25, "San Francisco"],
    ]
}

# Ask a question
question = "What is Alice's age?"
answer = model.answer(question, table)
print(answer)  # Output: 30
```

## Contents

```{toctree}
:maxdepth: 2
:caption: User Guide

guides/installation
guides/quickstart
guides/training
guides/evaluation
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/modules
```

```{toctree}
:maxdepth: 1
:caption: Development

GitHub Repository <https://github.com/gojiplus/tabula-rasa>
Contributing <https://github.com/gojiplus/tabula-rasa/blob/main/CONTRIBUTING.md>
Changelog <https://github.com/gojiplus/tabula-rasa/blob/main/CHANGELOG.md>
```

## Project Links

- **GitHub**: [github.com/gojiplus/tabula-rasa](https://github.com/gojiplus/tabula-rasa)
- **Issues**: [Issue Tracker](https://github.com/gojiplus/tabula-rasa/issues)
- **License**: MIT

## Indices and Tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
