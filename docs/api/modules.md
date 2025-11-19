# API Reference

This section contains the complete API documentation for Tabula Rasa.

## Core Modules

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst
   :recursive:

   tabula_rasa
```

## Main Classes

### TabulaRasa

The main model class for table question answering.

```{eval-rst}
.. autoclass:: tabula_rasa.TabulaRasa
   :members:
   :undoc-members:
   :show-inheritance:
```

### TableQADataset

Dataset class for table QA tasks.

```{eval-rst}
.. autoclass:: tabula_rasa.TableQADataset
   :members:
   :undoc-members:
   :show-inheritance:
```

## Training

Classes and functions for training models.

```{eval-rst}
.. automodule:: tabula_rasa.training
   :members:
   :undoc-members:
   :show-inheritance:
```

### Trainer

```{eval-rst}
.. autoclass:: tabula_rasa.training.Trainer
   :members:
   :undoc-members:
   :show-inheritance:
```

### TrainingArguments

```{eval-rst}
.. autoclass:: tabula_rasa.training.TrainingArguments
   :members:
   :undoc-members:
   :show-inheritance:
```

## Evaluation

Classes and functions for model evaluation.

```{eval-rst}
.. automodule:: tabula_rasa.evaluation
   :members:
   :undoc-members:
   :show-inheritance:
```

### Evaluator

```{eval-rst}
.. autoclass:: tabula_rasa.evaluation.Evaluator
   :members:
   :undoc-members:
   :show-inheritance:
```

### Metrics

```{eval-rst}
.. automodule:: tabula_rasa.evaluation.metrics
   :members:
   :undoc-members:
   :show-inheritance:
```

## Data Processing

Utilities for data processing and augmentation.

```{eval-rst}
.. automodule:: tabula_rasa.data
   :members:
   :undoc-members:
   :show-inheritance:
```

## Utilities

Helper functions and utilities.

```{eval-rst}
.. automodule:: tabula_rasa.utils
   :members:
   :undoc-members:
   :show-inheritance:
```

## CLI

Command-line interface documentation.

```{eval-rst}
.. automodule:: tabula_rasa.cli
   :members:
   :undoc-members:
   :show-inheritance:
```
