"""
Tabula Rasa: Production Table Knowledge LLM.

Teaching LLMs to accurately answer questions about tabular data through
statistical sketching and execution grounding.
"""

from .__version__ import __version__, __version_info__
from .core import AdvancedQueryExecutor, AdvancedStatSketch, Query
from .models import ProductionTableQA, StatisticalEncoder
from .training import ProductionTrainer, TableQADataset

__all__ = [
    "__version__",
    "__version_info__",
    "AdvancedStatSketch",
    "AdvancedQueryExecutor",
    "Query",
    "StatisticalEncoder",
    "ProductionTableQA",
    "TableQADataset",
    "ProductionTrainer",
]
