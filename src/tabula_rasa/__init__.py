"""
Tabula Rasa: Production Table Knowledge LLM.

Teaching LLMs to accurately answer questions about tabular data through
statistical sketching and execution grounding.
"""

from .core import AdvancedQueryExecutor, AdvancedStatSketch, Query
from .models import ProductionTableQA, StatisticalEncoder
from .training import ProductionTrainer, TableQADataset

__all__ = [
    "AdvancedStatSketch",
    "AdvancedQueryExecutor",
    "Query",
    "StatisticalEncoder",
    "ProductionTableQA",
    "TableQADataset",
    "ProductionTrainer",
]
