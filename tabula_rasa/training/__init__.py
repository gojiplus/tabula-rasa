"""Training components for table QA models."""

from .dataset import TableQADataset
from .trainer import ProductionTrainer

__all__ = [
    "TableQADataset",
    "ProductionTrainer",
]
