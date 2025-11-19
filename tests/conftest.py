"""Pytest fixtures for tabula-rasa tests."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_wine

from tabula_rasa import AdvancedStatSketch, ProductionTableQA


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "price": np.random.normal(100, 20, 100),
            "quantity": np.random.randint(1, 50, 100),
            "revenue": np.random.normal(500, 100, 100),
            "category": np.random.choice(["A", "B", "C"], 100),
        }
    )


@pytest.fixture
def wine_dataframe():
    """Load the wine dataset."""
    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    df["target"] = wine.target
    return df


@pytest.fixture
def sketch_extractor():
    """Create a sketch extractor instance."""
    return AdvancedStatSketch()


@pytest.fixture
def sample_sketch(sample_dataframe, sketch_extractor):
    """Extract sketch from sample DataFrame."""
    return sketch_extractor.extract(sample_dataframe, table_name="test_table")


@pytest.fixture
def qa_model():
    """Create a QA model instance."""
    return ProductionTableQA(model_name="t5-small", stat_dim=512)
