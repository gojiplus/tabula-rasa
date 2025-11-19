"""Tests for statistical sketch extraction."""

import numpy as np
import pandas as pd


class TestAdvancedStatSketch:
    """Test suite for AdvancedStatSketch."""

    def test_extract_basic(self, sample_dataframe, sketch_extractor):
        """Test basic sketch extraction."""
        sketch = sketch_extractor.extract(sample_dataframe, table_name="test")

        assert sketch["table_name"] == "test"
        assert sketch["n_rows"] == len(sample_dataframe)
        assert sketch["n_cols"] == len(sample_dataframe.columns)
        assert len(sketch["columns"]) == len(sample_dataframe.columns)

    def test_numeric_column_stats(self, sample_dataframe, sketch_extractor):
        """Test numeric column statistics."""
        sketch = sketch_extractor.extract(sample_dataframe)

        price_stats = sketch["columns"]["price"]
        assert price_stats["type"] == "numeric"
        assert "mean" in price_stats
        assert "std" in price_stats
        assert "min" in price_stats
        assert "max" in price_stats
        assert "quantiles" in price_stats
        assert "skewness" in price_stats
        assert "kurtosis" in price_stats

    def test_categorical_column_stats(self, sample_dataframe, sketch_extractor):
        """Test categorical column statistics."""
        sketch = sketch_extractor.extract(sample_dataframe)

        category_stats = sketch["columns"]["category"]
        assert category_stats["type"] == "categorical"
        assert "n_unique" in category_stats
        assert "mode" in category_stats
        assert "entropy" in category_stats
        assert "gini" in category_stats

    def test_correlations(self, sample_dataframe, sketch_extractor):
        """Test correlation computation."""
        sketch = sketch_extractor.extract(sample_dataframe)

        assert "correlations" in sketch
        # Should have some correlations between numeric columns
        assert isinstance(sketch["correlations"], dict)

    def test_copula_fitting(self, sample_dataframe, sketch_extractor):
        """Test Gaussian copula fitting."""
        sketch = sketch_extractor.extract(sample_dataframe)

        assert "copula" in sketch
        assert sketch["copula"] is not None
        if "error" not in sketch["copula"]:
            assert sketch["copula"]["type"] == "gaussian"
            assert "covariance" in sketch["copula"]
            assert "shrinkage" in sketch["copula"]

    def test_empty_dataframe(self, sketch_extractor):
        """Test sketch extraction on empty DataFrame."""
        df = pd.DataFrame()
        sketch = sketch_extractor.extract(df)

        assert sketch["n_rows"] == 0
        assert sketch["n_cols"] == 0

    def test_single_column(self, sketch_extractor):
        """Test sketch extraction with single column."""
        df = pd.DataFrame({"col1": [1, 2, 3, 4, 5]})
        sketch = sketch_extractor.extract(df)

        assert sketch["n_cols"] == 1
        assert "col1" in sketch["columns"]
        assert sketch["columns"]["col1"]["type"] == "numeric"

    def test_distribution_detection(self, sketch_extractor):
        """Test distribution type detection."""
        # Normal distribution
        df_normal = pd.DataFrame({"normal": np.random.normal(0, 1, 1000)})
        sketch_normal = sketch_extractor.extract(df_normal)
        assert sketch_normal["columns"]["normal"]["distribution_hint"] == "normal"

        # Right skewed
        df_skewed = pd.DataFrame({"skewed": np.random.exponential(1, 1000)})
        sketch_skewed = sketch_extractor.extract(df_skewed)
        assert sketch_skewed["columns"]["skewed"]["distribution_hint"] in [
            "right_skewed",
            "exponential",
        ]
