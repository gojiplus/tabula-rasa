"""Tests for query executor."""

import numpy as np
import pytest

from tabula_rasa import AdvancedQueryExecutor, Query


class TestAdvancedQueryExecutor:
    """Test suite for AdvancedQueryExecutor."""

    def test_aggregate_mean(self, sample_dataframe):
        """Test mean aggregation."""
        executor = AdvancedQueryExecutor(sample_dataframe)
        query = Query(query_type="aggregate", target_column="price", aggregation="mean")

        result = executor.execute(query)
        expected = sample_dataframe["price"].mean()

        assert abs(result - expected) < 1e-6

    def test_aggregate_sum(self, sample_dataframe):
        """Test sum aggregation."""
        executor = AdvancedQueryExecutor(sample_dataframe)
        query = Query(query_type="aggregate", target_column="revenue", aggregation="sum")

        result = executor.execute(query)
        expected = sample_dataframe["revenue"].sum()

        assert abs(result - expected) < 1e-6

    def test_aggregate_count(self, sample_dataframe):
        """Test count aggregation."""
        executor = AdvancedQueryExecutor(sample_dataframe)
        query = Query(query_type="aggregate", target_column="price", aggregation="count")

        result = executor.execute(query)
        expected = len(sample_dataframe)

        assert result == expected

    def test_aggregate_min_max(self, sample_dataframe):
        """Test min/max aggregations."""
        executor = AdvancedQueryExecutor(sample_dataframe)

        query_min = Query(query_type="aggregate", target_column="price", aggregation="min")
        result_min = executor.execute(query_min)
        assert abs(result_min - sample_dataframe["price"].min()) < 1e-6

        query_max = Query(query_type="aggregate", target_column="price", aggregation="max")
        result_max = executor.execute(query_max)
        assert abs(result_max - sample_dataframe["price"].max()) < 1e-6

    def test_conditional_query(self, sample_dataframe):
        """Test conditional aggregation."""
        executor = AdvancedQueryExecutor(sample_dataframe)
        query = Query(
            query_type="conditional",
            target_column="revenue",
            aggregation="mean",
            condition="price > 100",
        )

        result = executor.execute(query)
        expected = sample_dataframe[sample_dataframe["price"] > 100]["revenue"].mean()

        if not np.isnan(expected):
            assert abs(result - expected) < 1e-6

    def test_filter_query(self, sample_dataframe):
        """Test filter query (count matching rows)."""
        executor = AdvancedQueryExecutor(sample_dataframe)
        query = Query(query_type="filter", condition="quantity > 25")

        result = executor.execute(query)
        expected = (sample_dataframe["quantity"] > 25).sum()

        assert result == expected

    def test_condition_parsing(self, sample_dataframe):
        """Test condition parsing with different operators."""
        executor = AdvancedQueryExecutor(sample_dataframe)

        operators = [">", "<", ">=", "<=", "==", "!="]
        for op in operators:
            query = Query(query_type="filter", condition=f"price {op} 100")
            result = executor.execute(query)
            assert isinstance(result, int)
            assert result >= 0

    def test_invalid_query_type(self, sample_dataframe):
        """Test handling of invalid query type."""
        executor = AdvancedQueryExecutor(sample_dataframe)
        query = Query(query_type="invalid_type")

        with pytest.raises(ValueError):
            executor.execute(query)

    def test_invalid_aggregation(self, sample_dataframe):
        """Test handling of invalid aggregation."""
        executor = AdvancedQueryExecutor(sample_dataframe)
        query = Query(query_type="aggregate", target_column="price", aggregation="invalid_agg")

        with pytest.raises(ValueError):
            executor.execute(query)
