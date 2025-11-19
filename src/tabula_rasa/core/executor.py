"""Query execution on actual data for training and validation."""

import re
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class Query:
    """Structured query representation."""

    query_type: str  # 'aggregate', 'filter', 'conditional', 'join'
    target_column: str | None = None
    aggregation: str | None = None  # 'mean', 'sum', 'count', 'std', 'percentile'
    condition: str | None = None
    percentile: float | None = None
    group_by: str | None = None


class AdvancedQueryExecutor:
    """
    Execute queries on actual data.

    Supports: aggregations, filters, conditionals, group-by
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize query executor.

        Args:
            df: DataFrame to execute queries against
        """
        self.df = df

    def execute(self, query: Query) -> Any:
        """
        Execute structured query.

        Args:
            query: Query object specifying the operation

        Returns:
            Query result (float for aggregates, int for counts, dict for group-by)

        Raises:
            ValueError: If query type is unknown
        """
        if query.query_type == "aggregate":
            return self._execute_aggregate(query)
        elif query.query_type == "conditional":
            return self._execute_conditional(query)
        elif query.query_type == "filter":
            return self._execute_filter(query)
        elif query.query_type == "group_by":
            return self._execute_group_by(query)
        else:
            raise ValueError(f"Unknown query type: {query.query_type}")

    def _execute_aggregate(self, query: Query) -> float:
        """Execute aggregation query."""
        col_data = self.df[query.target_column]

        if query.aggregation == "mean":
            return float(col_data.mean())
        elif query.aggregation == "sum":
            return float(col_data.sum())
        elif query.aggregation == "count":
            return float(len(col_data))
        elif query.aggregation == "std":
            return float(col_data.std())
        elif query.aggregation == "min":
            return float(col_data.min())
        elif query.aggregation == "max":
            return float(col_data.max())
        elif query.aggregation == "percentile":
            return float(col_data.quantile(query.percentile))
        else:
            raise ValueError(f"Unknown aggregation: {query.aggregation}")

    def _execute_conditional(self, query: Query) -> float:
        """Execute conditional aggregation (e.g., mean of X where Y > 10)."""
        mask = self._parse_condition(query.condition)
        filtered_data = self.df[mask][query.target_column]

        if len(filtered_data) == 0:
            return float("nan")

        if query.aggregation == "mean":
            return float(filtered_data.mean())
        elif query.aggregation == "count":
            return float(len(filtered_data))
        elif query.aggregation == "std":
            return float(filtered_data.std())
        else:
            raise ValueError(f"Unknown aggregation: {query.aggregation}")

    def _execute_filter(self, query: Query) -> int:
        """Execute filter query (count matching rows)."""
        mask = self._parse_condition(query.condition)
        return int(mask.sum())

    def _execute_group_by(self, query: Query) -> dict:
        """Execute group-by aggregation."""
        grouped = self.df.groupby(query.group_by)[query.target_column]

        if query.aggregation == "mean":
            return grouped.mean().to_dict()
        elif query.aggregation == "count":
            return grouped.count().to_dict()
        elif query.aggregation == "sum":
            return grouped.sum().to_dict()
        else:
            raise ValueError(f"Unknown aggregation: {query.aggregation}")

    def _parse_condition(self, condition: str) -> pd.Series:
        """
        Parse condition string into boolean mask.

        Args:
            condition: String like "column > 10" or "price <= 100.5"

        Returns:
            Boolean Series mask

        Raises:
            ValueError: If condition cannot be parsed
        """
        # Support operators: >, <, >=, <=, ==, !=
        pattern = r"(\w+)\s*(>|<|>=|<=|==|!=)\s*([0-9.]+)"
        match = re.match(pattern, condition.strip())

        if not match:
            raise ValueError(f"Cannot parse condition: {condition}")

        col, op, val = match.groups()
        val = float(val)

        if op == ">":
            return self.df[col] > val
        elif op == "<":
            return self.df[col] < val
        elif op == ">=":
            return self.df[col] >= val
        elif op == "<=":
            return self.df[col] <= val
        elif op == "==":
            return self.df[col] == val
        elif op == "!=":
            return self.df[col] != val
        else:
            raise ValueError(f"Unknown operator: {op}")
