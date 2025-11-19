"""Dataset generation for table QA training."""

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from ..core.executor import AdvancedQueryExecutor, Query


class TableQADataset(Dataset):
    """Dataset for table QA with synthetic query generation."""

    def __init__(self, df: pd.DataFrame, sketch: dict, n_samples: int = 1000):
        """
        Initialize dataset with synthetic query generation.

        Args:
            df: Source DataFrame
            sketch: Statistical sketch of the DataFrame
            n_samples: Number of training samples to generate
        """
        self.df = df
        self.sketch = sketch
        self.executor = AdvancedQueryExecutor(df)
        self.samples = self._generate_samples(n_samples)

    def _generate_samples(self, n_samples: int) -> list[tuple]:
        """Generate diverse training samples."""
        samples = []
        numeric_cols = [
            c
            for c, s in self.sketch["columns"].items()
            if s["type"] == "numeric" and "error" not in s
        ]

        if len(numeric_cols) == 0:
            return samples

        query_types = ["aggregate", "conditional", "filter"]
        aggregations = ["mean", "std", "min", "max", "count"]

        for _ in range(n_samples):
            query_type = np.random.choice(query_types)

            if query_type == "aggregate":
                col = np.random.choice(numeric_cols)
                agg = np.random.choice(aggregations)

                question = self._format_question(query_type, col, agg)
                query = Query(query_type="aggregate", target_column=col, aggregation=agg)

                try:
                    answer = self.executor.execute(query)
                    if not (np.isnan(answer) or np.isinf(answer)):
                        samples.append((question, answer, query_type))
                except Exception:
                    continue

            elif query_type == "conditional":
                if len(numeric_cols) < 2:
                    continue

                target_col = np.random.choice(numeric_cols)
                condition_col = np.random.choice([c for c in numeric_cols if c != target_col])
                agg = np.random.choice(["mean", "count", "std"])

                # Random threshold
                threshold = self.df[condition_col].quantile(np.random.uniform(0.3, 0.7))
                op = np.random.choice([">", "<"])
                condition = f"{condition_col} {op} {threshold:.2f}"

                question = self._format_conditional_question(target_col, condition, agg)
                query = Query(
                    query_type="conditional",
                    target_column=target_col,
                    aggregation=agg,
                    condition=condition,
                )

                try:
                    answer = self.executor.execute(query)
                    if not (np.isnan(answer) or np.isinf(answer)):
                        samples.append((question, answer, query_type))
                except Exception:
                    continue

            elif query_type == "filter":
                col = np.random.choice(numeric_cols)
                threshold = self.df[col].quantile(np.random.uniform(0.3, 0.7))
                op = np.random.choice([">", "<"])
                condition = f"{col} {op} {threshold:.2f}"

                question = f"How many rows have {condition}?"
                query = Query(query_type="filter", condition=condition)

                try:
                    answer = self.executor.execute(query)
                    if not (np.isnan(answer) or np.isinf(answer)):
                        samples.append((question, answer, query_type))
                except Exception:
                    continue

        return samples

    def _format_question(self, _query_type: str, col: str, agg: str) -> str:
        """Format natural language question."""
        templates = {
            "mean": [
                f"What is the average {col}?",
                f"What is the mean value of {col}?",
                f"Calculate the average {col}",
            ],
            "std": [
                f"What is the standard deviation of {col}?",
                f"How much does {col} vary?",
            ],
            "min": [
                f"What is the minimum {col}?",
                f"What is the smallest value of {col}?",
            ],
            "max": [
                f"What is the maximum {col}?",
                f"What is the largest value of {col}?",
            ],
            "count": [
                "How many rows are there?",
                "What is the total number of rows?",
            ],
        }

        return np.random.choice(templates.get(agg, [f"What is the {agg} of {col}?"]))

    def _format_conditional_question(self, target: str, condition: str, agg: str) -> str:
        """Format conditional question."""
        templates = {
            "mean": f"What is the average {target} when {condition}?",
            "count": f"How many rows have {condition}?",
            "std": f"What is the standard deviation of {target} when {condition}?",
        }
        return templates.get(agg, f"What is the {agg} of {target} when {condition}?")

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """Get a single sample."""
        question, answer, query_type = self.samples[idx]

        # Query type to index
        type_to_idx = {"aggregate": 0, "conditional": 1, "filter": 2, "group_by": 3}
        query_type_idx = type_to_idx[query_type]

        return {"question": question, "answer": float(answer), "query_type": query_type_idx}
