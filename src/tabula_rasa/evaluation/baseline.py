"""
Baseline evaluator for comparing table knowledge system against raw LLM queries.

This module simulates what happens when you ask a base LLM questions about
a dataset without providing the data or using the table knowledge system.
The LLM has no access to the actual data, so it either:
1. Returns 0 (knows it can't answer)
2. Returns a random plausible-sounding number (hallucination)
3. Refuses to answer

For evaluation purposes, we simulate this by returning random values
or zeros, demonstrating the dramatic difference when an LLM lacks
ground truth data access.
"""

import random
from typing import Dict, List, Optional
import numpy as np


class BaselineLLMEvaluator:
    """
    Simulates a baseline LLM attempting to answer table queries without data access.

    This represents the performance you'd get from asking questions like:
    - "What is the mean of alcohol?"
    - "What is the maximum price?"

    Without providing the LLM any actual data to work with.
    """

    def __init__(self, mode: str = "random"):
        """
        Initialize baseline evaluator.

        Args:
            mode: How the baseline LLM responds
                - "zeros": Always returns 0 (knows it can't answer)
                - "random": Returns random plausible values (hallucination)
                - "refuse": Returns None (refuses to answer)
        """
        self.mode = mode
        self.random_state = random.Random(42)  # Reproducible randomness

    def answer_query(self, question: str, query_type: str = "aggregate") -> Optional[float]:
        """
        Simulate LLM answering a query without data access.

        Args:
            question: Natural language question (e.g., "What is the mean of alcohol?")
            query_type: Type of query (aggregate, conditional, filter)

        Returns:
            Simulated answer (or None if refuses to answer)
        """
        if self.mode == "zeros":
            # LLM knows it can't answer, returns 0
            return 0.0

        elif self.mode == "refuse":
            # LLM refuses to answer without data
            return None

        elif self.mode == "random":
            # LLM hallucinates a plausible-sounding answer
            # This simulates the dangerous scenario where LLM makes up numbers

            # Extract clues from the question to make plausible hallucination
            question_lower = question.lower()

            # For count/filter queries, return integer
            if "how many" in question_lower or "count" in question_lower:
                return float(self.random_state.randint(10, 1000))

            # For percentage/proportion queries
            if "percent" in question_lower or "proportion" in question_lower:
                return self.random_state.uniform(0, 100)

            # For std/variance (typically smaller positive values)
            if "std" in question_lower or "variance" in question_lower:
                return self.random_state.uniform(0.1, 50)

            # For mean/average/median (random reasonable range)
            if "mean" in question_lower or "average" in question_lower or "median" in question_lower:
                return self.random_state.uniform(1, 100)

            # For max/min (wider range)
            if "max" in question_lower or "min" in question_lower:
                return self.random_state.uniform(0, 200)

            # Default: random value
            return self.random_state.uniform(0, 100)

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def evaluate_on_queries(
        self,
        queries: List[Dict],
        ground_truths: List[float]
    ) -> Dict[str, float]:
        """
        Evaluate baseline on a set of queries.

        Args:
            queries: List of query dicts with 'question' and 'query_type' keys
            ground_truths: List of ground truth answers

        Returns:
            Dict of evaluation metrics matching the table knowledge system format
        """
        predictions = []
        valid_indices = []

        for i, query in enumerate(queries):
            answer = self.answer_query(
                question=query['question'],
                query_type=query.get('query_type', 'aggregate')
            )

            if answer is not None:
                predictions.append(answer)
                valid_indices.append(i)

        if not predictions:
            return {
                'mean_error_pct': float('inf'),
                'median_error_pct': float('inf'),
                'mae': float('inf'),
                'std_error_pct': float('inf'),
                'n_queries': len(queries),
                'n_answered': 0,
                'n_refused': len(queries)
            }

        # Calculate metrics against ground truth
        errors = []
        error_pcts = []

        for pred, idx in zip(predictions, valid_indices):
            truth = ground_truths[idx]
            error = abs(pred - truth)
            errors.append(error)

            # Percentage error
            if abs(truth) > 1e-6:
                error_pct = 100 * error / abs(truth)
            else:
                error_pct = 100 * error
            error_pcts.append(error_pct)

        return {
            'mean_error_pct': float(np.mean(error_pcts)),
            'median_error_pct': float(np.median(error_pcts)),
            'mae': float(np.mean(errors)),
            'std_error_pct': float(np.std(error_pcts)),
            'n_queries': len(queries),
            'n_answered': len(predictions),
            'n_refused': len(queries) - len(predictions)
        }


def create_baseline_comparison(
    table_knowledge_metrics: Dict[str, float],
    baseline_metrics: Dict[str, float],
    dataset_name: str
) -> Dict:
    """
    Create a comparison report between table knowledge system and baseline.

    Args:
        table_knowledge_metrics: Metrics from the table knowledge system
        baseline_metrics: Metrics from baseline LLM
        dataset_name: Name of the dataset

    Returns:
        Comprehensive comparison dict
    """
    # Calculate improvement factors
    mean_error_improvement = (
        baseline_metrics['mean_error_pct'] / table_knowledge_metrics['mean_error_pct']
        if table_knowledge_metrics['mean_error_pct'] > 0 else float('inf')
    )

    median_error_improvement = (
        baseline_metrics['median_error_pct'] / table_knowledge_metrics['median_error_pct']
        if table_knowledge_metrics['median_error_pct'] > 0 else float('inf')
    )

    mae_improvement = (
        baseline_metrics['mae'] / table_knowledge_metrics['mae']
        if table_knowledge_metrics['mae'] > 0 else float('inf')
    )

    return {
        'dataset': dataset_name,
        'table_knowledge': table_knowledge_metrics,
        'baseline': baseline_metrics,
        'improvement_factors': {
            'mean_error_pct': mean_error_improvement,
            'median_error_pct': median_error_improvement,
            'mae': mae_improvement
        },
        'summary': {
            'table_knowledge_mean_error': f"{table_knowledge_metrics['mean_error_pct']:.1f}%",
            'baseline_mean_error': f"{baseline_metrics['mean_error_pct']:.1f}%",
            'improvement': f"{mean_error_improvement:.1f}x better"
        }
    }
