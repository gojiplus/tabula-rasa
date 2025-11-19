#!/usr/bin/env python3
"""
Baseline Comparison Script

Compares the Table Knowledge LLM system against a baseline approach where
an LLM is asked questions about a dataset without providing the data.

This demonstrates the critical advantage of the table knowledge system:
- Table Knowledge System: ~5-13% error (learns from execution grounding)
- Baseline LLM: ~100%+ error (no data access, random guesses or zeros)

Usage:
    python compare_baseline.py --datasets Wine Diabetes --baseline-mode random
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np
from sklearn.datasets import load_wine, load_diabetes, fetch_california_housing, load_breast_cancer

from tabula_rasa.core.sketch import AdvancedStatSketch
from tabula_rasa.core.executor import QueryExecutor, Query
from tabula_rasa.models.qa_model import ProductionTableQA
from tabula_rasa.training.dataset import TableQADataset
from tabula_rasa.training.trainer import ProductionTrainer
from tabula_rasa.evaluation.baseline import BaselineLLMEvaluator, create_baseline_comparison


def load_dataset(name: str) -> pd.DataFrame:
    """Load a standard dataset by name."""
    if name == "Wine":
        data = load_wine()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
    elif name == "Diabetes":
        data = load_diabetes()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
    elif name == "Housing":
        data = fetch_california_housing()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['MedHouseVal'] = data.target
    elif name == "Cancer":
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
    else:
        raise ValueError(f"Unknown dataset: {name}")

    return df


def generate_test_queries(df: pd.DataFrame, executor: QueryExecutor) -> List[Dict]:
    """Generate comprehensive test queries and execute them for ground truth."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    queries = []

    # Aggregate queries: mean, max, min, count, std
    for col in numeric_cols[:5]:  # First 5 columns
        for agg in ['mean', 'max', 'min', 'std', 'count']:
            query_obj = Query(
                query_type='aggregate',
                target_column=col,
                aggregation=agg
            )
            try:
                ground_truth = executor.execute_query(query_obj)
                question = f"What is the {agg} of {col}?"
                queries.append({
                    'question': question,
                    'query_type': 'aggregate',
                    'query_obj': query_obj,
                    'ground_truth': ground_truth
                })
            except Exception as e:
                print(f"Skipping query {agg}({col}): {e}")

    # Conditional queries: "average X when Y > threshold"
    for target_col in numeric_cols[:3]:
        for cond_col in numeric_cols[:3]:
            if target_col != cond_col:
                threshold = df[cond_col].quantile(0.5)
                condition = f"{cond_col} > {threshold}"
                query_obj = Query(
                    query_type='conditional',
                    target_column=target_col,
                    aggregation='mean',
                    condition=condition
                )
                try:
                    ground_truth = executor.execute_query(query_obj)
                    question = f"What is the average {target_col} when {condition}?"
                    queries.append({
                        'question': question,
                        'query_type': 'conditional',
                        'query_obj': query_obj,
                        'ground_truth': ground_truth
                    })
                except Exception as e:
                    print(f"Skipping conditional query: {e}")

    # Filter queries: "count rows where X > threshold"
    for col in numeric_cols[:5]:
        threshold = df[col].quantile(0.6)
        condition = f"{col} > {threshold}"
        query_obj = Query(
            query_type='filter',
            condition=condition,
            aggregation='count'
        )
        try:
            ground_truth = executor.execute_query(query_obj)
            question = f"How many rows have {condition}?"
            queries.append({
                'question': question,
                'query_type': 'filter',
                'query_obj': query_obj,
                'ground_truth': ground_truth
            })
        except Exception as e:
            print(f"Skipping filter query: {e}")

    return queries


def evaluate_table_knowledge_system(
    df: pd.DataFrame,
    test_queries: List[Dict],
    epochs: int = 5,
    n_train: int = 200,
    n_val: int = 50
) -> Dict:
    """Train and evaluate the table knowledge system."""
    print("\n" + "="*80)
    print("EVALUATING TABLE KNOWLEDGE SYSTEM")
    print("="*80)

    start_time = time.time()

    # 1. Extract statistical sketch
    print(f"\n[1/4] Extracting statistical sketch from {df.shape[0]} rows...")
    sketch_extractor = AdvancedStatSketch()
    sketch = sketch_extractor.extract_sketch(df)
    print(f"✓ Sketch extracted: {len(sketch['columns'])} columns")

    # 2. Generate training data
    print(f"\n[2/4] Generating training dataset ({n_train} train, {n_val} val)...")
    executor = QueryExecutor(df)
    train_dataset = TableQADataset(df, executor, num_samples=n_train)
    val_dataset = TableQADataset(df, executor, num_samples=n_val, seed=42)
    print(f"✓ Training data: {len(train_dataset)} samples")
    print(f"✓ Validation data: {len(val_dataset)} samples")

    # 3. Train model
    print(f"\n[3/4] Training T5-based model ({epochs} epochs)...")
    model = ProductionTableQA(vocab_size=32128, max_columns=len(sketch['columns']))
    trainer = ProductionTrainer(
        model=model,
        sketch=sketch,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=epochs,
        batch_size=8,
        learning_rate=1e-4
    )
    history = trainer.train()
    print(f"✓ Training complete")
    print(f"  Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"  Final val loss: {history['val_loss'][-1]:.4f}")
    print(f"  Final val MAE: {history['val_mae'][-1]:.4f}")

    # 4. Evaluate on test queries
    print(f"\n[4/4] Evaluating on {len(test_queries)} test queries...")
    predictions = []
    ground_truths = []

    for query in test_queries:
        output = trainer.inference(query['question'])
        predictions.append(output['answer'])
        ground_truths.append(query['ground_truth'])

    # Calculate metrics
    errors = []
    error_pcts = []
    for pred, truth in zip(predictions, ground_truths):
        error = abs(pred - truth)
        errors.append(error)

        if abs(truth) > 1e-6:
            error_pct = 100 * error / abs(truth)
        else:
            error_pct = 100 * error
        error_pcts.append(error_pct)

    training_time = time.time() - start_time

    metrics = {
        'mean_error_pct': float(np.mean(error_pcts)),
        'median_error_pct': float(np.median(error_pcts)),
        'mae': float(np.mean(errors)),
        'std_error_pct': float(np.std(error_pcts)),
        'n_queries': len(test_queries),
        'training_time_seconds': training_time,
        'final_train_loss': float(history['train_loss'][-1]),
        'final_val_loss': float(history['val_loss'][-1]),
        'final_val_mae': float(history['val_mae'][-1])
    }

    print(f"✓ Evaluation complete")
    print(f"  Mean Error: {metrics['mean_error_pct']:.1f}%")
    print(f"  Median Error: {metrics['median_error_pct']:.1f}%")
    print(f"  MAE: {metrics['mae']:.2f}")
    print(f"  Training time: {training_time:.1f}s")

    return metrics


def evaluate_baseline(
    test_queries: List[Dict],
    mode: str = "random"
) -> Dict:
    """Evaluate baseline LLM without data access."""
    print("\n" + "="*80)
    print(f"EVALUATING BASELINE LLM (mode: {mode})")
    print("="*80)

    baseline = BaselineLLMEvaluator(mode=mode)

    print(f"\n[1/1] Evaluating on {len(test_queries)} test queries...")
    print(f"  (Simulating LLM answering questions WITHOUT access to data)")

    ground_truths = [q['ground_truth'] for q in test_queries]
    metrics = baseline.evaluate_on_queries(test_queries, ground_truths)

    print(f"✓ Baseline evaluation complete")
    print(f"  Mean Error: {metrics['mean_error_pct']:.1f}%")
    print(f"  Median Error: {metrics['median_error_pct']:.1f}%")
    print(f"  MAE: {metrics['mae']:.2f}")
    print(f"  Answered: {metrics['n_answered']}/{metrics['n_queries']}")

    return metrics


def print_comparison_report(comparison: Dict):
    """Print a beautiful comparison report."""
    print("\n" + "="*80)
    print(f"COMPARISON REPORT: {comparison['dataset']}")
    print("="*80)

    print("\n┌─────────────────────────────────────────────────────────────────────┐")
    print("│                        PERFORMANCE METRICS                          │")
    print("└─────────────────────────────────────────────────────────────────────┘")

    tk = comparison['table_knowledge']
    bl = comparison['baseline']
    imp = comparison['improvement_factors']

    # Create comparison table
    print("\nMetric                    Table Knowledge    Baseline LLM    Improvement")
    print("─" * 75)
    print(f"Mean Error %              {tk['mean_error_pct']:13.1f}%    {bl['mean_error_pct']:11.1f}%    {imp['mean_error_pct']:6.1f}x better")
    print(f"Median Error %            {tk['median_error_pct']:13.1f}%    {bl['median_error_pct']:11.1f}%    {imp['median_error_pct']:6.1f}x better")
    print(f"MAE                       {tk['mae']:13.2f}     {bl['mae']:11.2f}     {imp['mae']:6.1f}x better")
    print(f"Queries Evaluated         {tk['n_queries']:13d}     {bl['n_queries']:11d}")

    if 'training_time_seconds' in tk:
        print(f"Training Time (s)         {tk['training_time_seconds']:13.1f}     {'N/A':>11}     (one-time cost)")

    print("\n┌─────────────────────────────────────────────────────────────────────┐")
    print("│                             SUMMARY                                 │")
    print("└─────────────────────────────────────────────────────────────────────┘")
    print(f"""
The Table Knowledge System achieves {tk['mean_error_pct']:.1f}% mean error compared to
the baseline's {bl['mean_error_pct']:.1f}% error - a {imp['mean_error_pct']:.1f}x improvement!

This demonstrates the critical advantage of execution grounding:
  ✓ Table Knowledge: Learns from real query executions on actual data
  ✗ Baseline LLM: No data access, resorts to random guesses/hallucination
""")

    print("─" * 75)


def main():
    parser = argparse.ArgumentParser(
        description="Compare Table Knowledge System against Baseline LLM"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["Wine"],
        choices=["Wine", "Diabetes", "Housing", "Cancer"],
        help="Datasets to evaluate"
    )
    parser.add_argument(
        "--baseline-mode",
        choices=["zeros", "random", "refuse"],
        default="random",
        help="Baseline LLM behavior: zeros (returns 0), random (hallucinates), refuse (no answer)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Training epochs for table knowledge system"
    )
    parser.add_argument(
        "--n-train",
        type=int,
        default=200,
        help="Number of training samples"
    )
    parser.add_argument(
        "--n-val",
        type=int,
        default=50,
        help="Number of validation samples"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="baseline_comparison_results.json",
        help="Output JSON file for results"
    )

    args = parser.parse_args()

    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║        TABLE KNOWLEDGE vs BASELINE LLM COMPARISON                 ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print(f"\nDatasets: {', '.join(args.datasets)}")
    print(f"Baseline mode: {args.baseline_mode}")
    print(f"Training config: {args.epochs} epochs, {args.n_train} train samples")

    all_comparisons = []

    for dataset_name in args.datasets:
        print(f"\n\n{'#'*80}")
        print(f"# DATASET: {dataset_name}")
        print(f"{'#'*80}")

        # Load dataset
        df = load_dataset(dataset_name)
        print(f"\nLoaded {dataset_name}: {df.shape[0]} rows × {df.shape[1]} columns")

        # Generate test queries
        executor = QueryExecutor(df)
        test_queries = generate_test_queries(df, executor)
        print(f"Generated {len(test_queries)} test queries")

        # Evaluate Table Knowledge System
        tk_metrics = evaluate_table_knowledge_system(
            df=df,
            test_queries=test_queries,
            epochs=args.epochs,
            n_train=args.n_train,
            n_val=args.n_val
        )

        # Evaluate Baseline
        baseline_metrics = evaluate_baseline(
            test_queries=test_queries,
            mode=args.baseline_mode
        )

        # Create comparison
        comparison = create_baseline_comparison(
            table_knowledge_metrics=tk_metrics,
            baseline_metrics=baseline_metrics,
            dataset_name=dataset_name
        )
        all_comparisons.append(comparison)

        # Print report
        print_comparison_report(comparison)

    # Save results
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump({
            'comparisons': all_comparisons,
            'config': {
                'datasets': args.datasets,
                'baseline_mode': args.baseline_mode,
                'epochs': args.epochs,
                'n_train': args.n_train,
                'n_val': args.n_val
            },
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }, f, indent=2)

    print(f"\n✓ Results saved to: {output_path}")

    # Print aggregate summary
    if len(all_comparisons) > 1:
        print("\n" + "="*80)
        print("AGGREGATE SUMMARY ACROSS ALL DATASETS")
        print("="*80)

        avg_tk_error = np.mean([c['table_knowledge']['mean_error_pct'] for c in all_comparisons])
        avg_bl_error = np.mean([c['baseline']['mean_error_pct'] for c in all_comparisons])
        avg_improvement = np.mean([c['improvement_factors']['mean_error_pct'] for c in all_comparisons])

        print(f"\nAverage Mean Error:")
        print(f"  Table Knowledge: {avg_tk_error:.1f}%")
        print(f"  Baseline LLM: {avg_bl_error:.1f}%")
        print(f"  Average Improvement: {avg_improvement:.1f}x better")


if __name__ == "__main__":
    main()
