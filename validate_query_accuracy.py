#!/usr/bin/env python3
"""
Comprehensive validation script for Tabula Rasa query accuracy.

This script:
1. Trains models on 4 different datasets (Wine, Diabetes, Housing, Cancer)
2. Tests with comprehensive query sets (aggregate, conditional, filter)
3. Calculates actual metrics (mean error %, median error %, MAE)
4. Saves results to JSON for reproducibility
5. Generates a detailed validation report
"""

import numpy as np
import pandas as pd
import json
import time
from datetime import datetime
from sklearn.datasets import load_wine, load_diabetes, load_breast_cancer
from production_table_llm import (
    AdvancedStatSketch,
    AdvancedQueryExecutor,
    ProductionTableQA,
    ProductionTrainer,
    Query
)


def load_california_housing():
    """Load California housing dataset."""
    from sklearn.datasets import fetch_california_housing
    data = fetch_california_housing()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['MedHouseVal'] = data.target
    return df


def load_dataset(name):
    """Load a dataset by name."""
    if name == "Wine":
        data = load_wine()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        return df
    elif name == "Diabetes":
        data = load_diabetes()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        return df
    elif name == "Housing":
        return load_california_housing()
    elif name == "Cancer":
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        return df
    else:
        raise ValueError(f"Unknown dataset: {name}")


def generate_test_queries(df, dataset_name, n_queries=50):
    """
    Generate diverse test queries for validation.

    Returns a list of tuples: (query_text, ground_truth_answer)
    """
    queries = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Remove very large columns or problematic ones
    numeric_cols = [col for col in numeric_cols if df[col].nunique() > 1]

    if len(numeric_cols) < 2:
        return queries

    # Generate aggregate queries (mean, max, min, count, std)
    operations = ['mean', 'max', 'min', 'count', 'std']
    for op in operations:
        for col in numeric_cols[:min(5, len(numeric_cols))]:  # Limit to 5 columns
            query = f"What is the {op} of {col}?"
            if op == 'mean':
                answer = float(df[col].mean())
            elif op == 'max':
                answer = float(df[col].max())
            elif op == 'min':
                answer = float(df[col].min())
            elif op == 'count':
                answer = float(len(df))
            elif op == 'std':
                answer = float(df[col].std())
            queries.append((query, answer))

            if len(queries) >= n_queries:
                return queries

    # Generate conditional queries
    if len(numeric_cols) >= 2:
        for i, col1 in enumerate(numeric_cols[:3]):
            for col2 in numeric_cols[:3]:
                if col1 != col2:
                    threshold = df[col2].median()
                    query = f"What is the average {col1} when {col2} > {threshold:.2f}?"
                    filtered = df[df[col2] > threshold]
                    if len(filtered) > 0:
                        answer = float(filtered[col1].mean())
                        queries.append((query, answer))

                        if len(queries) >= n_queries:
                            return queries

    # Generate filter/count queries
    for col in numeric_cols[:5]:
        threshold = df[col].median()
        query = f"How many rows have {col} > {threshold:.2f}?"
        answer = float(len(df[df[col] > threshold]))
        queries.append((query, answer))

        if len(queries) >= n_queries:
            return queries

    return queries


def calculate_metrics(predictions, ground_truths):
    """Calculate error metrics."""
    errors = []
    error_pcts = []

    for pred, truth in zip(predictions, ground_truths):
        error = abs(pred - truth)
        errors.append(error)

        # Calculate percentage error (avoid division by zero)
        if abs(truth) > 1e-6:
            error_pct = 100 * error / abs(truth)
        else:
            error_pct = 100 * error  # For values near zero
        error_pcts.append(error_pct)

    return {
        'mean_error_pct': float(np.mean(error_pcts)),
        'median_error_pct': float(np.median(error_pcts)),
        'mae': float(np.mean(errors)),
        'std_error_pct': float(np.std(error_pcts)),
        'n_queries': len(predictions)
    }


def validate_dataset(dataset_name, n_train_samples=800, n_val_samples=200,
                     n_test_queries=50, epochs=10):
    """
    Train and validate on a single dataset.

    Returns validation results dictionary.
    """
    print(f"\n{'='*60}")
    print(f"Validating on {dataset_name} dataset")
    print(f"{'='*60}")

    # Load dataset
    df = load_dataset(dataset_name)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Extract statistical sketch
    print("\nExtracting statistical sketch...")
    sketcher = AdvancedStatSketch()
    sketch = sketcher.extract(df, table_name=dataset_name)
    print(f"  Extracted {len(sketch['columns'])} columns")

    # Initialize model
    print("\nInitializing model...")
    model = ProductionTableQA(model_name='t5-small', stat_dim=512)

    # Initialize trainer
    trainer = ProductionTrainer(
        model=model,
        df=df,
        sketch=sketch,
        lr=1e-4,
        batch_size=16,
        device='cpu'
    )

    # Train
    print(f"\nTraining with {n_train_samples} samples, {n_val_samples} validation samples...")
    start_time = time.time()

    best_val_loss, history = trainer.train(
        n_epochs=epochs,
        n_train_samples=n_train_samples,
        n_val_samples=n_val_samples
    )

    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")

    # Generate test queries
    print(f"\nGenerating {n_test_queries} test queries...")
    test_queries = generate_test_queries(df, dataset_name, n_test_queries)
    print(f"Generated {len(test_queries)} test queries")

    # Run inference on test queries
    print("\nRunning inference on test queries...")
    predictions = []
    ground_truths = []
    failed_queries = []

    model.eval()
    import torch
    with torch.no_grad():
        for query, ground_truth in test_queries:
            try:
                output = model(query, sketch)
                predicted = output['answer'].item()
                predictions.append(predicted)
                ground_truths.append(ground_truth)
            except Exception as e:
                failed_queries.append((query, str(e)))
                print(f"Failed query: {query} - {e}")

    # Calculate metrics
    if len(predictions) > 0:
        metrics = calculate_metrics(predictions, ground_truths)

        # Add additional info
        metrics['dataset_name'] = dataset_name
        metrics['dataset_shape'] = df.shape
        metrics['training_time_seconds'] = training_time
        metrics['n_failed_queries'] = len(failed_queries)
        metrics['final_train_loss'] = float(history['train_loss'][-1])
        metrics['final_val_loss'] = float(history['val_loss'][-1])
        metrics['final_val_mae'] = float(history['val_mae'][-1])

        # Print summary
        print(f"\n{'-'*60}")
        print(f"Results for {dataset_name}:")
        print(f"  Mean Error %:   {metrics['mean_error_pct']:.2f}%")
        print(f"  Median Error %: {metrics['median_error_pct']:.2f}%")
        print(f"  MAE:            {metrics['mae']:.4f}")
        print(f"  Std Error %:    {metrics['std_error_pct']:.2f}%")
        print(f"  Test queries:   {metrics['n_queries']}")
        print(f"  Failed queries: {metrics['n_failed_queries']}")
        print(f"  Final Val MAE:  {metrics['final_val_mae']:.4f}")
        print(f"{'-'*60}")

        return metrics
    else:
        print(f"ERROR: No successful predictions for {dataset_name}")
        return None


def main():
    """Run comprehensive validation on all datasets."""
    print("="*60)
    print("Tabula Rasa Query Accuracy Validation")
    print("="*60)
    print(f"Started at: {datetime.now().isoformat()}")

    datasets = ["Wine", "Diabetes", "Housing", "Cancer"]
    results = {}

    # Validate each dataset
    for dataset_name in datasets:
        try:
            metrics = validate_dataset(
                dataset_name=dataset_name,
                n_train_samples=800,
                n_val_samples=200,
                n_test_queries=50,
                epochs=10
            )
            if metrics:
                results[dataset_name] = metrics
        except Exception as e:
            print(f"ERROR validating {dataset_name}: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    output_file = "validation_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': results
        }, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Validation complete! Results saved to {output_file}")
    print(f"{'='*60}")

    # Generate summary report
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    print(f"\n{'Dataset':<12} {'Mean Err %':<12} {'Median Err %':<14} {'Val MAE':<10} {'N Queries':<12}")
    print("-"*60)

    for dataset_name in datasets:
        if dataset_name in results:
            r = results[dataset_name]
            print(f"{dataset_name:<12} {r['mean_error_pct']:>10.2f}% {r['median_error_pct']:>12.2f}% {r['final_val_mae']:>9.4f}  {r['n_queries']:>10}")

    # Calculate overall statistics
    if results:
        all_mean_errors = [r['mean_error_pct'] for r in results.values()]
        all_median_errors = [r['median_error_pct'] for r in results.values()]

        print("-"*60)
        print(f"\nOverall Statistics:")
        print(f"  Mean Error % Range:   {min(all_mean_errors):.2f}% - {max(all_mean_errors):.2f}%")
        print(f"  Median Error % Range: {min(all_median_errors):.2f}% - {max(all_median_errors):.2f}%")

    print("\n" + "="*60)
    print(f"Completed at: {datetime.now().isoformat()}")
    print("="*60)


if __name__ == "__main__":
    main()
