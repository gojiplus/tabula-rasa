"""Basic usage example for tabula-rasa."""

import pandas as pd
import torch
from sklearn.datasets import load_wine

from tabula_rasa import (
    AdvancedQueryExecutor,
    AdvancedStatSketch,
    ProductionTableQA,
    ProductionTrainer,
    Query,
)


def main():
    """Run basic usage example."""
    print("=" * 80)
    print("TABULA RASA - BASIC USAGE EXAMPLE")
    print("=" * 80)

    # Load sample data
    print("\n1. Loading Wine dataset...")
    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    df["target"] = wine.target
    print(f"   Loaded {len(df)} rows, {len(df.columns)} columns")

    # Extract statistical sketch
    print("\n2. Extracting statistical sketch...")
    sketcher = AdvancedStatSketch()
    sketch = sketcher.extract(df, table_name="wine")
    print(f"   Extracted {len(sketch['columns'])} columns")
    print(f"   Found {len(sketch['correlations'])} correlations")

    # Test query executor
    print("\n3. Testing query execution...")
    executor = AdvancedQueryExecutor(df)

    queries = [
        Query("aggregate", target_column="alcohol", aggregation="mean"),
        Query("aggregate", target_column="alcohol", aggregation="max"),
        Query("aggregate", target_column="alcohol", aggregation="min"),
    ]

    for query in queries:
        result = executor.execute(query)
        print(f"   {query.aggregation}(alcohol) = {result:.2f}")

    # Initialize model
    print("\n4. Initializing model...")
    model = ProductionTableQA(model_name="t5-small", stat_dim=512)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model has {total_params:,} parameters")

    # Train the model
    print("\n5. Training model...")
    trainer = ProductionTrainer(model, df, sketch, lr=1e-4, batch_size=8)
    best_loss, history = trainer.train(n_epochs=5, n_train_samples=100, n_val_samples=50)
    print(f"   Best validation loss: {best_loss:.4f}")
    print(f"   Final MAE: {history['val_mae'][-1]:.4f}")

    # Test inference
    print("\n6. Testing inference...")
    model.eval()

    test_questions = [
        ("What is the average alcohol?", Query("aggregate", "alcohol", "mean")),
        ("What is the maximum alcohol?", Query("aggregate", "alcohol", "max")),
    ]

    for question, query in test_questions:
        true_answer = executor.execute(query)

        with torch.no_grad():
            output = model(question, sketch)
            predicted = output["answer"].item()
            confidence = output["confidence"].item()

        error = abs(predicted - true_answer)
        error_pct = 100 * error / (abs(true_answer) + 1e-6)

        print(f"\n   Q: {question}")
        print(f"      True: {true_answer:.2f}")
        print(f"      Pred: {predicted:.2f} (error: {error_pct:.1f}%)")
        print(f"      Conf: {confidence:.2%}")

    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
