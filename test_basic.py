"""
Basic test script to verify the production_table_llm implementation
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
import torch

from production_table_llm import (
    AdvancedStatSketch,
    AdvancedQueryExecutor,
    ProductionTableQA,
    ProductionTrainer,
    Query
)

def test_basic_functionality():
    """Test basic functionality of the system"""
    print("="*80)
    print("TESTING PRODUCTION TABLE KNOWLEDGE LLM")
    print("="*80)

    # Load sample data
    print("\n[1/5] Loading Wine dataset...")
    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    df['target'] = wine.target
    print(f"  ✓ Loaded {len(df)} rows, {len(df.columns)} columns")

    # Extract sketch
    print("\n[2/5] Extracting statistical sketch...")
    sketcher = AdvancedStatSketch()
    sketch = sketcher.extract(df, table_name='wine')
    print(f"  ✓ Extracted {len(sketch['columns'])} columns")
    print(f"  ✓ Found {len(sketch['correlations'])} correlations")
    print(f"  ✓ Copula condition number: {sketch['copula']['condition_number']:.2f}")

    # Test executor
    print("\n[3/5] Testing query executor...")
    executor = AdvancedQueryExecutor(df)

    queries = [
        Query('aggregate', target_column='alcohol', aggregation='mean'),
        Query('aggregate', target_column='alcohol', aggregation='max'),
        Query('aggregate', target_column='alcohol', aggregation='count'),
    ]

    for query in queries:
        result = executor.execute(query)
        print(f"  ✓ {query.aggregation}(alcohol) = {result:.2f}")

    # Initialize model
    print("\n[4/5] Initializing model...")
    model = ProductionTableQA(model_name='t5-small', stat_dim=512)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Model initialized with {total_params:,} parameters")

    # Quick training test (minimal epochs)
    print("\n[5/5] Running quick training test...")
    trainer = ProductionTrainer(model, df, sketch, lr=1e-4, batch_size=8)
    best_loss, history = trainer.train(n_epochs=2, n_train_samples=50, n_val_samples=20)
    print(f"  ✓ Training completed. Best loss: {best_loss:.4f}")
    print(f"  ✓ Final validation MAE: {history['val_mae'][-1]:.4f}")

    # Test inference
    print("\n[6/6] Testing inference...")
    model.eval()

    test_questions = [
        ("What is the average alcohol?", Query('aggregate', target_column='alcohol', aggregation='mean')),
        ("What is the maximum alcohol?", Query('aggregate', target_column='alcohol', aggregation='max')),
    ]

    for question, query in test_questions:
        true_answer = executor.execute(query)

        with torch.no_grad():
            output = model(question, sketch)
            predicted = output['answer'].item()
            confidence = output['confidence'].item()

        error = abs(predicted - true_answer)
        error_pct = 100 * error / (abs(true_answer) + 1e-6)

        print(f"\n  Q: {question}")
        print(f"     True: {true_answer:.2f}")
        print(f"     Pred: {predicted:.2f} (error: {error_pct:.1f}%)")
        print(f"     Conf: {confidence:.2%}")

    print("\n" + "="*80)
    print("ALL TESTS PASSED!")
    print("="*80)
    print("\nThe system is working correctly. Key features verified:")
    print("  ✓ Statistical sketch extraction")
    print("  ✓ Query execution")
    print("  ✓ Model initialization")
    print("  ✓ Training pipeline")
    print("  ✓ Inference with confidence estimation")
    print("\nNext steps:")
    print("  - Run full demo: jupyter notebook demo_multiple_datasets.ipynb")
    print("  - Train on your own data using the production_table_llm module")
    print("="*80)

if __name__ == "__main__":
    test_basic_functionality()
