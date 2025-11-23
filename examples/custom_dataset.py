"""Example using a custom dataset."""

import pandas as pd

from tabula_rasa import AdvancedQueryExecutor, AdvancedStatSketch, Query


def main():
    """Run custom dataset example."""
    print("=" * 80)
    print("CUSTOM DATASET EXAMPLE")
    print("=" * 80)

    # Create a custom dataset
    print("\n1. Creating custom sales dataset...")
    df = pd.DataFrame(
        {
            "product": ["A", "B", "C", "A", "B", "C"] * 50,
            "price": [10.5, 15.0, 20.0, 10.0, 14.5, 19.5] * 50,
            "quantity": [100, 150, 75, 120, 130, 80] * 50,
            "region": ["North", "South", "East", "West", "North", "South"] * 50,
        }
    )

    # Add calculated columns
    df["revenue"] = df["price"] * df["quantity"]

    print(f"   Created dataset with {len(df)} rows, {len(df.columns)} columns")

    # Extract sketch
    print("\n2. Extracting statistical sketch...")
    sketcher = AdvancedStatSketch()
    sketch = sketcher.extract(df, table_name="sales")

    print(f"   Table: {sketch['table_name']}")
    print(f"   Rows: {sketch['n_rows']}")
    print(
        f"   Numeric columns: {len([c for c in sketch['columns'] if sketch['columns'][c]['type'] == 'numeric'])}"
    )
    print(
        f"   Categorical columns: {len([c for c in sketch['columns'] if sketch['columns'][c]['type'] == 'categorical'])}"
    )

    # Show column statistics
    print("\n3. Column statistics:")
    for col_name, col_stats in sketch["columns"].items():
        if col_stats["type"] == "numeric":
            print(f"\n   {col_name}:")
            print(f"      Mean: {col_stats['mean']:.2f}")
            print(f"      Std:  {col_stats['std']:.2f}")
            print(f"      Range: [{col_stats['min']:.2f}, {col_stats['max']:.2f}]")

    # Execute some queries
    print("\n4. Executing queries...")
    executor = AdvancedQueryExecutor(df)

    queries = [
        ("Average price", Query("aggregate", "price", "mean")),
        ("Total revenue", Query("aggregate", "revenue", "sum")),
        ("Max quantity", Query("aggregate", "quantity", "max")),
    ]

    for desc, query in queries:
        result = executor.execute(query)
        print(f"   {desc}: {result:.2f}")

    print("\n" + "=" * 80)
    print("Custom dataset example completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
