"""Integration tests for the full pipeline."""

import torch

from tabula_rasa import (
    AdvancedQueryExecutor,
    AdvancedStatSketch,
    ProductionTableQA,
    ProductionTrainer,
    Query,
)


class TestFullPipeline:
    """Test the complete pipeline from data to inference."""

    def test_end_to_end_pipeline(self, wine_dataframe):
        """Test complete pipeline: extract, train, infer."""
        # Extract sketch
        sketcher = AdvancedStatSketch()
        sketch = sketcher.extract(wine_dataframe, table_name="wine")

        assert sketch["n_rows"] == len(wine_dataframe)

        # Initialize model
        model = ProductionTableQA(model_name="t5-small", stat_dim=512)

        # Quick training
        trainer = ProductionTrainer(model, wine_dataframe, sketch, lr=1e-4, batch_size=8)
        best_loss, history = trainer.train(n_epochs=1, n_train_samples=20, n_val_samples=10)

        assert isinstance(best_loss, float)
        assert "train_loss" in history
        assert "val_loss" in history

        # Inference
        model.eval()
        question = "What is the average alcohol?"

        with torch.no_grad():
            output = model(question, sketch)

        assert "answer" in output
        assert isinstance(output["answer"].item(), float)

    def test_sketch_executor_consistency(self, sample_dataframe):
        """Test that sketch and executor give consistent results."""
        sketcher = AdvancedStatSketch()
        sketch = sketcher.extract(sample_dataframe)
        executor = AdvancedQueryExecutor(sample_dataframe)

        # Execute a query
        query = Query(query_type="aggregate", target_column="price", aggregation="mean")
        true_answer = executor.execute(query)

        # Check sketch contains relevant statistics
        price_stats = sketch["columns"]["price"]
        sketch_mean = price_stats["mean"]

        assert abs(true_answer - sketch_mean) < 1e-6

    def test_multiple_datasets(self, sample_dataframe, wine_dataframe):
        """Test that the same pipeline works on different datasets."""
        sketcher = AdvancedStatSketch()

        # Extract sketches from both datasets
        sketch1 = sketcher.extract(sample_dataframe, table_name="sample")
        sketch2 = sketcher.extract(wine_dataframe, table_name="wine")

        assert sketch1["table_name"] == "sample"
        assert sketch2["table_name"] == "wine"

        # Both should have the same structure
        assert set(sketch1.keys()) == set(sketch2.keys())
