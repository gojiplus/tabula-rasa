"""Main CLI entry point."""

import click
import pandas as pd
import torch

from .. import (
    AdvancedQueryExecutor,
    AdvancedStatSketch,
    ProductionTableQA,
    ProductionTrainer,
)
from ..__version__ import __version__


@click.group()
@click.version_option(version=__version__)
def cli():
    """Tabula Rasa: Production Table Knowledge LLM."""
    pass


@cli.command()
@click.argument("data_path", type=click.Path(exists=True))
@click.option("--model-name", default="t5-small", help="T5 model name")
@click.option("--epochs", default=10, help="Number of training epochs")
@click.option("--train-samples", default=1000, help="Number of training samples")
@click.option("--val-samples", default=200, help="Number of validation samples")
@click.option("--batch-size", default=16, help="Batch size")
@click.option("--lr", default=1e-4, help="Learning rate")
@click.option("--output", default="model.pt", help="Output model path")
def train(data_path, model_name, epochs, train_samples, val_samples, batch_size, lr, output):
    """Train a Table QA model on a CSV file."""
    click.echo(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    click.echo(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    click.echo("Extracting statistical sketch...")
    sketcher = AdvancedStatSketch()
    sketch = sketcher.extract(df, table_name="table")
    click.echo(f"Extracted {len(sketch['columns'])} columns")

    click.echo(f"Initializing model ({model_name})...")
    model = ProductionTableQA(model_name=model_name, stat_dim=768)
    total_params = sum(p.numel() for p in model.parameters())
    click.echo(f"Model has {total_params:,} parameters")

    click.echo("Training...")
    trainer = ProductionTrainer(model, df, sketch, lr=lr, batch_size=batch_size)
    best_loss, history = trainer.train(
        n_epochs=epochs, n_train_samples=train_samples, n_val_samples=val_samples
    )

    click.echo("\nTraining complete!")
    click.echo(f"Best validation loss: {best_loss:.4f}")
    click.echo(f"Final MAE: {history['val_mae'][-1]:.4f}")
    click.echo(f"Final MAPE: {history['val_mape'][-1]:.2%}")

    click.echo(f"\nSaving model to {output}...")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "sketch": sketch,
            "config": {"model_name": model_name},
        },
        output,
    )
    click.echo("Done!")


@cli.command()
@click.argument("data_path", type=click.Path(exists=True))
@click.argument("question")
def query(data_path, question):
    """Execute a query on a CSV file (without training)."""
    click.echo(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    click.echo("Extracting statistical sketch...")
    sketcher = AdvancedStatSketch()
    _ = sketcher.extract(df, table_name="table")

    # For simple queries, try to execute directly
    _ = AdvancedQueryExecutor(df)

    click.echo(f"\nQuestion: {question}")
    click.echo("Note: This is a direct execution. For ML-based inference, train a model first.")


@cli.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.argument("data_path", type=click.Path(exists=True))
@click.argument("question")
def inference(model_path, _data_path, question):
    """Run inference with a trained model."""
    click.echo(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location="cpu")

    model = ProductionTableQA(model_name=checkpoint["config"]["model_name"], stat_dim=768)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    sketch = checkpoint["sketch"]

    click.echo(f"\nQuestion: {question}")

    with torch.no_grad():
        output = model(question, sketch)
        answer = output["answer"].item()
        confidence = output["confidence"].item()

    click.echo(f"Answer: {answer:.2f}")
    click.echo(f"Confidence: {confidence:.2%}")


@cli.command()
@click.argument("data_path", type=click.Path(exists=True))
def analyze(data_path):
    """Analyze a CSV file and show statistical sketch."""
    click.echo(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    click.echo(f"Loaded {len(df)} rows, {len(df.columns)} columns\n")

    click.echo("Extracting statistical sketch...")
    sketcher = AdvancedStatSketch()
    sketch = sketcher.extract(df, table_name="table")

    click.echo("\n=== Table Statistics ===")
    click.echo(f"Rows: {sketch['n_rows']}")
    click.echo(f"Columns: {sketch['n_cols']}")

    click.echo("\n=== Column Details ===")
    for col_name, col_stats in sketch["columns"].items():
        click.echo(f"\n{col_name} ({col_stats['type']}):")
        if col_stats["type"] == "numeric":
            click.echo(f"  Mean: {col_stats['mean']:.2f}")
            click.echo(f"  Std: {col_stats['std']:.2f}")
            click.echo(f"  Range: [{col_stats['min']:.2f}, {col_stats['max']:.2f}]")
            click.echo(f"  Distribution: {col_stats['distribution_hint']}")
        else:
            click.echo(f"  Unique values: {col_stats['n_unique']}")
            click.echo(f"  Mode: {col_stats['mode']}")

    if sketch["correlations"]:
        click.echo("\n=== Strong Correlations ===")
        for pair, corr_data in list(sketch["correlations"].items())[:5]:
            col1, col2 = pair.split("|")
            click.echo(f"{col1} <-> {col2}: {corr_data['spearman']:.2f} ({corr_data['strength']})")


if __name__ == "__main__":
    cli()
