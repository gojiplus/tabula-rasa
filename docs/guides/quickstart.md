# Quick Start Guide

This guide will help you get started with Tabula Rasa in just a few minutes.

## Basic Example

Here's a simple example of using Tabula Rasa to answer questions about tabular data:

```python
from tabula_rasa import TabulaRasa

# Initialize the model
model = TabulaRasa()

# Define a table
table = {
    "columns": ["Product", "Price", "Stock"],
    "rows": [
        ["Laptop", 999.99, 15],
        ["Mouse", 29.99, 150],
        ["Keyboard", 79.99, 75],
    ]
}

# Ask questions
questions = [
    "What is the price of the laptop?",
    "How many keyboards are in stock?",
    "Which product is the most expensive?"
]

for question in questions:
    answer = model.answer(question, table)
    print(f"Q: {question}")
    print(f"A: {answer}\n")
```

## Working with DataFrames

Tabula Rasa integrates seamlessly with pandas DataFrames:

```python
import pandas as pd
from tabula_rasa import TabulaRasa

# Load data from CSV
df = pd.read_csv("sales_data.csv")

# Initialize model
model = TabulaRasa()

# Ask questions about your DataFrame
question = "What was the total sales in Q3?"
answer = model.answer_dataframe(question, df)
print(answer)
```

## Training a Custom Model

To train a model on your own data:

```python
from tabula_rasa import TabulaRasa, TableQADataset
from tabula_rasa.training import Trainer

# Load your training data
dataset = TableQADataset.from_json("training_data.json")

# Initialize model
model = TabulaRasa(
    model_name="t5-base",
    num_epochs=10,
    learning_rate=3e-4
)

# Create trainer
trainer = Trainer(
    model=model,
    train_dataset=dataset,
    eval_dataset=dataset,  # Use a separate eval set in practice
)

# Train
trainer.train()

# Save the model
model.save("./my_table_qa_model")
```

## Command Line Interface

Tabula Rasa includes a CLI for common tasks:

```bash
# Train a model
tabula-rasa train --data training_data.json --output ./model

# Evaluate a model
tabula-rasa eval --model ./model --data test_data.json

# Interactive mode
tabula-rasa interactive --model ./model
```

## Next Steps

- Learn about [training](training.md) models in detail
- Explore [evaluation](evaluation.md) metrics and techniques
- Check out the [API documentation](../api/modules.md)
- See more examples in the [GitHub repository](https://github.com/gojiplus/tabula-rasa/tree/main/examples)
