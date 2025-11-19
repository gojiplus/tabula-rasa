# Training Guide

This guide covers how to train custom table QA models using Tabula Rasa.

## Overview

Training a table QA model involves:

1. Preparing your training data
2. Configuring the model architecture
3. Setting up training parameters
4. Running the training loop
5. Evaluating and saving the model

## Data Format

Tabula Rasa expects training data in a specific JSON format:

```json
{
  "examples": [
    {
      "table": {
        "columns": ["Name", "Age", "City"],
        "rows": [
          ["Alice", 30, "New York"],
          ["Bob", 25, "San Francisco"]
        ]
      },
      "question": "How old is Alice?",
      "answer": "30"
    }
  ]
}
```

## Basic Training

```python
from tabula_rasa import TabulaRasa, TableQADataset
from tabula_rasa.training import Trainer, TrainingArguments

# Load data
train_dataset = TableQADataset.from_json("train.json")
eval_dataset = TableQADataset.from_json("eval.json")

# Configure training
args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=10,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=3e-4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    eval_steps=500,
    save_steps=1000,
    evaluation_strategy="steps",
)

# Initialize model
model = TabulaRasa(model_name="t5-base")

# Create trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Train
trainer.train()
```

## Advanced Configuration

### Custom Model Architecture

```python
from tabula_rasa import TabulaRasa

model = TabulaRasa(
    model_name="t5-large",  # Use a larger base model
    max_source_length=512,
    max_target_length=128,
    num_beams=4,  # Beam search for generation
)
```

### Data Augmentation

```python
from tabula_rasa.data import TableAugmenter

augmenter = TableAugmenter(
    shuffle_rows=True,
    shuffle_columns=True,
    drop_rows_prob=0.1,
    num_augmentations=3,
)

augmented_dataset = augmenter.augment(train_dataset)
```

### Learning Rate Scheduling

```python
from transformers import get_linear_schedule_with_warmup

args = TrainingArguments(
    # ... other args
    lr_scheduler_type="linear",
    warmup_ratio=0.1,
)
```

## Distributed Training

For multi-GPU training:

```bash
# Using torchrun
torchrun --nproc_per_node=4 train.py --config config.json

# Using DeepSpeed
deepspeed --num_gpus=4 train.py --config config.json --deepspeed ds_config.json
```

## Monitoring Training

### TensorBoard

Monitor training progress with TensorBoard:

```bash
tensorboard --logdir ./logs
```

### Weights & Biases

```python
args = TrainingArguments(
    # ... other args
    report_to="wandb",
    run_name="table-qa-experiment",
)
```

## Best Practices

1. **Start Small**: Begin with a smaller model (t5-small) to iterate quickly
2. **Validate Early**: Use a validation set to catch overfitting
3. **Tune Hyperparameters**: Experiment with learning rate, batch size, and warmup steps
4. **Use Mixed Precision**: Enable fp16 for faster training on modern GPUs
5. **Save Checkpoints**: Regularly save model checkpoints to resume training if interrupted

## Example Training Script

```python
#!/usr/bin/env python
"""Training script for table QA model."""

import argparse
from tabula_rasa import TabulaRasa, TableQADataset
from tabula_rasa.training import Trainer, TrainingArguments

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--eval-data", required=True)
    parser.add_argument("--output-dir", default="./output")
    parser.add_argument("--model-name", default="t5-base")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    args = parser.parse_args()

    # Load datasets
    train_dataset = TableQADataset.from_json(args.train_data)
    eval_dataset = TableQADataset.from_json(args.eval_data)

    # Configure training
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        logging_steps=100,
        eval_steps=500,
        save_steps=1000,
        fp16=True,
    )

    # Initialize and train
    model = TabulaRasa(model_name=args.model_name)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()
    model.save(args.output_dir)

if __name__ == "__main__":
    main()
```
