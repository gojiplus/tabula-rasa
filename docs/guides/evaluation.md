# Evaluation Guide

This guide covers how to evaluate table QA models using Tabula Rasa.

## Overview

Model evaluation helps you understand:

- How well your model performs on unseen data
- Which types of questions are challenging
- Where improvements are needed

## Basic Evaluation

```python
from tabula_rasa import TabulaRasa, TableQADataset
from tabula_rasa.evaluation import Evaluator

# Load model and test data
model = TabulaRasa.from_pretrained("./my_model")
test_dataset = TableQADataset.from_json("test.json")

# Evaluate
evaluator = Evaluator(model=model)
results = evaluator.evaluate(test_dataset)

print(f"Exact Match: {results['exact_match']:.2%}")
print(f"F1 Score: {results['f1']:.2%}")
```

## Metrics

### Exact Match (EM)

The percentage of predictions that match the ground truth exactly:

```python
# EM = (number of exact matches) / (total examples)
```

### F1 Score

Token-level F1 score between prediction and ground truth:

```python
# Precision = (true positives) / (true positives + false positives)
# Recall = (true positives) / (true positives + false negatives)
# F1 = 2 * (precision * recall) / (precision + recall)
```

### Custom Metrics

Define custom evaluation metrics:

```python
from tabula_rasa.evaluation import Metric

class TableAccuracyMetric(Metric):
    """Custom metric for table-specific accuracy."""

    def compute(self, predictions, references):
        correct = sum(
            pred.strip() == ref.strip()
            for pred, ref in zip(predictions, references)
        )
        return {"table_accuracy": correct / len(predictions)}

# Use in evaluation
evaluator = Evaluator(
    model=model,
    metrics=[TableAccuracyMetric()]
)
```

## Error Analysis

Identify where your model struggles:

```python
results = evaluator.evaluate(test_dataset, return_predictions=True)

# Analyze errors
errors = [
    (example, pred)
    for example, pred in zip(test_dataset, results['predictions'])
    if pred != example['answer']
]

# Group by error type
error_types = {
    'numerical': [],
    'textual': [],
    'aggregation': [],
}

for example, pred in errors:
    if example['question_type'] == 'numerical':
        error_types['numerical'].append((example, pred))
    # ... categorize other types
```

## Benchmarking

Compare your model against baselines:

```python
from tabula_rasa.evaluation import benchmark

models = {
    'baseline': TabulaRasa.from_pretrained('t5-small'),
    'fine-tuned': TabulaRasa.from_pretrained('./my_model'),
}

results = benchmark(
    models=models,
    dataset=test_dataset,
    metrics=['exact_match', 'f1'],
)

# Print comparison
for model_name, metrics in results.items():
    print(f"{model_name}:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.2%}")
```

## Cross-Validation

Perform k-fold cross-validation:

```python
from tabula_rasa.evaluation import cross_validate

results = cross_validate(
    model=model,
    dataset=full_dataset,
    n_splits=5,
    metrics=['exact_match', 'f1'],
)

print(f"Mean EM: {results['exact_match'].mean():.2%} ± {results['exact_match'].std():.2%}")
print(f"Mean F1: {results['f1'].mean():.2%} ± {results['f1'].std():.2%}")
```

## Performance Profiling

Measure inference speed:

```python
import time
from tabula_rasa.evaluation import profile

# Profile model
profile_results = profile(
    model=model,
    dataset=test_dataset,
    batch_sizes=[1, 4, 8, 16],
)

for batch_size, metrics in profile_results.items():
    print(f"Batch size {batch_size}:")
    print(f"  Throughput: {metrics['throughput']:.2f} examples/sec")
    print(f"  Latency: {metrics['latency']:.2f} ms/example")
```

## Evaluation on Specific Question Types

Evaluate performance on different question categories:

```python
# Group by question type
question_types = {}
for example in test_dataset:
    qtype = example.get('question_type', 'general')
    if qtype not in question_types:
        question_types[qtype] = []
    question_types[qtype].append(example)

# Evaluate each type
for qtype, examples in question_types.items():
    subset = TableQADataset(examples)
    results = evaluator.evaluate(subset)
    print(f"\n{qtype.upper()}:")
    print(f"  EM: {results['exact_match']:.2%}")
    print(f"  F1: {results['f1']:.2%}")
```

## CLI Evaluation

Use the command-line interface:

```bash
# Basic evaluation
tabula-rasa eval \
    --model ./my_model \
    --data test.json \
    --output results.json

# With specific metrics
tabula-rasa eval \
    --model ./my_model \
    --data test.json \
    --metrics exact_match f1 bleu \
    --output results.json

# Detailed error analysis
tabula-rasa eval \
    --model ./my_model \
    --data test.json \
    --error-analysis \
    --output results.json
```

## Best Practices

1. **Hold-Out Test Set**: Always evaluate on data the model hasn't seen during training
2. **Multiple Metrics**: Use multiple metrics to get a complete picture of performance
3. **Error Analysis**: Regularly analyze errors to identify improvement opportunities
4. **Stratified Sampling**: Ensure test set represents all question types
5. **Statistical Significance**: Use multiple runs and report confidence intervals
6. **Domain-Specific Evaluation**: Create custom metrics for your specific use case

## Example Evaluation Script

```python
#!/usr/bin/env python
"""Evaluation script for table QA model."""

import argparse
import json
from tabula_rasa import TabulaRasa, TableQADataset
from tabula_rasa.evaluation import Evaluator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="results.json")
    args = parser.parse_args()

    # Load model and data
    model = TabulaRasa.from_pretrained(args.model)
    dataset = TableQADataset.from_json(args.data)

    # Evaluate
    evaluator = Evaluator(model=model)
    results = evaluator.evaluate(
        dataset,
        return_predictions=True,
        return_confidence=True,
    )

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\nEvaluation Results:")
    print(f"  Exact Match: {results['exact_match']:.2%}")
    print(f"  F1 Score: {results['f1']:.2%}")
    print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()
```
