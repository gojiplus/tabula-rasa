# Usage Guide

This guide provides step-by-step instructions for using the Production Table Knowledge LLM system.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/gojiplus/tabula-rasa.git
cd tabula-rasa
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- PyTorch (deep learning framework)
- Transformers (T5 model)
- scikit-learn (datasets and utilities)
- pandas, numpy, scipy (data processing)
- matplotlib, seaborn (visualization)
- jupyter (notebooks)

**Note**: First run will download T5 model weights (~250MB). This is a one-time download.

## Quick Start

### Option 1: Run the Demo Notebook (Recommended)

The notebook provides a comprehensive demonstration across 4 real datasets:

```bash
jupyter notebook demo_multiple_datasets.ipynb
```

This will:
1. Load 4 real datasets (Wine, Diabetes, Housing, Cancer)
2. Extract statistical sketches for each
3. Train models with execution grounding
4. Evaluate on diverse query types
5. Visualize results and compare performance

**Expected runtime**: 15-30 minutes (depending on hardware)

### Option 2: Run Basic Test

To verify installation:

```bash
python test_basic.py
```

This runs a quick test on the Wine dataset with minimal training.

**Expected runtime**: 3-5 minutes

### Option 3: Use Programmatically

Create a new Python script:

```python
from production_table_llm import (
    AdvancedStatSketch,
    ProductionTableQA,
    ProductionTrainer,
    AdvancedQueryExecutor,
    Query
)
import pandas as pd

# 1. Load your data
df = pd.read_csv('your_data.csv')

# 2. Extract statistical sketch
sketcher = AdvancedStatSketch()
sketch = sketcher.extract(df, table_name='my_table')

print(f"Sketch extracted: {len(sketch['columns'])} columns")
print(f"Compression ratio: ~{df.memory_usage().sum() / len(str(sketch)):.1f}x")

# 3. Initialize model
model = ProductionTableQA(model_name='t5-small', stat_dim=768)

# 4. Train
device = 'cuda' if torch.cuda.is_available() else 'cpu'
trainer = ProductionTrainer(
    model=model,
    df=df,
    sketch=sketch,
    lr=1e-4,
    batch_size=16,
    device=device
)

best_loss, history = trainer.train(
    n_epochs=10,
    n_train_samples=1000,
    n_val_samples=200
)

# 5. Save model
torch.save(model.state_dict(), 'model.pt')

# 6. Ask questions
model.eval()
output = model("What is the average sales?", sketch)
print(f"Answer: {output['answer'].item():.2f}")
print(f"Confidence: {output['confidence'].item():.2%}")
```

## Common Use Cases

### 1. Training on Your Own Dataset

```python
import pandas as pd
from production_table_llm import AdvancedStatSketch, ProductionTableQA, ProductionTrainer

# Load data
df = pd.read_csv('my_data.csv')

# Extract sketch
sketcher = AdvancedStatSketch()
sketch = sketcher.extract(df, table_name='my_data')

# Train model
model = ProductionTableQA()
trainer = ProductionTrainer(model, df, sketch)
trainer.train(n_epochs=10, n_train_samples=1000)

# Ask questions
model.eval()
answer = model("What is the maximum price?", sketch)
```

### 2. Query Execution (Ground Truth)

```python
from production_table_llm import AdvancedQueryExecutor, Query

executor = AdvancedQueryExecutor(df)

# Aggregate query
query = Query('aggregate', target_column='sales', aggregation='mean')
result = executor.execute(query)
print(f"Average sales: {result}")

# Conditional query
query = Query('conditional',
              target_column='revenue',
              aggregation='mean',
              condition='sales > 1000')
result = executor.execute(query)
print(f"Average revenue when sales > 1000: {result}")

# Filter query
query = Query('filter', condition='price < 100')
result = executor.execute(query)
print(f"Number of rows with price < 100: {result}")
```

### 3. Analyzing Statistical Properties

```python
from production_table_llm import AdvancedStatSketch
import json

sketcher = AdvancedStatSketch()
sketch = sketcher.extract(df, table_name='my_data')

# View column statistics
for col_name, stats in sketch['columns'].items():
    if stats['type'] == 'numeric':
        print(f"\n{col_name}:")
        print(f"  Mean: {stats['mean']:.2f}")
        print(f"  Std: {stats['std']:.2f}")
        print(f"  Distribution: {stats['distribution_hint']}")
        print(f"  Outliers: {stats['outlier_rate']:.1%}")

# View correlations
print("\nStrong correlations:")
for pair, corr in sketch['correlations'].items():
    if corr['strength'] == 'strong':
        print(f"  {pair}: {corr['spearman']:.2f}")

# Save sketch
with open('sketch.json', 'w') as f:
    json.dump(sketch, f, indent=2)
```

### 4. Multi-Dataset Training

```python
# Train separate models for different datasets
from sklearn.datasets import load_wine, load_diabetes

datasets = {
    'wine': load_wine(),
    'diabetes': load_diabetes()
}

models = {}
sketches = {}

for name, dataset in datasets.items():
    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)

    # Extract sketch
    sketcher = AdvancedStatSketch()
    sketch = sketcher.extract(df, table_name=name)
    sketches[name] = sketch

    # Train model
    model = ProductionTableQA()
    trainer = ProductionTrainer(model, df, sketch)
    trainer.train(n_epochs=10)

    models[name] = model
    print(f"Trained model for {name}")

# Use models
for name, model in models.items():
    model.eval()
    answer = model("How many rows?", sketches[name])
    print(f"{name}: {answer['answer'].item():.0f} rows")
```

## Configuration Options

### Model Configuration

```python
# Small model (fast, less accurate)
model = ProductionTableQA(model_name='t5-small', stat_dim=512)

# Larger model (slower, more accurate)
model = ProductionTableQA(model_name='t5-base', stat_dim=768)
```

### Training Configuration

```python
trainer = ProductionTrainer(
    model=model,
    df=df,
    sketch=sketch,
    lr=1e-4,              # Learning rate
    batch_size=16,        # Batch size
    device='cpu'          # or 'cuda'
)

trainer.train(
    n_epochs=10,          # Number of epochs
    n_train_samples=1000, # Training queries
    n_val_samples=200     # Validation queries
)
```

### Sketch Configuration

```python
sketcher = AdvancedStatSketch(
    max_categories=50,      # Max categories to store
    confidence_level=0.95   # Confidence level for intervals
)
```

## Performance Tips

### 1. Use GPU if Available

```python
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
trainer = ProductionTrainer(model, df, sketch, device=device)
```

### 2. Adjust Batch Size

- Larger batch size = faster training, more memory
- Smaller batch size = slower training, less memory

```python
# For GPU with 8GB+ memory
trainer = ProductionTrainer(model, df, sketch, batch_size=32)

# For CPU or limited memory
trainer = ProductionTrainer(model, df, sketch, batch_size=4)
```

### 3. Early Stopping

Monitor validation loss and stop if not improving:

```python
best_loss, history = trainer.train(n_epochs=20)

# Plot validation loss
import matplotlib.pyplot as plt
plt.plot(history['val_loss'])
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.show()
```

## Troubleshooting

### Issue: Out of Memory

**Solution**: Reduce batch size or use smaller model

```python
trainer = ProductionTrainer(model, df, sketch, batch_size=4)
# or
model = ProductionTableQA(model_name='t5-small', stat_dim=256)
```

### Issue: Poor Performance

**Solutions**:
1. Increase training samples
2. Train for more epochs
3. Use larger model
4. Check data quality

```python
trainer.train(
    n_epochs=20,
    n_train_samples=2000,
    n_val_samples=400
)
```

### Issue: Slow Training

**Solutions**:
1. Use GPU
2. Increase batch size
3. Use smaller model
4. Reduce training samples

### Issue: Import Errors

**Solution**: Reinstall dependencies

```bash
pip install --upgrade -r requirements.txt
```

## FAQ

**Q: How much data do I need?**

A: Minimum ~100 rows, but 1000+ rows recommended for good performance.

**Q: What types of queries are supported?**

A: Currently supports:
- Aggregations: mean, sum, count, std, min, max, percentiles
- Conditional: mean/count/std with conditions
- Filter: count rows matching condition

**Q: Can it handle categorical data?**

A: Statistical sketch captures categorical data, but queries currently focus on numeric columns.

**Q: How long does training take?**

A:
- Small dataset (100-1000 rows): 2-5 minutes
- Medium dataset (1000-10000 rows): 5-15 minutes
- Large dataset (10000+ rows): 15-30 minutes

(with T5-small on GPU)

**Q: Can I use my own CSV files?**

A: Yes! Just load with pandas:

```python
df = pd.read_csv('your_file.csv')
sketcher = AdvancedStatSketch()
sketch = sketcher.extract(df)
```

**Q: How do I save/load models?**

A: Use PyTorch save/load:

```python
# Save
torch.save(model.state_dict(), 'model.pt')
torch.save(sketch, 'sketch.pkl')

# Load
model = ProductionTableQA()
model.load_state_dict(torch.load('model.pt'))
sketch = torch.load('sketch.pkl')
```

## Next Steps

- Experiment with different datasets
- Try different query types
- Tune hyperparameters for your use case
- Integrate into your application
- Contribute improvements to the repository

For more information, see README.md or open an issue on GitHub.
