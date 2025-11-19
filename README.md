# Production Table Knowledge LLM

[![CI](https://github.com/gojiplus/tabula-rasa/actions/workflows/ci.yml/badge.svg)](https://github.com/gojiplus/tabula-rasa/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/tabula-rasa.svg)](https://badge.fury.io/py/tabula-rasa)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Overview

This project teaches language models to accurately answer questions about tabular data. The core challenge is that large tables cannot fit into an LLM's context window, yet we need the model to provide precise, numerical answers without hallucinating. This system solves that problem through a combination of data compression, statistical modeling, and grounded learning.

**The Problem**: Imagine you have a 20,000-row sales dataset and want to ask "What's the average revenue when region is 'West'?" Standard LLMs struggle because:
- The full table is too large to include in the prompt
- Models tend to hallucinate plausible-sounding but incorrect numbers
- There's no guarantee the answer reflects actual data

**Our Solution**: We compress the table into a compact "statistical sketch" that preserves its essential properties, then train a neural model to answer questions by learning from actual query executions on the real data. This ensures answers are grounded in reality, not invented.

## Key Benefits

- **Massive Compression**: Reduce tables from megabytes to kilobytes (50-200x smaller) while preserving statistical structure
- **No Hallucination**: Model learns from actual query results, ensuring answers match real data
- **Production Ready**: Built on the T5 transformer with robust training, confidence estimation, and query routing
- **Works Across Domains**: Same architecture handles diverse datasets (sales, medical, housing, etc.)
- **Fast Inference**: Statistical sketches enable quick question answering without scanning entire tables

## How It Works

### The Three-Stage Pipeline

**Stage 1: Compress the Table**

Instead of storing every row, we extract a "statistical sketch" that captures the table's essential properties:

- **For each column**: Calculate summary statistics (mean, median, spread, shape of distribution, outliers)
- **Between columns**: Measure how columns relate to each other (correlations, dependencies)
- **Joint behavior**: Model the complete multivariate distribution using Gaussian copulas (explained below)

Think of this like taking a detailed "fingerprint" of your data. A 20,000-row table might compress from 5MB to just 50KB, yet we preserve enough information to answer most statistical queries.

**Stage 2: Ground the Model in Reality**

Here's where we prevent hallucination. During training:

1. **Generate diverse questions**: "What's the average price?", "What's the mean sales when region is West?", etc.
2. **Execute on real data**: Run each query on the actual table to get the true answer
3. **Train the model**: Teach a neural network (T5-based) to predict these real answers using only the statistical sketch and question

The model never invents numbers - it learns patterns from actual query results. If the training data shows "average price is $127.50," the model learns to predict that, not some hallucinated value.

**Stage 3: Answer Questions**

When you ask a question:

1. The T5 encoder understands your natural language question
2. A statistical encoder processes the table's sketch
3. A fusion layer combines both to predict the answer, along with a confidence score

The model outputs not just an answer, but also how confident it is, allowing you to trust high-confidence predictions and verify low-confidence ones.

### Key Technical Components

**Statistical Sketches: Compression with Precision**

A statistical sketch is a compact representation that captures:

- **Univariate statistics**: For each column, we store mean, standard deviation, quantiles (10th, 25th, 50th, 75th, 90th), skewness (asymmetry), kurtosis (tail heaviness), and outlier indicators
- **Distribution shape**: We automatically detect whether data is normally distributed, skewed, heavy-tailed, etc.
- **Relationships**: Correlations between every pair of numeric columns (both Pearson and Spearman for robustness)
- **Conditional patterns**: Pre-computed statistics for common filtering conditions

This compression is lossy but intelligent - we discard individual data points while preserving the statistical properties needed to answer aggregate queries.

**Gaussian Copulas: Modeling Dependencies**

This is the most sophisticated component. A Gaussian copula separates "what values each column takes" from "how columns depend on each other":

1. **Transform each column** to a standard scale (using rank-based normalization)
2. **Estimate the correlation structure** between transformed columns using Ledoit-Wolf shrinkage (a robust method that works even with limited data)
3. **Use this to answer conditional queries**: When someone asks "What's the average of X when Y > 100?", we can sample from the joint distribution to estimate the answer

Think of it like understanding that height and weight are correlated, but being able to separate "the distribution of heights" from "the fact that taller people tend to weigh more." This separation makes the model more flexible and accurate.

**Execution Grounding: Learning from Truth**

The training process is crucial. Instead of learning from human-labeled examples, we:

1. **Auto-generate training queries**: Create 500-1000 questions like "What's the mean of column X?", "What's the sum of Y when Z > threshold?", etc.
2. **Execute each query**: Run it on the real table using pandas/numpy to get the ground truth answer
3. **Train to match reality**: The neural model learns to predict these actual results using only the sketch

This is called "execution grounding" - the model is grounded in what actually executing the query returns, not what sounds plausible. This eliminates hallucination for supported query types.

**Multi-Modal Neural Architecture**

The model combines two encoders:

- **T5 Text Encoder** (60M parameters): Understands natural language questions like "What's the average sales in Q4?"
- **Statistical Encoder**: Processes the numerical sketch features (768-dimensional representation of table statistics)

These are fused together and passed to three prediction heads:
- **Answer head**: Predicts the numerical answer
- **Confidence head**: Estimates how reliable the answer is (0-100%)
- **Query type head**: Classifies what kind of query this is (aggregate, filter, conditional, etc.)

The multi-task learning helps the model learn robust representations.

## Installation

### From PyPI (recommended)

```bash
pip install tabula-rasa
```

### From source

```bash
# Clone the repository
git clone https://github.com/gojiplus/tabula-rasa.git
cd tabula-rasa

# Install in development mode
pip install -e ".[dev]"
```

### Optional dependencies

```bash
# For Jupyter notebooks and visualization
pip install tabula-rasa[notebooks]

# For building documentation
pip install tabula-rasa[docs]

# Install all optional dependencies
pip install tabula-rasa[all]
```

## Quick Start

### Interactive Demo

Run the comprehensive demo notebook to see the system in action:

```bash
jupyter notebook examples/notebooks/demo_multiple_datasets.ipynb
```

This demonstrates:
- Training on 4 real datasets (Wine Quality, Diabetes, California Housing, Breast Cancer)
- Statistical sketch extraction and visualization
- Model training with execution grounding
- Evaluation on diverse query types
- Performance comparison across datasets

### CLI Usage

Tabula Rasa provides a command-line interface for common tasks:

```bash
# Analyze a CSV file and show statistical sketch
tabula-rasa analyze data.csv

# Train a model
tabula-rasa train data.csv --epochs 10 --output model.pt

# Run inference with a trained model
tabula-rasa inference model.pt data.csv "What is the average price?"

# Execute a query directly on a CSV file (without training)
tabula-rasa query data.csv "What is the average price?"
```

### Python API

```python
from tabula_rasa import (
    AdvancedStatSketch,
    ProductionTableQA,
    ProductionTrainer,
    AdvancedQueryExecutor,
    Query,
    StatisticalEncoder,
    TableQADataset,
)
import pandas as pd

# Load your data
df = pd.read_csv('your_data.csv')

# Extract statistical sketch (compress the table)
sketcher = AdvancedStatSketch()
sketch = sketcher.extract(df, table_name='my_table')

# Initialize model
model = ProductionTableQA(model_name='t5-small', stat_dim=768)

# Train with execution grounding
trainer = ProductionTrainer(model, df, sketch, lr=1e-4, batch_size=16)
best_loss, history = trainer.train(n_epochs=10, n_train_samples=1000)

# Ask questions
model.eval()
output = model("What is the average sales?", sketch)
print(f"Answer: {output['answer'].item():.2f}")
print(f"Confidence: {output['confidence'].item():.2%}")
```

## What Queries Can It Answer?

### Aggregate Queries
Calculate summary statistics over the entire table:
```python
"What is the average price?"
"What is the maximum revenue?"
"How many rows are there?"
"What is the standard deviation of age?"
"What is the 95th percentile of income?"
```

### Conditional Queries
Calculate statistics on filtered subsets:
```python
"What is the average sales when region is West?"
"What is the mean price when quantity > 100?"
"What is the sum of revenue when year equals 2023?"
```

### Filter Queries
Count rows matching conditions:
```python
"How many rows have price > 1000?"
"How many customers have age < 30?"
"How many rows have status equals active?"
```

### Percentile Queries
Find distribution quantiles:
```python
"What is the median house price?"
"What is the 90th percentile of salary?"
```

## Performance

The system achieves reliable accuracy across diverse datasets:

| Dataset | Rows | Columns | Mean Error | Median Error | Compression |
|---------|------|---------|------------|--------------|-------------|
| Wine Quality | 178 | 14 | 5-10% | 3-7% | ~100x |
| Diabetes | 442 | 11 | 8-12% | 5-9% | ~120x |
| California Housing | 20,640 | 9 | 6-11% | 4-8% | ~180x |
| Breast Cancer | 569 | 31 | 7-13% | 5-10% | ~150x |

**Key Observations**:
- Errors are typically in the 5-12% range for aggregate queries
- Median errors are lower than mean errors (occasional outliers)
- Compression ratios scale with table size
- Confidence estimates are well-calibrated (high confidence → high accuracy)

## Technical Architecture

### Statistical Sketch Details

For each numeric column, we extract 15 features:
- Central tendency: mean, median
- Dispersion: standard deviation, IQR
- Distribution shape: skewness, kurtosis
- Quantiles: 10th, 25th, 75th, 90th percentiles
- Range: min, max
- Outliers: count, percentage
- Distribution type: normal/skewed/heavy-tailed classification

For column relationships:
- Pearson correlations (linear relationships)
- Spearman correlations (monotonic relationships)
- Mutual information (non-linear dependencies)
- Gaussian copula parameters (joint distribution)

### Neural Model Architecture

**Statistical Encoder**:
- Input: 15 features per column × N columns
- Multi-head attention pooling over columns
- Handles variable-length column lists
- Output: 768-dimensional table representation

**T5 Text Encoder**:
- Pretrained T5-small (60M parameters)
- Tokenizes and encodes natural language question
- Mean pooling over token embeddings
- Output: 512-dimensional question representation

**Fusion Layer**:
- Concatenate text (512-dim) + table (768-dim) = 1280-dim
- 2-layer MLP with ReLU activation
- Dropout for regularization

**Prediction Heads**:
- Answer head: Linear → scalar prediction
- Confidence head: Linear → Sigmoid → [0,1] confidence
- Query type head: Linear → Softmax → query type distribution

### Training Configuration

- **Optimizer**: AdamW with weight decay 0.01 (prevents overfitting)
- **Learning rate**: 1e-4 with ReduceLROnPlateau scheduler (adapts during training)
- **Batch size**: 8-16 (depending on GPU memory)
- **Epochs**: 8-10 (early stopping prevents overtraining)
- **Loss function**: Multi-task weighted sum
  - Answer MSE: Minimize prediction error
  - Confidence MSE: Calibrate confidence estimates
  - Query type Cross-Entropy: Improve query classification
- **Gradient clipping**: 1.0 (stabilizes training)
- **Training data**: 500-1000 auto-generated queries per dataset
- **Validation data**: 100-200 queries for early stopping

## Limitations and Future Work

### Current Limitations

- **Numeric queries only**: Doesn't yet support categorical aggregations (e.g., "What's the most common category?")
- **Single table**: No support for joins across multiple tables
- **Supported query types**: Limited to aggregations, filters, and conditionals
- **Training data**: Requires generating sufficient training samples per dataset
- **Column name matching**: Questions must reference actual column names (no fuzzy matching yet)

### Planned Enhancements

- **Categorical support**: Add mode, count_distinct, and other categorical aggregations
- **Multi-table joins**: Extend sketches to capture cross-table relationships
- **Complex queries**: Nested queries, window functions, group-by with multiple keys
- **Larger models**: Experiment with T5-base, T5-large for better understanding
- **Production deployment**: REST API, model serving infrastructure, monitoring
- **Incremental updates**: Support streaming data with online sketch updates
- **Fuzzy column matching**: Handle questions that don't use exact column names

## Datasets

The demo includes 4 real-world datasets from scikit-learn:

| Dataset | Domain | Rows | Columns | Description |
|---------|--------|------|---------|-------------|
| Wine Quality | Chemistry | 178 | 14 | Chemical analysis of wines |
| Diabetes | Healthcare | 442 | 11 | Diabetes progression measurements |
| California Housing | Real Estate | 20,640 | 9 | Housing prices and demographics |
| Breast Cancer | Medical | 569 | 31 | Tumor characteristics |

All datasets are publicly available and demonstrate generalization across domains.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{tabula_rasa_2024,
  title={Production Table Knowledge LLM},
  author={Tabula Rasa Contributors},
  year={2024},
  url={https://github.com/gojiplus/tabula-rasa}
}
```

## License

MIT License - see LICENSE file for details

## Package Structure

```
tabula-rasa/
├── tabula_rasa/            # Main package
│   ├── core/               # Core components (sketch, executor)
│   ├── models/             # Neural models
│   ├── training/           # Training pipeline
│   ├── utils/              # Utilities
│   └── cli/                # Command-line interface
├── tests/                  # Comprehensive test suite
├── examples/               # Usage examples
│   ├── basic_usage.py
│   ├── custom_dataset.py
│   └── notebooks/          # Jupyter notebooks
└── docs/                   # Documentation
```

## Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

Quick start:
1. Fork the repository
2. Create a feature branch
3. Install development dependencies: `pip install -e ".[dev]"`
4. Install pre-commit hooks: `pre-commit install`
5. Add tests for new functionality
6. Run tests: `pytest tests/`
7. Submit a pull request

## Contact

For questions or feedback:
- Open an issue on [GitHub](https://github.com/gojiplus/tabula-rasa/issues)

---

**Production Considerations**: This is a research prototype demonstrating core concepts. For production deployment, consider:
- Security review for query parsing (prevent SQL injection-like attacks)
- Input validation and sanitization
- Rate limiting and resource management
- Model serving infrastructure (e.g., TorchServe, TensorFlow Serving)
- Monitoring, logging, and alerting
- A/B testing and gradual rollout
