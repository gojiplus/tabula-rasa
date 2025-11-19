# Installation

## Requirements

Tabula Rasa requires Python 3.11 or later. We recommend using a virtual environment to manage dependencies.

## Standard Installation

Install Tabula Rasa using pip:

```bash
pip install tabula-rasa
```

## Development Installation

For development, clone the repository and install in editable mode with development dependencies:

```bash
git clone https://github.com/gojiplus/tabula-rasa.git
cd tabula-rasa
pip install -e ".[dev]"
```

## Optional Dependencies

Tabula Rasa provides several optional dependency groups:

### Notebooks

For Jupyter notebook support:

```bash
pip install tabula-rasa[notebooks]
```

### Documentation

For building documentation:

```bash
pip install tabula-rasa[docs]
```

### All Dependencies

To install all optional dependencies:

```bash
pip install tabula-rasa[all]
```

## Verifying Installation

After installation, verify that everything is working:

```python
import tabula_rasa
print(tabula_rasa.__version__)
```

Or use the CLI:

```bash
tabula-rasa --version
```

## GPU Support

Tabula Rasa uses PyTorch for deep learning. To enable GPU acceleration, ensure you have CUDA installed and install the appropriate PyTorch version:

```bash
# For CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Refer to the [PyTorch installation guide](https://pytorch.org/get-started/locally/) for more details.

## Troubleshooting

### Import Errors

If you encounter import errors, ensure all dependencies are installed:

```bash
pip install -e ".[all]"
```

### Version Conflicts

If you have dependency conflicts, try creating a fresh virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install tabula-rasa
```
