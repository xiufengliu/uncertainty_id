# Installation Guide

This guide provides detailed installation instructions for the Uncertainty-Aware Intrusion Detection System.

## Quick Installation

### From PyPI (Recommended)

```bash
pip install uncertainty-ids
```

### From Source

```bash
git clone https://github.com/research-team/uncertainty-ids.git
cd uncertainty-ids
pip install -e .
```

## Detailed Installation

### Prerequisites

- **Python**: 3.8 or higher
- **Operating System**: Linux, macOS, or Windows
- **Memory**: At least 4GB RAM (8GB+ recommended for training)
- **Storage**: At least 2GB free space

### System Dependencies

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install python3-dev python3-pip git
```

#### CentOS/RHEL/Fedora
```bash
sudo yum install python3-devel python3-pip git
# or for newer versions:
sudo dnf install python3-devel python3-pip git
```

#### macOS
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python and Git
brew install python git
```

#### Windows
1. Install Python from [python.org](https://www.python.org/downloads/)
2. Install Git from [git-scm.com](https://git-scm.com/download/win)
3. Open Command Prompt or PowerShell as Administrator

### Virtual Environment Setup

We strongly recommend using a virtual environment:

#### Using venv (Python built-in)
```bash
python -m venv uncertainty-ids-env
source uncertainty-ids-env/bin/activate  # Linux/macOS
# or
uncertainty-ids-env\Scripts\activate     # Windows
```

#### Using conda
```bash
conda create -n uncertainty-ids python=3.10
conda activate uncertainty-ids
```

### Installation Options

#### 1. Basic Installation
For basic usage and inference:
```bash
pip install uncertainty-ids
```

#### 2. Development Installation
For development, testing, and contributing:
```bash
git clone https://github.com/research-team/uncertainty-ids.git
cd uncertainty-ids
pip install -e ".[dev]"
```

#### 3. Full Installation
For all features including API, notebooks, and documentation:
```bash
pip install "uncertainty-ids[all]"
```

#### 4. Specific Feature Sets
```bash
# API server only
pip install "uncertainty-ids[api]"

# Jupyter notebooks
pip install "uncertainty-ids[notebooks]"

# Documentation building
pip install "uncertainty-ids[docs]"

# Testing tools
pip install "uncertainty-ids[test]"
```

### GPU Support (Optional)

For GPU acceleration with CUDA:

#### CUDA 11.8 (Recommended)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install uncertainty-ids
```

#### CUDA 12.1
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install uncertainty-ids
```

#### ROCm (AMD GPUs)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.4.2
pip install uncertainty-ids
```

#### Apple Silicon (M1/M2 Macs)
```bash
pip install torch torchvision
pip install uncertainty-ids
```

### Verification

Verify your installation:

```bash
python -c "import uncertainty_ids; print(f'Uncertainty-IDS version: {uncertainty_ids.__version__}')"
```

Check GPU availability (if installed):
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

Run a quick test:
```bash
python -c "
from uncertainty_ids import BayesianEnsembleIDS
model = BayesianEnsembleIDS(n_ensemble=3, d_model=32, max_seq_len=10, n_classes=2)
print('✅ Installation successful!')
"
```

## Post-Installation Setup

### 1. Download Sample Data
```bash
# Create data directory
mkdir -p data

# The package includes utilities to create synthetic data
python -c "
from uncertainty_ids.data import SyntheticIDSDataset
dataset = SyntheticIDSDataset.create_synthetic(n_samples=1000)
dataset.save('data/sample_data.csv')
print('Sample data created: data/sample_data.csv')
"
```

### 2. Configuration
Create a configuration directory:
```bash
mkdir -p ~/.uncertainty-ids
cp configs/default_config.yaml ~/.uncertainty-ids/config.yaml
```

### 3. Set Environment Variables (Optional)
```bash
export UNCERTAINTY_IDS_CONFIG_PATH=~/.uncertainty-ids/config.yaml
export UNCERTAINTY_IDS_DATA_PATH=./data
export UNCERTAINTY_IDS_MODEL_PATH=./models
```

## Quick Start

### 1. Basic Usage
```python
from uncertainty_ids import BayesianEnsembleIDS
from uncertainty_ids.data import SyntheticIDSDataset

# Create synthetic data
dataset = SyntheticIDSDataset.create_synthetic(n_samples=1000)

# Initialize model
model = BayesianEnsembleIDS(n_ensemble=5, d_model=64, max_seq_len=20, n_classes=2)

# Make predictions
sequences, queries, labels = dataset[0:10]
results = model.predict_with_uncertainty(sequences, queries)

print(f"Predictions: {results['predictions']}")
print(f"Uncertainties: {results['total_uncertainty']}")
```

### 2. Command Line Interface
```bash
# Train a model
uncertainty-ids-train --data-path data/sample_data.csv --epochs 50

# Start API server
uncertainty-ids-serve --model-path models/trained_model.pth

# Evaluate a model
uncertainty-ids-evaluate --model-path models/trained_model.pth --data-path data/test_data.csv
```

### 3. Run Examples
```bash
# Quick start example
python examples/quick_start.py

# NSL-KDD training example
python examples/train_nsl_kdd.py --epochs 20
```

## Troubleshooting

### Common Issues

#### 1. Import Error: No module named 'uncertainty_ids'
**Solution**: Make sure you've activated your virtual environment and installed the package:
```bash
source uncertainty-ids-env/bin/activate  # or your env name
pip install uncertainty-ids
```

#### 2. CUDA Out of Memory
**Solution**: Reduce batch size or model size:
```python
# Reduce batch size
training_config.batch_size = 16

# Reduce model dimension
model_config['d_model'] = 64
model_config['n_ensemble'] = 5
```

#### 3. Slow Training
**Solutions**:
- Enable GPU if available
- Reduce sequence length
- Use mixed precision training
- Reduce ensemble size for faster training

#### 4. Permission Denied (Linux/macOS)
**Solution**: Use user installation:
```bash
pip install --user uncertainty-ids
```

#### 5. SSL Certificate Error
**Solution**: Upgrade pip and certificates:
```bash
pip install --upgrade pip
pip install --upgrade certifi
```

### Getting Help

1. **Documentation**: https://uncertainty-ids.readthedocs.io/
2. **GitHub Issues**: https://github.com/research-team/uncertainty-ids/issues
3. **Discussions**: https://github.com/research-team/uncertainty-ids/discussions

### System Requirements Check

Run this script to check your system:

```python
import sys
import torch
import numpy as np
import pandas as pd
import sklearn

print("System Requirements Check")
print("=" * 30)
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"Scikit-learn version: {sklearn.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")

# Memory check
import psutil
memory = psutil.virtual_memory()
print(f"Total RAM: {memory.total / (1024**3):.1f} GB")
print(f"Available RAM: {memory.available / (1024**3):.1f} GB")

print("\n✅ System check complete!")
```

## Development Installation

For contributors and developers:

### 1. Clone and Setup
```bash
git clone https://github.com/research-team/uncertainty-ids.git
cd uncertainty-ids

# Create development environment
python -m venv venv
source venv/bin/activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### 2. Run Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=uncertainty_ids

# Run specific test categories
pytest -m "not slow"  # Skip slow tests
pytest -m integration  # Integration tests only
```

### 3. Code Quality
```bash
# Format code
black uncertainty_ids/ tests/
isort uncertainty_ids/ tests/

# Lint code
flake8 uncertainty_ids/ tests/
mypy uncertainty_ids/
```

### 4. Build Documentation
```bash
cd docs/
make html
```

## Next Steps

After installation:

1. **Try the Quick Start**: Run `python examples/quick_start.py`
2. **Read the Documentation**: Visit the online docs or build locally
3. **Explore Examples**: Check out the `examples/` directory
4. **Join the Community**: Participate in GitHub Discussions
5. **Contribute**: See `CONTRIBUTING.md` for guidelines

## Support

If you encounter any issues during installation:

1. Check this guide for common solutions
2. Search existing GitHub issues
3. Create a new issue with:
   - Your operating system and version
   - Python version
   - Complete error message
   - Steps to reproduce

We're here to help! 🚀
