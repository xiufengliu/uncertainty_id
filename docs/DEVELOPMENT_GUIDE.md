# Development Guide

This guide helps developers understand the codebase structure, contribute to the project, and extend the functionality.

## 📁 Project Structure

```
uncertainty_ids/
├── uncertainty_ids/           # Main package
│   ├── models/               # Model implementations
│   │   ├── transformer.py    # Bayesian ensemble transformer
│   │   ├── uncertainty.py    # Uncertainty quantification
│   │   └── baselines.py      # Baseline models
│   ├── training/             # Training components
│   │   ├── trainer.py        # Main trainer class
│   │   ├── losses.py         # Loss functions
│   │   └── optimizers.py     # Custom optimizers
│   ├── evaluation/           # Evaluation metrics and tools
│   │   ├── evaluator.py      # Model evaluator
│   │   ├── metrics.py        # Evaluation metrics
│   │   └── adversarial.py    # Adversarial evaluation
│   ├── data/                 # Data handling
│   │   ├── datasets.py       # Dataset classes
│   │   ├── preprocessing.py  # Data preprocessing
│   │   └── loaders.py        # Data loaders
│   └── utils/                # Utilities
│       ├── config.py         # Configuration handling
│       ├── visualization.py  # Plotting functions
│       ├── reproducibility.py # Reproducibility utilities
│       └── plotting_config.py # Plot configuration
├── configs/                  # Configuration files
├── scripts/                  # Experiment scripts
├── data/                     # Data directory
├── results/                  # Results and outputs
├── tests/                    # Unit tests
└── docs/                     # Documentation
```

## 🔧 Development Setup

### 1. Environment Setup

```bash
# Clone repository
git clone <repository-url>
cd uncertainty_ids

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install in development mode
pip install -e .

# Install development dependencies
pip install pytest black flake8 mypy pre-commit
```

### 2. Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

### 3. Code Style

We use Black for code formatting and flake8 for linting:

```bash
# Format code
black uncertainty_ids/ tests/

# Check linting
flake8 uncertainty_ids/ tests/

# Type checking
mypy uncertainty_ids/
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=uncertainty_ids

# Run specific test file
pytest tests/test_models.py

# Run specific test
pytest tests/test_models.py::test_transformer_forward
```

### Writing Tests

Create test files in the `tests/` directory:

```python
# tests/test_new_feature.py
import pytest
import torch
from uncertainty_ids.models.transformer import BayesianEnsembleTransformer

def test_model_initialization():
    """Test model can be initialized correctly."""
    model = BayesianEnsembleTransformer(
        input_dim=10,
        d_model=64,
        n_heads=4,
        n_layers=2,
        n_classes=2,
        ensemble_size=3
    )
    assert model.ensemble_size == 3
    assert model.d_model == 64

def test_model_forward():
    """Test model forward pass."""
    model = BayesianEnsembleTransformer(
        input_dim=10,
        d_model=64,
        n_heads=4,
        n_layers=2,
        n_classes=2,
        ensemble_size=3
    )
    
    x_cont = torch.randn(32, 10)
    x_cat = torch.zeros(32, 0)  # No categorical features
    
    logits, attention, individual = model(x_cont, x_cat, return_individual=True)
    
    assert logits.shape == (32, 2)
    assert individual.shape == (32, 3, 2)
```

## 🏗️ Architecture Overview

### Model Architecture

```python
# Core transformer with Bayesian ensemble
class BayesianEnsembleTransformer(nn.Module):
    def __init__(self, input_dim, d_model, n_heads, n_layers, 
                 n_classes, ensemble_size, dropout=0.1):
        # Multiple transformer models in ensemble
        self.ensemble = nn.ModuleList([
            TransformerModel(...) for _ in range(ensemble_size)
        ])
    
    def forward(self, x_cont, x_cat, return_individual=False):
        # Process through each ensemble member
        individual_outputs = []
        for model in self.ensemble:
            output = model(x_cont, x_cat)
            individual_outputs.append(output)
        
        # Aggregate ensemble predictions
        ensemble_logits = torch.stack(individual_outputs).mean(dim=0)
        
        return ensemble_logits, attention_weights, individual_outputs
```

### Uncertainty Quantification

```python
class UncertaintyQuantifier:
    def calculate_uncertainty(self, individual_preds, ensemble_logits):
        # Epistemic uncertainty (model uncertainty)
        epistemic = self._calculate_epistemic(individual_preds)
        
        # Aleatoric uncertainty (data uncertainty)
        aleatoric = self._calculate_aleatoric(ensemble_logits)
        
        # Total uncertainty
        total = epistemic + aleatoric
        
        return {
            'epistemic_uncertainty': epistemic,
            'aleatoric_uncertainty': aleatoric,
            'total_uncertainty': total
        }
```

## 🔄 Adding New Features

### 1. Adding a New Model

```python
# uncertainty_ids/models/new_model.py
import torch
import torch.nn as nn

class NewModel(nn.Module):
    """New model implementation."""
    
    def __init__(self, input_dim, hidden_dim, n_classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes)
        )
    
    def forward(self, x):
        return self.layers(x)

# Register in baselines.py
def create_new_model(config):
    return NewModel(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        n_classes=config['n_classes']
    )
```

### 2. Adding a New Dataset

```python
# uncertainty_ids/data/new_dataset.py
import torch
from torch.utils.data import Dataset
import pandas as pd

class NewDataset(Dataset):
    """New dataset implementation."""
    
    def __init__(self, data_dir, split='train', download=False):
        self.data_dir = data_dir
        self.split = split
        
        if download:
            self._download()
        
        self.data = self._load_data()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        features = torch.tensor(sample[:-1].values, dtype=torch.float32)
        label = torch.tensor(sample[-1], dtype=torch.long)
        return features, label
    
    def _download(self):
        # Implement download logic
        pass
    
    def _load_data(self):
        # Implement data loading logic
        return pd.read_csv(f"{self.data_dir}/{self.split}.csv")
```

### 3. Adding a New Metric

```python
# uncertainty_ids/evaluation/metrics.py
class NewMetrics:
    """New evaluation metrics."""
    
    @staticmethod
    def new_metric(predictions, targets):
        """Calculate new metric."""
        # Implement metric calculation
        return metric_value
```

### 4. Adding a New Loss Function

```python
# uncertainty_ids/training/losses.py
class NewLoss(nn.Module):
    """New loss function."""
    
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
    
    def forward(self, predictions, targets, uncertainties=None):
        # Implement loss calculation
        loss = self._calculate_loss(predictions, targets)
        
        if uncertainties is not None:
            # Add uncertainty regularization
            uncertainty_penalty = self._uncertainty_penalty(uncertainties)
            loss += self.weight * uncertainty_penalty
        
        return loss
```

## 📊 Experiment Management

### Configuration Management

```python
# configs/new_experiment.json
{
    "experiment_name": "new_experiment",
    "model": {
        "type": "BayesianEnsembleTransformer",
        "d_model": 256,
        "n_heads": 8,
        "n_layers": 6,
        "ensemble_size": 5
    },
    "training": {
        "batch_size": 256,
        "learning_rate": 0.001,
        "epochs": 100
    },
    "data": {
        "dataset": "nsl_kdd",
        "normalize": true,
        "validation_split": 0.2
    }
}
```

### Running Experiments

```python
# scripts/new_experiment.py
import argparse
from uncertainty_ids.utils.config import load_config
from uncertainty_ids.training.trainer import UncertaintyAwareTrainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--dataset', required=True)
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Run experiment
    trainer = UncertaintyAwareTrainer(config=config)
    results = trainer.run_experiment(args.dataset)
    
    # Save results
    trainer.save_results(results, f"results/{args.dataset}_results.json")

if __name__ == "__main__":
    main()
```

## 🐛 Debugging

### Common Issues

1. **CUDA Memory Issues**
   ```python
   # Add memory debugging
   torch.cuda.empty_cache()
   print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
   ```

2. **NaN Values in Loss**
   ```python
   # Add gradient clipping
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   
   # Check for NaN
   if torch.isnan(loss):
       print("NaN detected in loss!")
       breakpoint()
   ```

3. **Slow Training**
   ```python
   # Profile code
   import torch.profiler
   
   with torch.profiler.profile() as prof:
       # Training code here
       pass
   
   print(prof.key_averages().table(sort_by="cuda_time_total"))
   ```

### Logging and Monitoring

```python
import logging
from uncertainty_ids.utils.logging import setup_logger

# Setup detailed logging
logger = setup_logger('debug', level=logging.DEBUG)

# Log model statistics
logger.debug(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
logger.debug(f"Batch size: {batch_size}, Learning rate: {lr}")

# Monitor training
for epoch in range(epochs):
    logger.info(f"Epoch {epoch}: Loss = {loss:.4f}")
```

## 📈 Performance Optimization

### Model Optimization

```python
# Use mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    with autocast():
        outputs = model(batch)
        loss = criterion(outputs, targets)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### Data Loading Optimization

```python
# Optimize data loading
dataloader = DataLoader(
    dataset,
    batch_size=256,
    num_workers=8,        # Use multiple workers
    pin_memory=True,      # Pin memory for GPU transfer
    prefetch_factor=2     # Prefetch batches
)
```

## 🤝 Contributing

### Pull Request Process

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/new-feature`
3. **Make changes and add tests**
4. **Run tests**: `pytest`
5. **Format code**: `black .`
6. **Commit changes**: `git commit -m "Add new feature"`
7. **Push to branch**: `git push origin feature/new-feature`
8. **Create Pull Request**

### Code Review Checklist

- [ ] Code follows style guidelines
- [ ] Tests are included and passing
- [ ] Documentation is updated
- [ ] No breaking changes (or properly documented)
- [ ] Performance impact is considered
- [ ] Security implications are reviewed

### Issue Reporting

When reporting issues, include:

1. **Environment details** (Python version, PyTorch version, GPU)
2. **Minimal reproducible example**
3. **Expected vs actual behavior**
4. **Error messages and stack traces**
5. **Configuration files used**

## 📚 Resources

### Key Papers
- Original transformer paper: "Attention Is All You Need"
- Uncertainty quantification: "What Uncertainties Do We Need in Bayesian Deep Learning?"
- Ensemble methods: "Simple and Scalable Predictive Uncertainty Estimation"

### Useful Tools
- **Weights & Biases**: Experiment tracking
- **TensorBoard**: Training visualization
- **PyTorch Profiler**: Performance profiling
- **Memory Profiler**: Memory usage analysis

### Documentation
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Scikit-learn Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/index.html)
