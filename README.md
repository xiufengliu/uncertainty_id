# Uncertainty-Aware Intrusion Detection System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

A research implementation of an uncertainty-aware intrusion detection framework featuring Bayesian ensemble transformers with rigorous theoretical foundations and principled uncertainty quantification.

## Key Features

- **Theoretical Foundation**: Novel convergence guarantees and uncertainty bounds for transformer-based intrusion detection
- **Uncertainty Quantification**: Separates epistemic and aleatoric uncertainty for better decision-making
- **Bayesian Ensemble**: Multiple transformer models provide robust predictions with confidence estimates
- **Comprehensive Baselines**: Includes traditional ML, deep learning, and uncertainty-aware methods
- **Evidential Neural Networks**: State-of-the-art uncertainty quantification through evidential learning
- **Modular Design**: Clean, extensible architecture for research and development

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
- [Research & Development](#research--development)
- [License](#license)
- [Contributing](#contributing)
- [Support](#support)

## Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch 2.0 or higher
- CUDA (optional, for GPU acceleration)

### Install from Source

```bash
git clone https://github.com/xiufengliu/uncertainty_id.git
cd uncertainty_id
pip install -r requirements.txt
pip install -e .
```

## Project Structure

```
uncertainty-ids/
├── uncertainty_ids/           # Main package
│   ├── models/               # Bayesian ensemble transformer models
│   ├── data/                 # Data loading and preprocessing
│   ├── training/             # Training utilities and loops
│   ├── evaluation/           # Evaluation metrics and calibration
│   └── utils/                # Utility functions
├── data/                     # Dataset storage
│   └── processed/            # Preprocessed datasets (NSL-KDD, CICIDS2017, UNSW-NB15, SWaT)
├── figures/                  # Generated figures and visualizations
├── experiment_results/       # Experimental validation results
├── examples/                 # Usage examples and tutorials
├── scripts/                  # Utility scripts
├── configs/                  # Configuration files
├── requirements.txt          # Production dependencies
└── README.md                 # This file
```

## Quick Start

### Basic Usage

```python
from uncertainty_ids import BayesianEnsembleIDS
from uncertainty_ids.data import SyntheticIDSDataset
import torch

# Create synthetic data for demonstration
dataset = SyntheticIDSDataset.create_synthetic(n_samples=1000)

# Initialize model
model = BayesianEnsembleIDS(
    n_ensemble=10,
    d_model=128,
    max_seq_len=50,
    n_classes=2
)

# Make predictions with uncertainty
sequences, queries, labels = dataset[0:32]  # Batch of 32 samples
results = model.predict_with_uncertainty(sequences, queries)

print(f"Predictions: {results['predictions']}")
print(f"Confidence: {results['confidence']}")
print(f"Uncertainty: {results['total_uncertainty']}")
```

### Training a Model

```python
from uncertainty_ids.training import UncertaintyIDSTrainer
from uncertainty_ids.data import NSLKDDDataset
from torch.utils.data import DataLoader

# Load NSL-KDD dataset
dataset = NSLKDDDataset.from_file('data/NSL-KDD/KDDTrain+.txt')
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Initialize trainer
trainer = UncertaintyIDSTrainer({
    'n_ensemble': 10,
    'd_model': 128,
    'max_seq_len': 50,
    'n_classes': 2
})

# Train model
history = trainer.train(train_loader, val_loader, n_epochs=100)

# Save trained model
trainer.save_model('models/uncertainty_ids_model.pth')
```

## Architecture

### Mathematical Foundation

Our novel theoretical framework establishes convergence guarantees for transformer-based intrusion detection:

```
F(E; W^V, W^KQ) = E + W^V E · (E^T W^KQ E) / ρ
```

Where:
- **E**: Embedded network flow sequences
- **W^V, W^KQ**: Learnable transformer parameters
- **ρ**: Normalization factor (sequence length)

### Baseline Methods

The framework includes comprehensive baseline implementations:

- **Traditional ML**: Random Forest, SVM, Logistic Regression
- **Deep Learning**: MLP, LSTM, CNN
- **Uncertainty-Aware**: MC Dropout, Deep Ensemble, Variational Inference, **Evidential Neural Networks (ENN)**
- **Our Variants**: Single Transformer, Bayesian Ensemble Transformer

### Uncertainty Decomposition

Total uncertainty is decomposed into two components:

```
Total Uncertainty = Epistemic Uncertainty + Aleatoric Uncertainty
```

- **Epistemic Uncertainty**: Model uncertainty (reducible with more data)
- **Aleatoric Uncertainty**: Data uncertainty (irreducible noise)

### System Components

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Input    │───▶│  Feature         │───▶│  Bayesian       │
│  (Network       │    │  Embedding       │    │  Ensemble       │
│   Flows)        │    │                  │    │  Transformers   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Uncertainty   │◀───│  Calibration     │◀───│  Uncertainty    │
│   Quantified    │    │  & Decision      │    │  Estimation     │
│   Predictions   │    │  Support         │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```



## Research & Development

This work introduces an uncertainty-aware intrusion detection framework based on transformer in-context learning theory with Bayesian ensemble methods.

### Key Contributions

1. **Theoretical Framework**: Novel application of transformer ICL theory to cybersecurity
2. **Uncertainty Quantification**: Principled decomposition into epistemic and aleatoric components
3. **Convergence Guarantees**: Formal bounds for single-layer transformer ensembles
4. **Comprehensive Baselines**: Implementation of multiple uncertainty-aware methods
5. **Evidential Learning**: Integration of state-of-the-art evidential neural networks

### Technical Innovations

- **Bayesian Ensemble Transformers**: Multiple transformer models with uncertainty quantification
- **Theoretical Foundations**: Convergence guarantees and uncertainty bounds
- **Calibrated Predictions**: Well-calibrated confidence estimates for decision support
- **Real-time Processing**: Optimized for production deployment scenarios

### Related Work

- **Transformer Theory**: Builds on in-context learning theoretical foundations
- **Uncertainty Quantification**: Extends Bayesian deep learning to network security
- **Intrusion Detection**: Novel architecture for modern threat detection
- **Ensemble Methods**: Principled diversity regularization for improved calibration

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

For questions, issues, or support:
- Open an issue on GitHub
- Review the examples in the `examples/` directory
- Check the documentation and code comments
