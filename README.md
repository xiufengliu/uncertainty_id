# Uncertainty-Aware Intrusion Detection System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Tests](https://github.com/research-team/uncertainty-ids/workflows/Tests/badge.svg)](https://github.com/research-team/uncertainty-ids/actions)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://uncertainty-ids.readthedocs.io/)

A production-ready implementation of uncertainty-aware intrusion detection based on Bayesian ensemble transformers and in-context learning theory from "On the Training Convergence of Transformers for In-Context Classification".

## 🚀 Key Features

- **🎯 Theoretical Foundation**: Built on proven convergence guarantees from transformer in-context learning
- **🔬 Uncertainty Quantification**: Separates epistemic and aleatoric uncertainty for better decision-making
- **🤖 Bayesian Ensemble**: Multiple transformer models provide robust predictions with confidence estimates
- **⚡ Real-time Processing**: Optimized for production deployment in network security operations centers
- **📊 Comprehensive Evaluation**: Extensive metrics for both detection performance and uncertainty calibration
- **🐳 Production Ready**: Docker/Kubernetes deployment with monitoring and logging
- **🔧 Easy Integration**: REST API for seamless integration with existing security infrastructure

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Usage Examples](#usage-examples)
- [API Documentation](#api-documentation)
- [Evaluation](#evaluation)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [Research](#research)
- [License](#license)

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch 2.0 or higher
- CUDA (optional, for GPU acceleration)

### Install from PyPI

```bash
pip install uncertainty-ids
```

### Install from Source

```bash
git clone https://github.com/research-team/uncertainty-ids.git
cd uncertainty-ids
pip install -e .
```

### Development Installation

```bash
git clone https://github.com/research-team/uncertainty-ids.git
cd uncertainty-ids
pip install -e ".[dev]"
```

## 🚀 Quick Start

### 1. Basic Usage

```python
from uncertainty_ids import BayesianEnsembleIDS, NetworkDataProcessor
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

### 2. Training a Model

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

### 3. REST API Server

```python
from uncertainty_ids.api import create_app
import uvicorn

# Create FastAPI app
app = create_app(
    model_path='models/uncertainty_ids_model.pth',
    processor_path='preprocessors/'
)

# Run server
uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 4. Making API Requests

```python
import requests

# Prepare network flow data
flow_data = {
    "current_flow": {
        "duration": 0.5,
        "protocol_type": "tcp",
        "service": "http",
        "flag": "SF",
        "src_bytes": 1024,
        "dst_bytes": 512,
        # ... other 36 features
    },
    "return_uncertainty": True,
    "return_explanation": True
}

# Make prediction request
response = requests.post("http://localhost:8000/predict", json=flow_data)
result = response.json()

print(f"Prediction: {result['prediction_label']}")
print(f"Confidence: {result['confidence']:.3f}")
print(f"Requires Review: {result['requires_review']}")
```

## 🏗️ Architecture

### Mathematical Foundation

The system builds on the theoretical framework from "On the Training Convergence of Transformers for In-Context Classification":

```
F(E; W^V, W^KQ) = E + W^V E · (E^T W^KQ E) / ρ
```

Where:
- **E**: Embedded network flow sequences
- **W^V, W^KQ**: Learnable transformer parameters
- **ρ**: Normalization factor (sequence length)

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
