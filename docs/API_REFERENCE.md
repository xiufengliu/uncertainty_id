# API Reference

This document provides comprehensive API documentation for the Uncertainty-Aware Intrusion Detection System.

## 📋 Table of Contents

- [Core Models](#core-models)
- [Training Components](#training-components)
- [Evaluation Metrics](#evaluation-metrics)
- [Data Processing](#data-processing)
- [Uncertainty Quantification](#uncertainty-quantification)
- [Visualization](#visualization)
- [Configuration](#configuration)

## 🤖 Core Models

### BayesianEnsembleTransformer

The main model class implementing our Bayesian ensemble transformer architecture.

```python
from uncertainty_ids.models.transformer import BayesianEnsembleTransformer

model = BayesianEnsembleTransformer(
    input_dim=41,                    # Number of input features
    d_model=256,                     # Model dimension
    n_heads=8,                       # Number of attention heads
    n_layers=6,                      # Number of transformer layers
    n_classes=2,                     # Number of output classes
    ensemble_size=5,                 # Number of ensemble members
    dropout=0.1,                     # Dropout rate
    categorical_vocab_sizes=None     # Vocabulary sizes for categorical features
)

# Forward pass
logits, attention_weights, individual_preds = model(
    x_continuous,                    # Continuous features [batch_size, n_cont_features]
    x_categorical,                   # Categorical features [batch_size, n_cat_features]
    return_individual=True           # Return individual ensemble predictions
)
```

#### Methods

- **`forward(x_cont, x_cat, return_individual=False)`**: Forward pass through the model
- **`get_attention_weights(x_cont, x_cat)`**: Extract attention weights for interpretability
- **`enable_dropout()`**: Enable dropout for MC sampling
- **`disable_dropout()`**: Disable dropout for deterministic inference

### UncertaintyQuantifier

Handles uncertainty quantification for the ensemble predictions.

```python
from uncertainty_ids.models.uncertainty import UncertaintyQuantifier

uncertainty_quantifier = UncertaintyQuantifier(
    ensemble_size=5,
    mc_samples=100,
    temperature=1.0
)

# Calculate uncertainties
uncertainties = uncertainty_quantifier.calculate_uncertainty(
    individual_predictions,          # [batch_size, ensemble_size, n_classes]
    ensemble_logits                  # [batch_size, n_classes]
)

# Returns dictionary with:
# - 'epistemic_uncertainty': Model uncertainty
# - 'aleatoric_uncertainty': Data uncertainty  
# - 'total_uncertainty': Combined uncertainty
```

## 🎯 Training Components

### UncertaintyAwareTrainer

Main training class with uncertainty-aware loss functions.

```python
from uncertainty_ids.training.trainer import UncertaintyAwareTrainer

trainer = UncertaintyAwareTrainer(
    model=model,
    uncertainty_quantifier=uncertainty_quantifier,
    device='cuda',
    config=training_config
)

# Train the model
history = trainer.train(
    train_loader,                    # Training data loader
    val_loader,                      # Validation data loader
    epochs=100,                      # Number of training epochs
    save_path='models/best_model.pt' # Path to save best model
)
```

#### Methods

- **`train(train_loader, val_loader, epochs, save_path)`**: Main training loop
- **`validate(val_loader)`**: Validation step
- **`save_checkpoint(path, epoch, metrics)`**: Save model checkpoint
- **`load_checkpoint(path)`**: Load model checkpoint

### Loss Functions

```python
from uncertainty_ids.training.losses import (
    UncertaintyAwareLoss,
    EvidentialLoss,
    CalibrationLoss
)

# Uncertainty-aware loss combining classification and uncertainty
uncertainty_loss = UncertaintyAwareLoss(
    alpha=0.1,                       # Weight for uncertainty regularization
    beta=0.05                        # Weight for diversity regularization
)

# Evidential learning loss
evidential_loss = EvidentialLoss(
    num_classes=2,
    annealing_coeff=0.01
)

# Calibration loss for better uncertainty calibration
calibration_loss = CalibrationLoss(
    n_bins=10,
    weight=0.1
)
```

## 📊 Evaluation Metrics

### Classification Metrics

```python
from uncertainty_ids.evaluation.metrics import ClassificationMetrics

# Compute standard classification metrics
metrics = ClassificationMetrics.compute_metrics(
    y_true,                          # True labels
    y_pred,                          # Predicted labels
    y_proba,                         # Predicted probabilities
    average='binary'                 # Averaging strategy
)

# Returns: accuracy, precision, recall, f1_score, fpr, tpr, auc
```

### Uncertainty Metrics

```python
from uncertainty_ids.evaluation.metrics import UncertaintyMetrics

# Uncertainty-accuracy correlation
correlation = UncertaintyMetrics.uncertainty_accuracy_correlation(
    uncertainties,                   # Uncertainty estimates
    correctness                      # Binary correctness (1=correct, 0=incorrect)
)

# Mutual information between uncertainty and correctness
mi = UncertaintyMetrics.mutual_information(
    uncertainties,
    correctness,
    n_bins=10
)

# Area under risk-coverage curve
aurc = UncertaintyMetrics.area_under_risk_coverage_curve(
    uncertainties,
    correctness
)
```

### Calibration Metrics

```python
from uncertainty_ids.evaluation.metrics import CalibrationMetrics

# Expected Calibration Error
ece = CalibrationMetrics.expected_calibration_error(
    confidences,                     # Confidence scores [0, 1]
    correctness,                     # Binary correctness
    n_bins=10                        # Number of calibration bins
)

# Reliability diagram data
bin_centers, bin_accuracies, bin_confidences = CalibrationMetrics.reliability_diagram_data(
    confidences,
    correctness,
    n_bins=10
)

# Brier score
brier = CalibrationMetrics.brier_score(
    probabilities,                   # Predicted probabilities
    targets                          # True binary labels
)
```

## 📈 Data Processing

### Dataset Loaders

```python
from uncertainty_ids.data.datasets import (
    NSLKDDDataset,
    CICIDS2017Dataset,
    UNSWDataset
)

# Load NSL-KDD dataset
dataset = NSLKDDDataset(
    data_dir='data/',
    split='train',                   # 'train', 'val', or 'test'
    download=True,                   # Download if not exists
    normalize=True                   # Apply normalization
)

# Create data loader
from torch.utils.data import DataLoader
loader = DataLoader(
    dataset,
    batch_size=256,
    shuffle=True,
    num_workers=4
)
```

### Data Preprocessing

```python
from uncertainty_ids.data.preprocessing import DataPreprocessor

preprocessor = DataPreprocessor(
    normalize_continuous=True,       # Normalize continuous features
    encode_categorical=True,         # Encode categorical features
    handle_missing='mean'            # Missing value strategy
)

# Fit and transform data
X_processed = preprocessor.fit_transform(X_raw)

# Transform new data
X_new_processed = preprocessor.transform(X_new)
```

## 🎲 Uncertainty Quantification

### Monte Carlo Dropout

```python
from uncertainty_ids.models.uncertainty import MCDropoutSampler

mc_sampler = MCDropoutSampler(
    model=model,
    n_samples=100,                   # Number of MC samples
    dropout_rate=0.1                 # Dropout rate for sampling
)

# Get MC predictions
mc_predictions = mc_sampler.predict(x_cont, x_cat)
uncertainties = mc_sampler.calculate_uncertainty(mc_predictions)
```

### Ensemble Uncertainty

```python
from uncertainty_ids.models.uncertainty import EnsembleUncertainty

ensemble_uncertainty = EnsembleUncertainty(
    ensemble_size=5
)

# Calculate ensemble uncertainties
uncertainties = ensemble_uncertainty.calculate_uncertainty(
    individual_predictions           # [batch_size, ensemble_size, n_classes]
)
```

## 📊 Visualization

### Training Visualization

```python
from uncertainty_ids.utils.visualization import plot_training_history

# Plot training history
plot_training_history(
    history,                         # Training history dictionary
    output_path='results/training_history.pdf'
)
```

### Uncertainty Visualization

```python
from uncertainty_ids.utils.visualization import (
    plot_uncertainty_distribution,
    plot_calibration_diagram,
    plot_confusion_matrix
)

# Plot uncertainty distribution
plot_uncertainty_distribution(
    uncertainties,                   # Uncertainty values
    correctness,                     # Correctness indicators
    output_path='results/uncertainty_dist.pdf'
)

# Plot calibration diagram
plot_calibration_diagram(
    y_true,                          # True labels
    y_prob,                          # Predicted probabilities
    output_path='results/calibration.pdf'
)

# Plot confusion matrix
plot_confusion_matrix(
    y_true,                          # True labels
    y_pred,                          # Predicted labels
    class_names=['Normal', 'Attack'],
    output_path='results/confusion_matrix.pdf'
)
```

## ⚙️ Configuration

### Configuration Loading

```python
from uncertainty_ids.utils.config import load_config

# Load configuration from JSON file
config = load_config('configs/default.json')

# Access configuration values
model_config = config['model']
training_config = config['training']
```

### Configuration Schema

```python
{
    "model": {
        "d_model": 256,              # Model dimension
        "n_heads": 8,                # Number of attention heads
        "n_layers": 6,               # Number of transformer layers
        "ensemble_size": 5,          # Ensemble size
        "dropout": 0.1               # Dropout rate
    },
    "training": {
        "batch_size": 256,           # Batch size
        "learning_rate": 0.001,      # Learning rate
        "epochs": 100,               # Number of epochs
        "early_stopping_patience": 10, # Early stopping patience
        "weight_decay": 0.01,        # Weight decay
        "gradient_clipping": 1.0     # Gradient clipping threshold
    },
    "uncertainty": {
        "mc_samples": 100,           # MC dropout samples
        "temperature_scaling": true, # Use temperature scaling
        "calibration_method": "temperature" # Calibration method
    },
    "data": {
        "normalize": true,           # Normalize features
        "validation_split": 0.2,     # Validation split ratio
        "test_split": 0.2,           # Test split ratio
        "random_seed": 42            # Random seed for reproducibility
    }
}
```

## 🔧 Utilities

### Reproducibility

```python
from uncertainty_ids.utils.reproducibility import set_random_seeds

# Set all random seeds for reproducibility
set_random_seeds(
    seed=42,                         # Random seed
    deterministic=True               # Use deterministic algorithms
)
```

### Logging

```python
from uncertainty_ids.utils.logging import setup_logger

# Setup logger
logger = setup_logger(
    name='uncertainty_ids',
    log_file='logs/experiment.log',
    level='INFO'
)

logger.info("Starting experiment...")
```

## 🚀 Quick Start Example

```python
import torch
from uncertainty_ids.models.transformer import BayesianEnsembleTransformer
from uncertainty_ids.models.uncertainty import UncertaintyQuantifier
from uncertainty_ids.data.datasets import NSLKDDDataset
from uncertainty_ids.training.trainer import UncertaintyAwareTrainer
from uncertainty_ids.utils.config import load_config

# Load configuration
config = load_config('configs/default.json')

# Create model
model = BayesianEnsembleTransformer(
    input_dim=41,
    **config['model']
)

# Create uncertainty quantifier
uncertainty_quantifier = UncertaintyQuantifier(
    ensemble_size=config['model']['ensemble_size'],
    **config['uncertainty']
)

# Load data
train_dataset = NSLKDDDataset(split='train', download=True)
train_loader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=config['training']['batch_size'],
    shuffle=True
)

# Create trainer
trainer = UncertaintyAwareTrainer(
    model=model,
    uncertainty_quantifier=uncertainty_quantifier,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    config=config
)

# Train model
history = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=config['training']['epochs']
)
```
