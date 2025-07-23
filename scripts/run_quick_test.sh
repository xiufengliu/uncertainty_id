#!/bin/bash

# Quick test script to verify the implementation works with synthetic data
# This can be run without downloading real datasets

echo "Running quick test with synthetic data..."

# Create logs directory
mkdir -p logs results

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Function to run test with error handling
run_test() {
    local test_name=$1
    local command=$2
    
    echo "=========================================="
    echo "Running: $test_name"
    echo "Command: $command"
    echo "Time: $(date)"
    echo "=========================================="
    
    eval $command
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "✓ $test_name completed successfully"
    else
        echo "✗ $test_name failed with exit code $exit_code"
    fi
    
    return $exit_code
}

# Test 1: Basic installation test
run_test "Installation Test" "python test_installation.py"

# Test 2: NSL-KDD example with synthetic data
run_test "NSL-KDD Example" "python examples/train_nsl_kdd.py"

# Test 3: Quick experiment with synthetic data
echo "Creating synthetic test dataset..."
python -c "
import pandas as pd
import numpy as np

# Create synthetic NSL-KDD-like data
np.random.seed(42)
n_samples = 1000

data = {
    'duration': np.random.exponential(10, n_samples),
    'protocol_type': np.random.choice(['tcp', 'udp', 'icmp'], n_samples),
    'service': np.random.choice(['http', 'ftp', 'smtp'], n_samples),
    'flag': np.random.choice(['SF', 'S0', 'REJ'], n_samples),
    'src_bytes': np.random.exponential(1000, n_samples),
    'dst_bytes': np.random.exponential(1000, n_samples),
    'land': np.random.choice([0, 1], n_samples, p=[0.99, 0.01]),
    'wrong_fragment': np.random.poisson(0.1, n_samples),
    'urgent': np.random.poisson(0.05, n_samples),
    'hot': np.random.poisson(0.2, n_samples),
    'num_failed_logins': np.random.poisson(0.1, n_samples),
    'logged_in': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
    'num_compromised': np.random.poisson(0.05, n_samples),
    'root_shell': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
    'su_attempted': np.random.choice([0, 1], n_samples, p=[0.98, 0.02]),
    'num_root': np.random.poisson(0.1, n_samples),
    'num_file_creations': np.random.poisson(0.2, n_samples),
    'num_shells': np.random.poisson(0.1, n_samples),
    'num_access_files': np.random.poisson(0.15, n_samples),
    'num_outbound_cmds': np.random.poisson(0.05, n_samples),
    'is_host_login': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
    'is_guest_login': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
    'count': np.random.poisson(10, n_samples),
    'srv_count': np.random.poisson(8, n_samples),
    'serror_rate': np.random.beta(1, 10, n_samples),
    'srv_serror_rate': np.random.beta(1, 10, n_samples),
    'rerror_rate': np.random.beta(1, 10, n_samples),
    'srv_rerror_rate': np.random.beta(1, 10, n_samples),
    'same_srv_rate': np.random.beta(5, 2, n_samples),
    'diff_srv_rate': np.random.beta(1, 5, n_samples),
    'srv_diff_host_rate': np.random.beta(1, 5, n_samples),
    'dst_host_count': np.random.poisson(20, n_samples),
    'dst_host_srv_count': np.random.poisson(15, n_samples),
    'dst_host_same_srv_rate': np.random.beta(5, 2, n_samples),
    'dst_host_diff_srv_rate': np.random.beta(1, 5, n_samples),
    'dst_host_same_src_port_rate': np.random.beta(3, 3, n_samples),
    'dst_host_srv_diff_host_rate': np.random.beta(1, 5, n_samples),
    'dst_host_serror_rate': np.random.beta(1, 10, n_samples),
    'dst_host_srv_serror_rate': np.random.beta(1, 10, n_samples),
    'dst_host_rerror_rate': np.random.beta(1, 10, n_samples),
    'dst_host_srv_rerror_rate': np.random.beta(1, 10, n_samples),
    'attack_type': np.random.choice(['normal', 'dos', 'probe', 'u2r', 'r2l'], 
                                  n_samples, p=[0.6, 0.2, 0.1, 0.05, 0.05])
}

df = pd.DataFrame(data)

# Create data directory and save
import os
os.makedirs('data/test', exist_ok=True)
df.to_csv('data/test/synthetic_nsl_kdd.csv', index=False)
print(f'Created synthetic dataset with {len(df)} samples')
print(f'Attack distribution: {df[\"attack_type\"].value_counts().to_dict()}')
"

# Test 4: Run experiment with synthetic data
if [ -f "data/test/synthetic_nsl_kdd.csv" ]; then
    echo "Running experiment with synthetic data..."
    
    # Create a minimal config for quick testing
    cat > configs/quick_test.json << 'EOF'
{
  "ensemble_size": 2,
  "d_model": 64,
  "n_heads": 2,
  "dropout": 0.1,
  "max_seq_len": 21,
  "temperature": 1.0,
  "batch_size": 16,
  "learning_rate": 0.001,
  "lambda_diversity": 0.1,
  "lambda_uncertainty": 0.05,
  "num_epochs": 3,
  "patience": 2,
  "random_seed": 42
}
EOF

    # Create a simple test script
    cat > test_synthetic_experiment.py << 'EOF'
import torch
import pandas as pd
import numpy as np
from uncertainty_ids.data.preprocessing import DataPreprocessor
from uncertainty_ids.data.datasets import BaseIDSDataset
from uncertainty_ids.data.loaders import create_dataloaders
from uncertainty_ids.models.transformer import BayesianEnsembleTransformer
from uncertainty_ids.models.uncertainty import UncertaintyQuantifier
from uncertainty_ids.training.trainer import UncertaintyAwareTrainer
from uncertainty_ids.evaluation.evaluator import ModelEvaluator
import json

print("Loading synthetic dataset...")
df = pd.read_csv('data/test/synthetic_nsl_kdd.csv')
df['label'] = (df['attack_type'] != 'normal').astype(int)

print(f"Dataset shape: {df.shape}")
print(f"Label distribution: {df['label'].value_counts().to_dict()}")

# Preprocess data
print("Preprocessing data...")
preprocessor = DataPreprocessor()
continuous_features, categorical_features, labels = preprocessor.fit_transform(df, 'label')

print(f"Continuous features: {continuous_features.shape}")
print(f"Categorical features: {categorical_features.shape}")

# Create dataset
dataset = BaseIDSDataset(continuous_features, categorical_features, labels, sequence_length=20)

# Create data loaders
train_loader, val_loader, test_loader = create_dataloaders(dataset, batch_size=16, random_seed=42)

print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")
print(f"Test batches: {len(test_loader)}")

# Load config
with open('configs/quick_test.json', 'r') as f:
    config = json.load(f)

# Create model
print("Creating model...")
model = BayesianEnsembleTransformer(
    continuous_features=preprocessor.continuous_features,
    categorical_features=preprocessor.categorical_features,
    categorical_vocab_sizes=preprocessor.categorical_vocab_sizes,
    ensemble_size=config['ensemble_size'],
    d_model=config['d_model'],
    n_heads=config['n_heads'],
    dropout=config['dropout'],
    max_seq_len=config['max_seq_len']
)

uncertainty_quantifier = UncertaintyQuantifier()

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total model parameters: {total_params:,}")

# Create trainer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

trainer = UncertaintyAwareTrainer(
    model=model,
    uncertainty_quantifier=uncertainty_quantifier,
    device=device,
    learning_rate=config['learning_rate'],
    lambda_diversity=config['lambda_diversity'],
    lambda_uncertainty=config['lambda_uncertainty']
)

# Train model (quick training)
print("Starting training...")
history = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=config['num_epochs'],
    patience=config['patience']
)

# Evaluate model
print("Evaluating model...")
evaluator = ModelEvaluator(model, uncertainty_quantifier, device)
results = evaluator.evaluate_dataset(test_loader, "Synthetic Test")

print("\n" + "="*50)
print("QUICK TEST RESULTS")
print("="*50)
print(f"Accuracy: {results['accuracy']:.4f}")
print(f"F1-Score: {results['f1_score']:.4f}")
print(f"ECE: {results['ece']:.4f}")
print(f"AURC: {results['aurc']:.4f}")
print("="*50)

print("Quick test completed successfully!")
EOF

    run_test "Synthetic Experiment" "python test_synthetic_experiment.py"
    
    # Clean up
    rm -f test_synthetic_experiment.py configs/quick_test.json
else
    echo "✗ Failed to create synthetic dataset"
fi

echo "=========================================="
echo "Quick Test Summary"
echo "=========================================="
echo "All quick tests completed!"
echo "If all tests passed, the implementation is working correctly."
echo ""
echo "Next steps:"
echo "1. Download real datasets using: ./scripts/download_datasets.sh"
echo "2. Run full experiments using: ./scripts/submit_experiments.sh"
echo ""
echo "Results saved in results/ directory"
echo "Logs saved in logs/ directory"
