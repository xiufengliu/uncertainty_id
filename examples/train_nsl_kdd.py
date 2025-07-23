"""
Example script for training on NSL-KDD dataset.
Demonstrates the complete pipeline from data loading to evaluation.
"""

import torch
import pandas as pd
import numpy as np
import json
from pathlib import Path

from uncertainty_ids.data.preprocessing import DataPreprocessor
from uncertainty_ids.data.datasets import NSLKDDDataset
from uncertainty_ids.data.loaders import create_dataloaders
from uncertainty_ids.models.transformer import BayesianEnsembleTransformer
from uncertainty_ids.models.uncertainty import UncertaintyQuantifier
from uncertainty_ids.training.trainer import UncertaintyAwareTrainer
from uncertainty_ids.evaluation.evaluator import ModelEvaluator


def main():
    """Main training function for NSL-KDD dataset."""
    
    # Configuration (based on paper specifications)
    config = {
        'ensemble_size': 5,
        'd_model': 129,  # Must be divisible by n_heads (3)
        'n_heads': 3,
        'dropout': 0.1,
        'max_seq_len': 51,
        'batch_size': 32,
        'learning_rate': 1e-3,
        'lambda_diversity': 0.1,
        'lambda_uncertainty': 0.05,
        'num_epochs': 100,
        'patience': 10,
        'random_seed': 42
    }
    
    # Set random seeds for reproducibility
    torch.manual_seed(config['random_seed'])
    np.random.seed(config['random_seed'])
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load NSL-KDD dataset
    print("Loading NSL-KDD dataset...")
    
    # NSL-KDD column names
    columns = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
        'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
        'num_root', 'num_file_creations', 'num_shells', 'num_access_files',
        'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
        'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
        'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
        'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
        'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
        'dst_host_serror_rate', 'dst_host_srv_serror_rate',
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack_type'
    ]
    
    # For demonstration, create synthetic NSL-KDD-like data
    # In practice, you would load from: data/NSL-KDD/KDDTrain+.txt
    print("Creating synthetic NSL-KDD-like data for demonstration...")
    
    n_samples = 10000
    np.random.seed(42)
    
    # Generate synthetic data
    data = {
        'duration': np.random.exponential(10, n_samples),
        'protocol_type': np.random.choice(['tcp', 'udp', 'icmp'], n_samples),
        'service': np.random.choice(['http', 'ftp', 'smtp', 'telnet', 'other'], n_samples),
        'flag': np.random.choice(['SF', 'S0', 'REJ', 'RSTR'], n_samples),
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
    df['label'] = (df['attack_type'] != 'normal').astype(int)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Attack distribution: {df['attack_type'].value_counts()}")
    
    # Preprocess data
    print("Preprocessing data...")
    preprocessor = DataPreprocessor()
    continuous_features, categorical_features, labels = preprocessor.fit_transform(df, 'label')
    
    print(f"Continuous features: {len(preprocessor.continuous_features)}")
    print(f"Categorical features: {len(preprocessor.categorical_features)}")
    print(f"Categorical vocab sizes: {preprocessor.categorical_vocab_sizes}")
    
    # Create dataset
    dataset = NSLKDDDataset(continuous_features, categorical_features, labels, 
                           sequence_length=config['max_seq_len'] - 1)
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset, batch_size=config['batch_size'], random_seed=config['random_seed']
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
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
    print("Creating trainer...")
    trainer = UncertaintyAwareTrainer(
        model=model,
        uncertainty_quantifier=uncertainty_quantifier,
        device=device,
        learning_rate=config['learning_rate'],
        lambda_diversity=config['lambda_diversity'],
        lambda_uncertainty=config['lambda_uncertainty']
    )
    
    # Train model
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
    results = evaluator.evaluate_dataset(test_loader, "NSL-KDD")
    
    # Print results
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1-Score: {results['f1_score']:.4f}")
    print(f"FPR: {results['fpr']:.4f}")
    print(f"AUC: {results.get('auc', 'N/A'):.4f}")
    print(f"ECE: {results['ece']:.4f}")
    print(f"AURC: {results['aurc']:.4f}")
    print(f"Uncertainty-Accuracy Correlation: {results['uncertainty_accuracy_correlation']:.4f}")
    print(f"Mutual Information: {results['mutual_information']:.4f}")
    
    # Save results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    final_results = {
        'config': config,
        'training_history': history,
        'test_results': results,
        'model_info': {
            'total_parameters': total_params,
            'continuous_features': preprocessor.continuous_features,
            'categorical_features': preprocessor.categorical_features
        }
    }
    
    with open(output_dir / "nsl_kdd_example_results.json", 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    
    print(f"\nResults saved to {output_dir / 'nsl_kdd_example_results.json'}")
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
