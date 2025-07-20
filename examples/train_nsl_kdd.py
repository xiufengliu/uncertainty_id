#!/usr/bin/env python3
"""
NSL-KDD Training Example

This script demonstrates training the uncertainty-aware IDS on the NSL-KDD dataset,
which is a standard benchmark for intrusion detection systems.
"""

import sys
import os
import logging
import argparse
import numpy as np
import torch
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from uncertainty_ids import BayesianEnsembleIDS, NetworkDataProcessor
from uncertainty_ids.data import NSLKDDDataset, create_data_loaders
from uncertainty_ids.training import UncertaintyIDSTrainer, TrainingConfig
from uncertainty_ids.evaluation import ComprehensiveEvaluator
from uncertainty_ids.utils import setup_logging, save_config

# Setup logging
setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_nsl_kdd(data_dir: str = './data'):
    """
    Download NSL-KDD dataset if not already present.
    
    Args:
        data_dir: Directory to store the dataset
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    train_file = data_path / 'NSL-KDD' / 'KDDTrain+.txt'
    test_file = data_path / 'NSL-KDD' / 'KDDTest+.txt'
    
    if train_file.exists() and test_file.exists():
        logger.info("NSL-KDD dataset already exists")
        return str(train_file), str(test_file)
    
    logger.info("NSL-KDD dataset not found. Please download it manually:")
    logger.info("1. Visit: https://www.unb.ca/cic/datasets/nsl.html")
    logger.info("2. Download NSL-KDD dataset")
    logger.info(f"3. Extract to: {data_path / 'NSL-KDD'}")
    logger.info("4. Ensure files are named: KDDTrain+.txt and KDDTest+.txt")
    
    # For demonstration, create synthetic NSL-KDD-like data
    logger.info("Creating synthetic NSL-KDD-like data for demonstration...")
    
    nsl_kdd_dir = data_path / 'NSL-KDD'
    nsl_kdd_dir.mkdir(exist_ok=True)
    
    # Create synthetic training data
    create_synthetic_nsl_kdd(str(train_file), n_samples=10000, attack_rate=0.2)
    create_synthetic_nsl_kdd(str(test_file), n_samples=5000, attack_rate=0.15)
    
    return str(train_file), str(test_file)


def create_synthetic_nsl_kdd(filepath: str, n_samples: int = 10000, attack_rate: float = 0.2):
    """Create synthetic NSL-KDD-like data for demonstration."""
    np.random.seed(42)
    
    # NSL-KDD feature names (41 features + label)
    feature_names = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
        'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
        'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
        'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
        'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
        'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label'
    ]
    
    # Generate synthetic data
    data = []
    
    for _ in range(n_samples):
        # Generate base features
        row = np.random.randn(41)
        
        # Set categorical features to valid ranges
        row[1] = np.random.randint(0, 4)  # protocol_type
        row[2] = np.random.randint(0, 70)  # service
        row[3] = np.random.randint(0, 11)  # flag
        
        # Set binary features
        binary_features = [6, 7, 8, 11, 13, 14, 20, 21]  # land, wrong_fragment, etc.
        for idx in binary_features:
            row[idx] = np.random.randint(0, 2)
        
        # Set count features to non-negative
        count_features = [4, 5, 9, 10, 12, 15, 16, 17, 18, 19, 22, 23, 31, 32]
        for idx in count_features:
            row[idx] = abs(row[idx])
        
        # Set rate features to [0, 1]
        rate_features = [24, 25, 26, 27, 28, 29, 30, 33, 34, 35, 36, 37, 38, 39, 40]
        for idx in rate_features:
            row[idx] = np.clip(abs(row[idx]), 0, 1)
        
        # Generate label
        is_attack = np.random.random() < attack_rate
        
        # Modify features for attacks to make them distinguishable
        if is_attack:
            # Attacks tend to have different patterns
            row[4] = max(row[4], 2.0)  # Higher src_bytes
            row[22] = max(row[22], 5.0)  # Higher count
            row[24] = min(row[24] + 0.3, 1.0)  # Higher serror_rate
            label = np.random.choice(['dos', 'probe', 'r2l', 'u2r'])
        else:
            label = 'normal'
        
        # Add label
        row_with_label = np.append(row, 0 if label == 'normal' else 1)
        data.append(row_with_label)
    
    # Create DataFrame and save
    df = pd.DataFrame(data, columns=feature_names)
    
    # Save as CSV (NSL-KDD format is actually space-separated, but CSV is easier)
    df.to_csv(filepath, index=False)
    
    logger.info(f"Created synthetic NSL-KDD data: {filepath}")
    logger.info(f"  - Samples: {n_samples}")
    logger.info(f"  - Attack rate: {attack_rate:.2%}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Uncertainty-Aware IDS on NSL-KDD')
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Directory containing NSL-KDD dataset')
    parser.add_argument('--output-dir', type=str, default='./outputs',
                       help='Directory to save outputs')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--ensemble-size', type=int, default=10,
                       help='Number of models in ensemble')
    parser.add_argument('--model-dim', type=int, default=128,
                       help='Model dimension')
    parser.add_argument('--seq-len', type=int, default=50,
                       help='Sequence length for transformer')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    
    args = parser.parse_args()
    
    print("🚀 Training Uncertainty-Aware IDS on NSL-KDD Dataset")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Download/prepare NSL-KDD dataset
    print("\n📊 Step 1: Preparing NSL-KDD dataset...")
    
    train_file, test_file = download_nsl_kdd(args.data_dir)
    
    # Step 2: Load and preprocess data
    print("\n🔄 Step 2: Loading and preprocessing data...")
    
    processor = NetworkDataProcessor(
        sequence_length=args.seq_len,
        normalize_method='standard'
    )
    
    # Load training data
    train_X, train_y = processor.preprocess_data(
        train_file, 
        target_column='label',
        dataset_type='nsl-kdd'
    )
    
    # Load test data
    test_X, test_y = processor.preprocess_data(
        test_file,
        target_column='label', 
        dataset_type='nsl-kdd'
    )
    
    print(f"✅ Data loaded:")
    print(f"   - Training samples: {len(train_X)}")
    print(f"   - Test samples: {len(test_X)}")
    print(f"   - Features: {train_X.shape[1]}")
    print(f"   - Training attack rate: {train_y.mean():.2%}")
    print(f"   - Test attack rate: {test_y.mean():.2%}")
    
    # Create temporal sequences
    train_sequences, train_queries, train_labels = processor.create_temporal_sequences(train_X, train_y)
    test_sequences, test_queries, test_labels = processor.create_temporal_sequences(test_X, test_y)
    
    # Split training data into train/validation
    train_idx, val_idx = train_test_split(
        range(len(train_sequences)), 
        test_size=0.2, 
        random_state=42,
        stratify=train_labels
    )
    
    # Create datasets
    from torch.utils.data import TensorDataset, DataLoader
    
    train_dataset = TensorDataset(
        torch.FloatTensor(train_sequences[train_idx]),
        torch.FloatTensor(train_queries[train_idx]),
        torch.LongTensor(train_labels[train_idx])
    )
    
    val_dataset = TensorDataset(
        torch.FloatTensor(train_sequences[val_idx]),
        torch.FloatTensor(train_queries[val_idx]),
        torch.LongTensor(train_labels[val_idx])
    )
    
    test_dataset = TensorDataset(
        torch.FloatTensor(test_sequences),
        torch.FloatTensor(test_queries),
        torch.LongTensor(test_labels)
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"✅ Data loaders created:")
    print(f"   - Training batches: {len(train_loader)}")
    print(f"   - Validation batches: {len(val_loader)}")
    print(f"   - Test batches: {len(test_loader)}")
    
    # Step 3: Configure and train model
    print("\n🤖 Step 3: Configuring model...")
    
    model_config = {
        'n_ensemble': args.ensemble_size,
        'd_model': args.model_dim,
        'max_seq_len': args.seq_len,
        'n_classes': 2,
        'dropout_rate': 0.1
    }
    
    training_config = TrainingConfig(
        model_type='bayesian_ensemble',
        model_params=model_config,
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=1e-5,
        early_stopping_patience=10,
        log_every=5,
        validate_every=1,
        calibrate_uncertainty=True,
        checkpoint_dir=str(output_dir / 'checkpoints'),
        device=args.device
    )
    
    print(f"✅ Model configuration:")
    print(f"   - Ensemble size: {model_config['n_ensemble']}")
    print(f"   - Model dimension: {model_config['d_model']}")
    print(f"   - Sequence length: {model_config['max_seq_len']}")
    print(f"   - Training epochs: {training_config.n_epochs}")
    
    # Step 4: Train the model
    print("\n🏋️ Step 4: Training the model...")
    
    trainer = UncertaintyIDSTrainer(training_config)
    
    try:
        history = trainer.train(train_loader, val_loader)
        print("✅ Training completed successfully!")
        
        # Save training history
        import json
        with open(output_dir / 'training_history.json', 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            history_json = {k: [float(x) for x in v] for k, v in history.items()}
            json.dump(history_json, f, indent=2)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return
    
    # Step 5: Evaluate the model
    print("\n📈 Step 5: Evaluating the model...")
    
    try:
        evaluation_results = trainer.evaluate(test_loader)
        
        print("✅ Evaluation Results:")
        print(f"   - Accuracy: {evaluation_results.detection_metrics['accuracy']:.4f}")
        print(f"   - Precision: {evaluation_results.detection_metrics['precision']:.4f}")
        print(f"   - Recall: {evaluation_results.detection_metrics['recall']:.4f}")
        print(f"   - F1-Score: {evaluation_results.detection_metrics['f1_score']:.4f}")
        print(f"   - False Positive Rate: {evaluation_results.detection_metrics['false_positive_rate']:.4f}")
        
        if 'auc_roc' in evaluation_results.detection_metrics:
            print(f"   - AUC-ROC: {evaluation_results.detection_metrics['auc_roc']:.4f}")
        
        if evaluation_results.uncertainty_metrics:
            print(f"   - Avg Uncertainty: {evaluation_results.uncertainty_metrics['mean_uncertainty']:.4f}")
            print(f"   - Uncertainty-Accuracy Correlation: {evaluation_results.uncertainty_metrics['uncertainty_accuracy_correlation']:.4f}")
        
        if evaluation_results.calibration_metrics:
            print(f"   - Expected Calibration Error: {evaluation_results.calibration_metrics['expected_calibration_error']:.4f}")
        
        # Save evaluation results
        import pickle
        with open(output_dir / 'evaluation_results.pkl', 'wb') as f:
            pickle.dump(evaluation_results, f)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return
    
    # Step 6: Save model and preprocessor
    print("\n💾 Step 6: Saving model and preprocessor...")
    
    try:
        # Save model
        model_path = output_dir / 'uncertainty_ids_nsl_kdd.pth'
        trainer.save_model(str(model_path))
        
        # Save preprocessor
        processor_path = output_dir / 'preprocessor'
        processor.save_preprocessors(str(processor_path))
        
        # Save configuration
        config_path = output_dir / 'config.json'
        save_config(training_config.to_dict(), str(config_path))
        
        print(f"✅ Model saved to: {model_path}")
        print(f"✅ Preprocessor saved to: {processor_path}")
        print(f"✅ Configuration saved to: {config_path}")
        
    except Exception as e:
        logger.error(f"Saving failed: {e}")
        return
    
    # Summary
    print("\n🎉 NSL-KDD Training Completed Successfully!")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print("Files created:")
    print("  - uncertainty_ids_nsl_kdd.pth (trained model)")
    print("  - preprocessor/ (data preprocessor)")
    print("  - training_history.json (training metrics)")
    print("  - evaluation_results.pkl (evaluation results)")
    print("  - config.json (training configuration)")
    print("\nNext steps:")
    print("1. Start API server with trained model")
    print("2. Run additional evaluations")
    print("3. Compare with baseline methods")


if __name__ == "__main__":
    main()
