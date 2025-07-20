#!/usr/bin/env python3
"""
Command-line interface for training uncertainty-aware IDS models.

This script provides a comprehensive CLI for training models with various
configurations and datasets.
"""

import argparse
import logging
import sys
import os
from pathlib import Path
import json
import yaml
from typing import Dict, Any

from ..training import UncertaintyIDSTrainer, TrainingConfig
from ..data import NetworkDataProcessor, create_data_loaders
from ..utils import setup_logging, load_config, save_config

logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for training CLI."""
    parser = argparse.ArgumentParser(
        description='Train Uncertainty-Aware Intrusion Detection Models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    data_group = parser.add_argument_group('Data Configuration')
    data_group.add_argument('--data-path', type=str, required=True,
                           help='Path to training data CSV file')
    data_group.add_argument('--test-data-path', type=str,
                           help='Path to test data CSV file (optional)')
    data_group.add_argument('--dataset-type', type=str, default='auto',
                           choices=['auto', 'nsl-kdd', 'cicids2017', 'unsw-nb15', 'generic'],
                           help='Type of dataset')
    data_group.add_argument('--target-column', type=str, default='label',
                           help='Name of target column')
    data_group.add_argument('--sequence-length', type=int, default=50,
                           help='Length of temporal sequences')
    data_group.add_argument('--normalize-method', type=str, default='standard',
                           choices=['standard', 'minmax', 'robust'],
                           help='Feature normalization method')
    
    # Model arguments
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument('--model-type', type=str, default='bayesian_ensemble',
                            choices=['bayesian_ensemble', 'single_transformer'],
                            help='Type of model to train')
    model_group.add_argument('--ensemble-size', type=int, default=10,
                            help='Number of models in ensemble')
    model_group.add_argument('--model-dim', type=int, default=128,
                            help='Model dimension')
    model_group.add_argument('--dropout-rate', type=float, default=0.1,
                            help='Dropout rate')
    
    # Training arguments
    train_group = parser.add_argument_group('Training Configuration')
    train_group.add_argument('--batch-size', type=int, default=64,
                            help='Batch size for training')
    train_group.add_argument('--epochs', type=int, default=100,
                            help='Number of training epochs')
    train_group.add_argument('--learning-rate', type=float, default=1e-3,
                            help='Learning rate')
    train_group.add_argument('--weight-decay', type=float, default=1e-5,
                            help='Weight decay for regularization')
    train_group.add_argument('--optimizer', type=str, default='adam',
                            choices=['adam', 'adamw', 'sgd'],
                            help='Optimizer type')
    train_group.add_argument('--scheduler', type=str, default='reduce_on_plateau',
                            choices=['reduce_on_plateau', 'cosine', 'step', 'none'],
                            help='Learning rate scheduler')
    train_group.add_argument('--gradient-clip-norm', type=float, default=1.0,
                            help='Gradient clipping norm')
    
    # Regularization arguments
    reg_group = parser.add_argument_group('Regularization')
    reg_group.add_argument('--ensemble-diversity-weight', type=float, default=0.01,
                          help='Weight for ensemble diversity loss')
    reg_group.add_argument('--uncertainty-regularization-weight', type=float, default=0.01,
                          help='Weight for uncertainty regularization')
    reg_group.add_argument('--adversarial-training', action='store_true',
                          help='Enable adversarial training for robustness')
    reg_group.add_argument('--adversarial-epsilon', type=float, default=0.01,
                          help='Epsilon for adversarial perturbations')
    reg_group.add_argument('--spectral-normalization', action='store_true',
                          help='Enable spectral normalization for stability')
    
    # Early stopping and checkpointing
    checkpoint_group = parser.add_argument_group('Checkpointing and Early Stopping')
    checkpoint_group.add_argument('--early-stopping-patience', type=int, default=20,
                                 help='Early stopping patience')
    checkpoint_group.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                                 help='Directory to save checkpoints')
    checkpoint_group.add_argument('--save-checkpoint-every', type=int, default=10,
                                 help='Save checkpoint every N epochs')
    
    # Calibration
    calib_group = parser.add_argument_group('Uncertainty Calibration')
    calib_group.add_argument('--calibrate-uncertainty', action='store_true', default=True,
                            help='Enable uncertainty calibration')
    calib_group.add_argument('--calibration-method', type=str, default='temperature',
                            choices=['temperature', 'platt', 'isotonic'],
                            help='Calibration method')
    
    # Output and logging
    output_group = parser.add_argument_group('Output and Logging')
    output_group.add_argument('--output-dir', type=str, default='./outputs',
                             help='Output directory for results')
    output_group.add_argument('--model-name', type=str, default='uncertainty_ids_model',
                             help='Name for saved model')
    output_group.add_argument('--log-level', type=str, default='INFO',
                             choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                             help='Logging level')
    output_group.add_argument('--log-every', type=int, default=10,
                             help='Log every N epochs')
    output_group.add_argument('--validate-every', type=int, default=1,
                             help='Validate every N epochs')
    
    # Device and performance
    perf_group = parser.add_argument_group('Performance')
    perf_group.add_argument('--device', type=str, default='auto',
                           choices=['auto', 'cpu', 'cuda', 'mps'],
                           help='Device to use for training')
    perf_group.add_argument('--num-workers', type=int, default=4,
                           help='Number of data loader workers')
    
    # Configuration file
    parser.add_argument('--config', type=str,
                       help='Path to configuration file (YAML or JSON)')
    parser.add_argument('--save-config', type=str,
                       help='Save configuration to file')
    
    # Miscellaneous
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--resume', type=str,
                       help='Path to checkpoint to resume training from')
    
    return parser


def load_config_file(config_path: str) -> Dict[str, Any]:
    """Load configuration from file."""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    if config_path.suffix.lower() in ['.yaml', '.yml']:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    elif config_path.suffix.lower() == '.json':
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")


def create_training_config(args: argparse.Namespace) -> TrainingConfig:
    """Create training configuration from arguments."""
    model_params = {
        'n_ensemble': args.ensemble_size if args.model_type == 'bayesian_ensemble' else 1,
        'd_model': args.model_dim,
        'max_seq_len': args.sequence_length,
        'n_classes': 2,
        'dropout_rate': args.dropout_rate
    }
    
    scheduler_params = {
        'mode': 'min',
        'factor': 0.5,
        'patience': 10,
        'verbose': True
    }
    
    config = TrainingConfig(
        model_type=args.model_type,
        model_params=model_params,
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.gradient_clip_norm,
        optimizer=args.optimizer,
        scheduler=args.scheduler,
        scheduler_params=scheduler_params,
        ensemble_diversity_weight=args.ensemble_diversity_weight,
        uncertainty_regularization_weight=args.uncertainty_regularization_weight,
        early_stopping_patience=args.early_stopping_patience,
        save_checkpoint_every=args.save_checkpoint_every,
        checkpoint_dir=args.checkpoint_dir,
        calibrate_uncertainty=args.calibrate_uncertainty,
        calibration_method=args.calibration_method,
        log_every=args.log_every,
        validate_every=args.validate_every,
        device=args.device
    )
    
    return config


def main():
    """Main training function."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=getattr(logging, args.log_level.upper()))
    
    # Load configuration from file if provided
    if args.config:
        file_config = load_config_file(args.config)
        
        # Update args with file config (command line args take precedence)
        for key, value in file_config.items():
            if not hasattr(args, key) or getattr(args, key) is None:
                setattr(args, key, value)
    
    # Set random seed
    import torch
    import numpy as np
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("🚀 Starting Uncertainty-Aware IDS Training")
    logger.info(f"Output directory: {output_dir}")
    
    # Create training configuration
    training_config = create_training_config(args)
    
    # Save configuration if requested
    if args.save_config:
        save_config(training_config.to_dict(), args.save_config)
        logger.info(f"Configuration saved to: {args.save_config}")
    
    # Load and preprocess data
    logger.info("📊 Loading and preprocessing data...")
    
    processor = NetworkDataProcessor(
        sequence_length=args.sequence_length,
        normalize_method=args.normalize_method
    )
    
    # Load training data
    train_X, train_y = processor.preprocess_data(
        args.data_path,
        target_column=args.target_column,
        dataset_type=args.dataset_type
    )
    
    logger.info(f"Training data loaded: {len(train_X)} samples, {train_X.shape[1]} features")
    logger.info(f"Attack rate: {train_y.mean():.2%}")
    
    # Load test data if provided
    test_X, test_y = None, None
    if args.test_data_path:
        test_X, test_y = processor.preprocess_data(
            args.test_data_path,
            target_column=args.target_column,
            dataset_type=args.dataset_type
        )
        logger.info(f"Test data loaded: {len(test_X)} samples")
    
    # Create temporal sequences
    train_sequences, train_queries, train_labels = processor.create_temporal_sequences(train_X, train_y)
    
    if test_X is not None:
        test_sequences, test_queries, test_labels = processor.create_temporal_sequences(test_X, test_y)
    else:
        test_sequences = test_queries = test_labels = None
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        sequences=train_sequences,
        queries=train_queries,
        labels=train_labels,
        test_sequences=test_sequences,
        test_queries=test_queries,
        test_labels=test_labels,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_split=0.8,
        val_split=0.2
    )
    
    logger.info(f"Data loaders created:")
    logger.info(f"  - Training batches: {len(train_loader)}")
    logger.info(f"  - Validation batches: {len(val_loader)}")
    if test_loader:
        logger.info(f"  - Test batches: {len(test_loader)}")
    
    # Initialize trainer
    logger.info("🤖 Initializing trainer...")
    trainer = UncertaintyIDSTrainer(training_config)
    
    # Resume from checkpoint if requested
    if args.resume:
        logger.info(f"Resuming training from: {args.resume}")
        trainer.load_model(args.resume, load_optimizer=True)
    
    # Train the model
    logger.info("🏋️ Starting training...")
    
    try:
        history = trainer.train(train_loader, val_loader)
        logger.info("✅ Training completed successfully!")
        
        # Save training history
        history_path = output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            history_json = {k: [float(x) for x in v] for k, v in history.items()}
            json.dump(history_json, f, indent=2)
        
        logger.info(f"Training history saved to: {history_path}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)
    
    # Evaluate on test set if available
    if test_loader:
        logger.info("📈 Evaluating on test set...")
        
        try:
            evaluation_results = trainer.evaluate(test_loader)
            
            logger.info("✅ Evaluation Results:")
            logger.info(f"  - Accuracy: {evaluation_results.detection_metrics['accuracy']:.4f}")
            logger.info(f"  - F1-Score: {evaluation_results.detection_metrics['f1_score']:.4f}")
            logger.info(f"  - False Positive Rate: {evaluation_results.detection_metrics['false_positive_rate']:.4f}")
            
            if evaluation_results.uncertainty_metrics:
                logger.info(f"  - Avg Uncertainty: {evaluation_results.uncertainty_metrics['mean_uncertainty']:.4f}")
            
            if evaluation_results.calibration_metrics:
                logger.info(f"  - Calibration Error: {evaluation_results.calibration_metrics['expected_calibration_error']:.4f}")
            
            # Save evaluation results
            import pickle
            eval_path = output_dir / 'evaluation_results.pkl'
            with open(eval_path, 'wb') as f:
                pickle.dump(evaluation_results, f)
            
            logger.info(f"Evaluation results saved to: {eval_path}")
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
    
    # Save final model and preprocessor
    logger.info("💾 Saving model and preprocessor...")
    
    try:
        # Save model
        model_path = output_dir / f'{args.model_name}.pth'
        trainer.save_model(str(model_path))
        
        # Save preprocessor
        processor_path = output_dir / 'preprocessor'
        processor.save_preprocessors(str(processor_path))
        
        # Save final configuration
        config_path = output_dir / 'config.json'
        save_config(training_config.to_dict(), str(config_path))
        
        logger.info(f"✅ Model saved to: {model_path}")
        logger.info(f"✅ Preprocessor saved to: {processor_path}")
        logger.info(f"✅ Configuration saved to: {config_path}")
        
    except Exception as e:
        logger.error(f"Saving failed: {e}")
        sys.exit(1)
    
    logger.info("🎉 Training pipeline completed successfully!")
    logger.info(f"All outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
