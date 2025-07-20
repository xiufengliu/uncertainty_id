#!/usr/bin/env python3
"""
Quick Start Example for Uncertainty-Aware Intrusion Detection System

This script demonstrates the basic usage of the uncertainty-aware IDS,
including data loading, model training, and evaluation.
"""

import sys
import os
import logging
import numpy as np
import torch
from pathlib import Path

# Add the parent directory to the path so we can import uncertainty_ids
sys.path.insert(0, str(Path(__file__).parent.parent))

from uncertainty_ids import BayesianEnsembleIDS, NetworkDataProcessor
from uncertainty_ids.data import SyntheticIDSDataset, create_data_loaders
from uncertainty_ids.training import UncertaintyIDSTrainer, TrainingConfig
from uncertainty_ids.evaluation import ComprehensiveEvaluator
from uncertainty_ids.utils import setup_logging

# Setup logging
setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main function demonstrating the uncertainty-aware IDS."""
    
    print("🚀 Uncertainty-Aware Intrusion Detection System - Quick Start")
    print("=" * 60)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Step 1: Create synthetic data for demonstration
    print("\n📊 Step 1: Creating synthetic network data...")
    
    dataset = SyntheticIDSDataset.create_synthetic(
        n_samples=5000,
        sequence_length=20,  # Shorter sequences for faster training
        n_features=41,
        attack_rate=0.15,
        random_state=42
    )
    
    print(f"✅ Created dataset with {len(dataset)} samples")
    print(f"   - Sequence length: {dataset.sequences.shape[1]}")
    print(f"   - Features: {dataset.sequences.shape[2]}")
    print(f"   - Attack rate: {dataset.get_attack_rate():.2%}")
    
    # Step 2: Create data loaders
    print("\n🔄 Step 2: Creating data loaders...")
    
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset,
        batch_size=32,
        train_split=0.7,
        val_split=0.15,
        test_split=0.15,
        shuffle=True,
        random_state=42
    )
    
    print(f"✅ Data loaders created:")
    print(f"   - Training batches: {len(train_loader)}")
    print(f"   - Validation batches: {len(val_loader)}")
    print(f"   - Test batches: {len(test_loader)}")
    
    # Step 3: Initialize model
    print("\n🤖 Step 3: Initializing Bayesian Ensemble Model...")
    
    model_config = {
        'n_ensemble': 5,  # Smaller ensemble for faster training
        'd_model': 64,    # Smaller model for faster training
        'max_seq_len': 20,
        'n_classes': 2,
        'dropout_rate': 0.1
    }
    
    model = BayesianEnsembleIDS(**model_config)
    
    print(f"✅ Model initialized:")
    print(f"   - Ensemble size: {model.n_ensemble}")
    print(f"   - Model dimension: {model.d_model}")
    print(f"   - Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Step 4: Configure training
    print("\n⚙️ Step 4: Configuring training...")
    
    training_config = TrainingConfig(
        model_type='bayesian_ensemble',
        model_params=model_config,
        batch_size=32,
        n_epochs=10,  # Fewer epochs for quick demo
        learning_rate=1e-3,
        weight_decay=1e-5,
        early_stopping_patience=5,
        log_every=2,
        validate_every=1,
        calibrate_uncertainty=True,
        checkpoint_dir='./checkpoints'
    )
    
    print(f"✅ Training configuration:")
    print(f"   - Epochs: {training_config.n_epochs}")
    print(f"   - Batch size: {training_config.batch_size}")
    print(f"   - Learning rate: {training_config.learning_rate}")
    
    # Step 5: Train the model
    print("\n🏋️ Step 5: Training the model...")
    
    trainer = UncertaintyIDSTrainer(training_config)
    
    try:
        history = trainer.train(train_loader, val_loader)
        print("✅ Training completed successfully!")
        
        # Print training summary
        final_train_loss = history['train_loss'][-1]
        final_val_loss = history['val_loss'][-1] if history['val_loss'] else 0
        final_val_acc = history['val_accuracy'][-1] if history['val_accuracy'] else 0
        
        print(f"   - Final training loss: {final_train_loss:.4f}")
        print(f"   - Final validation loss: {final_val_loss:.4f}")
        print(f"   - Final validation accuracy: {final_val_acc:.4f}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return
    
    # Step 6: Evaluate the model
    print("\n📈 Step 6: Evaluating the model...")
    
    try:
        evaluation_results = trainer.evaluate(test_loader)
        
        print("✅ Evaluation completed!")
        print(f"   - Test Accuracy: {evaluation_results.detection_metrics['accuracy']:.4f}")
        print(f"   - Test F1-Score: {evaluation_results.detection_metrics['f1_score']:.4f}")
        print(f"   - False Positive Rate: {evaluation_results.detection_metrics['false_positive_rate']:.4f}")
        
        if evaluation_results.uncertainty_metrics:
            print(f"   - Average Uncertainty: {evaluation_results.uncertainty_metrics['mean_uncertainty']:.4f}")
            print(f"   - Uncertainty-Accuracy Correlation: {evaluation_results.uncertainty_metrics['uncertainty_accuracy_correlation']:.4f}")
        
        if evaluation_results.calibration_metrics:
            print(f"   - Expected Calibration Error: {evaluation_results.calibration_metrics['expected_calibration_error']:.4f}")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return
    
    # Step 7: Demonstrate real-time prediction
    print("\n🔮 Step 7: Demonstrating real-time prediction...")
    
    try:
        # Get a sample from the test set
        sample_batch = next(iter(test_loader))
        sequences, queries, true_labels = sample_batch
        
        # Take first sample
        sample_sequence = sequences[0:1]  # Keep batch dimension
        sample_query = queries[0:1]
        true_label = true_labels[0].item()
        
        # Make prediction with uncertainty
        model.eval()
        with torch.no_grad():
            results = model.predict_with_uncertainty(sample_sequence, sample_query)
        
        prediction = results['predictions'].item()
        confidence = results['confidence'].item()
        total_uncertainty = results['total_uncertainty'].item()
        epistemic_uncertainty = results['epistemic_uncertainty'].item()
        aleatoric_uncertainty = results['aleatoric_uncertainty'].item()
        
        print("✅ Sample prediction:")
        print(f"   - True label: {'Attack' if true_label == 1 else 'Normal'}")
        print(f"   - Predicted: {'Attack' if prediction == 1 else 'Normal'}")
        print(f"   - Confidence: {confidence:.4f}")
        print(f"   - Total Uncertainty: {total_uncertainty:.4f}")
        print(f"   - Epistemic Uncertainty: {epistemic_uncertainty:.4f}")
        print(f"   - Aleatoric Uncertainty: {aleatoric_uncertainty:.4f}")
        print(f"   - Requires Review: {'Yes' if total_uncertainty > 0.2 else 'No'}")
        
    except Exception as e:
        print(f"❌ Prediction demo failed: {e}")
        return
    
    # Step 8: Save the trained model
    print("\n💾 Step 8: Saving the trained model...")
    
    try:
        model_save_path = './models/uncertainty_ids_quickstart.pth'
        os.makedirs('./models', exist_ok=True)
        
        trainer.save_model(model_save_path)
        print(f"✅ Model saved to: {model_save_path}")
        
    except Exception as e:
        print(f"❌ Model saving failed: {e}")
    
    # Summary
    print("\n🎉 Quick Start Completed Successfully!")
    print("=" * 60)
    print("Next steps:")
    print("1. Try the API server: python -m uncertainty_ids.api.server")
    print("2. Explore the notebooks in examples/notebooks/")
    print("3. Train on real datasets using examples/train_nsl_kdd.py")
    print("4. Check out the comprehensive evaluation tools")
    print("\nFor more information, visit: https://github.com/research-team/uncertainty-ids")


if __name__ == "__main__":
    main()
