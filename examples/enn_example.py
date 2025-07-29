#!/usr/bin/env python3
"""
Example usage of Evidential Neural Networks (ENN) for uncertainty-aware intrusion detection.

This example demonstrates how to use the ENN implementation for:
1. Training an evidential neural network
2. Making predictions with uncertainty estimates
3. Evaluating uncertainty quality
"""

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uncertainty_ids.models.evidential_neural_network import EvidentialNeuralNetwork
from uncertainty_ids.training.enn_trainer import ENNTrainer

def create_intrusion_detection_data(n_samples=1000):
    """Create synthetic intrusion detection data."""
    np.random.seed(42)
    
    # Simulate network flow features
    # Normal traffic features
    normal_features = np.random.normal(0, 1, (n_samples//2, 10))
    normal_labels = np.zeros(n_samples//2)
    
    # Malicious traffic features (shifted distribution)
    malicious_features = np.random.normal(2, 1.5, (n_samples//2, 10))
    malicious_labels = np.ones(n_samples//2)
    
    # Combine data
    X = np.vstack([normal_features, malicious_features])
    y = np.hstack([normal_labels, malicious_labels])
    
    # Shuffle data
    indices = np.random.permutation(n_samples)
    X, y = X[indices], y[indices]
    
    return torch.FloatTensor(X), torch.LongTensor(y)

def main():
    """Main example function."""
    print("Evidential Neural Networks (ENN) Example")
    print("=" * 50)
    
    # 1. Create synthetic intrusion detection data
    print("1. Creating synthetic intrusion detection data...")
    X, y = create_intrusion_detection_data(n_samples=1000)
    
    # Split data
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    test_size = len(X) - train_size - val_size
    
    train_X, train_y = X[:train_size], y[:train_size]
    val_X, val_y = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    test_X, test_y = X[train_size+val_size:], y[train_size+val_size:]
    
    # Create data loaders
    train_dataset = TensorDataset(train_X, train_y)
    val_dataset = TensorDataset(val_X, val_y)
    test_dataset = TensorDataset(test_X, test_y)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Validation samples: {len(val_dataset)}")
    print(f"   Test samples: {len(test_dataset)}")
    
    # 2. Create ENN model
    print("\n2. Creating Evidential Neural Network...")
    model = EvidentialNeuralNetwork(
        input_dim=10,
        hidden_dims=[64, 32],
        num_classes=2,
        dropout_rate=0.1
    )
    
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 3. Create trainer
    print("\n3. Setting up ENN trainer...")
    trainer = ENNTrainer(
        model=model,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        learning_rate=0.001,
        weight_decay=1e-4,
        annealing_step=10
    )
    
    print(f"   Device: {trainer.device}")
    print(f"   Learning rate: {trainer.learning_rate}")
    print(f"   Annealing step: {trainer.annealing_step}")
    
    # 4. Train the model
    print("\n4. Training ENN model...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=20
    )
    
    print(f"   Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"   Final train accuracy: {history['train_accuracy'][-1]:.4f}")
    print(f"   Final val loss: {history['val_loss'][-1]:.4f}")
    print(f"   Final val accuracy: {history['val_accuracy'][-1]:.4f}")
    
    # 5. Evaluate on test set
    print("\n5. Evaluating on test set...")
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_uncertainties = []
    all_confidences = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(trainer.device)
            
            # Get predictions with uncertainty
            outputs = model.predict_with_uncertainty(batch_x)
            
            all_predictions.extend(outputs['predictions'].cpu().numpy())
            all_labels.extend(batch_y.numpy())
            all_uncertainties.extend(outputs['total_uncertainty'].cpu().numpy())
            all_confidences.extend(outputs['confidence'].cpu().numpy())
            all_probabilities.extend(outputs['max_probability'].cpu().numpy())
    
    # Convert to numpy arrays
    predictions = np.array(all_predictions)
    labels = np.array(all_labels)
    uncertainties = np.array(all_uncertainties)
    confidences = np.array(all_confidences)
    probabilities = np.array(all_probabilities)
    
    # Compute metrics
    accuracy = np.mean(predictions == labels)
    correct_mask = (predictions == labels)
    
    print(f"   Test Accuracy: {accuracy:.4f}")
    print(f"   Mean Uncertainty: {np.mean(uncertainties):.4f}")
    print(f"   Mean Confidence: {np.mean(confidences):.4f}")
    
    # 6. Analyze uncertainty quality
    print("\n6. Analyzing uncertainty quality...")
    
    # Uncertainty should be higher for incorrect predictions
    correct_uncertainty = uncertainties[correct_mask]
    incorrect_uncertainty = uncertainties[~correct_mask]
    
    print(f"   Uncertainty for correct predictions: {np.mean(correct_uncertainty):.4f} ± {np.std(correct_uncertainty):.4f}")
    print(f"   Uncertainty for incorrect predictions: {np.mean(incorrect_uncertainty):.4f} ± {np.std(incorrect_uncertainty):.4f}")
    
    # Correlation between uncertainty and correctness (should be negative)
    uncertainty_correlation = np.corrcoef(uncertainties, correct_mask.astype(float))[0, 1]
    print(f"   Uncertainty-correctness correlation: {uncertainty_correlation:.4f} (should be negative)")
    
    # 7. Example predictions
    print("\n7. Example predictions with uncertainty:")
    print("   Sample | True | Pred | Confidence | Uncertainty | Correct")
    print("   " + "-" * 55)
    
    for i in range(min(10, len(predictions))):
        is_correct = "✓" if predictions[i] == labels[i] else "✗"
        print(f"   {i:6d} | {labels[i]:4d} | {predictions[i]:4d} | {confidences[i]:10.4f} | {uncertainties[i]:11.4f} | {is_correct:7s}")
    
    print("\n" + "=" * 50)
    print("ENN Example completed successfully!")
    print(f"✓ Trained ENN with {accuracy:.1%} accuracy")
    print(f"✓ Uncertainty correlation: {uncertainty_correlation:.3f}")
    print("✓ ENN provides principled uncertainty quantification")

if __name__ == "__main__":
    main()
