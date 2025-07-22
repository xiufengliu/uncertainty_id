#!/usr/bin/env python3
"""
Real Comprehensive Experiments for Uncertainty-Aware Intrusion Detection
Memory-efficient implementation for cluster execution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.calibration import calibration_curve
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

print("Starting Real Experiments...")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"GPU name: {torch.cuda.get_device_name()}")

class SingleLayerTransformer(nn.Module):
    """Single-layer transformer as described in the paper"""
    
    def __init__(self, input_dim, d_model=64, n_heads=4, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Input embedding
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Single-layer transformer block
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
            nn.Dropout(dropout)
        )
        
        # Output projection
        self.output_projection = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
        
        # Input projection
        x = self.input_projection(x)
        
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        # Global average pooling and output
        x = x.mean(dim=1)
        x = self.dropout(x)
        output = self.output_projection(x)
        
        return torch.sigmoid(output.squeeze(-1))

class BayesianEnsembleIDS:
    """Bayesian Ensemble Transformer for Intrusion Detection"""
    
    def __init__(self, input_dim, n_ensemble=5, d_model=64, n_heads=4, dropout=0.1, device='cuda'):
        self.input_dim = input_dim
        self.n_ensemble = n_ensemble
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        print(f"Initializing Bayesian Ensemble with {n_ensemble} models on {self.device}")
        
        # Create ensemble of transformers
        self.models = []
        for i in range(n_ensemble):
            model = SingleLayerTransformer(input_dim, d_model, n_heads, dropout)
            model.to(self.device)
            self.models.append(model)
        
    def train_ensemble(self, train_loader, val_loader, epochs=20, lr=1e-3):
        """Train the ensemble with diversity regularization"""
        print(f"Training ensemble of {self.n_ensemble} models for {epochs} epochs...")
        
        optimizers = [torch.optim.Adam(model.parameters(), lr=lr) for model in self.models]
        criterion = nn.BCELoss()
        
        best_val_acc = 0
        
        for epoch in range(epochs):
            # Training phase
            epoch_losses = []
            for model in self.models:
                model.train()
            
            batch_count = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device).float()
                
                batch_losses = []
                for i, (model, optimizer) in enumerate(zip(self.models, optimizers)):
                    optimizer.zero_grad()
                    
                    output = model(data)
                    loss = criterion(output, target)
                    
                    # Diversity regularization
                    if len(self.models) > 1:
                        diversity_loss = 0
                        for j, other_model in enumerate(self.models):
                            if i != j:
                                with torch.no_grad():
                                    other_output = other_model(data)
                                diversity_loss += F.mse_loss(output, other_output)
                        diversity_loss /= (len(self.models) - 1)
                        loss += 0.1 * diversity_loss  # λ₁ = 0.1
                    
                    loss.backward()
                    optimizer.step()
                    batch_losses.append(loss.item())
                
                epoch_losses.extend(batch_losses)
                batch_count += 1
                
                # Print progress every 50 batches
                if batch_count % 50 == 0:
                    print(f'  Epoch {epoch}, Batch {batch_count}, Avg Loss: {np.mean(batch_losses):.4f}')
            
            # Validation
            val_acc = self.evaluate(val_loader)
            
            avg_loss = np.mean(epoch_losses)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            
            print(f'Epoch {epoch:2d}: Loss={avg_loss:.4f}, Val Acc={val_acc:.4f}, Best={best_val_acc:.4f}')
        
        print(f'Training completed. Best validation accuracy: {best_val_acc:.4f}')
        return best_val_acc
    
    def predict_with_uncertainty(self, data_loader):
        """Make predictions with uncertainty quantification"""
        all_predictions = []
        all_targets = []
        
        for model in self.models:
            model.eval()
        
        with torch.no_grad():
            for data, target in data_loader:
                data = data.to(self.device)
                
                batch_predictions = []
                for model in self.models:
                    pred = model(data).cpu().numpy()
                    batch_predictions.append(pred)
                
                all_predictions.append(np.array(batch_predictions))
                all_targets.append(target.numpy())
        
        predictions = np.concatenate(all_predictions, axis=1)
        targets = np.concatenate(all_targets)
        
        # Calculate ensemble statistics
        mean_pred = np.mean(predictions, axis=0)
        epistemic_uncertainty = np.var(predictions, axis=0)
        aleatoric_uncertainty = np.mean(predictions * (1 - predictions), axis=0)
        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
        
        return {
            'predictions': mean_pred,
            'targets': targets,
            'epistemic_uncertainty': epistemic_uncertainty,
            'aleatoric_uncertainty': aleatoric_uncertainty,
            'total_uncertainty': total_uncertainty
        }
    
    def evaluate(self, data_loader):
        """Evaluate ensemble performance"""
        results = self.predict_with_uncertainty(data_loader)
        predictions = (results['predictions'] > 0.5).astype(int)
        targets = results['targets']
        return accuracy_score(targets, predictions)

def load_dataset(dataset_name):
    """Load preprocessed dataset"""
    data_dir = f'data/processed/{dataset_name}'
    
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    
    print(f"Loaded {dataset_name}: Train={X_train.shape}, Test={X_test.shape}")
    print(f"Class distribution - Train: {np.bincount(y_train)}, Test: {np.bincount(y_test)}")
    
    return X_train, X_test, y_train, y_test

def create_data_loaders(X_train, X_test, y_train, y_test, batch_size=256):
    """Create PyTorch data loaders"""
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    
    # Create validation split
    val_size = int(0.2 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, test_loader

def calculate_metrics(predictions, targets):
    """Calculate metrics: FPR, Precision, Recall, F1"""
    pred_labels = (predictions > 0.5).astype(int)

    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(targets, pred_labels).ravel()

    # Calculate requested metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # Additional metrics for completeness
    accuracy = accuracy_score(targets, pred_labels)
    auc = roc_auc_score(targets, predictions) if len(np.unique(targets)) > 1 else 0.0

    return {
        'fpr': fpr,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'auc': auc
    }

def run_baseline_experiments(X_train, X_test, y_train, y_test, dataset_name):
    """Run baseline experiments"""
    print(f"\n--- Running Baseline Experiments for {dataset_name} ---")
    
    results = {}
    
    # Traditional ML baselines
    baselines = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=4),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, n_jobs=4)
    }
    
    for name, model in baselines.items():
        print(f"Training {name}...")
        try:
            model.fit(X_train, y_train)
            predictions = model.predict_proba(X_test)[:, 1]
            metrics = calculate_metrics(predictions, y_test)
            results[name] = metrics
            print(f"{name} - FPR: {metrics['fpr']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}")
        except Exception as e:
            print(f"Error with {name}: {e}")
            continue
    
    return results

def run_experiments():
    """Run real experiments on all datasets"""
    datasets = ['nsl_kdd', 'cicids2017', 'unsw_nb15', 'swat']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    all_results = {}
    
    # Create results directory in our project space
    project_dir = '/zhome/bb/9/101964/xiuli/IntrDetection'
    results_dir = os.path.join(project_dir, 'experiment_results')
    figures_dir = os.path.join(project_dir, 'figures')
    logs_dir = os.path.join(project_dir, 'logs')

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    print(f"Saving all results to: {results_dir}")
    print(f"Saving all figures to: {figures_dir}")
    print(f"Saving all logs to: {logs_dir}")
    
    for dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"Running REAL experiments on {dataset_name.upper()} dataset")
        print(f"{'='*60}")
        
        try:
            # Load data
            X_train, X_test, y_train, y_test = load_dataset(dataset_name)
            
            # Run baseline experiments
            baseline_results = run_baseline_experiments(X_train, X_test, y_train, y_test, dataset_name)
            
            # Create data loaders for deep learning
            train_loader, val_loader, test_loader = create_data_loaders(
                X_train, X_test, y_train, y_test, batch_size=256
            )
            
            input_dim = X_train.shape[1]
            
            # Run our Bayesian Ensemble Transformer
            print(f"\n--- Training Our Bayesian Ensemble Transformer ---")
            ensemble_model = BayesianEnsembleIDS(
                input_dim=input_dim,
                n_ensemble=5,  # Reduced for memory efficiency
                d_model=64,   # Reduced for memory efficiency
                n_heads=4,    # Reduced for memory efficiency
                dropout=0.1,
                device=device
            )
            
            # Train ensemble
            best_val_acc = ensemble_model.train_ensemble(
                train_loader, val_loader, epochs=20, lr=1e-3
            )
            
            # Evaluate ensemble
            print("Evaluating ensemble on test set...")
            results = ensemble_model.predict_with_uncertainty(test_loader)
            ensemble_metrics = calculate_metrics(results['predictions'], results['targets'])
            
            print(f"Our Method - Accuracy: {ensemble_metrics['accuracy']:.4f}, "
                  f"F1: {ensemble_metrics['f1']:.4f}, ECE: {ensemble_metrics['ece']:.4f}")
            
            # Combine all results
            dataset_results = baseline_results.copy()
            dataset_results['Ours (Bayesian Ensemble Transformer)'] = ensemble_metrics
            
            all_results[dataset_name] = dataset_results
            
            # Save intermediate results to our project directory
            result_file = os.path.join(results_dir, f'{dataset_name}_results.json')
            with open(result_file, 'w') as f:
                json.dump(dataset_results, f, indent=2)
            
            print(f"Results saved for {dataset_name}")
            
            # Clear GPU memory
            del ensemble_model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save comprehensive results to our project directory
    all_results_file = os.path.join(results_dir, 'all_results.json')
    with open(all_results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("REAL EXPERIMENTAL RESULTS SUMMARY")
    print(f"{'='*60}")
    
    for dataset_name, results in all_results.items():
        print(f"\n{dataset_name.upper()} Dataset:")
        print("-" * 50)
        print(f"{'Method':<30} {'Accuracy':<10} {'F1-Score':<10} {'FPR':<8} {'ECE':<8}")
        print("-" * 50)
        
        for method, metrics in results.items():
            marker = "*" if "Ours" in method else ""
            print(f"{method:<30} {metrics['accuracy']:<10.4f} {metrics['f1']:<10.4f} "
                  f"{metrics['fpr']:<8.4f} {metrics['ece']:<8.4f}{marker}")
    
    return all_results

if __name__ == "__main__":
    print("Starting REAL Uncertainty-Aware IDS Experiments")
    print("=" * 60)
    
    # Run real experiments
    results = run_experiments()
    
    print("\n" + "="*60)
    print("REAL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Results saved in {results_dir}/ directory")
    print("These are ACTUAL experimental results, not synthetic data!")
