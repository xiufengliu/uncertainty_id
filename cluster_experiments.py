#!/usr/bin/env python3
"""
Cluster-Optimized Uncertainty-Aware Intrusion Detection Experiments
Enhanced version for DTU GPU cluster with checkpointing and robust error handling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import json
import time
import logging
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

# Setup logging
def setup_logging():
    """Setup comprehensive logging"""
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'cluster_experiments_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

logger = setup_logging()

class SingleLayerTransformer(nn.Module):
    """Single-layer transformer optimized for cluster execution"""
    
    def __init__(self, input_dim, d_model=64, n_heads=4, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Input embedding
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Single transformer layer
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed forward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        
        # Output layer
        self.output = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Input projection
        x = self.input_projection(x)
        
        # Add sequence dimension for attention
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # (batch, 1, d_model)
        
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Feed forward
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        # Output
        x = x.squeeze(1)  # Remove sequence dimension
        x = self.dropout(x)
        output = self.output(x)
        
        return torch.sigmoid(output.squeeze(-1))  # Only squeeze last dimension

class BayesianEnsembleTransformer:
    """Bayesian ensemble of transformers with uncertainty quantification"""
    
    def __init__(self, input_dim, ensemble_size=5, device='cuda'):
        self.ensemble_size = ensemble_size
        self.device = device
        self.models = []
        
        for i in range(ensemble_size):
            model = SingleLayerTransformer(input_dim).to(device)
            self.models.append(model)
    
    def train_ensemble(self, train_loader, val_loader, epochs=50):
        """Train the ensemble with early stopping"""
        logger.info(f"Training ensemble of {self.ensemble_size} models...")
        
        for i, model in enumerate(self.models):
            logger.info(f"Training model {i+1}/{self.ensemble_size}")
            
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
            criterion = nn.BCELoss()
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
            
            best_val_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(epochs):
                # Training
                model.train()
                train_loss = 0
                for batch_x, batch_y in train_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.float().to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                # Validation
                model.eval()
                val_loss = 0
                val_batches = 0
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        try:
                            batch_x, batch_y = batch_x.to(self.device), batch_y.float().to(self.device)
                            outputs = model(batch_x)
                            loss = criterion(outputs, batch_y)
                            val_loss += loss.item()
                            val_batches += 1
                        except Exception as batch_error:
                            logger.warning(f"Validation batch error: {batch_error}")
                            continue

                if val_batches > 0:
                    val_loss /= val_batches
                else:
                    val_loss = float('inf')
                    logger.warning("No valid validation batches processed")

                scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= 10:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        break
                
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Model {i+1} - Epoch {epoch+1}: Val Loss = {val_loss:.4f}")
    
    def predict_with_uncertainty(self, data_loader):
        """Generate predictions with uncertainty estimates"""
        all_predictions = []
        all_targets = []
        
        for model in self.models:
            model.eval()
            predictions = []
            targets = []
            
            with torch.no_grad():
                for batch_x, batch_y in data_loader:
                    batch_x = batch_x.to(self.device)
                    outputs = model(batch_x)
                    predictions.extend(outputs.cpu().numpy())
                    targets.extend(batch_y.numpy())
            
            all_predictions.append(predictions)
            all_targets = targets  # Same for all models
        
        # Calculate ensemble statistics
        predictions_array = np.array(all_predictions)  # (ensemble_size, n_samples)
        mean_predictions = np.mean(predictions_array, axis=0)
        uncertainty = np.std(predictions_array, axis=0)
        
        return {
            'predictions': mean_predictions,
            'uncertainty': uncertainty,
            'targets': np.array(all_targets)
        }

def calculate_metrics(predictions, targets):
    """Calculate metrics: FPR, Precision, Recall, F1"""
    try:
        pred_labels = (predictions > 0.5).astype(int)

        # Ensure targets and predictions are numpy arrays
        targets = np.array(targets)
        predictions = np.array(predictions)
        pred_labels = np.array(pred_labels)

        # Calculate confusion matrix with explicit labels to handle missing classes
        cm = confusion_matrix(targets, pred_labels, labels=[0, 1])

        # Handle different confusion matrix shapes
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        elif cm.shape == (1, 1):
            # Only one class present
            if np.unique(targets)[0] == 0:
                tn, fp, fn, tp = cm[0, 0], 0, 0, 0
            else:
                tn, fp, fn, tp = 0, 0, 0, cm[0, 0]
        else:
            # Fallback
            tn = fp = fn = tp = 0

        # Calculate requested metrics with safe division
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        # Additional metrics with error handling
        accuracy = accuracy_score(targets, pred_labels)
        try:
            auc = roc_auc_score(targets, predictions) if len(np.unique(targets)) > 1 else 0.0
        except ValueError:
            auc = 0.0

        return {
            'fpr': float(fpr),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'accuracy': float(accuracy),
            'auc': float(auc)
        }

    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        return {
            'fpr': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'accuracy': 0.0,
            'auc': 0.0
        }

def load_dataset(dataset_name):
    """Load preprocessed dataset with error handling"""
    try:
        data_dir = f'data/processed/{dataset_name}'
        
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
        
        X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
        X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
        y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
        y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
        
        logger.info(f"Loaded {dataset_name}: Train={X_train.shape}, Test={X_test.shape}")
        logger.info(f"Class distribution - Train: {np.bincount(y_train)}, Test: {np.bincount(y_test)}")
        
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        logger.error(f"Error loading dataset {dataset_name}: {e}")
        raise

def create_data_loaders(X_train, X_test, y_train, y_test, batch_size=256):
    """Create PyTorch data loaders with robust error handling"""
    try:
        # Validate input data
        if len(X_train) == 0 or len(X_test) == 0:
            raise ValueError("Empty dataset provided")

        if X_train.shape[1] != X_test.shape[1]:
            raise ValueError("Feature dimension mismatch between train and test")

        # Check for NaN or infinite values
        if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
            logger.warning("NaN or infinite values detected in training data")
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=-1e6)

        if np.any(np.isnan(X_test)) or np.any(np.isinf(X_test)):
            logger.warning("NaN or infinite values detected in test data")
            X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e6, neginf=-1e6)

        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))

        # Create validation split with minimum size check
        val_size = max(1, int(0.2 * len(train_dataset)))  # At least 1 sample
        train_size = len(train_dataset) - val_size

        if train_size <= 0:
            raise ValueError("Training dataset too small for validation split")

        train_subset, val_subset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        # Use num_workers=0 for cluster compatibility
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        logger.info(f"Created data loaders: Train={len(train_subset)}, Val={len(val_subset)}, Test={len(test_dataset)}")

        return train_loader, val_loader, test_loader

    except Exception as e:
        logger.error(f"Error creating data loaders: {e}")
        raise

def run_baseline_experiments(X_train, X_test, y_train, y_test, dataset_name):
    """Run baseline experiments with robust error handling"""
    logger.info(f"Running baseline experiments for {dataset_name}")

    results = {}

    # Get optimal number of jobs (use available cores but cap at 8)
    n_jobs = min(8, os.cpu_count() or 1)

    # Traditional ML baselines with memory-efficient settings
    baselines = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=n_jobs,
            max_depth=20,  # Prevent overfitting and memory issues
            min_samples_split=5,
            min_samples_leaf=2
        ),
        'SVM': SVC(
            kernel='rbf',
            probability=True,
            random_state=42,
            cache_size=1000,  # Limit memory usage
            max_iter=10000
        ),
        'Logistic Regression': LogisticRegression(
            random_state=42,
            max_iter=2000,
            n_jobs=n_jobs,
            solver='liblinear'  # More memory efficient for large datasets
        )
    }

    for name, model in baselines.items():
        try:
            logger.info(f"Training {name}...")
            start_time = time.time()

            # Check data size and adjust if necessary
            if len(X_train) > 100000 and name == 'SVM':
                logger.info(f"Large dataset detected for SVM, using subset for training")
                # Use stratified subset for SVM to avoid memory issues
                from sklearn.model_selection import train_test_split
                X_train_sub, _, y_train_sub, _ = train_test_split(
                    X_train, y_train, train_size=50000, random_state=42, stratify=y_train
                )
                model.fit(X_train_sub, y_train_sub)
            else:
                model.fit(X_train, y_train)

            # Get predictions with error handling
            try:
                predictions = model.predict_proba(X_test)[:, 1]
            except Exception as pred_error:
                logger.warning(f"predict_proba failed for {name}, using decision_function: {pred_error}")
                if hasattr(model, 'decision_function'):
                    raw_predictions = model.decision_function(X_test)
                    # Convert to probabilities using sigmoid
                    predictions = 1 / (1 + np.exp(-raw_predictions))
                else:
                    # Fallback to binary predictions
                    binary_preds = model.predict(X_test)
                    predictions = binary_preds.astype(float)

            metrics = calculate_metrics(predictions, y_test)
            results[name] = metrics

            training_time = time.time() - start_time
            logger.info(f"{name:<25} - FPR: {metrics['fpr']:.4f}, Precision: {metrics['precision']:.4f}, "
                       f"Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f} (Time: {training_time:.2f}s)")

            # Clear model from memory if it's large
            if hasattr(model, 'n_features_in_') and model.n_features_in_ > 50:
                del model

        except Exception as e:
            logger.error(f"Error with {name}: {e}")
            # Add fallback metrics
            results[name] = {
                'fpr': 0.0, 'precision': 0.0, 'recall': 0.0,
                'f1': 0.0, 'accuracy': 0.0, 'auc': 0.0
            }
            continue

    return results

def save_checkpoint(results, dataset_name, checkpoint_dir='checkpoints'):
    """Save intermediate results as checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_file = os.path.join(checkpoint_dir, f'{dataset_name}_checkpoint.json')
    
    with open(checkpoint_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Checkpoint saved: {checkpoint_file}")

def main():
    """Main experiment function"""
    logger.info("Starting Cluster-Optimized Uncertainty-Aware IDS Experiments")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        logger.info(f"Current GPU: {torch.cuda.current_device()}")
        logger.info(f"GPU name: {torch.cuda.get_device_name()}")

        # Clear GPU cache
        torch.cuda.empty_cache()

        # Log GPU memory
        device_id = torch.cuda.current_device()
        gpu_memory = torch.cuda.get_device_properties(device_id).total_memory / 1024**3
        logger.info(f"GPU memory: {gpu_memory:.2f} GB")

    # Configuration
    datasets = ['nsl_kdd', 'cicids2017', 'unsw_nb15', 'swat']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # Set memory management for CUDA
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    # Create output directories
    results_dir = 'experiment_results'
    figures_dir = 'figures'
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    all_results = {}
    
    for dataset_name in datasets:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {dataset_name.upper()} dataset")
        logger.info(f"{'='*60}")
        
        try:
            # Load data
            X_train, X_test, y_train, y_test = load_dataset(dataset_name)
            
            # Run baseline experiments
            baseline_results = run_baseline_experiments(X_train, X_test, y_train, y_test, dataset_name)
            
            # Create data loaders for deep learning
            train_loader, val_loader, test_loader = create_data_loaders(
                X_train, X_test, y_train, y_test, batch_size=256
            )
            
            # Clear GPU memory before training
            if device == 'cuda':
                torch.cuda.empty_cache()

            # Train Bayesian Ensemble Transformer
            logger.info("Training Bayesian Ensemble Transformer...")
            input_dim = X_train.shape[1]

            try:
                ensemble = BayesianEnsembleTransformer(input_dim, ensemble_size=5, device=device)
                ensemble.train_ensemble(train_loader, val_loader, epochs=50)

                # Evaluate ensemble
                logger.info("Evaluating ensemble...")
                results = ensemble.predict_with_uncertainty(test_loader)
                ensemble_metrics = calculate_metrics(results['predictions'], results['targets'])

                logger.info(f"{'Bayesian Ensemble Transformer':<25} - FPR: {ensemble_metrics['fpr']:.4f}, "
                           f"Precision: {ensemble_metrics['precision']:.4f}, "
                           f"Recall: {ensemble_metrics['recall']:.4f}, F1: {ensemble_metrics['f1']:.4f}")

                # Clean up ensemble models to free memory
                del ensemble
                if device == 'cuda':
                    torch.cuda.empty_cache()

            except Exception as ensemble_error:
                logger.error(f"Error with Bayesian Ensemble Transformer: {ensemble_error}")
                # Use fallback metrics
                ensemble_metrics = {
                    'fpr': 0.0, 'precision': 0.0, 'recall': 0.0,
                    'f1': 0.0, 'accuracy': 0.0, 'auc': 0.0
                }

            # Combine results
            dataset_results = baseline_results.copy()
            dataset_results['Bayesian Ensemble Transformer'] = ensemble_metrics
            all_results[dataset_name] = dataset_results

            # Save checkpoint after each dataset
            save_checkpoint(all_results, dataset_name)

            # Log memory usage if CUDA
            if device == 'cuda':
                allocated = torch.cuda.memory_allocated() / 1024**3
                cached = torch.cuda.memory_reserved() / 1024**3
                logger.info(f"GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")

        except Exception as e:
            logger.error(f"Error processing {dataset_name}: {e}")
            # Save partial results even if this dataset failed
            if baseline_results:
                all_results[dataset_name] = baseline_results
                save_checkpoint(all_results, dataset_name)
            continue
    
    # Save final results
    results_file = os.path.join(results_dir, 'all_results.json')
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"All results saved to: {results_file}")
    logger.info("Experiments completed successfully!")
    
    return all_results

if __name__ == "__main__":
    try:
        results = main()
        logger.info("All experiments completed successfully!")
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        raise
