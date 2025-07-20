"""
Training and Deployment Pipeline for Uncertainty-Aware Intrusion Detection System
Implements the complete workflow from data preprocessing to model deployment
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from uncertainty_ids_core import BayesianEnsembleIDS, UncertaintyCalibrator
from evaluation_framework import ComprehensiveEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NetworkDataProcessor:
    """Data preprocessing pipeline for network traffic data"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = self._get_feature_names()
        
    def _get_feature_names(self) -> list:
        """Define the 41 standard network intrusion detection features"""
        return [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
            'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
            'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
            'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
            'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
            'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
            'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
            'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
            'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
        ]
    
    def preprocess_data(self, data_path: str, target_column: str = 'label') -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess network traffic data
        
        Args:
            data_path: Path to CSV file containing network data
            target_column: Name of the target column
            
        Returns:
            X: Preprocessed features
            y: Encoded labels
        """
        logger.info(f"Loading data from {data_path}")
        
        # Load data
        df = pd.read_csv(data_path)
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Handle categorical features
        categorical_columns = ['protocol_type', 'service', 'flag']
        for col in categorical_columns:
            if col in X.columns:
                X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        
        # Convert to numpy arrays
        X = X.values.astype(np.float32)
        
        # Encode labels (binary: normal=0, attack=1)
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        logger.info(f"Data preprocessing complete. Shape: {X_scaled.shape}, Classes: {len(np.unique(y_encoded))}")
        
        return X_scaled, y_encoded
    
    def create_temporal_sequences(self, X: np.ndarray, y: np.ndarray, 
                                sequence_length: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create temporal sequences for transformer input
        
        Args:
            X: Feature matrix
            y: Labels
            sequence_length: Length of input sequences
            
        Returns:
            sequences: Input sequences (batch_size, seq_len, n_features)
            queries: Query samples (batch_size, n_features)
            labels: Corresponding labels
        """
        sequences = []
        queries = []
        labels = []
        
        for i in range(sequence_length, len(X)):
            # Historical sequence
            seq = X[i-sequence_length:i]
            # Current query
            query = X[i]
            # Label for current query
            label = y[i]
            
            sequences.append(seq)
            queries.append(query)
            labels.append(label)
        
        return np.array(sequences), np.array(queries), np.array(labels)
    
    def save_preprocessors(self, save_dir: str):
        """Save preprocessing objects"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.scaler, save_path / 'scaler.pkl')
        joblib.dump(self.label_encoder, save_path / 'label_encoder.pkl')
        
        logger.info(f"Preprocessors saved to {save_dir}")

class UncertaintyIDSTrainer:
    """Training pipeline for uncertainty-aware IDS"""
    
    def __init__(self, model_config: Dict):
        self.config = model_config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize model
        self.model = BayesianEnsembleIDS(**model_config).to(self.device)
        
        # Initialize calibrator
        self.calibrator = UncertaintyCalibrator()
        
        # Initialize evaluator
        self.evaluator = ComprehensiveEvaluator()
        
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              n_epochs: int = 100, learning_rate: float = 1e-3) -> Dict:
        """
        Train the uncertainty-aware IDS model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            n_epochs: Number of training epochs
            learning_rate: Learning rate
            
        Returns:
            training_history: Dictionary containing training metrics
        """
        logger.info("Starting model training...")
        
        # Initialize optimizer and loss function
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        # Training history
        history = {
            'train_loss': [], 'val_loss': [], 'val_accuracy': [],
            'val_uncertainty': [], 'val_calibration_error': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 20
        
        for epoch in range(n_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_batches = 0
            
            for batch_idx, (sequences, queries, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
                sequences = sequences.to(self.device)
                queries = queries.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass through ensemble
                logits, uncertainties = self.model(sequences, queries)
                
                # Compute loss
                loss = criterion(logits, labels)
                
                # Add uncertainty regularization
                uncertainty_reg = torch.mean(uncertainties) * 0.01
                total_loss = loss + uncertainty_reg
                
                # Backward pass
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += total_loss.item()
                train_batches += 1
            
            avg_train_loss = train_loss / train_batches
            
            # Validation phase
            val_metrics = self.validate(val_loader)
            
            # Update learning rate
            scheduler.step(val_metrics['loss'])
            
            # Save metrics
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_metrics['loss'])
            history['val_accuracy'].append(val_metrics['accuracy'])
            history['val_uncertainty'].append(val_metrics['avg_uncertainty'])
            history['val_calibration_error'].append(val_metrics['calibration_error'])
            
            # Early stopping
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                # Save best model
                self.save_model('best_model.pth')
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
            
            # Log progress
            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{n_epochs} - "
                    f"Train Loss: {avg_train_loss:.4f}, "
                    f"Val Loss: {val_metrics['loss']:.4f}, "
                    f"Val Acc: {val_metrics['accuracy']:.4f}, "
                    f"Val Uncertainty: {val_metrics['avg_uncertainty']:.4f}"
                )
        
        logger.info("Training completed!")
        return history
    
    def validate(self, val_loader: DataLoader) -> Dict:
        """Validate the model and compute metrics"""
        self.model.eval()
        
        all_predictions = []
        all_probabilities = []
        all_uncertainties = []
        all_confidences = []
        all_labels = []
        val_loss = 0.0
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for sequences, queries, labels in val_loader:
                sequences = sequences.to(self.device)
                queries = queries.to(self.device)
                labels = labels.to(self.device)
                
                # Get predictions with uncertainty
                results = self.model.predict_with_uncertainty(sequences, queries)
                
                # Compute loss
                logits, _ = self.model(sequences, queries)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                
                # Collect results
                all_predictions.extend(results['predictions'].cpu().numpy())
                all_probabilities.extend(results['probabilities'].cpu().numpy())
                all_uncertainties.extend(results['total_uncertainty'].cpu().numpy())
                all_confidences.extend(results['confidence'].cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        all_uncertainties = np.array(all_uncertainties)
        all_confidences = np.array(all_confidences)
        all_labels = np.array(all_labels)
        
        # Compute metrics
        accuracy = (all_predictions == all_labels).mean()
        avg_uncertainty = all_uncertainties.mean()
        
        # Compute calibration error
        correctness = (all_predictions == all_labels).astype(float)
        calibration_error = self.calibrator.compute_calibration_error(
            all_confidences, correctness
        )
        
        return {
            'loss': val_loss / len(val_loader),
            'accuracy': accuracy,
            'avg_uncertainty': avg_uncertainty,
            'calibration_error': calibration_error
        }
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': self.config,
            'calibrator': self.calibrator
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.calibrator = checkpoint['calibrator']
        logger.info(f"Model loaded from {filepath}")

def main():
    """Main training and evaluation pipeline"""
    
    # Configuration
    config = {
        'data_path': 'network_data.csv',  # Path to your network data
        'model_config': {
            'n_ensemble': 10,
            'd_model': 128,
            'max_seq_len': 50,
            'n_classes': 2
        },
        'training': {
            'batch_size': 64,
            'n_epochs': 100,
            'learning_rate': 1e-3,
            'sequence_length': 50
        }
    }
    
    # Initialize data processor
    processor = NetworkDataProcessor()
    
    # Load and preprocess data (assuming you have network_data.csv)
    # For demonstration, we'll create synthetic data
    logger.info("Creating synthetic network data for demonstration...")
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 10000
    n_features = 41
    
    # Generate synthetic network features
    X_synthetic = np.random.randn(n_samples, n_features).astype(np.float32)
    
    # Generate synthetic labels (10% attacks)
    y_synthetic = np.random.binomial(1, 0.1, n_samples)
    
    # Create temporal sequences
    sequences, queries, labels = processor.create_temporal_sequences(
        X_synthetic, y_synthetic, config['training']['sequence_length']
    )
    
    # Split data
    train_seq, test_seq, train_queries, test_queries, train_labels, test_labels = train_test_split(
        sequences, queries, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    train_seq, val_seq, train_queries, val_queries, train_labels, val_labels = train_test_split(
        train_seq, train_queries, train_labels, test_size=0.2, random_state=42, stratify=train_labels
    )
    
    # Create data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(train_seq),
        torch.FloatTensor(train_queries),
        torch.LongTensor(train_labels)
    )
    
    val_dataset = TensorDataset(
        torch.FloatTensor(val_seq),
        torch.FloatTensor(val_queries),
        torch.LongTensor(val_labels)
    )
    
    test_dataset = TensorDataset(
        torch.FloatTensor(test_seq),
        torch.FloatTensor(test_queries),
        torch.LongTensor(test_labels)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    
    # Initialize trainer
    trainer = UncertaintyIDSTrainer(config['model_config'])
    
    # Train model
    history = trainer.train(
        train_loader, val_loader,
        n_epochs=config['training']['n_epochs'],
        learning_rate=config['training']['learning_rate']
    )
    
    # Load best model
    trainer.load_model('best_model.pth')
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    
    trainer.model.eval()
    all_predictions = []
    all_probabilities = []
    all_uncertainties = []
    all_confidences = []
    all_labels = []
    
    with torch.no_grad():
        for sequences, queries, labels in test_loader:
            sequences = sequences.to(trainer.device)
            queries = queries.to(trainer.device)
            
            results = trainer.model.predict_with_uncertainty(sequences, queries)
            
            all_predictions.extend(results['predictions'].cpu().numpy())
            all_probabilities.extend(results['probabilities'].cpu().numpy())
            all_uncertainties.extend(results['total_uncertainty'].cpu().numpy())
            all_confidences.extend(results['confidence'].cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    all_uncertainties = np.array(all_uncertainties)
    all_confidences = np.array(all_confidences)
    all_labels = np.array(all_labels)
    
    # Comprehensive evaluation
    evaluator = ComprehensiveEvaluator()
    results = evaluator.evaluate_model(
        all_labels, all_predictions, all_probabilities, 
        all_uncertainties, all_confidences
    )
    
    # Generate and print report
    report = evaluator.generate_evaluation_report(results)
    print(report)
    
    # Plot results
    evaluator.plot_evaluation_results(results, save_path='evaluation_results.png')
    
    # Save preprocessors
    processor.save_preprocessors('preprocessors/')
    
    logger.info("Training and evaluation pipeline completed successfully!")

if __name__ == "__main__":
    main()
