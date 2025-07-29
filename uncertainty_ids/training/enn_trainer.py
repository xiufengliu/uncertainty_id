"""
Trainer for Evidential Neural Networks (ENN).

This module provides training utilities specifically designed for ENN models,
including evidential loss computation and uncertainty-aware evaluation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm
import os

from ..models.evidential_neural_network import EvidentialNeuralNetwork, evidential_loss
from ..evaluation.metrics import CalibrationMetrics

logger = logging.getLogger(__name__)


class ENNTrainer:
    """
    Trainer class for Evidential Neural Networks.
    """
    
    def __init__(
        self,
        model: EvidentialNeuralNetwork,
        device: torch.device = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        annealing_step: int = 10
    ):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.annealing_step = annealing_step
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Initialize scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_uncertainty': []
        }
        
        # Calibration metrics
        self.calibration_metrics = CalibrationMetrics()
    
    def train_epoch(
        self, 
        train_loader: DataLoader, 
        epoch: int
    ) -> Dict[str, float]:
        """
        Train the model for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        total_classification_loss = 0.0
        total_kl_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, (data, targets) in enumerate(progress_bar):
            data, targets = data.to(self.device), targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(data)
            
            # Compute evidential loss
            loss_dict = evidential_loss(
                outputs['alpha'],
                targets,
                epoch,
                self.model.num_classes,
                self.annealing_step
            )
            
            # Backward pass
            loss_dict['total_loss'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss_dict['total_loss'].item()
            total_classification_loss += loss_dict['classification_loss'].item()
            total_kl_loss += loss_dict['kl_loss'].item()
            
            # Compute accuracy
            predictions = torch.argmax(outputs['probabilities'], dim=1)
            correct_predictions += (predictions == targets).sum().item()
            total_samples += targets.size(0)
            
            # Update progress bar
            current_accuracy = correct_predictions / total_samples
            progress_bar.set_postfix({
                'Loss': f'{loss_dict["total_loss"].item():.4f}',
                'Acc': f'{current_accuracy:.4f}',
                'KL_coef': f'{loss_dict["annealing_coef"]:.3f}'
            })
        
        # Compute epoch metrics
        epoch_metrics = {
            'loss': total_loss / len(train_loader),
            'classification_loss': total_classification_loss / len(train_loader),
            'kl_loss': total_kl_loss / len(train_loader),
            'accuracy': correct_predictions / total_samples
        }
        
        return epoch_metrics
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        all_uncertainties = []
        all_predictions = []
        all_targets = []
        all_confidences = []
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model(data)
                
                # Compute loss
                loss_dict = evidential_loss(
                    outputs['alpha'],
                    targets,
                    epoch=100,  # Use high epoch for full KL weight
                    num_classes=self.model.num_classes,
                    annealing_step=self.annealing_step
                )
                
                total_loss += loss_dict['total_loss'].item()
                
                # Compute accuracy
                predictions = torch.argmax(outputs['probabilities'], dim=1)
                correct_predictions += (predictions == targets).sum().item()
                total_samples += targets.size(0)
                
                # Collect uncertainty metrics
                all_uncertainties.extend(outputs['total_uncertainty'].cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_confidences.extend(outputs['confidence'].cpu().numpy())
        
        # Compute validation metrics
        val_metrics = {
            'loss': total_loss / len(val_loader),
            'accuracy': correct_predictions / total_samples,
            'mean_uncertainty': np.mean(all_uncertainties),
            'mean_confidence': np.mean(all_confidences)
        }
        
        # Compute uncertainty quality metrics
        uncertainty_quality = self._evaluate_uncertainty_quality(
            np.array(all_predictions),
            np.array(all_targets),
            np.array(all_uncertainties),
            np.array(all_confidences)
        )
        val_metrics.update(uncertainty_quality)
        
        return val_metrics
    
    def _evaluate_uncertainty_quality(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        uncertainties: np.ndarray,
        confidences: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate the quality of uncertainty estimates.
        
        Args:
            predictions: Model predictions
            targets: Ground truth labels
            uncertainties: Uncertainty estimates
            confidences: Confidence estimates
            
        Returns:
            Dictionary with uncertainty quality metrics
        """
        # Compute correctness
        correct = (predictions == targets).astype(float)
        
        # Uncertainty-accuracy correlation (should be negative)
        uncertainty_correlation = np.corrcoef(uncertainties, correct)[0, 1]
        
        # Confidence-accuracy correlation (should be positive)
        confidence_correlation = np.corrcoef(confidences, correct)[0, 1]
        
        # Expected Calibration Error (ECE)
        ece = CalibrationMetrics.expected_calibration_error(
            confidences, correct, n_bins=10
        )
        
        return {
            'uncertainty_correlation': uncertainty_correlation,
            'confidence_correlation': confidence_correlation,
            'ece': ece
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        save_dir: str = None
    ) -> Dict[str, List[float]]:
        """
        Train the ENN model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
            save_dir: Directory to save model checkpoints
            
        Returns:
            Training history
        """
        logger.info(f"Starting ENN training for {num_epochs} epochs")
        
        best_val_loss = float('inf')
        
        for epoch in range(1, num_epochs + 1):
            # Train epoch
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_metrics['loss'])
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_accuracy'].append(train_metrics['accuracy'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_accuracy'].append(val_metrics['accuracy'])
            self.history['val_uncertainty'].append(val_metrics['mean_uncertainty'])
            
            # Log metrics
            logger.info(
                f"Epoch {epoch}/{num_epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.4f}, "
                f"Val Uncertainty: {val_metrics['mean_uncertainty']:.4f}, "
                f"ECE: {val_metrics['ece']:.4f}"
            )
            
            # Save best model
            if save_dir and val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                self.save_model(os.path.join(save_dir, 'best_enn_model.pth'))
        
        logger.info("ENN training completed")
        return self.history
    
    def save_model(self, filepath: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)
        logger.info(f"Model loaded from {filepath}")
