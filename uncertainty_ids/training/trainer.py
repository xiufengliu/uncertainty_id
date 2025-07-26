"""
Main trainer for uncertainty-aware intrusion detection.
Based on the training procedure described in Section 3.3 of the paper.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
import numpy as np
from tqdm import tqdm
import logging
import os
from pathlib import Path

from ..models.transformer import BayesianEnsembleTransformer
from ..models.uncertainty import UncertaintyQuantifier, TemperatureScaling
from .losses import CompositeLoss


class UncertaintyAwareTrainer:
    """
    Trainer for uncertainty-aware intrusion detection model.
    
    Implements the training procedure from Section 3.3 including:
    - Composite loss function (classification + diversity + uncertainty)
    - Temperature scaling for calibration
    - Early stopping and model checkpointing
    """
    
    def __init__(
        self,
        model: BayesianEnsembleTransformer,
        uncertainty_quantifier: UncertaintyQuantifier,
        device: torch.device,
        learning_rate: float = 1e-3,
        lambda_diversity: float = 0.1,
        lambda_uncertainty: float = 0.05,
        class_weights: Optional[torch.Tensor] = None,
        checkpoint_dir: str = "checkpoints"
    ):
        """
        Initialize trainer.
        
        Args:
            model: Bayesian ensemble transformer model
            uncertainty_quantifier: Uncertainty quantification module
            device: Training device (CPU/GPU)
            learning_rate: Learning rate (default 1e-3 as per paper)
            lambda_diversity: Diversity regularization weight (λ₁ = 0.1)
            lambda_uncertainty: Uncertainty regularization weight (λ₂ = 0.05)
            class_weights: Optional class weights for imbalanced datasets
            checkpoint_dir: Directory for saving checkpoints
        """
        self.model = model.to(device)
        self.uncertainty_quantifier = uncertainty_quantifier.to(device)
        self.device = device
        
        # Loss function
        self.criterion = CompositeLoss(
            lambda_diversity=lambda_diversity,
            lambda_uncertainty=lambda_uncertainty,
            class_weights=class_weights
        )
        
        # Optimizer (Adam as commonly used for transformers)
        self.optimizer = torch.optim.Adam(
            list(model.parameters()) + list(uncertainty_quantifier.parameters()),
            lr=learning_rate
        )
        
        # Temperature scaling for calibration
        self.temperature_scaler = TemperatureScaling()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Logging
        self.train_losses = []
        self.val_losses = []
        
        # Checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def train_epoch(self, train_loader: DataLoader, epoch: int = None) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader
            epoch: Optional epoch number for progress bar display

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        self.uncertainty_quantifier.train()

        # Update current epoch if provided
        if epoch is not None:
            self.current_epoch = epoch

        epoch_losses = []
        epoch_metrics = {
            'total_loss': 0.0,
            'ce_loss': 0.0,
            'diversity_loss': 0.0,
            'uncertainty_loss': 0.0
        }

        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, (cont_features, cat_features, labels) in enumerate(progress_bar):
            # Move to device
            cont_features = cont_features.to(self.device)
            cat_features = cat_features.to(self.device)
            labels = labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            ensemble_logits, _, individual_logits = self.model(
                cont_features, cat_features, return_individual=True
            )
            
            # Uncertainty quantification
            predictions, epistemic_unc, aleatoric_unc, total_unc, ensemble_probs = \
                self.uncertainty_quantifier(ensemble_logits, individual_logits)
            
            # Compute loss
            loss, loss_dict = self.criterion(
                ensemble_logits, individual_logits, labels, total_unc, predictions
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.uncertainty_quantifier.parameters(), max_norm=1.0)
            
            # Update parameters
            self.optimizer.step()
            
            # Update metrics
            for key, value in loss_dict.items():
                epoch_metrics[key] += value
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'CE': f"{loss_dict['ce_loss']:.4f}",
                'Div': f"{loss_dict['diversity_loss']:.4f}",
                'Unc': f"{loss_dict['uncertainty_loss']:.4f}"
            })
        
        # Average metrics over epoch
        num_batches = len(train_loader)
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        return epoch_metrics
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        self.uncertainty_quantifier.eval()
        
        val_metrics = {
            'total_loss': 0.0,
            'ce_loss': 0.0,
            'diversity_loss': 0.0,
            'uncertainty_loss': 0.0,
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0
        }
        
        all_predictions = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for cont_features, cat_features, labels in tqdm(val_loader, desc="Validation"):
                # Move to device
                cont_features = cont_features.to(self.device)
                cat_features = cat_features.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                ensemble_logits, _, individual_logits = self.model(
                    cont_features, cat_features, return_individual=True
                )
                
                # Uncertainty quantification
                predictions, epistemic_unc, aleatoric_unc, total_unc, ensemble_probs = \
                    self.uncertainty_quantifier(ensemble_logits, individual_logits)
                
                # Compute loss
                loss, loss_dict = self.criterion(
                    ensemble_logits, individual_logits, labels, total_unc, predictions
                )
                
                # Update metrics
                for key, value in loss_dict.items():
                    val_metrics[key] += value
                
                # Collect predictions for metrics
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(ensemble_probs.cpu().numpy())
        
        # Average losses
        num_batches = len(val_loader)
        for key in ['total_loss', 'ce_loss', 'diversity_loss', 'uncertainty_loss']:
            val_metrics[key] /= num_batches
        
        # Compute classification metrics
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        accuracy = (all_predictions == all_labels).mean()
        
        # Binary classification metrics
        tp = ((all_predictions == 1) & (all_labels == 1)).sum()
        fp = ((all_predictions == 1) & (all_labels == 0)).sum()
        fn = ((all_predictions == 0) & (all_labels == 1)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        val_metrics.update({
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        })
        
        return val_metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 100,
        patience: int = 10,
        save_best: bool = True
    ) -> Dict[str, List[float]]:
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Maximum number of epochs
            patience: Early stopping patience
            save_best: Whether to save best model
            
        Returns:
            Training history
        """
        self.logger.info(f"Starting training for {num_epochs} epochs")
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_f1_score': []
        }
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Training
            train_metrics = self.train_epoch(train_loader)
            
            # Validation
            val_metrics = self.validate(val_loader)
            
            # Update history
            history['train_loss'].append(train_metrics['total_loss'])
            history['val_loss'].append(val_metrics['total_loss'])
            history['val_accuracy'].append(val_metrics['accuracy'])
            history['val_f1_score'].append(val_metrics['f1_score'])
            
            # Logging
            self.logger.info(
                f"Epoch {epoch}: "
                f"Train Loss: {train_metrics['total_loss']:.4f}, "
                f"Val Loss: {val_metrics['total_loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.4f}, "
                f"Val F1: {val_metrics['f1_score']:.4f}"
            )
            
            # Early stopping and model saving
            if val_metrics['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['total_loss']
                self.patience_counter = 0
                
                if save_best:
                    self.save_checkpoint('best_model.pth', val_metrics)
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= patience:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break
        
        return history
    
    def save_checkpoint(self, filename: str, metrics: Optional[Dict] = None):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'uncertainty_quantifier_state_dict': self.uncertainty_quantifier.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'metrics': metrics
        }
        
        torch.save(checkpoint, self.checkpoint_dir / filename)
        self.logger.info(f"Checkpoint saved: {filename}")
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        checkpoint = torch.load(self.checkpoint_dir / filename, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.uncertainty_quantifier.load_state_dict(checkpoint['uncertainty_quantifier_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        
        self.logger.info(f"Checkpoint loaded: {filename}")
        
        return checkpoint.get('metrics', {})
