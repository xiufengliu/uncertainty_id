"""
Main trainer class for uncertainty-aware intrusion detection models.

This module provides a comprehensive training framework with support for
uncertainty quantification, ensemble training, and calibration.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import json
from tqdm import tqdm

from ..models import BayesianEnsembleIDS, SingleLayerTransformerIDS
from ..utils import UncertaintyCalibrator, ModelCheckpoint, EarlyStopping
from ..evaluation import ComprehensiveEvaluator

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    
    # Model configuration
    model_type: str = 'bayesian_ensemble'
    model_params: Dict[str, Any] = field(default_factory=lambda: {
        'n_ensemble': 10,
        'd_model': 128,
        'max_seq_len': 50,
        'n_classes': 2,
        'dropout_rate': 0.1
    })
    
    # Training parameters
    batch_size: int = 64
    n_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 1.0
    
    # Optimization
    optimizer: str = 'adam'
    scheduler: str = 'reduce_on_plateau'
    scheduler_params: Dict[str, Any] = field(default_factory=lambda: {
        'mode': 'min',
        'factor': 0.5,
        'patience': 10,
        'verbose': True
    })
    
    # Regularization
    ensemble_diversity_weight: float = 0.01
    uncertainty_regularization_weight: float = 0.01
    
    # Early stopping
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 1e-4
    
    # Checkpointing
    save_best_model: bool = True
    save_checkpoint_every: int = 10
    checkpoint_dir: str = 'checkpoints'
    
    # Calibration
    calibrate_uncertainty: bool = True
    calibration_method: str = 'temperature'
    
    # Logging
    log_every: int = 10
    validate_every: int = 1
    
    # Device
    device: str = 'auto'  # 'auto', 'cpu', 'cuda'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'model_type': self.model_type,
            'model_params': self.model_params,
            'batch_size': self.batch_size,
            'n_epochs': self.n_epochs,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'gradient_clip_norm': self.gradient_clip_norm,
            'optimizer': self.optimizer,
            'scheduler': self.scheduler,
            'scheduler_params': self.scheduler_params,
            'ensemble_diversity_weight': self.ensemble_diversity_weight,
            'uncertainty_regularization_weight': self.uncertainty_regularization_weight,
            'early_stopping_patience': self.early_stopping_patience,
            'early_stopping_min_delta': self.early_stopping_min_delta,
            'save_best_model': self.save_best_model,
            'save_checkpoint_every': self.save_checkpoint_every,
            'checkpoint_dir': self.checkpoint_dir,
            'calibrate_uncertainty': self.calibrate_uncertainty,
            'calibration_method': self.calibration_method,
            'log_every': self.log_every,
            'validate_every': self.validate_every,
            'device': self.device,
        }


class UncertaintyIDSTrainer:
    """
    Comprehensive trainer for uncertainty-aware intrusion detection models.
    
    Supports training of both single models and Bayesian ensembles with
    uncertainty quantification, calibration, and comprehensive evaluation.
    """
    
    def __init__(self, config: Union[TrainingConfig, Dict[str, Any]]):
        """
        Initialize trainer.
        
        Args:
            config: Training configuration
        """
        if isinstance(config, dict):
            self.config = TrainingConfig(**config)
        else:
            self.config = config
        
        # Set device
        self.device = self._get_device()
        
        # Initialize model
        self.model = self._create_model()
        
        # Initialize training components
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.CrossEntropyLoss()
        
        # Initialize utilities
        self.calibrator = UncertaintyCalibrator(method=self.config.calibration_method)
        self.evaluator = ComprehensiveEvaluator()
        self.checkpoint_manager = ModelCheckpoint(
            save_dir=self.config.checkpoint_dir,
            save_best=self.config.save_best_model
        )
        self.early_stopping = EarlyStopping(
            patience=self.config.early_stopping_patience,
            min_delta=self.config.early_stopping_min_delta
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_f1_score': [],
            'val_uncertainty': [],
            'val_calibration_error': [],
            'learning_rate': []
        }
        
        logger.info(f"UncertaintyIDSTrainer initialized with {self.config.model_type} on {self.device}")
    
    def _get_device(self) -> torch.device:
        """Get training device."""
        if self.config.device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                return torch.device('cpu')
        else:
            return torch.device(self.config.device)
    
    def _create_model(self):
        """Create model based on configuration."""
        if self.config.model_type == 'bayesian_ensemble':
            model = BayesianEnsembleIDS(**self.config.model_params)
        elif self.config.model_type == 'single_transformer':
            model = SingleLayerTransformerIDS(**self.config.model_params)
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")
        
        return model.to(self.device)
    
    def _create_optimizer(self):
        """Create optimizer."""
        if self.config.optimizer == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        if self.config.scheduler == 'reduce_on_plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, **self.config.scheduler_params
            )
        elif self.config.scheduler == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config.n_epochs
            )
        elif self.config.scheduler == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.1
            )
        elif self.config.scheduler == 'none':
            self.scheduler = None
        else:
            raise ValueError(f"Unknown scheduler: {self.config.scheduler}")
    
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            
        Returns:
            Training history dictionary
        """
        logger.info("Starting model training...")
        
        # Create optimizer and scheduler
        self._create_optimizer()
        self._create_scheduler()
        
        # Create checkpoint directory
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        # Training loop
        for epoch in range(self.config.n_epochs):
            epoch_start_time = time.time()
            
            # Training phase
            train_metrics = self._train_epoch(train_loader, epoch)
            
            # Validation phase
            val_metrics = {}
            if val_loader is not None and epoch % self.config.validate_every == 0:
                val_metrics = self._validate_epoch(val_loader, epoch)
            
            # Update learning rate
            if self.scheduler is not None:
                if self.config.scheduler == 'reduce_on_plateau' and val_metrics:
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            if val_metrics:
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_accuracy'].append(val_metrics['accuracy'])
                self.history['val_f1_score'].append(val_metrics.get('f1_score', 0.0))
                self.history['val_uncertainty'].append(val_metrics.get('avg_uncertainty', 0.0))
                self.history['val_calibration_error'].append(val_metrics.get('calibration_error', 0.0))
            
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            # Checkpointing
            if epoch % self.config.save_checkpoint_every == 0:
                self.checkpoint_manager.save(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    metrics=val_metrics or train_metrics,
                    config=self.config.to_dict()
                )
            
            # Early stopping
            if val_metrics and self.early_stopping.should_stop(val_metrics['loss']):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
            
            # Logging
            if epoch % self.config.log_every == 0:
                epoch_time = time.time() - epoch_start_time
                log_msg = f"Epoch {epoch}/{self.config.n_epochs} ({epoch_time:.2f}s) - "
                log_msg += f"Train Loss: {train_metrics['loss']:.4f}"
                
                if val_metrics:
                    log_msg += f", Val Loss: {val_metrics['loss']:.4f}"
                    log_msg += f", Val Acc: {val_metrics['accuracy']:.4f}"
                    if 'avg_uncertainty' in val_metrics:
                        log_msg += f", Val Uncertainty: {val_metrics['avg_uncertainty']:.4f}"
                
                logger.info(log_msg)
        
        # Final calibration
        if self.config.calibrate_uncertainty and val_loader is not None:
            logger.info("Performing final uncertainty calibration...")
            self._calibrate_model(val_loader)
        
        # Save final model
        self.checkpoint_manager.save(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=epoch,
            metrics=val_metrics or train_metrics,
            config=self.config.to_dict(),
            is_final=True
        )
        
        logger.info("Training completed!")
        return self.history
    
    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        n_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
        
        for batch_idx, (sequences, queries, targets) in enumerate(pbar):
            sequences = sequences.to(self.device)
            queries = queries.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            if isinstance(self.model, BayesianEnsembleIDS):
                logits, epistemic_uncertainty = self.model(sequences, queries)
                
                # Compute loss with uncertainty regularization
                classification_loss = self.criterion(logits, targets)
                
                # Ensemble diversity regularization
                diversity_loss = self._compute_diversity_loss()
                
                # Uncertainty regularization
                uncertainty_reg = torch.mean(epistemic_uncertainty) * self.config.uncertainty_regularization_weight
                
                total_loss_batch = (
                    classification_loss + 
                    diversity_loss * self.config.ensemble_diversity_weight +
                    uncertainty_reg
                )
            else:
                logits, _ = self.model(sequences, queries)
                total_loss_batch = self.criterion(logits, targets)
            
            # Backward pass
            total_loss_batch.backward()
            
            # Gradient clipping
            if self.config.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.gradient_clip_norm
                )
            
            self.optimizer.step()
            
            total_loss += total_loss_batch.item()
            n_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': total_loss_batch.item()})
        
        return {'loss': total_loss / n_batches}
    
    def _validate_epoch(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        
        all_predictions = []
        all_probabilities = []
        all_uncertainties = []
        all_confidences = []
        all_targets = []
        total_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False)
            
            for sequences, queries, targets in pbar:
                sequences = sequences.to(self.device)
                queries = queries.to(self.device)
                targets = targets.to(self.device)
                
                # Get predictions with uncertainty
                results = self.model.predict_with_uncertainty(sequences, queries)
                
                # Compute loss
                if isinstance(self.model, BayesianEnsembleIDS):
                    logits, _ = self.model(sequences, queries)
                else:
                    logits, _ = self.model(sequences, queries)
                
                loss = self.criterion(logits, targets)
                total_loss += loss.item()
                n_batches += 1
                
                # Collect results
                all_predictions.extend(results['predictions'].cpu().numpy())
                all_probabilities.extend(results['probabilities'].cpu().numpy())
                all_confidences.extend(results['confidence'].cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
                if 'total_uncertainty' in results:
                    all_uncertainties.extend(results['total_uncertainty'].cpu().numpy())
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        all_confidences = np.array(all_confidences)
        all_targets = np.array(all_targets)
        
        # Compute metrics
        accuracy = (all_predictions == all_targets).mean()
        
        # Compute F1 score
        from sklearn.metrics import f1_score
        f1 = f1_score(all_targets, all_predictions, average='weighted', zero_division=0)
        
        metrics = {
            'loss': total_loss / n_batches,
            'accuracy': accuracy,
            'f1_score': f1,
        }
        
        # Add uncertainty metrics if available
        if all_uncertainties:
            all_uncertainties = np.array(all_uncertainties)
            metrics['avg_uncertainty'] = np.mean(all_uncertainties)
            
            # Compute calibration error
            correctness = (all_predictions == all_targets).astype(float)
            calibration_error = self.calibrator.compute_calibration_error(
                all_confidences, correctness
            )['expected_calibration_error']
            metrics['calibration_error'] = calibration_error
        
        return metrics
    
    def _compute_diversity_loss(self) -> torch.Tensor:
        """Compute ensemble diversity loss."""
        if not isinstance(self.model, BayesianEnsembleIDS):
            return torch.tensor(0.0, device=self.device)
        
        # Simple diversity loss based on parameter differences
        diversity_loss = 0.0
        n_pairs = 0
        
        for i in range(len(self.model.ensemble_models)):
            for j in range(i + 1, len(self.model.ensemble_models)):
                model_i = self.model.ensemble_models[i]
                model_j = self.model.ensemble_models[j]
                
                # Compute parameter similarity
                similarity = 0.0
                n_params = 0
                
                for p1, p2 in zip(model_i.parameters(), model_j.parameters()):
                    if p1.shape == p2.shape:
                        similarity += torch.sum((p1 - p2) ** 2)
                        n_params += p1.numel()
                
                if n_params > 0:
                    diversity_loss += torch.exp(-similarity / n_params)
                    n_pairs += 1
        
        return diversity_loss / n_pairs if n_pairs > 0 else torch.tensor(0.0, device=self.device)
    
    def _calibrate_model(self, val_loader: DataLoader):
        """Calibrate model uncertainty on validation data."""
        self.model.eval()
        
        all_logits = []
        all_targets = []
        
        with torch.no_grad():
            for sequences, queries, targets in val_loader:
                sequences = sequences.to(self.device)
                queries = queries.to(self.device)
                targets = targets.to(self.device)
                
                if isinstance(self.model, BayesianEnsembleIDS):
                    logits, _ = self.model(sequences, queries)
                else:
                    logits, _ = self.model(sequences, queries)
                
                all_logits.append(logits)
                all_targets.append(targets)
        
        all_logits = torch.cat(all_logits, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Fit calibrator
        self.calibrator.fit(all_logits, all_targets)
        
        # Apply calibration to ensemble if applicable
        if isinstance(self.model, BayesianEnsembleIDS):
            # Update temperature parameter
            if hasattr(self.calibrator.calibrator, 'temperature'):
                self.model.temperature.data = self.calibrator.calibrator.temperature.data
    
    def save_model(self, filepath: str, include_optimizer: bool = False):
        """Save trained model."""
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'model_config': self.config.model_params,
            'training_config': self.config.to_dict(),
            'history': self.history,
            'model_class': self.model.__class__.__name__,
        }
        
        if include_optimizer and self.optimizer is not None:
            save_dict['optimizer_state_dict'] = self.optimizer.state_dict()
        
        if hasattr(self, 'calibrator') and self.calibrator.fitted:
            save_dict['calibrator'] = self.calibrator
        
        torch.save(save_dict, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str, load_optimizer: bool = False):
        """Load trained model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load training history
        if 'history' in checkpoint:
            self.history = checkpoint['history']
        
        # Load optimizer if requested
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            if self.optimizer is None:
                self._create_optimizer()
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load calibrator
        if 'calibrator' in checkpoint:
            self.calibrator = checkpoint['calibrator']
        
        logger.info(f"Model loaded from {filepath}")
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, Any]:
        """Evaluate model on test data."""
        logger.info("Evaluating model...")
        
        self.model.eval()
        
        all_predictions = []
        all_probabilities = []
        all_uncertainties = []
        all_confidences = []
        all_targets = []
        
        with torch.no_grad():
            for sequences, queries, targets in tqdm(test_loader, desc="Evaluating"):
                sequences = sequences.to(self.device)
                queries = queries.to(self.device)
                
                results = self.model.predict_with_uncertainty(sequences, queries)
                
                all_predictions.extend(results['predictions'].cpu().numpy())
                all_probabilities.extend(results['probabilities'].cpu().numpy())
                all_confidences.extend(results['confidence'].cpu().numpy())
                all_targets.extend(targets.numpy())
                
                if 'total_uncertainty' in results:
                    all_uncertainties.extend(results['total_uncertainty'].cpu().numpy())
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        all_confidences = np.array(all_confidences)
        all_targets = np.array(all_targets)
        
        # Comprehensive evaluation
        evaluation_results = self.evaluator.evaluate_model(
            y_true=all_targets,
            y_pred=all_predictions,
            y_prob=all_probabilities,
            uncertainties=np.array(all_uncertainties) if all_uncertainties else None,
            confidences=all_confidences
        )
        
        logger.info("Evaluation completed!")
        return evaluation_results
