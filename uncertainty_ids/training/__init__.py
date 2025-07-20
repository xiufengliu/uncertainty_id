"""
Training module for Uncertainty-Aware Intrusion Detection System.

This module provides comprehensive training capabilities including:
- Model trainers with uncertainty quantification
- Training loops with early stopping and checkpointing
- Hyperparameter optimization
- Distributed training support
"""

from .trainer import UncertaintyIDSTrainer, TrainingConfig
from .loops import TrainingLoop, ValidationLoop
from .optimizers import create_optimizer, create_scheduler
from .losses import UncertaintyLoss, EnsembleLoss, CalibrationLoss

__all__ = [
    'UncertaintyIDSTrainer',
    'TrainingConfig',
    'TrainingLoop',
    'ValidationLoop',
    'create_optimizer',
    'create_scheduler',
    'UncertaintyLoss',
    'EnsembleLoss',
    'CalibrationLoss',
]
