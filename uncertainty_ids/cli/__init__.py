"""
Command Line Interface for Uncertainty-Aware Intrusion Detection System.

This module provides command-line tools for training, evaluation,
preprocessing, and serving the uncertainty-aware IDS models.
"""

from .train import main as train_main
from .evaluate import main as evaluate_main
from .serve import main as serve_main
from .preprocess import main as preprocess_main

__all__ = [
    'train_main',
    'evaluate_main', 
    'serve_main',
    'preprocess_main',
]
