"""
Visualization utilities for uncertainty-aware intrusion detection.
All plots are saved in PDF format for publication quality.
"""

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import torch

from .plotting_config import configure_plotting_for_pdf, COLORS, MARKERS, LINESTYLES

# Configure plotting for PDF output
configure_plotting_for_pdf()


def save_figure(fig: plt.Figure, filepath: str, **kwargs) -> None:
    """
    Save figure in PDF format with consistent settings.
    
    Args:
        fig: Matplotlib figure object
        filepath: Path to save the figure (will add .pdf extension if missing)
        **kwargs: Additional arguments for savefig
    """
    # Ensure PDF extension
    filepath = Path(filepath)
    if filepath.suffix != '.pdf':
        filepath = filepath.with_suffix('.pdf')
    
    # Create directory if it doesn't exist
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Default save parameters for publication quality
    save_params = {
        'format': 'pdf',
        'dpi': 300,
        'bbox_inches': 'tight',
        'pad_inches': 0.1,
        'facecolor': 'white',
        'edgecolor': 'none'
    }
    save_params.update(kwargs)
    
    fig.savefig(filepath, **save_params)
    print(f"Figure saved to: {filepath}")


def plot_training_history(history: Dict[str, List[float]], 
                         output_path: str = "results/training_history.pdf") -> None:
    """
    Plot training history with loss and metrics over epochs.
    
    Args:
        history: Dictionary containing training history
        output_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Training History', fontsize=16, fontweight='bold')
    
    # Plot training and validation loss
    if 'train_loss' in history and 'val_loss' in history:
        axes[0, 0].plot(history['train_loss'], label='Training Loss', linewidth=2)
        axes[0, 0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # Plot accuracy
    if 'train_acc' in history and 'val_acc' in history:
        axes[0, 1].plot(history['train_acc'], label='Training Accuracy', linewidth=2)
        axes[0, 1].plot(history['val_acc'], label='Validation Accuracy', linewidth=2)
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot F1 score
    if 'train_f1' in history and 'val_f1' in history:
        axes[1, 0].plot(history['train_f1'], label='Training F1', linewidth=2)
        axes[1, 0].plot(history['val_f1'], label='Validation F1', linewidth=2)
        axes[1, 0].set_title('F1 Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot uncertainty metrics
    if 'train_ece' in history and 'val_ece' in history:
        axes[1, 1].plot(history['train_ece'], label='Training ECE', linewidth=2)
        axes[1, 1].plot(history['val_ece'], label='Validation ECE', linewidth=2)
        axes[1, 1].set_title('Expected Calibration Error')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('ECE')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         class_names: Optional[List[str]] = None,
                         output_path: str = "results/confusion_matrix.pdf") -> None:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes
        output_path: Path to save the plot
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, cbar_kws={'label': 'Count'})
    
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    
    plt.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)


def plot_uncertainty_distribution(uncertainties: np.ndarray, 
                                 correctness: np.ndarray,
                                 output_path: str = "results/uncertainty_distribution.pdf") -> None:
    """
    Plot uncertainty distributions for correct vs incorrect predictions.
    
    Args:
        uncertainties: Array of uncertainty values
        correctness: Array of correctness (1 for correct, 0 for incorrect)
        output_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Separate uncertainties by correctness
    correct_uncertainties = uncertainties[correctness == 1]
    incorrect_uncertainties = uncertainties[correctness == 0]
    
    # Plot histograms
    ax.hist(correct_uncertainties, bins=50, alpha=0.7, label='Correct Predictions', 
            color='blue', density=True, edgecolor='black', linewidth=0.5)
    ax.hist(incorrect_uncertainties, bins=50, alpha=0.7, label='Incorrect Predictions', 
            color='red', density=True, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Uncertainty', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Uncertainty Distribution: Correct vs Incorrect Predictions', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    correct_mean = np.mean(correct_uncertainties)
    incorrect_mean = np.mean(incorrect_uncertainties)
    ax.axvline(correct_mean, color='blue', linestyle='--', alpha=0.8, 
               label=f'Correct Mean: {correct_mean:.3f}')
    ax.axvline(incorrect_mean, color='red', linestyle='--', alpha=0.8, 
               label=f'Incorrect Mean: {incorrect_mean:.3f}')
    
    plt.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)


def plot_calibration_diagram(y_true: np.ndarray, y_prob: np.ndarray, 
                           n_bins: int = 10,
                           output_path: str = "results/calibration_diagram.pdf") -> None:
    """
    Plot reliability diagram for calibration analysis.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_bins: Number of bins for calibration
        output_path: Path to save the plot
    """
    from sklearn.calibration import calibration_curve
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Reliability diagram
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy='uniform'
    )
    
    ax1.plot(mean_predicted_value, fraction_of_positives, "s-", 
             linewidth=2, markersize=8, label='Model')
    ax1.plot([0, 1], [0, 1], "k:", label='Perfect Calibration')
    ax1.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax1.set_ylabel('Fraction of Positives', fontsize=12)
    ax1.set_title('Reliability Diagram', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Confidence histogram
    ax2.hist(y_prob, bins=n_bins, alpha=0.7, color='skyblue', 
             edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Predicted Probability', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Confidence Histogram', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)


def plot_ensemble_size_analysis(ensemble_sizes: List[int], 
                               f1_scores: List[float], 
                               ece_scores: List[float],
                               f1_stds: Optional[List[float]] = None,
                               ece_stds: Optional[List[float]] = None,
                               output_path: str = "results/ensemble_size_analysis.pdf") -> None:
    """
    Plot ensemble size analysis showing F1 vs ECE trade-off.
    
    Args:
        ensemble_sizes: List of ensemble sizes
        f1_scores: List of F1 scores
        ece_scores: List of ECE scores
        f1_stds: Standard deviations for F1 scores
        ece_stds: Standard deviations for ECE scores
        output_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # F1 Score vs Ensemble Size
    if f1_stds is not None:
        ax1.errorbar(ensemble_sizes, f1_scores, yerr=f1_stds, 
                     marker='o', linewidth=2, markersize=8, capsize=5)
    else:
        ax1.plot(ensemble_sizes, f1_scores, 'o-', linewidth=2, markersize=8)
    
    ax1.set_xlabel('Ensemble Size', fontsize=12)
    ax1.set_ylabel('F1 Score', fontsize=12)
    ax1.set_title('F1 Score vs Ensemble Size', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # ECE vs Ensemble Size
    if ece_stds is not None:
        ax2.errorbar(ensemble_sizes, ece_scores, yerr=ece_stds, 
                     marker='s', linewidth=2, markersize=8, capsize=5, color='red')
    else:
        ax2.plot(ensemble_sizes, ece_scores, 's-', linewidth=2, markersize=8, color='red')
    
    ax2.set_xlabel('Ensemble Size', fontsize=12)
    ax2.set_ylabel('Expected Calibration Error', fontsize=12)
    ax2.set_title('ECE vs Ensemble Size', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)


def plot_convergence_analysis(epochs: List[int], 
                            empirical_loss: List[float],
                            theoretical_bound: List[float],
                            output_path: str = "results/convergence_analysis.pdf") -> None:
    """
    Plot convergence analysis comparing empirical vs theoretical rates.
    
    Args:
        epochs: List of epoch numbers
        empirical_loss: Empirical training loss
        theoretical_bound: Theoretical convergence bound
        output_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.semilogy(epochs, empirical_loss, 'b-', linewidth=2, label='Empirical Loss')
    ax.semilogy(epochs, theoretical_bound, 'r--', linewidth=2, label='Theoretical Bound')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (log scale)', fontsize=12)
    ax.set_title('Convergence Analysis: Empirical vs Theoretical', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)
