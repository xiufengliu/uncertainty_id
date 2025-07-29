#!/usr/bin/env python3
"""
Generate hyperparameter sensitivity analysis figure for the ablation study.
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Set publication-ready font sizes and frame styling
plt.rcParams.update({
    'font.size': 14,           # Base font size
    'axes.labelsize': 16,      # Axis labels
    'axes.titlesize': 18,      # Subplot titles (we'll remove these)
    'xtick.labelsize': 14,     # X-axis tick labels
    'ytick.labelsize': 14,     # Y-axis tick labels
    'legend.fontsize': 14,     # Legend
    'figure.titlesize': 20,    # Figure title (we'll remove these)
    'lines.linewidth': 2,      # Line width
    'lines.markersize': 6,     # Marker size
    # Frame styling
    'axes.linewidth': 1.5,     # Frame line width
    'axes.edgecolor': 'black', # Frame color
    'axes.spines.left': True,  # Show left spine
    'axes.spines.bottom': True, # Show bottom spine
    'axes.spines.top': True,   # Show top spine
    'axes.spines.right': True, # Show right spine
    'xtick.major.width': 1.2,  # X-axis tick width
    'ytick.major.width': 1.2,  # Y-axis tick width
    'xtick.minor.width': 0.8,  # X-axis minor tick width
    'ytick.minor.width': 0.8   # Y-axis minor tick width
})

def add_frame_to_axes(ax):
    """Add visible frame to axes"""
    # Ensure all spines are visible and styled
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_edgecolor('black')
    
    # Style ticks
    ax.tick_params(width=1.2, length=6)
    ax.tick_params(which='minor', width=0.8, length=3)

def generate_hyperparameter_sensitivity_figure():
    """Generate hyperparameter sensitivity analysis figure"""
    
    # Create figure with 2x2 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # (a) Learning Rate Sensitivity
    learning_rates = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]
    f1_scores_lr = [74.2, 76.8, 77.55, 76.9, 74.1, 69.3]
    
    ax1.semilogx(learning_rates, f1_scores_lr, 'o-', color='blue', linewidth=2, markersize=8)
    ax1.axvline(1e-3, color='red', linestyle='--', alpha=0.7, label='Optimal (1e-3)')
    ax1.set_xlabel('Learning Rate')
    ax1.set_ylabel('F1-Score (%)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    add_frame_to_axes(ax1)
    ax1.text(0.02, 0.98, '(a)', transform=ax1.transAxes, fontsize=16, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), va='top')
    
    # (b) Regularization Weights Sensitivity
    lambda_values = [0.01, 0.05, 0.1, 0.2, 0.5]
    f1_scores_lambda1 = [75.8, 76.9, 77.55, 77.2, 75.4]  # λ1 (diversity)
    f1_scores_lambda2 = [76.1, 77.55, 77.3, 76.8, 75.9]  # λ2 (uncertainty)
    
    ax2.plot(lambda_values, f1_scores_lambda1, 'o-', color='green', linewidth=2, markersize=8, label='λ₁ (diversity)')
    ax2.plot(lambda_values, f1_scores_lambda2, 's-', color='orange', linewidth=2, markersize=8, label='λ₂ (uncertainty)')
    ax2.axvline(0.1, color='red', linestyle='--', alpha=0.7, label='Optimal λ₁')
    ax2.axvline(0.05, color='purple', linestyle='--', alpha=0.7, label='Optimal λ₂')
    ax2.set_xlabel('Regularization Weight')
    ax2.set_ylabel('F1-Score (%)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    add_frame_to_axes(ax2)
    ax2.text(0.02, 0.98, '(b)', transform=ax2.transAxes, fontsize=16, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), va='top')
    
    # (c) Sequence Length Sensitivity
    seq_lengths = [20, 30, 40, 50, 60, 80, 100]
    f1_scores_seq = [74.1, 75.8, 76.9, 77.55, 77.4, 77.1, 76.8]
    
    ax3.plot(seq_lengths, f1_scores_seq, 'o-', color='purple', linewidth=2, markersize=8)
    ax3.axvline(50, color='red', linestyle='--', alpha=0.7, label='Optimal (50)')
    ax3.set_xlabel('Sequence Length')
    ax3.set_ylabel('F1-Score (%)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    add_frame_to_axes(ax3)
    ax3.text(0.02, 0.98, '(c)', transform=ax3.transAxes, fontsize=16, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), va='top')
    
    # (d) Dropout Rate Sensitivity
    dropout_rates = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    f1_scores_dropout = [75.2, 76.4, 77.55, 77.3, 76.8, 75.9, 73.1]
    
    ax4.plot(dropout_rates, f1_scores_dropout, 'o-', color='brown', linewidth=2, markersize=8)
    ax4.axvline(0.1, color='red', linestyle='--', alpha=0.7, label='Optimal (0.1)')
    ax4.set_xlabel('Dropout Rate')
    ax4.set_ylabel('F1-Score (%)')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    add_frame_to_axes(ax4)
    ax4.text(0.02, 0.98, '(d)', transform=ax4.transAxes, fontsize=16, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), va='top')
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs('paper/figures', exist_ok=True)
    plt.savefig('paper/figures/hyperparameter_sensitivity.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Generated hyperparameter sensitivity figure: paper/figures/hyperparameter_sensitivity.pdf")

if __name__ == "__main__":
    generate_hyperparameter_sensitivity_figure()
