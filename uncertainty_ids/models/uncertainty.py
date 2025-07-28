"""
Uncertainty quantification and calibration for intrusion detection.
Based on Section 3.4 and Algorithm 2 (Uncertainty-Aware Prediction) in the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
from scipy.optimize import minimize_scalar


class UncertaintyQuantifier(nn.Module):
    """
    Uncertainty quantification module implementing epistemic and aleatoric uncertainty decomposition.
    
    Based on Algorithm 2: Uncertainty-Aware Prediction
    """
    
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        # Temperature parameter for calibration (learnable)
        self.temperature = nn.Parameter(torch.tensor(temperature))
        
    def forward(
        self,
        ensemble_logits: torch.Tensor,
        individual_logits: torch.Tensor,
        tau_base: float = 0.5,
        alpha: float = 0.1
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute uncertainty-aware predictions.
        
        Implements Algorithm 2 from the paper:
        1. Apply temperature scaling to individual logits
        2. Compute ensemble mean probability
        3. Compute epistemic uncertainty (model disagreement)
        4. Compute aleatoric uncertainty (data uncertainty)
        5. Compute total uncertainty and adaptive threshold
        
        Args:
            ensemble_logits: Ensemble logits [batch_size, 2]
            individual_logits: Individual model logits [ensemble_size, batch_size, 2]
            tau_base: Base classification threshold (default 0.5)
            alpha: Sensitivity hyperparameter for uncertainty contribution
            
        Returns:
            predictions: Final binary predictions [batch_size]
            epistemic_uncertainty: Epistemic uncertainty [batch_size]
            aleatoric_uncertainty: Aleatoric uncertainty [batch_size]
            total_uncertainty: Total uncertainty [batch_size]
            ensemble_probs: Ensemble probabilities [batch_size, 2]
        """
        ensemble_size, batch_size, num_classes = individual_logits.shape
        
        # Apply temperature scaling to individual logits (Step 5 in Algorithm 2)
        scaled_logits = individual_logits / self.temperature.clamp(min=1e-8)
        individual_probs = F.softmax(scaled_logits, dim=-1)  # [M, B, 2]
        
        # Compute ensemble mean probability (Step 6 in Algorithm 2)
        ensemble_probs = individual_probs.mean(dim=0)  # [B, 2]
        
        # Focus on positive class probability for binary classification
        individual_pos_probs = individual_probs[:, :, 1]  # [M, B]
        ensemble_pos_prob = ensemble_probs[:, 1]  # [B]
        
        # Compute epistemic uncertainty (Step 7 in Algorithm 2)
        # σ_epistemic^2 = (1/M) Σ (p_m - p̄)^2
        epistemic_variance = ((individual_pos_probs - ensemble_pos_prob.unsqueeze(0)) ** 2).mean(dim=0)
        epistemic_uncertainty = torch.sqrt(epistemic_variance + 1e-8)  # [B]
        
        # Compute aleatoric uncertainty (Step 8 in Algorithm 2)
        # σ_aleatoric^2 = (1/M) Σ p_m(1 - p_m)
        aleatoric_variance = (individual_pos_probs * (1 - individual_pos_probs)).mean(dim=0)
        aleatoric_uncertainty = torch.sqrt(aleatoric_variance + 1e-8)  # [B]
        
        # Compute total uncertainty (Step 9 in Algorithm 2)
        # σ_total^2 = σ_epistemic^2 + σ_aleatoric^2
        total_uncertainty = torch.sqrt(epistemic_variance + aleatoric_variance + 1e-8)  # [B]
        
        # Determine adaptive threshold (Step 10 in Algorithm 2)
        # τ = τ_base - α · σ_total
        adaptive_threshold = tau_base - alpha * total_uncertainty  # [B]
        adaptive_threshold = torch.clamp(adaptive_threshold, min=0.1, max=0.9)  # Reasonable bounds
        
        # Make final prediction (Step 11 in Algorithm 2)
        # ŷ = I[p̄ > τ]
        predictions = (ensemble_pos_prob > adaptive_threshold).long()  # [B]
        
        return predictions, epistemic_uncertainty, aleatoric_uncertainty, total_uncertainty, ensemble_probs


class TemperatureScaling(nn.Module):
    """
    Temperature scaling for calibration.
    Based on Algorithm 3: Uncertainty Calibration in the paper.
    """
    
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))
        
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply temperature scaling to logits.
        
        Args:
            logits: Raw logits [batch_size, num_classes]
            
        Returns:
            Calibrated probabilities [batch_size, num_classes]
        """
        return F.softmax(logits / self.temperature, dim=-1)
    
    def calibrate(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 50
    ) -> float:
        """
        Calibrate temperature parameter on validation set.
        
        Implements Algorithm 3 from the paper:
        Minimize negative log-likelihood: L_cal(T) = -Σ [y_i log σ(z̄_i/T) + (1-y_i) log(1-σ(z̄_i/T))]
        
        Args:
            logits: Validation logits [n_samples, num_classes]
            labels: True labels [n_samples]
            lr: Learning rate for temperature optimization
            max_iter: Maximum optimization iterations
            
        Returns:
            Optimal temperature value
        """
        # Convert to numpy for scipy optimization
        logits_np = logits.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        
        def calibration_loss(temperature):
            """Negative log-likelihood loss for temperature scaling."""
            if temperature <= 0:
                return 1e6  # Invalid temperature
                
            # Apply temperature scaling
            scaled_logits = logits_np / temperature
            probs = np.exp(scaled_logits) / np.sum(np.exp(scaled_logits), axis=1, keepdims=True)
            
            # Compute negative log-likelihood
            # For binary classification, use positive class probability
            pos_probs = probs[:, 1] if probs.shape[1] == 2 else probs[np.arange(len(labels_np)), labels_np]
            pos_probs = np.clip(pos_probs, 1e-8, 1 - 1e-8)  # Numerical stability
            
            if probs.shape[1] == 2:  # Binary classification
                nll = -(labels_np * np.log(pos_probs) + (1 - labels_np) * np.log(1 - pos_probs))
            else:  # Multi-class
                nll = -np.log(pos_probs)
                
            return np.mean(nll)
        
        # Optimize temperature
        result = minimize_scalar(calibration_loss, bounds=(0.1, 10.0), method='bounded')
        optimal_temp = result.x
        
        # Update parameter
        self.temperature.data = torch.tensor(optimal_temp)
        
        return optimal_temp


def expected_calibration_error(
    probs: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 10
) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    Args:
        probs: Predicted probabilities [n_samples, num_classes]
        labels: True labels [n_samples]
        n_bins: Number of bins for calibration
        
    Returns:
        ECE value
    """
    # Get confidence (max probability) and predictions
    confidences, predictions = torch.max(probs, dim=1)
    accuracies = predictions.eq(labels)
    
    ece = 0.0
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Find samples in this bin
        in_bin = confidences.gt(bin_lower.item()) & confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece.item()
