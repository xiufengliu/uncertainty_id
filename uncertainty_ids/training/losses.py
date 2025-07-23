"""
Loss functions for uncertainty-aware intrusion detection.
Based on Section 3.3 Training Procedure in the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class DiversityLoss(nn.Module):
    """
    Diversity regularization loss for ensemble training.
    
    Based on Equation: L_diversity = -(1/(M(M-1))) Σ KL(p_m || p_m')
    Promotes diversity among ensemble members to prevent mode collapse.
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(self, individual_probs: torch.Tensor) -> torch.Tensor:
        """
        Compute diversity loss.
        
        Args:
            individual_probs: Individual model probabilities [ensemble_size, batch_size, num_classes]
            
        Returns:
            Diversity loss value
        """
        ensemble_size, batch_size, num_classes = individual_probs.shape
        
        if ensemble_size <= 1:
            return torch.tensor(0.0, device=individual_probs.device)
        
        total_kl = 0.0
        count = 0
        
        # Compute pairwise KL divergences
        for m in range(ensemble_size):
            for m_prime in range(ensemble_size):
                if m != m_prime:
                    # KL(p_m || p_m')
                    p_m = individual_probs[m] + 1e-8  # Add small epsilon for numerical stability
                    p_m_prime = individual_probs[m_prime] + 1e-8
                    
                    kl_div = F.kl_div(
                        torch.log(p_m_prime), p_m, reduction='batchmean'
                    )
                    total_kl += kl_div
                    count += 1
        
        # Average over all pairs and negate (we want to maximize diversity)
        diversity_loss = -total_kl / count if count > 0 else torch.tensor(0.0)
        
        return diversity_loss


class UncertaintyLoss(nn.Module):
    """
    Uncertainty regularization loss.
    
    Based on the formulation in Section 3.3:
    L_uncertainty,i = {
        σ_total(x_i)     if y_i ≠ ŷ_i (misclassified)
        1 - σ_total(x_i) if y_i = ŷ_i (correctly classified)
    }
    
    Encourages higher uncertainty for misclassified samples and lower uncertainty for correct ones.
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(
        self,
        total_uncertainty: torch.Tensor,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute uncertainty regularization loss.
        
        Args:
            total_uncertainty: Total uncertainty estimates [batch_size]
            predictions: Model predictions [batch_size]
            targets: True labels [batch_size]
            
        Returns:
            Uncertainty loss value
        """
        # Determine correct/incorrect predictions
        correct_mask = (predictions == targets).float()
        incorrect_mask = 1.0 - correct_mask
        
        # Apply uncertainty loss formulation
        uncertainty_loss = (
            incorrect_mask * total_uncertainty +  # Higher uncertainty for incorrect
            correct_mask * (1.0 - total_uncertainty)  # Lower uncertainty for correct
        )
        
        return uncertainty_loss.mean()


class CompositeLoss(nn.Module):
    """
    Composite loss function combining classification, diversity, and uncertainty losses.
    
    Based on Equation: L_total = L_CE + λ₁ L_diversity + λ₂ L_uncertainty
    """
    
    def __init__(
        self,
        lambda_diversity: float = 0.1,
        lambda_uncertainty: float = 0.05,
        class_weights: Optional[torch.Tensor] = None
    ):
        """
        Initialize composite loss.
        
        Args:
            lambda_diversity: Weight for diversity regularization (λ₁)
            lambda_uncertainty: Weight for uncertainty regularization (λ₂)
            class_weights: Optional class weights for imbalanced datasets
        """
        super().__init__()
        
        self.lambda_diversity = lambda_diversity
        self.lambda_uncertainty = lambda_uncertainty
        
        # Classification loss
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        
        # Regularization losses
        self.diversity_loss = DiversityLoss()
        self.uncertainty_loss = UncertaintyLoss()
        
    def forward(
        self,
        ensemble_logits: torch.Tensor,
        individual_logits: torch.Tensor,
        targets: torch.Tensor,
        total_uncertainty: torch.Tensor,
        predictions: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute composite loss.
        
        Args:
            ensemble_logits: Ensemble predictions [batch_size, num_classes]
            individual_logits: Individual model predictions [ensemble_size, batch_size, num_classes]
            targets: True labels [batch_size]
            total_uncertainty: Total uncertainty estimates [batch_size]
            predictions: Hard predictions [batch_size]
            
        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary of individual loss components
        """
        # Classification loss
        ce_loss = self.ce_loss(ensemble_logits, targets)
        
        # Convert logits to probabilities for diversity loss
        individual_probs = F.softmax(individual_logits, dim=-1)
        diversity_loss = self.diversity_loss(individual_probs)
        
        # Uncertainty loss
        uncertainty_loss = self.uncertainty_loss(total_uncertainty, predictions, targets)
        
        # Total loss
        total_loss = (
            ce_loss + 
            self.lambda_diversity * diversity_loss + 
            self.lambda_uncertainty * uncertainty_loss
        )
        
        # Loss dictionary for logging
        loss_dict = {
            'total_loss': total_loss.item(),
            'ce_loss': ce_loss.item(),
            'diversity_loss': diversity_loss.item(),
            'uncertainty_loss': uncertainty_loss.item()
        }
        
        return total_loss, loss_dict


# Alias for backward compatibility
UncertaintyAwareLoss = CompositeLoss


class ICLRegularizationLoss(nn.Module):
    """
    ICL regularization loss for meta-learning.
    
    Based on the ICL regularization term in Algorithm 1:
    ICL_Regularization = ||Attention(x_q, C) - GradientStep(x_q, C)||²
    """
    
    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight
        
    def forward(
        self,
        attention_weights: torch.Tensor,
        context_features: torch.Tensor,
        query_features: torch.Tensor,
        context_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute ICL regularization loss.
        
        This is a simplified implementation. In practice, this would involve
        computing explicit gradient steps and comparing with attention patterns.
        
        Args:
            attention_weights: Attention weights [batch_size, n_heads, seq_len, seq_len]
            context_features: Context features
            query_features: Query features  
            context_labels: Context labels
            
        Returns:
            ICL regularization loss
        """
        if attention_weights is None:
            return torch.tensor(0.0)
        
        # Simplified: encourage attention to focus on relevant context examples
        # In full implementation, this would compare attention with gradient-based updates
        
        batch_size, n_heads, seq_len, _ = attention_weights.shape
        
        # Focus on query-to-context attention (last position to context positions)
        query_attention = attention_weights[:, :, -1, :-1]  # [B, H, context_len]
        
        # Target: uniform attention as baseline (could be improved with gradient computation)
        target_attention = torch.ones_like(query_attention) / query_attention.shape[-1]
        
        # L2 loss
        icl_loss = F.mse_loss(query_attention, target_attention)
        
        return self.weight * icl_loss
