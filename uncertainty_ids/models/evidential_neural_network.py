"""
Evidential Neural Networks (ENN) for uncertainty-aware intrusion detection.

This module implements Evidential Neural Networks that learn a Dirichlet distribution
over class probabilities, providing principled uncertainty quantification through
evidential learning.

References:
- Sensoy et al. "Evidential Deep Learning to Quantify Classification Uncertainty" (NeurIPS 2018)
- Amini et al. "Deep Evidential Regression" (NeurIPS 2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class EvidentialLayer(nn.Module):
    """
    Evidential layer that outputs Dirichlet parameters (evidence).
    """
    
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.linear = nn.Linear(input_dim, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute evidence (Dirichlet parameters).
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            evidence: Evidence tensor of shape (batch_size, num_classes)
        """
        # Use exponential activation to ensure positive evidence
        evidence = torch.exp(self.linear(x))
        return evidence


class EvidentialNeuralNetwork(nn.Module):
    """
    Evidential Neural Network for uncertainty-aware classification.
    
    The network learns to output evidence (Dirichlet parameters) rather than
    class probabilities, enabling principled uncertainty quantification.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [128, 64],
        num_classes: int = 2,
        dropout_rate: float = 0.1,
        activation: str = 'relu'
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Build feature extraction layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                self._get_activation(activation),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
            
        self.feature_extractor = nn.Sequential(*layers)
        
        # Evidential output layer
        self.evidential_layer = EvidentialLayer(prev_dim, num_classes)
        
        # Initialize weights
        self._initialize_weights()
        
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU(),
            'gelu': nn.GELU()
        }
        return activations.get(activation, nn.ReLU())
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the evidential neural network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Dictionary containing:
                - evidence: Dirichlet parameters (batch_size, num_classes)
                - alpha: Concentration parameters (batch_size, num_classes)
                - probabilities: Expected probabilities (batch_size, num_classes)
                - uncertainty: Total uncertainty (batch_size,)
                - aleatoric_uncertainty: Data uncertainty (batch_size,)
                - epistemic_uncertainty: Model uncertainty (batch_size,)
        """
        # Extract features
        features = self.feature_extractor(x)
        
        # Get evidence (Dirichlet parameters)
        evidence = self.evidential_layer(features)
        
        # Compute concentration parameters (alpha = evidence + 1)
        alpha = evidence + 1.0
        
        # Compute Dirichlet strength (sum of concentration parameters)
        strength = torch.sum(alpha, dim=1, keepdim=True)
        
        # Expected probabilities under Dirichlet distribution
        probabilities = alpha / strength
        
        # Uncertainty quantification
        uncertainty_dict = self._compute_uncertainties(alpha, strength)
        
        return {
            'evidence': evidence,
            'alpha': alpha,
            'probabilities': probabilities,
            'strength': strength.squeeze(-1),
            **uncertainty_dict
        }
    
    def _compute_uncertainties(
        self, 
        alpha: torch.Tensor, 
        strength: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute different types of uncertainties from Dirichlet parameters.
        
        Args:
            alpha: Concentration parameters (batch_size, num_classes)
            strength: Dirichlet strength (batch_size, 1)
            
        Returns:
            Dictionary with uncertainty measures
        """
        batch_size = alpha.size(0)
        
        # Expected probabilities
        probs = alpha / strength
        
        # Total uncertainty (entropy of expected probabilities)
        total_uncertainty = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        
        # Aleatoric uncertainty (expected entropy)
        # E[H[p]] where p ~ Dir(alpha)
        digamma_sum = torch.digamma(strength.squeeze(-1))
        digamma_alpha = torch.digamma(alpha)
        aleatoric_uncertainty = torch.sum(
            (alpha / strength) * (digamma_sum.unsqueeze(-1) - digamma_alpha), 
            dim=1
        )
        
        # Epistemic uncertainty (mutual information)
        epistemic_uncertainty = total_uncertainty - aleatoric_uncertainty
        
        # Confidence (inverse of total uncertainty)
        confidence = 1.0 / (1.0 + total_uncertainty)
        
        return {
            'total_uncertainty': total_uncertainty,
            'aleatoric_uncertainty': aleatoric_uncertainty,
            'epistemic_uncertainty': epistemic_uncertainty,
            'confidence': confidence
        }
    
    def predict_with_uncertainty(
        self, 
        x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Make predictions with uncertainty estimates.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary with predictions and uncertainties
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            
            # Get predicted classes
            predictions = torch.argmax(outputs['probabilities'], dim=1)
            
            # Get prediction confidence (max probability)
            max_probs = torch.max(outputs['probabilities'], dim=1)[0]
            
            return {
                'predictions': predictions,
                'probabilities': outputs['probabilities'],
                'confidence': outputs['confidence'],
                'max_probability': max_probs,
                'total_uncertainty': outputs['total_uncertainty'],
                'aleatoric_uncertainty': outputs['aleatoric_uncertainty'],
                'epistemic_uncertainty': outputs['epistemic_uncertainty'],
                'evidence': outputs['evidence'],
                'alpha': outputs['alpha']
            }


def evidential_loss(
    alpha: torch.Tensor,
    targets: torch.Tensor,
    epoch: int,
    num_classes: int,
    annealing_step: int = 10
) -> Dict[str, torch.Tensor]:
    """
    Compute evidential loss for training ENN.
    
    Args:
        alpha: Concentration parameters (batch_size, num_classes)
        targets: Ground truth labels (batch_size,)
        epoch: Current training epoch
        num_classes: Number of classes
        annealing_step: KL annealing step
        
    Returns:
        Dictionary with loss components
    """
    # Convert targets to one-hot
    targets_one_hot = F.one_hot(targets, num_classes).float()
    
    # Dirichlet strength
    S = torch.sum(alpha, dim=1, keepdim=True)
    
    # Expected log-likelihood (classification loss)
    expected_probs = alpha / S
    classification_loss = torch.sum(
        targets_one_hot * (torch.digamma(S) - torch.digamma(alpha)), 
        dim=1
    )
    
    # KL divergence regularization
    # KL(Dir(alpha) || Dir(1, 1, ..., 1))
    alpha_hat = targets_one_hot + (1 - targets_one_hot) * alpha
    S_hat = torch.sum(alpha_hat, dim=1, keepdim=True)
    
    kl_loss = torch.lgamma(S_hat) - torch.sum(torch.lgamma(alpha_hat), dim=1, keepdim=True) + \
              torch.sum((alpha_hat - 1) * (torch.digamma(alpha_hat) - torch.digamma(S_hat)), dim=1, keepdim=True)
    
    # Annealing coefficient
    annealing_coef = min(1.0, epoch / annealing_step)
    
    # Total loss
    total_loss = torch.mean(classification_loss + annealing_coef * kl_loss.squeeze(-1))
    
    return {
        'total_loss': total_loss,
        'classification_loss': torch.mean(classification_loss),
        'kl_loss': torch.mean(kl_loss),
        'annealing_coef': annealing_coef
    }
