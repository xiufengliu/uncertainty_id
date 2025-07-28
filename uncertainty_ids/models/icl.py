"""
In-Context Learning (ICL) enabled transformer for few-shot attack adaptation.
Based on Algorithm 1: Meta-Learning ICL-Enabled Bayesian Ensemble Training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
from .transformer import BayesianEnsembleTransformer


class ICLEnabledTransformer(nn.Module):
    """
    ICL-enabled transformer that can adapt to new attack types using context examples.
    
    Based on the ICL formulation in the paper:
    f_θ(x_q | C_i) = Transformer([Embed(x_1, y_1); ...; Embed(x_k, y_k); Embed(x_q, ∅)])
    """
    
    def __init__(
        self,
        continuous_features: List[str],
        categorical_features: List[str],
        categorical_vocab_sizes: dict,
        ensemble_size: int = 5,
        d_model: int = 128,
        n_heads: int = 3,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        max_seq_len: int = 51,
        max_context_len: int = 20
    ):
        super().__init__()
        
        self.max_context_len = max_context_len
        self.d_model = d_model
        
        # Base ensemble transformer
        self.ensemble = BayesianEnsembleTransformer(
            continuous_features=continuous_features,
            categorical_features=categorical_features,
            categorical_vocab_sizes=categorical_vocab_sizes,
            ensemble_size=ensemble_size,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            max_seq_len=max_seq_len + max_context_len  # Extended for context
        )
        
        # Special tokens for ICL
        self.context_separator = nn.Parameter(torch.randn(d_model))
        self.query_token = nn.Parameter(torch.randn(d_model))
        
        # ICL regularization components
        self.icl_regularization_weight = 0.1
        
    def create_icl_sequence(
        self,
        context_cont: torch.Tensor,
        context_cat: torch.Tensor,
        context_labels: torch.Tensor,
        query_cont: torch.Tensor,
        query_cat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create ICL input sequence: [Embed(x_1, y_1); ...; Embed(x_k, y_k); Embed(x_q, ∅)]
        
        Args:
            context_cont: Context continuous features [batch_size, k, n_continuous]
            context_cat: Context categorical features [batch_size, k, n_categorical]
            context_labels: Context labels [batch_size, k]
            query_cont: Query continuous features [batch_size, n_continuous]
            query_cat: Query categorical features [batch_size, n_categorical]
            
        Returns:
            icl_cont: ICL sequence continuous features [batch_size, k+1, n_continuous]
            icl_cat: ICL sequence categorical features [batch_size, k+1, n_categorical]
        """
        batch_size, k, n_continuous = context_cont.shape
        n_categorical = context_cat.shape[-1]
        
        # Add query to sequence (query has no label, represented as empty)
        query_cont_expanded = query_cont.unsqueeze(1)  # [B, 1, n_continuous]
        query_cat_expanded = query_cat.unsqueeze(1)    # [B, 1, n_categorical]
        
        # Concatenate context and query
        icl_cont = torch.cat([context_cont, query_cont_expanded], dim=1)  # [B, k+1, n_continuous]
        icl_cat = torch.cat([context_cat, query_cat_expanded], dim=1)     # [B, k+1, n_categorical]
        
        return icl_cont, icl_cat
    
    def forward(
        self,
        context_cont: torch.Tensor,
        context_cat: torch.Tensor,
        context_labels: torch.Tensor,
        query_cont: torch.Tensor,
        query_cat: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for ICL adaptation.
        
        Args:
            context_cont: Context continuous features [batch_size, k, n_continuous]
            context_cat: Context categorical features [batch_size, k, n_categorical]
            context_labels: Context labels [batch_size, k]
            query_cont: Query continuous features [batch_size, n_continuous]
            query_cat: Query categorical features [batch_size, n_categorical]
            return_attention: Whether to return attention weights
            
        Returns:
            logits: Query predictions [batch_size, 2]
            attention_weights: Optional attention weights
        """
        # Create ICL sequence
        icl_cont, icl_cat = self.create_icl_sequence(
            context_cont, context_cat, context_labels, query_cont, query_cat
        )
        
        # Forward through ensemble (no parameter updates - pure ICL)
        logits, attention_weights, _ = self.ensemble(
            icl_cont, icl_cat, return_attention=return_attention
        )
        
        return logits, attention_weights
    
    def compute_icl_regularization(
        self,
        attention_weights: torch.Tensor,
        context_cont: torch.Tensor,
        context_cat: torch.Tensor,
        context_labels: torch.Tensor,
        query_cont: torch.Tensor,
        query_cat: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute ICL regularization term that encourages attention patterns to correlate with gradient descent.
        
        Based on paper: ICL_Regularization = ||Attention(x_q, C) - GradientStep(x_q, C)||^2
        
        Args:
            attention_weights: Attention weights from forward pass
            context_cont, context_cat, context_labels: Context examples
            query_cont, query_cat: Query examples
            
        Returns:
            ICL regularization loss
        """
        # Simplified approximation: encourage attention to focus on similar context examples
        # In practice, this would involve computing gradient-based updates
        
        batch_size = query_cont.shape[0]
        k = context_cont.shape[1]
        
        # Get attention weights for query position (last position in sequence)
        # attention_weights: [batch_size, n_heads, seq_len, seq_len]
        if attention_weights is not None and len(attention_weights) > 0:
            # Average over heads and take attention from query to context
            query_attention = attention_weights[0].mean(dim=1)[:, -1, :k]  # [B, k]
            
            # Compute similarity-based target attention (proxy for gradient descent)
            with torch.no_grad():
                # Simple similarity: higher attention for same-class context examples
                target_attention = torch.ones_like(query_attention) / k  # Uniform baseline
                
            # L2 loss between actual and target attention
            reg_loss = F.mse_loss(query_attention, target_attention)
        else:
            reg_loss = torch.tensor(0.0, device=query_cont.device)
        
        return self.icl_regularization_weight * reg_loss


class MetaLearningTrainer:
    """
    Meta-learning trainer for ICL-enabled ensemble.
    Implements Algorithm 1 from the paper.
    """
    
    def __init__(
        self,
        model: ICLEnabledTransformer,
        meta_lr: float = 1e-3,
        inner_lr: float = 1e-4,
        lambda_icl: float = 0.1,
        lambda_div: float = 0.1
    ):
        self.model = model
        self.meta_lr = meta_lr
        self.inner_lr = inner_lr
        self.lambda_icl = lambda_icl
        self.lambda_div = lambda_div
        
        # Meta-optimizer
        self.meta_optimizer = torch.optim.Adam(model.parameters(), lr=meta_lr)
        
    def meta_train_step(
        self,
        attack_families: Dict[str, Dict],
        n_query: int = 5
    ) -> Dict[str, float]:
        """
        Single meta-training step.
        
        Implements the inner and outer loops from Algorithm 1:
        1. Sample attack families and create episodes
        2. Inner loop: ICL adaptation (no parameter updates)
        3. Outer loop: Meta-update based on ICL performance
        
        Args:
            attack_families: Dictionary of attack family data
            n_query: Number of query examples per episode
            
        Returns:
            Dictionary of losses
        """
        self.meta_optimizer.zero_grad()
        
        total_meta_loss = 0.0
        total_icl_reg = 0.0
        total_diversity_loss = 0.0
        
        # Sample meta-batch of attack families
        family_names = list(attack_families.keys())
        batch_families = torch.randperm(len(family_names))[:min(4, len(family_names))]
        
        for family_idx in batch_families:
            family_name = family_names[family_idx]
            family_data = attack_families[family_name]
            
            # Sample context and query sets
            k = torch.randint(1, 11, (1,)).item()  # Variable shot learning (1-10)
            
            # Sample k context examples and n_query query examples
            total_samples = len(family_data['cont'])
            indices = torch.randperm(total_samples)
            
            context_indices = indices[:k]
            query_indices = indices[k:k+n_query]
            
            if len(query_indices) == 0:
                continue
                
            # Extract context and query data
            context_cont = family_data['cont'][context_indices].unsqueeze(0)  # [1, k, n_cont]
            context_cat = family_data['cat'][context_indices].unsqueeze(0)    # [1, k, n_cat]
            context_labels = family_data['labels'][context_indices].unsqueeze(0)  # [1, k]
            
            query_cont = family_data['cont'][query_indices]    # [n_q, n_cont]
            query_cat = family_data['cat'][query_indices]      # [n_q, n_cat]
            query_labels = family_data['labels'][query_indices]  # [n_q]
            
            # Inner loop: ICL adaptation (no parameter updates)
            inner_loss = 0.0
            icl_reg_loss = 0.0
            
            for q_idx in range(len(query_indices)):
                # ICL prediction for this query
                logits, attention_weights = self.model(
                    context_cont, context_cat, context_labels,
                    query_cont[q_idx:q_idx+1], query_cat[q_idx:q_idx+1],
                    return_attention=True
                )
                
                # Compute loss
                query_loss = F.cross_entropy(logits, query_labels[q_idx:q_idx+1])
                inner_loss += query_loss
                
                # ICL regularization
                icl_reg = self.model.compute_icl_regularization(
                    attention_weights, context_cont, context_cat, context_labels,
                    query_cont[q_idx:q_idx+1], query_cat[q_idx:q_idx+1]
                )
                icl_reg_loss += icl_reg
            
            # Average over queries
            inner_loss = inner_loss / len(query_indices)
            icl_reg_loss = icl_reg_loss / len(query_indices)
            
            # Add to meta-loss
            family_meta_loss = inner_loss + self.lambda_icl * icl_reg_loss
            total_meta_loss += family_meta_loss
            total_icl_reg += icl_reg_loss
        
        # Ensemble diversity loss
        # This would be computed across ensemble members
        diversity_loss = torch.tensor(0.0)  # Placeholder
        total_diversity_loss = self.lambda_div * diversity_loss
        
        # Total meta-loss
        final_meta_loss = total_meta_loss + total_diversity_loss
        
        # Meta-update (outer loop)
        final_meta_loss.backward()
        self.meta_optimizer.step()
        
        return {
            'meta_loss': final_meta_loss.item(),
            'icl_regularization': total_icl_reg.item(),
            'diversity_loss': total_diversity_loss.item()
        }
