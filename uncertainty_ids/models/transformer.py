"""
Single-layer transformer implementation for uncertainty-aware intrusion detection.
Based on Section 3.2 Architecture Design and Table 1 in the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List
from .embedding import HeterogeneousEmbedding, PositionalEncoding


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention mechanism.
    Based on paper specification: 3 attention heads, d_model = 128.
    """
    
    def __init__(self, d_model: int = 128, n_heads: int = 3, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for multi-head self-attention.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Optional attention mask [batch_size, seq_len, seq_len]
            
        Returns:
            output: Attention output [batch_size, seq_len, d_model]
            attention_weights: Attention weights [batch_size, n_heads, seq_len, seq_len]
        """
        batch_size, seq_len, d_model = x.shape
        
        # Linear projections
        Q = self.w_q(x)  # [B, T, d_model]
        K = self.w_k(x)  # [B, T, d_model]
        V = self.w_v(x)  # [B, T, d_model]
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)  # [B, H, T, d_k]
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)  # [B, H, T, d_k]
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)  # [B, H, T, d_k]
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, H, T, T]
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention_weights = F.softmax(scores, dim=-1)  # [B, H, T, T]
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)  # [B, H, T, d_k]
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )  # [B, T, d_model]
        
        # Final linear projection
        output = self.w_o(context)
        
        return output, attention_weights


class FeedForwardNetwork(nn.Module):
    """
    Position-wise feed-forward network.
    Based on paper specification: d_ff = 4 * d_model.
    """
    
    def __init__(self, d_model: int = 128, d_ff: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
            
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for feed-forward network.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class SingleLayerTransformer(nn.Module):
    """
    Single-layer transformer block for intrusion detection.
    
    Based on Table 1 architecture specifications:
    - Input Embedding: d_input × d_model
    - Position Encoding: (T+1) × d_model  
    - Multi-Head Self-Attention: d_model × d_model (3 heads)
    - Feed-Forward Network: d_model × d_ff × d_model
    - Classification Head: d_model × 2
    - Total Parameters per model: ~0.2M
    """
    
    def __init__(
        self,
        continuous_features: List[str],
        categorical_features: List[str],
        categorical_vocab_sizes: dict,
        d_model: int = 128,
        n_heads: int = 3,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        max_seq_len: int = 51  # T+1 where T=50
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Embedding layer for heterogeneous features
        self.embedding = HeterogeneousEmbedding(
            continuous_features=continuous_features,
            categorical_features=categorical_features,
            categorical_vocab_sizes=categorical_vocab_sizes,
            d_model=d_model,
            dropout=dropout
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Multi-head self-attention
        self.attention = MultiHeadSelfAttention(d_model, n_heads, dropout)
        
        # Feed-forward network
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Classification head (d_model × 2 for binary classification)
        self.classifier = nn.Linear(d_model, 2)
        
    def forward(
        self, 
        x_cont: torch.Tensor, 
        x_cat: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for single-layer transformer.
        
        Args:
            x_cont: Continuous features [batch_size, seq_len, n_continuous]
            x_cat: Categorical features [batch_size, seq_len, n_categorical]
            mask: Optional attention mask [batch_size, seq_len, seq_len]
            return_attention: Whether to return attention weights
            
        Returns:
            logits: Classification logits [batch_size, 2]
            attention_weights: Optional attention weights [batch_size, n_heads, seq_len, seq_len]
        """
        # Embedding
        x = self.embedding(x_cont, x_cat)  # [B, T, d_model]
        
        # Add positional encoding
        x = self.pos_encoding(x)  # [B, T, d_model]
        
        # Multi-head self-attention with residual connection
        attn_output, attention_weights = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward network with residual connection
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        
        # Global average pooling for sequence-level representation
        # Take mean over sequence dimension (handles both seq_len=1 and seq_len>1)
        pooled = x.mean(dim=1)  # [B, d_model]
        
        # Classification
        logits = self.classifier(pooled)  # [B, 2]
        
        if return_attention:
            return logits, attention_weights
        else:
            return logits, None


class BayesianEnsembleTransformer(nn.Module):
    """
    Bayesian ensemble of single-layer transformers.

    Based on Section 3.2 and Equation: p_ensemble(x) = Σ w_m · p_m(x)
    where w_m are learned ensemble weights satisfying Σ w_m = 1.
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
        max_seq_len: int = 51
    ):
        super().__init__()

        self.ensemble_size = ensemble_size
        self.d_model = d_model

        # Create ensemble of transformers with different random initializations
        self.models = nn.ModuleList([
            SingleLayerTransformer(
                continuous_features=continuous_features,
                categorical_features=categorical_features,
                categorical_vocab_sizes=categorical_vocab_sizes,
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                max_seq_len=max_seq_len
            ) for _ in range(ensemble_size)
        ])

        # Learned ensemble weights (initialized to uniform)
        self.ensemble_weights = nn.Parameter(torch.ones(ensemble_size) / ensemble_size)

    def forward(
        self,
        x_cont: torch.Tensor,
        x_cat: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_individual: bool = False,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[torch.Tensor]]]:
        """
        Forward pass for Bayesian ensemble.

        Args:
            x_cont: Continuous features [batch_size, seq_len, n_continuous]
            x_cat: Categorical features [batch_size, seq_len, n_categorical]
            mask: Optional attention mask
            return_individual: Whether to return individual model predictions
            return_attention: Whether to return attention weights

        Returns:
            ensemble_logits: Ensemble prediction [batch_size, 2]
            attention_weights: Optional attention weights from all models
            individual_logits: Optional individual model predictions
        """
        individual_logits = []
        attention_weights_list = []

        # Get predictions from each model
        for model in self.models:
            logits, attn_weights = model(x_cont, x_cat, mask, return_attention)
            individual_logits.append(logits)
            if return_attention:
                attention_weights_list.append(attn_weights)

        # Stack individual predictions
        individual_logits = torch.stack(individual_logits, dim=0)  # [M, B, 2]

        # Normalize ensemble weights to sum to 1
        normalized_weights = F.softmax(self.ensemble_weights, dim=0)  # [M]

        # Weighted ensemble prediction
        ensemble_logits = torch.sum(
            normalized_weights.view(-1, 1, 1) * individual_logits, dim=0
        )  # [B, 2]

        # Prepare return values
        attention_weights = attention_weights_list if return_attention else None
        individual_preds = individual_logits if return_individual else None

        return ensemble_logits, attention_weights, individual_preds
