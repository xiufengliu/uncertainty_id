"""
Heterogeneous embedding layer for network flow features.
Based on Section 3.2 Architecture Design in the paper.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional


class HeterogeneousEmbedding(nn.Module):
    """
    Specialized embedding function that processes heterogeneous network flow data.
    
    Implements Equation: φ(x) = Concat(φ_cont(x_cont), φ_cat(x_cat))
    where φ_cont applies linear projection to continuous features after normalization,
    while φ_cat employs learned embeddings for categorical features.
    """
    
    def __init__(
        self,
        continuous_features: List[str],
        categorical_features: List[str],
        categorical_vocab_sizes: Dict[str, int],
        d_model: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.continuous_features = continuous_features
        self.categorical_features = categorical_features
        self.d_model = d_model
        
        # Continuous feature processing (only if we have continuous features)
        self.n_continuous = len(continuous_features)
        if self.n_continuous > 0:
            self.continuous_projection = nn.Linear(self.n_continuous, d_model // 2)
            self.continuous_norm = nn.LayerNorm(self.n_continuous)
        else:
            self.continuous_projection = None
            self.continuous_norm = None
        
        # Categorical feature embeddings
        self.categorical_embeddings = nn.ModuleDict()
        for feature, vocab_size in categorical_vocab_sizes.items():
            if feature in categorical_features:
                # Each categorical feature gets embedding_dim dimensions
                embedding_dim = max(4, min(50, vocab_size // 2))
                self.categorical_embeddings[feature] = nn.Embedding(vocab_size, embedding_dim)
        
        # Calculate total categorical embedding dimension
        total_cat_dim = sum(
            self.categorical_embeddings[feat].embedding_dim
            for feat in categorical_features
        )

        # Project concatenated categorical embeddings to d_model // 2 (only if we have categorical features)
        if total_cat_dim > 0:
            self.categorical_projection = nn.Linear(total_cat_dim, d_model // 2)
        else:
            self.categorical_projection = None
        
        # Final projection to d_model
        self.final_projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x_cont: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for heterogeneous embedding.

        Args:
            x_cont: Continuous features [batch_size, seq_len, n_continuous] or [batch_size, n_continuous]
            x_cat: Categorical features [batch_size, seq_len, n_categorical] or [batch_size, n_categorical]

        Returns:
            Embedded features [batch_size, seq_len, d_model] or [batch_size, 1, d_model]
        """
        # Handle both tabular and sequence data
        if x_cont.dim() == 2:
            # Tabular data: add sequence dimension
            x_cont = x_cont.unsqueeze(1)  # [batch_size, 1, n_continuous]
            x_cat = x_cat.unsqueeze(1)    # [batch_size, 1, n_categorical]

        batch_size, seq_len = x_cont.shape[:2]

        # Process continuous features (if any)
        if self.n_continuous > 0 and x_cont.shape[-1] > 0:
            # Handle dynamic input size
            if x_cont.shape[-1] != self.n_continuous:
                # If input size doesn't match, create a new projection layer
                if not hasattr(self, '_dynamic_continuous_projection'):
                    self._dynamic_continuous_projection = nn.Linear(x_cont.shape[-1], self.d_model // 2).to(x_cont.device)
                    self._dynamic_continuous_norm = nn.LayerNorm(x_cont.shape[-1]).to(x_cont.device)

                x_cont_norm = self._dynamic_continuous_norm(x_cont)
                cont_embedded = self._dynamic_continuous_projection(x_cont_norm)
            else:
                x_cont_norm = self.continuous_norm(x_cont)
                cont_embedded = self.continuous_projection(x_cont_norm)  # [B, T, d_model//2]
        else:
            # Create zero tensor for continuous features if none exist
            cont_embedded = torch.zeros(batch_size, seq_len, self.d_model // 2,
                                      device=x_cont.device, dtype=x_cont.dtype)
        
        # Process categorical features
        cat_embeddings = []
        for i, feature in enumerate(self.categorical_features):
            # Extract feature column and embed
            feat_indices = x_cat[:, :, i].long()  # [B, T]
            feat_embedded = self.categorical_embeddings[feature](feat_indices)  # [B, T, embed_dim]
            cat_embeddings.append(feat_embedded)
        
        # Concatenate all categorical embeddings
        if cat_embeddings and self.categorical_projection is not None:
            cat_concat = torch.cat(cat_embeddings, dim=-1)  # [B, T, total_cat_dim]
            cat_embedded = self.categorical_projection(cat_concat)  # [B, T, d_model//2]
        else:
            cat_embedded = torch.zeros(batch_size, seq_len, self.d_model // 2,
                                     device=x_cont.device, dtype=x_cont.dtype)
        
        # Concatenate continuous and categorical embeddings
        combined = torch.cat([cont_embedded, cat_embedded], dim=-1)  # [B, T, d_model]
        
        # Final projection and dropout
        output = self.final_projection(combined)
        output = self.dropout(output)
        
        return output


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer input.
    Based on standard transformer positional encoding.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Ensure we don't go beyond d_model for odd dimensions
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           (-np.log(10000.0) / d_model))

        # Handle odd d_model by only using available dimensions
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            # Only apply cosine to available dimensions
            cos_dim = min(len(div_term), d_model // 2 + d_model % 2)
            pe[:, 1::2][:, :cos_dim] = torch.cos(position * div_term[:cos_dim])

        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Input embeddings [batch_size, seq_len, d_model]
            
        Returns:
            Position-encoded embeddings [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].transpose(0, 1)
