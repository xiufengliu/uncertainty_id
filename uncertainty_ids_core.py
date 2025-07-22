"""
Uncertainty-Aware Intrusion Detection System
Core Implementation based on "Training Convergence of Transformers for In-Context Classification"

This implementation adapts the paper's single-layer transformer framework for 
network intrusion detection with Bayesian ensemble uncertainty quantification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
from collections import deque
import time

@dataclass
class NetworkFlow:
    """Network flow representation with standard intrusion detection features"""
    duration: float
    protocol_type: int  # 0: tcp, 1: udp, 2: icmp
    service: int        # Service type (0-69)
    flag: int          # Connection flag (0-10)
    src_bytes: int
    dst_bytes: int
    land: int          # 1 if connection is from/to same host/port
    wrong_fragment: int
    urgent: int
    hot: int           # Number of "hot" indicators
    num_failed_logins: int
    logged_in: int     # 1 if successfully logged in
    num_compromised: int
    root_shell: int    # 1 if root shell obtained
    su_attempted: int  # 1 if "su root" command attempted
    num_root: int      # Number of "root" accesses
    num_file_creations: int
    num_shells: int    # Number of shell prompts
    num_access_files: int
    num_outbound_cmds: int
    is_host_login: int
    is_guest_login: int
    count: int         # Number of connections to same host in past 2 seconds
    srv_count: int     # Number of connections to same service in past 2 seconds
    serror_rate: float # % of connections with "SYN" errors
    srv_serror_rate: float
    rerror_rate: float # % of connections with "REJ" errors
    srv_rerror_rate: float
    same_srv_rate: float
    diff_srv_rate: float
    srv_diff_host_rate: float
    dst_host_count: int
    dst_host_srv_count: int
    dst_host_same_srv_rate: float
    dst_host_diff_srv_rate: float
    dst_host_same_src_port_rate: float
    dst_host_srv_diff_host_rate: float
    dst_host_serror_rate: float
    dst_host_srv_serror_rate: float
    dst_host_rerror_rate: float
    dst_host_srv_rerror_rate: float

class NetworkFeatureEmbedding(nn.Module):
    """
    Embedding layer for network traffic features
    Converts raw network features into transformer-compatible embeddings
    """
    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Categorical feature embeddings
        self.protocol_embed = nn.Embedding(4, 8)   # tcp, udp, icmp, other
        self.service_embed = nn.Embedding(70, 16)  # 70 different services
        self.flag_embed = nn.Embedding(11, 8)     # 11 different flags
        
        # Continuous feature normalization
        self.continuous_norm = nn.BatchNorm1d(38)  # 38 continuous features
        
        # Final projection to embedding dimension
        self.projection = nn.Linear(38 + 8 + 16 + 8, embed_dim)
        
    def forward(self, flows: torch.Tensor) -> torch.Tensor:
        """
        Args:
            flows: Tensor of shape (batch_size, 41) containing network flow features
        Returns:
            embeddings: Tensor of shape (batch_size, embed_dim)
        """
        batch_size = flows.shape[0]
        
        # Extract categorical features
        protocol = self.protocol_embed(flows[:, 1].long())
        service = self.service_embed(flows[:, 2].long())
        flag = self.flag_embed(flows[:, 3].long())
        
        # Extract and normalize continuous features
        continuous_features = torch.cat([
            flows[:, 0:1],    # duration
            flows[:, 4:]      # remaining continuous features
        ], dim=1)
        
        continuous_normalized = self.continuous_norm(continuous_features)
        
        # Combine all features
        combined = torch.cat([continuous_normalized, protocol, service, flag], dim=1)
        
        # Project to embedding dimension
        embeddings = self.projection(combined)
        
        return embeddings

class SingleLayerTransformerIDS(nn.Module):
    """
    Single-layer transformer adapted for intrusion detection
    Based on the paper's simplified transformer architecture
    """
    def __init__(self, d_model: int = 128, max_seq_len: int = 100, n_classes: int = 2):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.n_classes = n_classes
        
        # Feature embedding
        self.feature_embedding = NetworkFeatureEmbedding(d_model)
        
        # Transformer parameters (sparse structure from paper)
        self.W_V = nn.Parameter(torch.randn(d_model, d_model) * 0.02)
        self.W_KQ = nn.Parameter(torch.randn(d_model, d_model) * 0.02)
        
        # Normalization factor (set to sequence length as in paper)
        self.register_buffer('rho', torch.tensor(max_seq_len, dtype=torch.float32))
        
        # Classification head
        self.classifier = nn.Linear(d_model, n_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, network_prompt: torch.Tensor, query_flow: torch.Tensor) -> torch.Tensor:
        """
        Args:
            network_prompt: Tensor of shape (batch_size, seq_len, 41) - historical flows
            query_flow: Tensor of shape (batch_size, 41) - current flow to classify
        Returns:
            logits: Tensor of shape (batch_size, n_classes)
        """
        batch_size, seq_len, _ = network_prompt.shape
        
        # Embed all flows
        prompt_embeddings = self.feature_embedding(network_prompt.view(-1, 41))
        prompt_embeddings = prompt_embeddings.view(batch_size, seq_len, self.d_model)
        
        query_embedding = self.feature_embedding(query_flow).unsqueeze(1)
        
        # Combine prompt and query
        E = torch.cat([prompt_embeddings, query_embedding], dim=1)  # (batch, seq_len+1, d_model)
        E = E.transpose(1, 2)  # (batch, d_model, seq_len+1)
        
        # Apply transformer operation (Equation 2 from paper)
        # F(E; W^V, W^KQ) = E + W^V E · (E^T W^KQ E) / ρ
        attention_weights = torch.matmul(
            E.transpose(1, 2), 
            torch.matmul(self.W_KQ, E)
        ) / self.rho
        
        output = E + torch.matmul(
            self.W_V, 
            torch.matmul(E, attention_weights.transpose(1, 2))
        )
        
        # Extract query representation (last position)
        query_repr = output[:, :, -1]  # (batch, d_model)
        query_repr = self.dropout(query_repr)
        
        # Classify
        logits = self.classifier(query_repr)
        
        return logits

class BayesianEnsembleIDS(nn.Module):
    """
    Bayesian ensemble of transformers for uncertainty-aware intrusion detection
    """
    def __init__(self, n_ensemble: int = 10, d_model: int = 128, 
                 max_seq_len: int = 100, n_classes: int = 2):
        super().__init__()
        self.n_ensemble = n_ensemble
        self.n_classes = n_classes
        
        # Create ensemble of transformers with different initializations
        self.ensemble_models = nn.ModuleList([
            self._create_diverse_model(d_model, max_seq_len, n_classes, i)
            for i in range(n_ensemble)
        ])
        
        # Temperature parameter for calibration
        self.temperature = nn.Parameter(torch.ones(1))
        
    def _create_diverse_model(self, d_model: int, max_seq_len: int, 
                            n_classes: int, seed: int) -> SingleLayerTransformerIDS:
        """Create a transformer with diverse initialization"""
        torch.manual_seed(seed * 42)  # Different seed for each model
        model = SingleLayerTransformerIDS(d_model, max_seq_len, n_classes)
        
        # Add diversity through different dropout rates
        model.dropout.p = 0.1 + 0.05 * (seed % 3)
        
        return model
    
    def forward(self, network_prompt: torch.Tensor, 
                query_flow: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through ensemble
        Returns:
            mean_logits: Average predictions across ensemble
            uncertainty: Epistemic uncertainty estimate
        """
        ensemble_logits = []
        
        for model in self.ensemble_models:
            logits = model(network_prompt, query_flow)
            ensemble_logits.append(logits)
        
        # Stack predictions
        ensemble_logits = torch.stack(ensemble_logits, dim=0)  # (n_ensemble, batch, n_classes)
        
        # Compute mean and uncertainty
        mean_logits = torch.mean(ensemble_logits, dim=0)
        
        # Epistemic uncertainty (variance across ensemble members)
        epistemic_uncertainty = torch.var(ensemble_logits, dim=0)
        
        # Apply temperature scaling for calibration
        calibrated_logits = mean_logits / self.temperature
        
        return calibrated_logits, epistemic_uncertainty
    
    def predict_with_uncertainty(self, network_prompt: torch.Tensor, 
                               query_flow: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Make predictions with comprehensive uncertainty quantification
        """
        self.eval()
        with torch.no_grad():
            # Get ensemble predictions
            mean_logits, epistemic_uncertainty = self.forward(network_prompt, query_flow)
            
            # Convert to probabilities
            probs = F.softmax(mean_logits, dim=-1)
            
            # Compute aleatoric uncertainty (entropy of predictions)
            aleatoric_uncertainty = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
            
            # Total uncertainty
            total_uncertainty = epistemic_uncertainty.mean(dim=-1) + aleatoric_uncertainty
            
            # Confidence (1 - normalized uncertainty)
            confidence = 1.0 - (total_uncertainty / torch.log(torch.tensor(self.n_classes)))
            
            # Predictions
            predictions = torch.argmax(probs, dim=-1)
            
            return {
                'predictions': predictions,
                'probabilities': probs,
                'confidence': confidence,
                'epistemic_uncertainty': epistemic_uncertainty.mean(dim=-1),
                'aleatoric_uncertainty': aleatoric_uncertainty,
                'total_uncertainty': total_uncertainty
            }

class UncertaintyCalibrator:
    """
    Calibrates uncertainty estimates using temperature scaling and Platt scaling
    """
    def __init__(self):
        self.temperature = 1.0
        self.platt_a = 1.0
        self.platt_b = 0.0
        
    def calibrate_temperature(self, logits: torch.Tensor, 
                            targets: torch.Tensor) -> float:
        """
        Calibrate using temperature scaling
        """
        temperature = nn.Parameter(torch.ones(1))
        optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)
        
        def eval_loss():
            loss = F.cross_entropy(logits / temperature, targets)
            loss.backward()
            return loss
        
        optimizer.step(eval_loss)
        
        self.temperature = temperature.item()
        return self.temperature
    
    def compute_calibration_error(self, probs: torch.Tensor, 
                                targets: torch.Tensor, n_bins: int = 10) -> float:
        """
        Compute Expected Calibration Error (ECE)
        """
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        confidences = torch.max(probs, dim=1)[0]
        predictions = torch.argmax(probs, dim=1)
        accuracies = predictions.eq(targets)
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece.item()

# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    batch_size = 32
    seq_len = 50
    n_features = 41
    
    # Sample network flows
    network_prompt = torch.randn(batch_size, seq_len, n_features)
    query_flow = torch.randn(batch_size, n_features)
    
    # Create model
    model = BayesianEnsembleIDS(n_ensemble=5, d_model=64)
    
    # Make predictions with uncertainty
    results = model.predict_with_uncertainty(network_prompt, query_flow)
    
    print("Prediction results:")
    print(f"Predictions shape: {results['predictions'].shape}")
    print(f"Confidence range: [{results['confidence'].min():.3f}, {results['confidence'].max():.3f}]")
    print(f"Total uncertainty range: [{results['total_uncertainty'].min():.3f}, {results['total_uncertainty'].max():.3f}]")
