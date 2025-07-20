"""
Unit tests for model components.

Tests the core transformer and ensemble models for correctness,
performance, and uncertainty quantification quality.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

from uncertainty_ids.models import (
    SingleLayerTransformerIDS, BayesianEnsembleIDS, 
    NetworkFeatureEmbedding, BaseUncertaintyModel
)
from tests import (
    assert_tensor_shape, assert_model_output_format, 
    assert_uncertainty_tensor, get_test_config
)


class TestNetworkFeatureEmbedding:
    """Test cases for NetworkFeatureEmbedding."""
    
    def test_embedding_initialization(self):
        """Test embedding layer initialization."""
        embed_dim = 64
        embedding = NetworkFeatureEmbedding(embed_dim=embed_dim)
        
        assert embedding.embed_dim == embed_dim
        assert embedding.protocol_embed.num_embeddings == 4
        assert embedding.service_embed.num_embeddings == 70
        assert embedding.flag_embed.num_embeddings == 11
        assert embedding.projection.out_features == embed_dim
    
    def test_embedding_forward_single(self):
        """Test forward pass with single sample."""
        embed_dim = 64
        batch_size = 1
        n_features = 41
        
        embedding = NetworkFeatureEmbedding(embed_dim=embed_dim)
        
        # Create test input
        flows = torch.randn(batch_size, n_features)
        flows[:, 1] = 0  # protocol_type
        flows[:, 2] = 1  # service
        flows[:, 3] = 2  # flag
        
        # Forward pass
        output = embedding(flows)
        
        # Check output shape
        assert_tensor_shape(output, (batch_size, embed_dim))
        assert not torch.isnan(output).any()
    
    def test_embedding_forward_batch(self):
        """Test forward pass with batch."""
        embed_dim = 64
        batch_size = 32
        n_features = 41
        
        embedding = NetworkFeatureEmbedding(embed_dim=embed_dim)
        
        # Create test input
        flows = torch.randn(batch_size, n_features)
        flows[:, 1] = torch.randint(0, 4, (batch_size,))  # protocol_type
        flows[:, 2] = torch.randint(0, 70, (batch_size,))  # service
        flows[:, 3] = torch.randint(0, 11, (batch_size,))  # flag
        
        # Forward pass
        output = embedding(flows)
        
        # Check output shape
        assert_tensor_shape(output, (batch_size, embed_dim))
        assert not torch.isnan(output).any()
    
    def test_embedding_forward_sequence(self):
        """Test forward pass with sequence input."""
        embed_dim = 64
        batch_size = 16
        seq_len = 10
        n_features = 41
        
        embedding = NetworkFeatureEmbedding(embed_dim=embed_dim)
        
        # Create test input
        flows = torch.randn(batch_size, seq_len, n_features)
        flows[:, :, 1] = torch.randint(0, 4, (batch_size, seq_len))
        flows[:, :, 2] = torch.randint(0, 70, (batch_size, seq_len))
        flows[:, :, 3] = torch.randint(0, 11, (batch_size, seq_len))
        
        # Forward pass
        output = embedding(flows)
        
        # Check output shape
        assert_tensor_shape(output, (batch_size, seq_len, embed_dim))
        assert not torch.isnan(output).any()


class TestSingleLayerTransformerIDS:
    """Test cases for SingleLayerTransformerIDS."""
    
    def test_model_initialization(self):
        """Test model initialization."""
        config = get_test_config()['model']
        model = SingleLayerTransformerIDS(**config)
        
        assert model.d_model == config['d_model']
        assert model.max_seq_len == config['max_seq_len']
        assert model.n_classes == config['n_classes']
        
        # Check parameter shapes
        assert model.W_V.shape == (config['d_model'], config['d_model'])
        assert model.W_KQ.shape == (config['d_model'], config['d_model'])
    
    def test_forward_pass(self):
        """Test forward pass."""
        config = get_test_config()['model']
        model = SingleLayerTransformerIDS(**config)
        
        batch_size = 8
        seq_len = config['max_seq_len']
        n_features = 41
        
        # Create test input
        network_prompt = torch.randn(batch_size, seq_len, n_features)
        query_flow = torch.randn(batch_size, n_features)
        
        # Set categorical features to valid ranges
        network_prompt[:, :, 1] = torch.randint(0, 4, (batch_size, seq_len))
        network_prompt[:, :, 2] = torch.randint(0, 70, (batch_size, seq_len))
        network_prompt[:, :, 3] = torch.randint(0, 11, (batch_size, seq_len))
        
        query_flow[:, 1] = torch.randint(0, 4, (batch_size,))
        query_flow[:, 2] = torch.randint(0, 70, (batch_size,))
        query_flow[:, 3] = torch.randint(0, 11, (batch_size,))
        
        # Forward pass
        logits, attention_weights = model(network_prompt, query_flow)
        
        # Check output shapes
        assert_tensor_shape(logits, (batch_size, config['n_classes']))
        assert_tensor_shape(attention_weights, (batch_size, seq_len + 1, seq_len + 1))
        
        # Check for NaN values
        assert not torch.isnan(logits).any()
        assert not torch.isnan(attention_weights).any()
    
    def test_predict_with_uncertainty(self):
        """Test uncertainty prediction."""
        config = get_test_config()['model']
        model = SingleLayerTransformerIDS(**config)
        
        batch_size = 4
        seq_len = config['max_seq_len']
        n_features = 41
        
        # Create test input
        network_prompt = torch.randn(batch_size, seq_len, n_features)
        query_flow = torch.randn(batch_size, n_features)
        
        # Set categorical features
        network_prompt[:, :, 1:4] = torch.randint(0, 4, (batch_size, seq_len, 3))
        query_flow[:, 1:4] = torch.randint(0, 4, (batch_size, 3))
        
        # Predict with uncertainty
        results = model.predict_with_uncertainty(network_prompt, query_flow)
        
        # Check output format
        assert_model_output_format(results, batch_size, config['n_classes'])
        
        # For single model, epistemic uncertainty should be zero
        assert torch.allclose(results['epistemic_uncertainty'], torch.zeros(batch_size))
    
    def test_attention_weights(self):
        """Test attention weights extraction."""
        config = get_test_config()['model']
        model = SingleLayerTransformerIDS(**config)
        
        batch_size = 2
        seq_len = config['max_seq_len']
        n_features = 41
        
        # Create test input
        network_prompt = torch.randn(batch_size, seq_len, n_features)
        query_flow = torch.randn(batch_size, n_features)
        
        # Set categorical features
        network_prompt[:, :, 1:4] = torch.randint(0, 4, (batch_size, seq_len, 3))
        query_flow[:, 1:4] = torch.randint(0, 4, (batch_size, 3))
        
        # Get attention weights
        attention_weights = model.get_attention_weights(network_prompt, query_flow)
        
        # Check shape and properties
        assert_tensor_shape(attention_weights, (batch_size, seq_len + 1, seq_len + 1))
        assert not torch.isnan(attention_weights).any()


class TestBayesianEnsembleIDS:
    """Test cases for BayesianEnsembleIDS."""
    
    def test_ensemble_initialization(self):
        """Test ensemble initialization."""
        config = get_test_config()['model']
        model = BayesianEnsembleIDS(**config)
        
        assert model.n_ensemble == config['n_ensemble']
        assert len(model.ensemble_models) == config['n_ensemble']
        assert model.temperature.shape == (1,)
        assert model.ensemble_weights.shape == (config['n_ensemble'],)
    
    def test_ensemble_diversity(self):
        """Test that ensemble members are diverse."""
        config = get_test_config()['model']
        model = BayesianEnsembleIDS(**config)
        
        # Check that models have different parameters
        model1_params = list(model.ensemble_models[0].parameters())
        model2_params = list(model.ensemble_models[1].parameters())
        
        # At least some parameters should be different
        differences = []
        for p1, p2 in zip(model1_params, model2_params):
            if p1.shape == p2.shape:
                diff = torch.norm(p1 - p2).item()
                differences.append(diff)
        
        assert any(diff > 1e-6 for diff in differences), "Ensemble members are not diverse"
    
    def test_ensemble_forward(self):
        """Test ensemble forward pass."""
        config = get_test_config()['model']
        model = BayesianEnsembleIDS(**config)
        
        batch_size = 4
        seq_len = config['max_seq_len']
        n_features = 41
        
        # Create test input
        network_prompt = torch.randn(batch_size, seq_len, n_features)
        query_flow = torch.randn(batch_size, n_features)
        
        # Set categorical features
        network_prompt[:, :, 1:4] = torch.randint(0, 4, (batch_size, seq_len, 3))
        query_flow[:, 1:4] = torch.randint(0, 4, (batch_size, 3))
        
        # Forward pass
        logits, epistemic_uncertainty = model(network_prompt, query_flow)
        
        # Check output shapes
        assert_tensor_shape(logits, (batch_size, config['n_classes']))
        assert_tensor_shape(epistemic_uncertainty, (batch_size,))
        
        # Check for valid values
        assert not torch.isnan(logits).any()
        assert_uncertainty_tensor(epistemic_uncertainty)
    
    def test_ensemble_predict_with_uncertainty(self):
        """Test ensemble uncertainty prediction."""
        config = get_test_config()['model']
        model = BayesianEnsembleIDS(**config)
        
        batch_size = 4
        seq_len = config['max_seq_len']
        n_features = 41
        
        # Create test input
        network_prompt = torch.randn(batch_size, seq_len, n_features)
        query_flow = torch.randn(batch_size, n_features)
        
        # Set categorical features
        network_prompt[:, :, 1:4] = torch.randint(0, 4, (batch_size, seq_len, 3))
        query_flow[:, 1:4] = torch.randint(0, 4, (batch_size, 3))
        
        # Predict with uncertainty
        results = model.predict_with_uncertainty(network_prompt, query_flow)
        
        # Check output format
        assert_model_output_format(results, batch_size, config['n_classes'])
        
        # For ensemble, epistemic uncertainty should be non-zero (usually)
        assert results['epistemic_uncertainty'].sum() >= 0
        
        # Check additional ensemble-specific outputs
        assert 'requires_review' in results
        assert 'high_confidence' in results
        assert 'ensemble_weights' in results
        assert 'temperature' in results
    
    def test_ensemble_diversity_metrics(self):
        """Test ensemble diversity computation."""
        config = get_test_config()['model']
        model = BayesianEnsembleIDS(**config)
        
        batch_size = 2
        seq_len = config['max_seq_len']
        n_features = 41
        
        # Create test input
        network_prompt = torch.randn(batch_size, seq_len, n_features)
        query_flow = torch.randn(batch_size, n_features)
        
        # Set categorical features
        network_prompt[:, :, 1:4] = torch.randint(0, 4, (batch_size, seq_len, 3))
        query_flow[:, 1:4] = torch.randint(0, 4, (batch_size, 3))
        
        # Get diversity metrics
        diversity = model.get_ensemble_diversity(network_prompt, query_flow)
        
        # Check that all expected metrics are present
        expected_keys = ['average_disagreement', 'mean_entropy', 'mean_individual_entropy', 
                        'mutual_information', 'ensemble_size']
        for key in expected_keys:
            assert key in diversity
            assert isinstance(diversity[key], (int, float))
            assert not np.isnan(diversity[key])
    
    def test_ensemble_member_management(self):
        """Test adding/removing ensemble members."""
        config = get_test_config()['model']
        model = BayesianEnsembleIDS(**config)
        
        initial_size = model.n_ensemble
        
        # Test adding member
        model.add_ensemble_member()
        assert model.n_ensemble == initial_size + 1
        assert len(model.ensemble_models) == initial_size + 1
        
        # Test removing member
        model.remove_ensemble_member(0)
        assert model.n_ensemble == initial_size
        assert len(model.ensemble_models) == initial_size
    
    def test_ensemble_weights_management(self):
        """Test ensemble weights setting and getting."""
        config = get_test_config()['model']
        model = BayesianEnsembleIDS(**config)
        
        # Test setting custom weights
        custom_weights = [0.5, 0.3, 0.2]
        model.set_ensemble_weights(custom_weights)
        
        # Get normalized weights
        weights = model.get_ensemble_weights()
        
        # Check that weights are normalized
        assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-6)
        assert len(weights) == config['n_ensemble']


class TestModelIntegration:
    """Integration tests for model components."""
    
    def test_model_save_load(self, test_model, temp_dir):
        """Test model saving and loading."""
        import os
        
        model_path = os.path.join(temp_dir, 'test_model.pth')
        
        # Save model
        test_model.save_checkpoint(model_path, version='test')
        
        # Check file exists
        assert os.path.exists(model_path)
        
        # Load model
        loaded_model, checkpoint = BayesianEnsembleIDS.load_checkpoint(
            model_path, **get_test_config()['model']
        )
        
        # Check that models have same architecture
        assert loaded_model.n_ensemble == test_model.n_ensemble
        assert loaded_model.d_model == test_model.d_model
        assert checkpoint['version'] == 'test'
    
    def test_model_memory_usage(self, test_model):
        """Test model memory usage computation."""
        memory_info = test_model.get_memory_usage()
        
        assert 'parameters_mb' in memory_info
        assert 'buffers_mb' in memory_info
        assert 'total_mb' in memory_info
        
        assert memory_info['parameters_mb'] > 0
        assert memory_info['total_mb'] >= memory_info['parameters_mb']
    
    def test_model_parameter_count(self, test_model):
        """Test parameter counting."""
        param_info = test_model.count_parameters()
        
        assert 'total_parameters' in param_info
        assert 'trainable_parameters' in param_info
        assert 'non_trainable_parameters' in param_info
        
        assert param_info['total_parameters'] > 0
        assert param_info['trainable_parameters'] <= param_info['total_parameters']
    
    @pytest.mark.slow
    def test_model_performance_benchmark(self, test_model):
        """Benchmark model inference performance."""
        from tests import benchmark_function
        
        config = get_test_config()['model']
        batch_size = 32
        seq_len = config['max_seq_len']
        n_features = 41
        
        # Create test input
        network_prompt = torch.randn(batch_size, seq_len, n_features)
        query_flow = torch.randn(batch_size, n_features)
        
        # Set categorical features
        network_prompt[:, :, 1:4] = torch.randint(0, 4, (batch_size, seq_len, 3))
        query_flow[:, 1:4] = torch.randint(0, 4, (batch_size, 3))
        
        # Benchmark prediction
        benchmark_results = benchmark_function(
            test_model.predict_with_uncertainty,
            network_prompt, query_flow,
            n_runs=10
        )
        
        # Check that inference is reasonably fast
        assert benchmark_results['mean_time'] < 1.0, "Model inference too slow"
        assert benchmark_results['std_time'] < 0.5, "Model inference time too variable"
        
        print(f"Average inference time: {benchmark_results['mean_time']:.4f}s")


if __name__ == "__main__":
    pytest.main([__file__])
