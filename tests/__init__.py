"""
Test suite for Uncertainty-Aware Intrusion Detection System.

This package contains comprehensive unit tests and integration tests
for all major components of the uncertainty-aware IDS system.
"""

import pytest
import torch
import numpy as np
import tempfile
import os
from pathlib import Path
from typing import Dict, Any, Tuple

# Test configuration
TEST_CONFIG = {
    'model': {
        'n_ensemble': 3,  # Smaller ensemble for faster testing
        'd_model': 64,    # Smaller model for faster testing
        'max_seq_len': 10,
        'n_classes': 2,
    },
    'data': {
        'n_samples': 1000,
        'n_features': 41,
        'sequence_length': 10,
        'attack_rate': 0.1,
    },
    'training': {
        'batch_size': 32,
        'n_epochs': 2,  # Minimal training for testing
        'learning_rate': 1e-3,
    }
}


def get_test_config() -> Dict[str, Any]:
    """Get test configuration."""
    return TEST_CONFIG.copy()


def create_temp_dir() -> str:
    """Create temporary directory for test files."""
    return tempfile.mkdtemp()


def cleanup_temp_dir(temp_dir: str):
    """Clean up temporary directory."""
    import shutil
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


# Test fixtures and utilities
@pytest.fixture
def test_config():
    """Pytest fixture for test configuration."""
    return get_test_config()


@pytest.fixture
def temp_dir():
    """Pytest fixture for temporary directory."""
    temp_path = create_temp_dir()
    yield temp_path
    cleanup_temp_dir(temp_path)


@pytest.fixture
def synthetic_data():
    """Pytest fixture for synthetic test data."""
    from uncertainty_ids.data.processor import create_synthetic_ids_data
    
    config = get_test_config()
    X, y = create_synthetic_ids_data(
        n_samples=config['data']['n_samples'],
        n_features=config['data']['n_features'],
        attack_rate=config['data']['attack_rate'],
        random_state=42
    )
    return X, y


@pytest.fixture
def processed_data(synthetic_data):
    """Pytest fixture for processed test data."""
    from uncertainty_ids.data import NetworkDataProcessor
    
    X, y = synthetic_data
    config = get_test_config()
    
    processor = NetworkDataProcessor(
        sequence_length=config['data']['sequence_length']
    )
    
    # Create temporal sequences
    sequences, queries, labels = processor.create_temporal_sequences(X, y)
    
    return sequences, queries, labels, processor


@pytest.fixture
def test_model(test_config):
    """Pytest fixture for test model."""
    from uncertainty_ids.models import BayesianEnsembleIDS
    
    model = BayesianEnsembleIDS(**test_config['model'])
    return model


@pytest.fixture
def trained_model(test_model, processed_data, temp_dir):
    """Pytest fixture for trained test model."""
    from uncertainty_ids.training import UncertaintyIDSTrainer
    from torch.utils.data import DataLoader, TensorDataset
    
    sequences, queries, labels, processor = processed_data
    config = get_test_config()
    
    # Create data loaders
    dataset = TensorDataset(
        torch.FloatTensor(sequences),
        torch.FloatTensor(queries),
        torch.LongTensor(labels)
    )
    
    train_loader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=False)
    
    # Train model
    trainer = UncertaintyIDSTrainer(config['model'])
    trainer.train(train_loader, val_loader, n_epochs=config['training']['n_epochs'])
    
    # Save model
    model_path = os.path.join(temp_dir, 'test_model.pth')
    trainer.save_model(model_path)
    
    # Save processor
    processor_path = os.path.join(temp_dir, 'preprocessors')
    processor.save_preprocessors(processor_path)
    
    return trainer.model, model_path, processor_path


# Test utilities
def assert_tensor_shape(tensor: torch.Tensor, expected_shape: Tuple[int, ...]):
    """Assert tensor has expected shape."""
    assert tensor.shape == expected_shape, f"Expected shape {expected_shape}, got {tensor.shape}"


def assert_tensor_range(tensor: torch.Tensor, min_val: float, max_val: float):
    """Assert tensor values are in expected range."""
    assert tensor.min() >= min_val, f"Tensor minimum {tensor.min()} < {min_val}"
    assert tensor.max() <= max_val, f"Tensor maximum {tensor.max()} > {max_val}"


def assert_probability_tensor(tensor: torch.Tensor):
    """Assert tensor contains valid probabilities."""
    assert_tensor_range(tensor, 0.0, 1.0)
    if tensor.dim() > 1:
        # Check that probabilities sum to 1 along last dimension
        sums = torch.sum(tensor, dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6), \
            "Probabilities don't sum to 1"


def assert_uncertainty_tensor(tensor: torch.Tensor):
    """Assert tensor contains valid uncertainty values."""
    assert tensor.min() >= 0, f"Uncertainty values must be non-negative, got min: {tensor.min()}"
    assert not torch.isnan(tensor).any(), "Uncertainty tensor contains NaN values"
    assert not torch.isinf(tensor).any(), "Uncertainty tensor contains infinite values"


def assert_model_output_format(output: Dict[str, torch.Tensor], batch_size: int, n_classes: int):
    """Assert model output has correct format."""
    required_keys = ['predictions', 'probabilities', 'confidence', 
                    'epistemic_uncertainty', 'aleatoric_uncertainty', 'total_uncertainty']
    
    for key in required_keys:
        assert key in output, f"Missing key '{key}' in model output"
    
    # Check shapes
    assert_tensor_shape(output['predictions'], (batch_size,))
    assert_tensor_shape(output['probabilities'], (batch_size, n_classes))
    assert_tensor_shape(output['confidence'], (batch_size,))
    assert_tensor_shape(output['epistemic_uncertainty'], (batch_size,))
    assert_tensor_shape(output['aleatoric_uncertainty'], (batch_size,))
    assert_tensor_shape(output['total_uncertainty'], (batch_size,))
    
    # Check value ranges
    assert_probability_tensor(output['probabilities'])
    assert_tensor_range(output['confidence'], 0.0, 1.0)
    assert_uncertainty_tensor(output['epistemic_uncertainty'])
    assert_uncertainty_tensor(output['aleatoric_uncertainty'])
    assert_uncertainty_tensor(output['total_uncertainty'])
    
    # Check predictions are valid class indices
    assert output['predictions'].min() >= 0, "Predictions contain negative values"
    assert output['predictions'].max() < n_classes, f"Predictions exceed number of classes ({n_classes})"


def create_mock_network_flow() -> Dict[str, Any]:
    """Create mock network flow data for testing."""
    return {
        'duration': 0.5,
        'protocol_type': 0,  # tcp
        'service': 1,
        'flag': 2,
        'src_bytes': 1024,
        'dst_bytes': 512,
        'land': 0,
        'wrong_fragment': 0,
        'urgent': 0,
        'hot': 0,
        'num_failed_logins': 0,
        'logged_in': 1,
        'num_compromised': 0,
        'root_shell': 0,
        'su_attempted': 0,
        'num_root': 0,
        'num_file_creations': 0,
        'num_shells': 0,
        'num_access_files': 0,
        'num_outbound_cmds': 0,
        'is_host_login': 0,
        'is_guest_login': 0,
        'count': 5,
        'srv_count': 3,
        'serror_rate': 0.0,
        'srv_serror_rate': 0.0,
        'rerror_rate': 0.0,
        'srv_rerror_rate': 0.0,
        'same_srv_rate': 0.8,
        'diff_srv_rate': 0.2,
        'srv_diff_host_rate': 0.1,
        'dst_host_count': 10,
        'dst_host_srv_count': 8,
        'dst_host_same_srv_rate': 0.8,
        'dst_host_diff_srv_rate': 0.2,
        'dst_host_same_src_port_rate': 0.1,
        'dst_host_srv_diff_host_rate': 0.1,
        'dst_host_serror_rate': 0.0,
        'dst_host_srv_serror_rate': 0.0,
        'dst_host_rerror_rate': 0.0,
        'dst_host_srv_rerror_rate': 0.0,
    }


def create_mock_attack_flow() -> Dict[str, Any]:
    """Create mock attack network flow data for testing."""
    flow = create_mock_network_flow()
    # Modify to look more like an attack
    flow.update({
        'duration': 0.0,  # Very short duration
        'src_bytes': 0,
        'dst_bytes': 0,
        'count': 100,  # High connection count
        'srv_count': 100,
        'serror_rate': 1.0,  # High error rate
        'srv_serror_rate': 1.0,
    })
    return flow


# Performance testing utilities
class PerformanceTimer:
    """Context manager for timing code execution."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        print(f"{self.name} took {self.duration:.4f} seconds")


def benchmark_function(func, *args, n_runs: int = 10, **kwargs):
    """Benchmark a function over multiple runs."""
    import time
    
    times = []
    for _ in range(n_runs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        times.append(end_time - start_time)
    
    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'result': result  # Result from last run
    }
