"""
Uncertainty-Aware Intrusion Detection System (UncertaintyIDS)

A production-ready implementation of uncertainty-aware intrusion detection
based on Bayesian ensemble transformers and in-context learning theory.

This package provides:
- Bayesian ensemble transformers for network intrusion detection
- Comprehensive uncertainty quantification (epistemic + aleatoric)
- Production-ready REST API for real-time inference
- Evaluation framework for detection and uncertainty metrics
- Data processing pipelines for standard IDS datasets

Author: Research Team
License: MIT
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Research Team"
__email__ = "research@example.com"
__license__ = "MIT"

# Core imports
from .models import BayesianEnsembleIDS, SingleLayerTransformerIDS
from .data import NetworkDataProcessor, IDSDataset
from .evaluation import ComprehensiveEvaluator, UncertaintyMetrics
from .utils import UncertaintyCalibrator, ModelCheckpoint

# API imports
from .api import UncertaintyIDSAPI

__all__ = [
    # Core models
    'BayesianEnsembleIDS',
    'SingleLayerTransformerIDS',
    
    # Data processing
    'NetworkDataProcessor', 
    'IDSDataset',
    
    # Evaluation
    'ComprehensiveEvaluator',
    'UncertaintyMetrics',
    
    # Utilities
    'UncertaintyCalibrator',
    'ModelCheckpoint',
    
    # API
    'UncertaintyIDSAPI',
]

# Package metadata
PACKAGE_INFO = {
    'name': 'uncertainty-ids',
    'version': __version__,
    'description': 'Uncertainty-Aware Intrusion Detection System',
    'long_description': __doc__,
    'author': __author__,
    'author_email': __email__,
    'license': __license__,
    'url': 'https://github.com/research-team/uncertainty-ids',
    'keywords': [
        'intrusion detection', 'uncertainty quantification', 
        'bayesian deep learning', 'transformers', 'cybersecurity'
    ],
    'classifiers': [
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Security',
        'Topic :: System :: Networking :: Monitoring',
    ],
}

# Configuration defaults
DEFAULT_CONFIG = {
    'model': {
        'n_ensemble': 10,
        'd_model': 128,
        'max_seq_len': 50,
        'n_classes': 2,
        'dropout_rate': 0.1,
    },
    'training': {
        'batch_size': 64,
        'learning_rate': 1e-3,
        'n_epochs': 100,
        'patience': 20,
        'weight_decay': 1e-5,
    },
    'data': {
        'sequence_length': 50,
        'test_size': 0.2,
        'val_size': 0.2,
        'random_state': 42,
    },
    'uncertainty': {
        'calibration_method': 'temperature_scaling',
        'uncertainty_threshold': 0.2,
        'confidence_threshold': 0.8,
    },
    'api': {
        'host': '0.0.0.0',
        'port': 8000,
        'workers': 4,
        'timeout': 30,
    },
    'logging': {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file': 'uncertainty_ids.log',
    }
}

def get_config():
    """Get default configuration dictionary."""
    return DEFAULT_CONFIG.copy()

def get_version():
    """Get package version."""
    return __version__

def get_package_info():
    """Get package metadata."""
    return PACKAGE_INFO.copy()
