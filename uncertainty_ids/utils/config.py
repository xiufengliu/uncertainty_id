"""
Configuration utilities.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Union


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from JSON or YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        if config_path.suffix.lower() == '.json':
            config = json.load(f)
        elif config_path.suffix.lower() in ['.yaml', '.yml']:
            config = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
    
    return config


def save_config(config: Dict[str, Any], config_path: Union[str, Path]):
    """
    Save configuration to JSON or YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration file
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        if config_path.suffix.lower() == '.json':
            json.dump(config, f, indent=2)
        elif config_path.suffix.lower() in ['.yaml', '.yml']:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration based on paper specifications.
    
    Returns:
        Default configuration dictionary
    """
    return {
        "model": {
            "ensemble_size": 5,
            "d_model": 128,
            "n_heads": 3,
            "d_ff": 512,
            "dropout": 0.1,
            "max_seq_len": 51,
            "temperature": 1.0
        },
        "training": {
            "batch_size": 32,
            "learning_rate": 0.001,
            "lambda_diversity": 0.1,
            "lambda_uncertainty": 0.05,
            "num_epochs": 100,
            "patience": 10,
            "random_seed": 42
        },
        "data": {
            "sequence_length": 50,
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "test_ratio": 0.15
        },
        "icl": {
            "k_shot_range": [1, 10],
            "n_query": 5,
            "meta_batch_size": 4,
            "meta_lr": 0.001,
            "inner_lr": 0.0001,
            "lambda_icl": 0.1,
            "lambda_div": 0.1
        },
        "evaluation": {
            "n_bins": 10,
            "n_thresholds": 100
        }
    }
