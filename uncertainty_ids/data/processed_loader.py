"""
Loader for pre-processed datasets.
This module handles loading data from the data/processed directory.
"""

import os
import numpy as np
import torch
import pickle
import json
from typing import Tuple, Dict, List, Optional
from .datasets import BaseIDSDataset


def _detect_feature_types(X_train: np.ndarray, feature_names: List[str]) -> Tuple[List[str], List[str], List[int]]:
    """
    Detect feature types based on data characteristics.

    Args:
        X_train: Training data
        feature_names: List of feature names

    Returns:
        continuous_features: List of continuous feature names
        categorical_features: List of categorical feature names
        categorical_indices: List of categorical feature indices
    """
    continuous_features = []
    categorical_features = []
    categorical_indices = []

    for i, feature in enumerate(feature_names):
        # Get column data
        col_data = X_train[:, i]
        unique_values = np.unique(col_data)

        # Heuristics for categorical detection:
        # 1. Small number of unique values (< 20)
        # 2. All values are integers
        # 3. Values are in a small range
        is_categorical = (
            len(unique_values) < 20 and
            np.all(np.mod(col_data, 1) == 0) and
            (np.max(col_data) - np.min(col_data)) < 100
        )

        if is_categorical:
            categorical_features.append(feature)
            categorical_indices.append(i)
        else:
            continuous_features.append(feature)

    # Ensure we have at least some continuous features
    if len(continuous_features) == 0:
        # If no continuous features detected, treat all as continuous except last few
        continuous_features = feature_names[:-min(5, len(feature_names)//4)]
        categorical_features = feature_names[-min(5, len(feature_names)//4):]
        categorical_indices = list(range(len(continuous_features), len(feature_names)))

    return continuous_features, categorical_features, categorical_indices


def load_processed_dataset(
    dataset_name: str,
    processed_dir: str = "data/processed",
    sequence_length: int = 50
) -> Tuple[BaseIDSDataset, BaseIDSDataset]:
    """
    Load pre-processed dataset from the processed directory.
    
    Args:
        dataset_name: Name of dataset ('nsl_kdd', 'cicids2017', 'unsw_nb15', 'swat')
        processed_dir: Directory containing processed data
        sequence_length: Length of temporal sequences
        
    Returns:
        train_dataset: Training dataset
        test_dataset: Test dataset
    """
    # Validate dataset name
    valid_datasets = ['nsl_kdd', 'cicids2017', 'unsw_nb15', 'swat']
    if dataset_name not in valid_datasets:
        raise ValueError(f"Unknown dataset: {dataset_name}. Valid options: {valid_datasets}")
    
    # Construct dataset path
    dataset_path = os.path.join(processed_dir, dataset_name)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Processed dataset not found at {dataset_path}")
    
    # Load data
    X_train = np.load(os.path.join(dataset_path, "X_train.npy"))
    y_train = np.load(os.path.join(dataset_path, "y_train.npy"))
    X_test = np.load(os.path.join(dataset_path, "X_test.npy"))
    y_test = np.load(os.path.join(dataset_path, "y_test.npy"))
    
    # Load feature names
    with open(os.path.join(dataset_path, "feature_names.txt"), "r") as f:
        feature_names = [line.strip() for line in f.readlines()]
    
    # Determine continuous and categorical features
    continuous_features = []
    categorical_features = []
    categorical_indices = []
    
    # Try to load preprocessor if available
    preprocessor_path = os.path.join(dataset_path, "preprocessors.pkl")
    if os.path.exists(preprocessor_path):
        with open(preprocessor_path, "rb") as f:
            preprocessors = pickle.load(f)

        if hasattr(preprocessors, 'continuous_features') and hasattr(preprocessors, 'categorical_features'):
            continuous_features = preprocessors.continuous_features
            categorical_features = preprocessors.categorical_features

            # Get indices of categorical features
            for i, feature in enumerate(feature_names):
                if feature in categorical_features:
                    categorical_indices.append(i)
        else:
            # Fallback to heuristic approach
            continuous_features, categorical_features, categorical_indices = _detect_feature_types(X_train, feature_names)
    else:
        # If no preprocessor, make a best guess based on data types
        continuous_features, categorical_features, categorical_indices = _detect_feature_types(X_train, feature_names)
    
    # Split features into continuous and categorical
    if categorical_indices:
        X_train_cont = np.delete(X_train, categorical_indices, axis=1)
        X_train_cat = X_train[:, categorical_indices]
        X_test_cont = np.delete(X_test, categorical_indices, axis=1)
        X_test_cat = X_test[:, categorical_indices]
    else:
        # If no categorical features, create a dummy categorical feature
        X_train_cont = X_train
        X_train_cat = np.zeros((X_train.shape[0], 1), dtype=int)  # Dummy categorical feature
        X_test_cont = X_test
        X_test_cat = np.zeros((X_test.shape[0], 1), dtype=int)    # Dummy categorical feature

        # Add dummy categorical feature to the lists
        categorical_features = ['dummy_categorical']
        categorical_indices = [X_train.shape[1]]  # Index beyond actual features
    
    # Convert to PyTorch tensors
    X_train_cont_tensor = torch.FloatTensor(X_train_cont)
    X_train_cat_tensor = torch.LongTensor(X_train_cat)
    y_train_tensor = torch.LongTensor(y_train)
    
    X_test_cont_tensor = torch.FloatTensor(X_test_cont)
    X_test_cat_tensor = torch.LongTensor(X_test_cat)
    y_test_tensor = torch.LongTensor(y_test)
    
    # Create datasets
    train_dataset = BaseIDSDataset(
        X_train_cont_tensor, X_train_cat_tensor, y_train_tensor, sequence_length
    )
    
    test_dataset = BaseIDSDataset(
        X_test_cont_tensor, X_test_cat_tensor, y_test_tensor, sequence_length
    )
    
    print(f"Loaded {dataset_name} dataset:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    print(f"  Continuous features: {len(continuous_features)}")
    print(f"  Categorical features: {len(categorical_features)}")
    
    return train_dataset, test_dataset


def get_categorical_vocab_sizes(
    dataset_name: str,
    processed_dir: str = "data/processed"
) -> Dict[str, int]:
    """
    Get vocabulary sizes for categorical features.
    
    Args:
        dataset_name: Name of dataset
        processed_dir: Directory containing processed data
        
    Returns:
        Dictionary mapping feature names to vocabulary sizes
    """
    dataset_path = os.path.join(processed_dir, dataset_name)
    
    # Try to load preprocessor if available
    preprocessor_path = os.path.join(dataset_path, "preprocessors.pkl")
    if os.path.exists(preprocessor_path):
        with open(preprocessor_path, "rb") as f:
            preprocessors = pickle.load(f)
            
        if hasattr(preprocessors, 'categorical_vocab_sizes'):
            return preprocessors.categorical_vocab_sizes
    
    # If no preprocessor or no vocab sizes, load data and compute
    X_train = np.load(os.path.join(dataset_path, "X_train.npy"))
    
    # Load feature names
    with open(os.path.join(dataset_path, "feature_names.txt"), "r") as f:
        feature_names = [line.strip() for line in f.readlines()]
    
    # Determine categorical features and their vocab sizes
    categorical_vocab_sizes = {}

    for i, feature in enumerate(feature_names):
        # Check if column contains integers with small number of unique values
        unique_values = np.unique(X_train[:, i])
        if len(unique_values) < 20 and np.all(np.mod(X_train[:, i], 1) == 0):
            # Add 1 to max value to account for 0-indexing
            vocab_size = int(np.max(unique_values)) + 1
            categorical_vocab_sizes[feature] = max(2, vocab_size)  # Minimum vocab size of 2

    # If no categorical features found, add dummy categorical feature
    if not categorical_vocab_sizes:
        categorical_vocab_sizes['dummy_categorical'] = 2

    return categorical_vocab_sizes


def get_dataset_info(
    dataset_name: str,
    processed_dir: str = "data/processed"
) -> Dict[str, any]:
    """
    Get dataset information.
    
    Args:
        dataset_name: Name of dataset
        processed_dir: Directory containing processed data
        
    Returns:
        Dictionary with dataset information
    """
    dataset_path = os.path.join(processed_dir, dataset_name)
    
    # Try to load dataset info if available
    info_path = os.path.join(dataset_path, "dataset_info.json")
    if os.path.exists(info_path):
        with open(info_path, "r") as f:
            return json.load(f)
    
    # If no info file, create basic info
    X_train = np.load(os.path.join(dataset_path, "X_train.npy"))
    y_train = np.load(os.path.join(dataset_path, "y_train.npy"))
    X_test = np.load(os.path.join(dataset_path, "X_test.npy"))
    y_test = np.load(os.path.join(dataset_path, "y_test.npy"))
    
    # Load feature names
    with open(os.path.join(dataset_path, "feature_names.txt"), "r") as f:
        feature_names = [line.strip() for line in f.readlines()]
    
    # Count classes
    train_classes, train_counts = np.unique(y_train, return_counts=True)
    test_classes, test_counts = np.unique(y_test, return_counts=True)
    
    # Create info dictionary
    info = {
        "name": dataset_name,
        "train_samples": X_train.shape[0],
        "test_samples": X_test.shape[0],
        "features": len(feature_names),
        "feature_names": feature_names,
        "train_class_distribution": {int(c): int(count) for c, count in zip(train_classes, train_counts)},
        "test_class_distribution": {int(c): int(count) for c, count in zip(test_classes, test_counts)}
    }
    
    return info
