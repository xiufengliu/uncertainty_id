"""
Data processing module for Uncertainty-Aware Intrusion Detection System.

This module provides comprehensive data preprocessing pipelines for standard
intrusion detection datasets including NSL-KDD, CICIDS2017, and UNSW-NB15.
"""

from .processor import NetworkDataProcessor
from .datasets import IDSDataset, NSLKDDDataset, CICIDS2017Dataset, UNSWNB15Dataset
from .loaders import create_data_loaders, SequentialDataLoader
from .transforms import NetworkTransforms, TemporalSequenceTransform
from .utils import download_dataset, validate_dataset, get_dataset_info

__all__ = [
    'NetworkDataProcessor',
    'IDSDataset',
    'NSLKDDDataset', 
    'CICIDS2017Dataset',
    'UNSWNB15Dataset',
    'create_data_loaders',
    'SequentialDataLoader',
    'NetworkTransforms',
    'TemporalSequenceTransform',
    'download_dataset',
    'validate_dataset',
    'get_dataset_info',
]
