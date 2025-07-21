"""
Dataset classes for intrusion detection data.

This module provides PyTorch dataset classes for various intrusion detection
datasets with support for temporal sequences and data augmentation.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List, Union
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class IDSDataset(Dataset):
    """
    Base PyTorch dataset class for intrusion detection data.
    
    Supports both single samples and temporal sequences for transformer models.
    """
    
    def __init__(self, sequences: np.ndarray, queries: np.ndarray, 
                 labels: np.ndarray, transform=None):
        """
        Initialize the dataset.
        
        Args:
            sequences: Historical network flows (n_samples, seq_len, n_features)
            queries: Current flows to classify (n_samples, n_features)
            labels: Target labels (n_samples,)
            transform: Optional data transformation function
        """
        self.sequences = torch.FloatTensor(sequences)
        self.queries = torch.FloatTensor(queries)
        self.labels = torch.LongTensor(labels)
        self.transform = transform
        
        # Validate data shapes
        assert len(self.sequences) == len(self.queries) == len(self.labels), \
            "All arrays must have the same length"
        
        logger.info(f"Initialized IDSDataset with {len(self)} samples")
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sequence = self.sequences[idx]
        query = self.queries[idx]
        label = self.labels[idx]
        
        if self.transform:
            sequence, query = self.transform(sequence, query)
        
        return sequence, query, label
    
    def get_class_distribution(self) -> Dict[int, int]:
        """Get the distribution of classes in the dataset."""
        unique, counts = torch.unique(self.labels, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))
    
    def get_attack_rate(self) -> float:
        """Get the proportion of attack samples."""
        return (self.labels == 1).float().mean().item()
    
    def get_dataset_info(self) -> Dict:
        """Get comprehensive dataset information."""
        return {
            'n_samples': len(self),
            'sequence_length': self.sequences.shape[1],
            'n_features': self.sequences.shape[2],
            'n_classes': len(torch.unique(self.labels)),
            'class_distribution': self.get_class_distribution(),
            'attack_rate': self.get_attack_rate(),
            'data_types': {
                'sequences': str(self.sequences.dtype),
                'queries': str(self.queries.dtype),
                'labels': str(self.labels.dtype)
            }
        }


class NSLKDDDataset(IDSDataset):
    """
    NSL-KDD dataset class with specific preprocessing for NSL-KDD data.
    """
    
    FEATURE_NAMES = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
        'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
        'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
        'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
        'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
        'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
    ]
    
    ATTACK_TYPES = {
        'normal': 0,
        # DoS attacks
        'back': 1, 'land': 1, 'neptune': 1, 'pod': 1, 'smurf': 1, 'teardrop': 1,
        # R2L attacks  
        'ftp_write': 1, 'guess_passwd': 1, 'imap': 1, 'multihop': 1, 'phf': 1,
        'spy': 1, 'warezclient': 1, 'warezmaster': 1,
        # U2R attacks
        'buffer_overflow': 1, 'loadmodule': 1, 'perl': 1, 'rootkit': 1,
        # Probe attacks
        'ipsweep': 1, 'nmap': 1, 'portsweep': 1, 'satan': 1
    }
    
    @classmethod
    def from_file(cls, filepath: str, sequence_length: int = 50, **kwargs):
        """
        Create NSL-KDD dataset from file.
        
        Args:
            filepath: Path to NSL-KDD CSV file
            sequence_length: Length of temporal sequences
            **kwargs: Additional arguments for dataset creation
        """
        from .processor import NetworkDataProcessor
        
        processor = NetworkDataProcessor(sequence_length=sequence_length)
        X, y = processor.preprocess_data(filepath, target_column='label', dataset_type='nsl-kdd')
        
        sequences, queries, labels = processor.create_temporal_sequences(X, y)
        
        return cls(sequences, queries, labels, **kwargs)


class CICIDS2017Dataset(IDSDataset):
    """
    CICIDS2017 dataset class with specific preprocessing for CICIDS2017 data.
    """
    
    ATTACK_TYPES = {
        'BENIGN': 0,
        'DoS Hulk': 1, 'DoS GoldenEye': 1, 'DoS slowloris': 1, 'DoS Slowhttptest': 1,
        'DDoS': 1, 'FTP-Patator': 1, 'SSH-Patator': 1, 'Web Attack – Brute Force': 1,
        'Web Attack – XSS': 1, 'Web Attack – Sql Injection': 1, 'Infiltration': 1,
        'Bot': 1, 'PortScan': 1, 'Heartbleed': 1
    }
    
    @classmethod
    def from_file(cls, filepath: str, sequence_length: int = 50, **kwargs):
        """Create CICIDS2017 dataset from file."""
        from .processor import NetworkDataProcessor
        
        processor = NetworkDataProcessor(sequence_length=sequence_length)
        X, y = processor.preprocess_data(filepath, target_column='Label', dataset_type='cicids2017')
        
        sequences, queries, labels = processor.create_temporal_sequences(X, y)
        
        return cls(sequences, queries, labels, **kwargs)


class UNSWNB15Dataset(IDSDataset):
    """
    UNSW-NB15 dataset class with specific preprocessing for UNSW-NB15 data.
    """
    
    ATTACK_CATEGORIES = {
        'Normal': 0,
        'Generic': 1, 'Exploits': 1, 'Fuzzers': 1, 'DoS': 1, 'Reconnaissance': 1,
        'Analysis': 1, 'Backdoor': 1, 'Shellcode': 1, 'Worms': 1
    }
    
    @classmethod
    def from_file(cls, filepath: str, sequence_length: int = 50, **kwargs):
        """Create UNSW-NB15 dataset from file."""
        from .processor import NetworkDataProcessor
        
        processor = NetworkDataProcessor(sequence_length=sequence_length)
        X, y = processor.preprocess_data(filepath, target_column='label', dataset_type='unsw-nb15')
        
        sequences, queries, labels = processor.create_temporal_sequences(X, y)
        
        return cls(sequences, queries, labels, **kwargs)


class SyntheticIDSDataset(IDSDataset):
    """
    Synthetic IDS dataset for testing and demonstration purposes.
    """
    
    @classmethod
    def create_synthetic(cls, n_samples: int = 10000, sequence_length: int = 50,
                        n_features: int = 41, attack_rate: float = 0.1, 
                        random_state: int = 42, **kwargs):
        """
        Create synthetic IDS dataset.
        
        Args:
            n_samples: Number of samples to generate
            sequence_length: Length of temporal sequences
            n_features: Number of features
            attack_rate: Proportion of attack samples
            random_state: Random seed
            **kwargs: Additional arguments for dataset creation
        """
        from .processor import NetworkDataProcessor, create_synthetic_ids_data
        
        # Generate synthetic data
        X, y = create_synthetic_ids_data(
            n_samples=n_samples + sequence_length,  # Extra samples for sequences
            n_features=n_features,
            attack_rate=attack_rate,
            random_state=random_state
        )
        
        # Create processor and temporal sequences
        processor = NetworkDataProcessor(sequence_length=sequence_length)
        sequences, queries, labels = processor.create_temporal_sequences(X, y)
        
        return cls(sequences, queries, labels, **kwargs)


class BalancedIDSDataset(Dataset):
    """
    Balanced dataset wrapper that ensures equal representation of classes.
    """
    
    def __init__(self, dataset: IDSDataset, balance_method: str = 'oversample'):
        """
        Initialize balanced dataset.
        
        Args:
            dataset: Original IDS dataset
            balance_method: Method to balance classes ('oversample', 'undersample', 'weighted')
        """
        self.original_dataset = dataset
        self.balance_method = balance_method
        
        # Get class distribution
        class_dist = dataset.get_class_distribution()
        self.classes = list(class_dist.keys())
        self.class_counts = list(class_dist.values())
        
        # Create balanced indices
        self.balanced_indices = self._create_balanced_indices()
        
        logger.info(f"Created balanced dataset with {len(self.balanced_indices)} samples "
                   f"using {balance_method} method")
    
    def _create_balanced_indices(self) -> List[int]:
        """Create indices for balanced sampling."""
        if self.balance_method == 'oversample':
            return self._oversample_indices()
        elif self.balance_method == 'undersample':
            return self._undersample_indices()
        else:
            raise ValueError(f"Unknown balance method: {self.balance_method}")
    
    def _oversample_indices(self) -> List[int]:
        """Create indices by oversampling minority class."""
        # Find indices for each class
        class_indices = {}
        for class_id in self.classes:
            mask = self.original_dataset.labels == class_id
            class_indices[class_id] = torch.where(mask)[0].tolist()
        
        # Oversample to match majority class
        max_count = max(self.class_counts)
        balanced_indices = []
        
        for class_id in self.classes:
            indices = class_indices[class_id]
            # Repeat indices to reach max_count
            n_repeats = max_count // len(indices)
            remainder = max_count % len(indices)
            
            balanced_indices.extend(indices * n_repeats)
            balanced_indices.extend(indices[:remainder])
        
        return balanced_indices
    
    def _undersample_indices(self) -> List[int]:
        """Create indices by undersampling majority class."""
        # Find indices for each class
        class_indices = {}
        for class_id in self.classes:
            mask = self.original_dataset.labels == class_id
            class_indices[class_id] = torch.where(mask)[0].tolist()
        
        # Undersample to match minority class
        min_count = min(self.class_counts)
        balanced_indices = []
        
        for class_id in self.classes:
            indices = class_indices[class_id][:min_count]
            balanced_indices.extend(indices)
        
        return balanced_indices
    
    def __len__(self) -> int:
        return len(self.balanced_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        original_idx = self.balanced_indices[idx]
        return self.original_dataset[original_idx]
    
    def get_class_distribution(self) -> Dict[int, int]:
        """Get the distribution of classes in the balanced dataset."""
        labels = [self.original_dataset.labels[idx] for idx in self.balanced_indices]
        unique, counts = torch.unique(torch.tensor(labels), return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))


class StreamingIDSDataset(Dataset):
    """
    Streaming dataset for real-time intrusion detection.
    
    Simulates continuous network traffic for online learning and evaluation.
    """
    
    def __init__(self, base_dataset: IDSDataset, stream_length: int = 1000,
                 concept_drift: bool = False, drift_points: Optional[List[int]] = None):
        """
        Initialize streaming dataset.
        
        Args:
            base_dataset: Base dataset to stream from
            stream_length: Length of the stream
            concept_drift: Whether to simulate concept drift
            drift_points: Points in the stream where drift occurs
        """
        self.base_dataset = base_dataset
        self.stream_length = stream_length
        self.concept_drift = concept_drift
        self.drift_points = drift_points or []
        
        # Create streaming indices
        self.stream_indices = self._create_stream_indices()
        
        logger.info(f"Created streaming dataset with {stream_length} samples, "
                   f"concept drift: {concept_drift}")
    
    def _create_stream_indices(self) -> List[int]:
        """Create indices for streaming data."""
        # Simple approach: cycle through base dataset
        base_length = len(self.base_dataset)
        indices = []
        
        for i in range(self.stream_length):
            base_idx = i % base_length
            
            # Apply concept drift if specified
            if self.concept_drift and any(i >= dp for dp in self.drift_points):
                # Simple drift: add noise to features
                pass  # Drift will be applied in __getitem__
            
            indices.append(base_idx)
        
        return indices
    
    def __len__(self) -> int:
        return self.stream_length
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        base_idx = self.stream_indices[idx]
        sequence, query, label = self.base_dataset[base_idx]
        
        # Apply concept drift if needed
        if self.concept_drift and any(idx >= dp for dp in self.drift_points):
            # Add noise to simulate drift
            drift_noise = torch.randn_like(query) * 0.1
            query = query + drift_noise
            sequence = sequence + torch.randn_like(sequence) * 0.05
        
        return sequence, query, label
    
    def get_current_position(self) -> int:
        """Get current position in the stream."""
        return len(self.stream_indices)
    
    def is_drift_point(self, idx: int) -> bool:
        """Check if current index is a drift point."""
        return idx in self.drift_points
