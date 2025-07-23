"""
Data loaders for training and evaluation.
"""

import torch
from torch.utils.data import DataLoader, random_split
from typing import Tuple, Dict, Optional, List
from .datasets import BaseIDSDataset
import numpy as np


def create_dataloaders(
    dataset: BaseIDSDataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    batch_size: int = 32,
    num_workers: int = 4,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        dataset: Dataset to split
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes
        random_seed: Random seed for reproducibility
        
    Returns:
        train_loader: Training dataloader
        val_loader: Validation dataloader
        test_loader: Test dataloader
    """
    # Set random seed for reproducibility
    torch.manual_seed(random_seed)
    
    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


class ICLDataLoader:
    """
    Data loader for In-Context Learning episodes.
    
    Based on the ICL experimental setup in Section 4.3 of the paper.
    """
    
    def __init__(
        self,
        attack_families: Dict[str, Dict[str, torch.Tensor]],
        k_shot_range: Tuple[int, int] = (1, 10),
        n_query: int = 5,
        batch_size: int = 4
    ):
        """
        Initialize ICL data loader.
        
        Args:
            attack_families: Dictionary of attack family data
            k_shot_range: Range for number of context examples (k-shot)
            n_query: Number of query examples per episode
            batch_size: Number of families per meta-batch
        """
        self.attack_families = attack_families
        self.family_names = list(attack_families.keys())
        self.k_shot_range = k_shot_range
        self.n_query = n_query
        self.batch_size = batch_size
        
    def sample_episode(self, family_name: str) -> Dict[str, torch.Tensor]:
        """
        Sample an ICL episode from a specific attack family.
        
        Args:
            family_name: Name of the attack family
            
        Returns:
            Dictionary containing context and query data
        """
        family_data = self.attack_families[family_name]
        total_samples = len(family_data['cont'])
        
        # Sample k (number of context examples)
        k = np.random.randint(self.k_shot_range[0], self.k_shot_range[1] + 1)
        
        # Ensure we have enough samples
        if total_samples < k + self.n_query:
            # If not enough samples, reduce k or n_query
            max_k = max(1, total_samples - self.n_query)
            k = min(k, max_k)
            n_query = min(self.n_query, total_samples - k)
        else:
            n_query = self.n_query
        
        # Sample indices
        indices = torch.randperm(total_samples)
        context_indices = indices[:k]
        query_indices = indices[k:k + n_query]
        
        # Extract data
        episode = {
            'context_cont': family_data['cont'][context_indices],
            'context_cat': family_data['cat'][context_indices],
            'context_labels': family_data['labels'][context_indices],
            'query_cont': family_data['cont'][query_indices],
            'query_cat': family_data['cat'][query_indices],
            'query_labels': family_data['labels'][query_indices],
            'family_name': family_name,
            'k_shot': k,
            'n_query': n_query
        }
        
        return episode
    
    def sample_meta_batch(self) -> List[Dict[str, torch.Tensor]]:
        """
        Sample a meta-batch of ICL episodes.
        
        Returns:
            List of ICL episodes
        """
        # Sample families for this meta-batch
        sampled_families = np.random.choice(
            self.family_names, 
            size=min(self.batch_size, len(self.family_names)),
            replace=False
        )
        
        episodes = []
        for family_name in sampled_families:
            episode = self.sample_episode(family_name)
            episodes.append(episode)
        
        return episodes
    
    def __iter__(self):
        """Iterator for continuous episode sampling."""
        while True:
            yield self.sample_meta_batch()


def create_icl_dataloaders(
    train_families: Dict[str, Dict[str, torch.Tensor]],
    val_families: Dict[str, Dict[str, torch.Tensor]],
    test_families: Dict[str, Dict[str, torch.Tensor]],
    k_shot_range: Tuple[int, int] = (1, 10),
    n_query: int = 5,
    batch_size: int = 4
) -> Tuple[ICLDataLoader, ICLDataLoader, ICLDataLoader]:
    """
    Create ICL dataloaders for meta-learning.
    
    Args:
        train_families: Meta-training attack families
        val_families: Meta-validation attack families
        test_families: Meta-test attack families
        k_shot_range: Range for k-shot learning
        n_query: Number of query examples per episode
        batch_size: Meta-batch size
        
    Returns:
        train_loader: Meta-training dataloader
        val_loader: Meta-validation dataloader
        test_loader: Meta-test dataloader
    """
    train_loader = ICLDataLoader(
        train_families, k_shot_range, n_query, batch_size
    )
    
    val_loader = ICLDataLoader(
        val_families, k_shot_range, n_query, batch_size
    )
    
    test_loader = ICLDataLoader(
        test_families, k_shot_range, n_query, batch_size
    )
    
    return train_loader, val_loader, test_loader


def collate_icl_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for ICL episodes.
    
    Args:
        batch: List of ICL episodes
        
    Returns:
        Batched ICL data
    """
    # Find maximum dimensions for padding
    max_k = max(episode['k_shot'] for episode in batch)
    max_n_query = max(episode['n_query'] for episode in batch)
    
    # Get feature dimensions
    cont_dim = batch[0]['context_cont'].shape[-1]
    cat_dim = batch[0]['context_cat'].shape[-1]
    
    # Initialize batched tensors
    batch_size = len(batch)
    
    batched_context_cont = torch.zeros(batch_size, max_k, cont_dim)
    batched_context_cat = torch.zeros(batch_size, max_k, cat_dim, dtype=torch.long)
    batched_context_labels = torch.zeros(batch_size, max_k, dtype=torch.long)
    
    batched_query_cont = torch.zeros(batch_size, max_n_query, cont_dim)
    batched_query_cat = torch.zeros(batch_size, max_n_query, cat_dim, dtype=torch.long)
    batched_query_labels = torch.zeros(batch_size, max_n_query, dtype=torch.long)
    
    # Fill batched tensors
    for i, episode in enumerate(batch):
        k = episode['k_shot']
        n_q = episode['n_query']
        
        batched_context_cont[i, :k] = episode['context_cont']
        batched_context_cat[i, :k] = episode['context_cat']
        batched_context_labels[i, :k] = episode['context_labels']
        
        batched_query_cont[i, :n_q] = episode['query_cont']
        batched_query_cat[i, :n_q] = episode['query_cat']
        batched_query_labels[i, :n_q] = episode['query_labels']
    
    return {
        'context_cont': batched_context_cont,
        'context_cat': batched_context_cat,
        'context_labels': batched_context_labels,
        'query_cont': batched_query_cont,
        'query_cat': batched_query_cat,
        'query_labels': batched_query_labels,
        'family_names': [episode['family_name'] for episode in batch],
        'k_shots': [episode['k_shot'] for episode in batch],
        'n_queries': [episode['n_query'] for episode in batch]
    }
