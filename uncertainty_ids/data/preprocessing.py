"""
Data preprocessing utilities for intrusion detection datasets.
Based on the heterogeneous feature processing described in Section 3.2.
"""

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional, Any
import pickle
import os


class DataPreprocessor:
    """
    Data preprocessor for heterogeneous network flow features.
    
    Handles both continuous and categorical features as described in the paper:
    φ(x) = Concat(φ_cont(x_cont), φ_cat(x_cat))
    """
    
    def __init__(self):
        self.continuous_scaler = StandardScaler()
        self.categorical_encoders = {}
        self.continuous_features = []
        self.categorical_features = []
        self.categorical_vocab_sizes = {}
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
        
    def identify_feature_types(self, df: pd.DataFrame, target_column: str) -> Tuple[List[str], List[str]]:
        """
        Automatically identify continuous and categorical features.
        
        Args:
            df: Input dataframe
            target_column: Name of target column
            
        Returns:
            continuous_features: List of continuous feature names
            categorical_features: List of categorical feature names
        """
        feature_columns = [col for col in df.columns if col != target_column]
        
        continuous_features = []
        categorical_features = []
        
        for col in feature_columns:
            if df[col].dtype in ['int64', 'float64']:
                # Check if it's actually categorical (small number of unique values)
                unique_values = df[col].nunique()
                if unique_values <= 20 and df[col].dtype == 'int64':
                    categorical_features.append(col)
                else:
                    continuous_features.append(col)
            else:
                categorical_features.append(col)
                
        return continuous_features, categorical_features
    
    def fit(self, df: pd.DataFrame, target_column: str) -> 'DataPreprocessor':
        """
        Fit preprocessor on training data.
        
        Args:
            df: Training dataframe
            target_column: Name of target column
            
        Returns:
            Self for method chaining
        """
        # Identify feature types
        self.continuous_features, self.categorical_features = self.identify_feature_types(df, target_column)
        
        # Fit continuous feature scaler
        if self.continuous_features:
            continuous_data = df[self.continuous_features].values
            self.continuous_scaler.fit(continuous_data)
        
        # Fit categorical encoders
        for feature in self.categorical_features:
            encoder = LabelEncoder()
            # Handle missing values
            feature_data = df[feature].fillna('unknown')
            encoder.fit(feature_data)
            self.categorical_encoders[feature] = encoder
            self.categorical_vocab_sizes[feature] = len(encoder.classes_)
        
        # Fit label encoder
        self.label_encoder.fit(df[target_column])
        
        self.is_fitted = True
        return self
    
    def transform(self, df: pd.DataFrame, target_column: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Transform data using fitted preprocessor.
        
        Args:
            df: Input dataframe
            target_column: Name of target column
            
        Returns:
            continuous_features: Normalized continuous features [n_samples, n_continuous]
            categorical_features: Encoded categorical features [n_samples, n_categorical]
            labels: Encoded labels [n_samples]
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        # Process continuous features
        if self.continuous_features:
            continuous_data = df[self.continuous_features].values
            continuous_normalized = self.continuous_scaler.transform(continuous_data)
            continuous_tensor = torch.FloatTensor(continuous_normalized)
        else:
            continuous_tensor = torch.empty(len(df), 0)
        
        # Process categorical features
        categorical_data = []
        for feature in self.categorical_features:
            feature_data = df[feature].fillna('unknown')
            # Handle unseen categories
            encoded_data = []
            for value in feature_data:
                try:
                    encoded_value = self.categorical_encoders[feature].transform([value])[0]
                except ValueError:
                    # Unseen category, assign to 0 (or create unknown class)
                    encoded_value = 0
                encoded_data.append(encoded_value)
            categorical_data.append(encoded_data)
        
        if categorical_data:
            categorical_tensor = torch.LongTensor(np.array(categorical_data).T)
        else:
            categorical_tensor = torch.empty(len(df), 0, dtype=torch.long)
        
        # Process labels
        labels = self.label_encoder.transform(df[target_column])
        labels_tensor = torch.LongTensor(labels)
        
        return continuous_tensor, categorical_tensor, labels_tensor
    
    def fit_transform(self, df: pd.DataFrame, target_column: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Fit preprocessor and transform data in one step.
        
        Args:
            df: Input dataframe
            target_column: Name of target column
            
        Returns:
            continuous_features: Normalized continuous features
            categorical_features: Encoded categorical features  
            labels: Encoded labels
        """
        return self.fit(df, target_column).transform(df, target_column)
    
    def save(self, filepath: str):
        """Save fitted preprocessor to file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'DataPreprocessor':
        """Load fitted preprocessor from file."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


class AttackFamilyProcessor:
    """
    Processor for organizing data by attack families for ICL training.
    Based on the meta-learning setup described in Section 4.3.
    """
    
    def __init__(self):
        self.attack_family_mapping = {
            # NSL-KDD attack families
            'normal': 'Normal',
            'dos': 'DoS',
            'probe': 'Probe', 
            'u2r': 'U2R',
            'r2l': 'R2L',
            
            # CICIDS2017 attack families
            'benign': 'Normal',
            'ddos': 'DoS',
            'portscan': 'Probe',
            'botnet': 'Botnet',
            'infiltration': 'Infiltration',
            'web_attack': 'Web_Attack',
            'brute_force': 'Brute_Force',
            
            # UNSW-NB15 attack families
            'analysis': 'Analysis',
            'backdoor': 'Backdoor',
            'exploits': 'Exploits',
            'fuzzers': 'Fuzzers',
            'generic': 'Generic',
            'reconnaissance': 'Reconnaissance',
            'shellcode': 'Shellcode',
            'worms': 'Worms'
        }
    
    def create_attack_families(
        self,
        continuous_features: torch.Tensor,
        categorical_features: torch.Tensor,
        labels: torch.Tensor,
        attack_types: List[str]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Organize data by attack families for meta-learning.
        
        Args:
            continuous_features: Continuous features [n_samples, n_continuous]
            categorical_features: Categorical features [n_samples, n_categorical]
            labels: Labels [n_samples]
            attack_types: List of attack type names for each sample
            
        Returns:
            Dictionary mapping family names to their data
        """
        families = {}
        
        for i, attack_type in enumerate(attack_types):
            # Map specific attack to family
            family = self.attack_family_mapping.get(attack_type.lower(), 'Unknown')
            
            if family not in families:
                families[family] = {
                    'cont': [],
                    'cat': [],
                    'labels': []
                }
            
            families[family]['cont'].append(continuous_features[i])
            families[family]['cat'].append(categorical_features[i])
            families[family]['labels'].append(labels[i])
        
        # Convert lists to tensors
        for family in families:
            families[family]['cont'] = torch.stack(families[family]['cont'])
            families[family]['cat'] = torch.stack(families[family]['cat'])
            families[family]['labels'] = torch.stack(families[family]['labels'])
        
        return families
    
    def split_families_for_meta_learning(
        self,
        families: Dict[str, Dict[str, torch.Tensor]],
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2,
        random_seed: int = 42
    ) -> Tuple[Dict, Dict, Dict]:
        """
        Split attack families for meta-learning.
        
        Based on paper setup:
        - Meta-Training Families (60%): DoS variants, Probe attacks, Normal traffic
        - Meta-Validation Families (20%): U2R attacks, specific malware families  
        - Meta-Test Families (20%): R2L attacks, APTs, zero-day exploits
        
        Args:
            families: Dictionary of attack families
            train_ratio: Ratio for meta-training families
            val_ratio: Ratio for meta-validation families
            test_ratio: Ratio for meta-test families
            random_seed: Random seed for reproducibility
            
        Returns:
            train_families: Meta-training families
            val_families: Meta-validation families
            test_families: Meta-test families
        """
        np.random.seed(random_seed)
        
        family_names = list(families.keys())
        n_families = len(family_names)
        
        # Shuffle families
        shuffled_families = np.random.permutation(family_names)
        
        # Split indices
        n_train = int(n_families * train_ratio)
        n_val = int(n_families * val_ratio)
        
        train_names = shuffled_families[:n_train]
        val_names = shuffled_families[n_train:n_train + n_val]
        test_names = shuffled_families[n_train + n_val:]
        
        # Create family dictionaries
        train_families = {name: families[name] for name in train_names}
        val_families = {name: families[name] for name in val_names}
        test_families = {name: families[name] for name in test_names}
        
        return train_families, val_families, test_families
