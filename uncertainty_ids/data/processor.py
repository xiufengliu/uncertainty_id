"""
Network data processor for intrusion detection datasets.

This module provides comprehensive preprocessing capabilities for network
traffic data, including feature engineering, normalization, and temporal
sequence creation for transformer-based models.
"""

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, List, Optional, Union
import logging
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class NetworkDataProcessor:
    """
    Comprehensive network data processor for intrusion detection.
    
    Handles preprocessing of standard IDS datasets with support for:
    - Feature normalization and encoding
    - Temporal sequence creation
    - Data validation and cleaning
    - Multiple dataset formats
    """
    
    def __init__(self, sequence_length: int = 50, normalize_method: str = 'standard'):
        """
        Initialize the data processor.
        
        Args:
            sequence_length: Length of temporal sequences for transformer input
            normalize_method: Normalization method ('standard', 'minmax', 'robust')
        """
        self.sequence_length = sequence_length
        self.normalize_method = normalize_method
        
        # Preprocessing components
        self.scaler = self._get_scaler(normalize_method)
        self.label_encoder = LabelEncoder()
        self.categorical_encoders = {}
        
        # Feature information
        self.feature_names = self._get_standard_feature_names()
        self.categorical_features = ['protocol_type', 'service', 'flag']
        self.continuous_features = None
        
        # Dataset statistics
        self.dataset_stats = {}
        
        logger.info(f"Initialized NetworkDataProcessor with sequence_length={sequence_length}, "
                   f"normalize_method={normalize_method}")
    
    def _get_scaler(self, method: str):
        """Get the appropriate scaler based on method."""
        if method == 'standard':
            return StandardScaler()
        elif method == 'minmax':
            return MinMaxScaler()
        elif method == 'robust':
            from sklearn.preprocessing import RobustScaler
            return RobustScaler()
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def _get_standard_feature_names(self) -> List[str]:
        """Get the standard 41 network intrusion detection features."""
        return [
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
    
    def preprocess_data(self, data: Union[str, pd.DataFrame], 
                       target_column: str = 'label',
                       dataset_type: str = 'auto') -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess network traffic data.
        
        Args:
            data: Path to CSV file or pandas DataFrame
            target_column: Name of the target column
            dataset_type: Type of dataset ('nsl-kdd', 'cicids2017', 'unsw-nb15', 'auto')
            
        Returns:
            X: Preprocessed features (n_samples, n_features)
            y: Encoded labels (n_samples,)
        """
        logger.info("Starting data preprocessing...")
        
        # Load data if path is provided
        if isinstance(data, str):
            df = pd.read_csv(data)
            logger.info(f"Loaded data from {data} with shape {df.shape}")
        else:
            df = data.copy()
        
        # Dataset-specific preprocessing
        if dataset_type == 'auto':
            dataset_type = self._detect_dataset_type(df)
        
        df = self._apply_dataset_specific_preprocessing(df, dataset_type)
        
        # Separate features and target
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Validate and clean data
        X, y = self._validate_and_clean_data(X, y)
        
        # Handle categorical features
        X = self._encode_categorical_features(X)
        
        # Ensure we have the right number of features
        X = self._standardize_features(X)
        
        # Encode labels
        y_encoded = self._encode_labels(y)
        
        # Scale continuous features
        X_scaled = self._scale_features(X)
        
        # Store dataset statistics
        self._compute_dataset_statistics(X_scaled, y_encoded)
        
        logger.info(f"Preprocessing complete. Final shape: {X_scaled.shape}, "
                   f"Classes: {len(np.unique(y_encoded))}")
        
        return X_scaled, y_encoded
    
    def _detect_dataset_type(self, df: pd.DataFrame) -> str:
        """Automatically detect dataset type based on columns and characteristics."""
        columns = set(df.columns)
        
        # NSL-KDD detection
        if 'difficulty' in columns or len(df.columns) == 43:  # 41 features + label + difficulty
            return 'nsl-kdd'
        
        # CICIDS2017 detection
        if 'Flow Duration' in columns or 'Total Fwd Packets' in columns:
            return 'cicids2017'
        
        # UNSW-NB15 detection
        if 'id' in columns or 'attack_cat' in columns:
            return 'unsw-nb15'
        
        # Default to generic
        return 'generic'
    
    def _apply_dataset_specific_preprocessing(self, df: pd.DataFrame, 
                                            dataset_type: str) -> pd.DataFrame:
        """Apply dataset-specific preprocessing steps."""
        if dataset_type == 'nsl-kdd':
            return self._preprocess_nsl_kdd(df)
        elif dataset_type == 'cicids2017':
            return self._preprocess_cicids2017(df)
        elif dataset_type == 'unsw-nb15':
            return self._preprocess_unsw_nb15(df)
        else:
            return df
    
    def _preprocess_nsl_kdd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess NSL-KDD dataset."""
        # Remove difficulty column if present
        if 'difficulty' in df.columns:
            df = df.drop(columns=['difficulty'])
        
        # Ensure column names match standard features
        if len(df.columns) == 42:  # 41 features + label
            df.columns = self.feature_names + ['label']
        
        # Convert attack types to binary (normal vs attack)
        if 'label' in df.columns:
            df['label'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)
        
        return df
    
    def _preprocess_cicids2017(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess CICIDS2017 dataset."""
        # Map CICIDS2017 features to standard features (simplified mapping)
        # This is a basic mapping - in practice, you'd want more sophisticated feature engineering
        
        # Handle infinite and NaN values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)
        
        # Convert to binary classification
        if 'Label' in df.columns:
            df['label'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
            df = df.drop(columns=['Label'])
        
        # Select and rename features to match standard format
        # This is a simplified approach - you might want more sophisticated mapping
        feature_mapping = self._get_cicids2017_feature_mapping()
        
        if feature_mapping:
            available_features = [f for f in feature_mapping.keys() if f in df.columns]
            df_mapped = df[available_features].rename(columns=feature_mapping)
            
            # Add missing features with zeros
            for feature in self.feature_names:
                if feature not in df_mapped.columns:
                    df_mapped[feature] = 0
            
            # Reorder columns
            df = df_mapped[self.feature_names + ['label']]
        
        return df
    
    def _preprocess_unsw_nb15(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess UNSW-NB15 dataset."""
        # Remove ID column if present
        if 'id' in df.columns:
            df = df.drop(columns=['id'])
        
        # Handle attack categories
        if 'attack_cat' in df.columns:
            df = df.drop(columns=['attack_cat'])
        
        # Convert to binary classification
        if 'label' in df.columns:
            df['label'] = df['label'].astype(int)  # Already binary in UNSW-NB15
        
        return df
    
    def _get_cicids2017_feature_mapping(self) -> Dict[str, str]:
        """Get feature mapping from CICIDS2017 to standard features."""
        # This is a simplified mapping - in practice, you'd want domain expertise
        return {
            'Flow Duration': 'duration',
            'Protocol': 'protocol_type',
            'Total Fwd Packets': 'count',
            'Total Backward Packets': 'srv_count',
            # Add more mappings as needed
        }
    
    def _validate_and_clean_data(self, X: pd.DataFrame, 
                                y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Validate and clean the data."""
        initial_shape = X.shape
        
        # Remove rows with all NaN values
        valid_rows = ~X.isnull().all(axis=1)
        X = X[valid_rows]
        y = y[valid_rows]
        
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values with appropriate defaults
        for col in X.columns:
            if X[col].dtype in ['object', 'category']:
                X[col] = X[col].fillna('unknown')
            else:
                X[col] = X[col].fillna(0)
        
        # Remove duplicate rows
        before_dedup = len(X)
        X = X.drop_duplicates()
        y = y[X.index]
        after_dedup = len(X)
        
        if before_dedup != after_dedup:
            logger.info(f"Removed {before_dedup - after_dedup} duplicate rows")
        
        logger.info(f"Data validation complete. Shape: {initial_shape} -> {X.shape}")
        
        return X, y
    
    def _encode_categorical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features."""
        X_encoded = X.copy()
        
        for feature in self.categorical_features:
            if feature in X_encoded.columns:
                if feature not in self.categorical_encoders:
                    self.categorical_encoders[feature] = LabelEncoder()
                
                # Handle unknown categories
                unique_values = X_encoded[feature].unique()
                self.categorical_encoders[feature].fit(unique_values)
                
                X_encoded[feature] = self.categorical_encoders[feature].transform(
                    X_encoded[feature]
                )
        
        return X_encoded
    
    def _standardize_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Ensure we have exactly 41 features in the correct order."""
        X_standard = pd.DataFrame()
        
        for feature in self.feature_names:
            if feature in X.columns:
                X_standard[feature] = X[feature]
            else:
                # Add missing features with default values
                if feature in self.categorical_features:
                    X_standard[feature] = 0  # Default category
                else:
                    X_standard[feature] = 0.0  # Default continuous value
        
        return X_standard
    
    def _encode_labels(self, y: pd.Series) -> np.ndarray:
        """Encode target labels."""
        if y.dtype == 'object' or y.dtype.name == 'category':
            y_encoded = self.label_encoder.fit_transform(y)
        else:
            y_encoded = y.values
        
        # Ensure binary classification
        unique_labels = np.unique(y_encoded)
        if len(unique_labels) > 2:
            logger.warning(f"Found {len(unique_labels)} unique labels. "
                          "Converting to binary classification (normal vs attack).")
            y_encoded = (y_encoded > 0).astype(int)
        
        return y_encoded
    
    def _scale_features(self, X: pd.DataFrame) -> np.ndarray:
        """Scale continuous features."""
        # Identify continuous features (non-categorical)
        self.continuous_features = [f for f in X.columns if f not in self.categorical_features]
        
        X_scaled = X.copy()
        
        # Scale only continuous features
        if self.continuous_features:
            X_scaled[self.continuous_features] = self.scaler.fit_transform(
                X_scaled[self.continuous_features]
            )
        
        return X_scaled.values.astype(np.float32)
    
    def _compute_dataset_statistics(self, X: np.ndarray, y: np.ndarray):
        """Compute and store dataset statistics."""
        self.dataset_stats = {
            'n_samples': len(X),
            'n_features': X.shape[1],
            'n_classes': len(np.unique(y)),
            'class_distribution': dict(zip(*np.unique(y, return_counts=True))),
            'feature_means': np.mean(X, axis=0),
            'feature_stds': np.std(X, axis=0),
            'attack_rate': np.mean(y),
        }
        
        logger.info(f"Dataset statistics: {self.dataset_stats['n_samples']} samples, "
                   f"{self.dataset_stats['n_features']} features, "
                   f"attack rate: {self.dataset_stats['attack_rate']:.3f}")
    
    def create_temporal_sequences(self, X: np.ndarray, y: np.ndarray, 
                                sequence_length: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create temporal sequences for transformer input.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (n_samples,)
            sequence_length: Length of sequences (uses self.sequence_length if None)
            
        Returns:
            sequences: Historical sequences (n_sequences, seq_len, n_features)
            queries: Query samples (n_sequences, n_features)
            labels: Corresponding labels (n_sequences,)
        """
        if sequence_length is None:
            sequence_length = self.sequence_length
        
        if len(X) < sequence_length + 1:
            raise ValueError(f"Not enough samples ({len(X)}) to create sequences of length {sequence_length}")
        
        sequences = []
        queries = []
        labels = []
        
        for i in range(sequence_length, len(X)):
            # Historical sequence
            seq = X[i-sequence_length:i]
            # Current query
            query = X[i]
            # Label for current query
            label = y[i]
            
            sequences.append(seq)
            queries.append(query)
            labels.append(label)
        
        sequences = np.array(sequences, dtype=np.float32)
        queries = np.array(queries, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        
        logger.info(f"Created {len(sequences)} temporal sequences of length {sequence_length}")
        
        return sequences, queries, labels
    
    def split_data(self, X: np.ndarray, y: np.ndarray, 
                  test_size: float = 0.2, val_size: float = 0.2, 
                  random_state: int = 42, stratify: bool = True) -> Dict[str, np.ndarray]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            X: Features
            y: Labels
            test_size: Proportion of test set
            val_size: Proportion of validation set (from remaining data)
            random_state: Random seed
            stratify: Whether to stratify splits
            
        Returns:
            Dictionary with train, validation, and test splits
        """
        stratify_y = y if stratify else None
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify_y
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        stratify_temp = y_temp if stratify else None
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            random_state=random_state, stratify=stratify_temp
        )
        
        splits = {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }
        
        logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return splits
    
    def save_preprocessors(self, save_dir: str):
        """Save preprocessing objects for later use."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save scalers and encoders
        joblib.dump(self.scaler, save_path / 'scaler.pkl')
        joblib.dump(self.label_encoder, save_path / 'label_encoder.pkl')
        joblib.dump(self.categorical_encoders, save_path / 'categorical_encoders.pkl')
        
        # Save configuration and statistics
        config = {
            'sequence_length': self.sequence_length,
            'normalize_method': self.normalize_method,
            'feature_names': self.feature_names,
            'categorical_features': self.categorical_features,
            'continuous_features': self.continuous_features,
            'dataset_stats': self.dataset_stats
        }
        joblib.dump(config, save_path / 'processor_config.pkl')
        
        logger.info(f"Preprocessors saved to {save_dir}")
    
    def load_preprocessors(self, save_dir: str):
        """Load preprocessing objects."""
        save_path = Path(save_dir)
        
        # Load scalers and encoders
        self.scaler = joblib.load(save_path / 'scaler.pkl')
        self.label_encoder = joblib.load(save_path / 'label_encoder.pkl')
        self.categorical_encoders = joblib.load(save_path / 'categorical_encoders.pkl')
        
        # Load configuration
        config = joblib.load(save_path / 'processor_config.pkl')
        self.sequence_length = config['sequence_length']
        self.normalize_method = config['normalize_method']
        self.feature_names = config['feature_names']
        self.categorical_features = config['categorical_features']
        self.continuous_features = config['continuous_features']
        self.dataset_stats = config['dataset_stats']
        
        logger.info(f"Preprocessors loaded from {save_dir}")
    
    def transform_new_data(self, X: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted preprocessors."""
        # Apply same preprocessing steps
        X_processed = self._encode_categorical_features(X)
        X_processed = self._standardize_features(X_processed)
        
        # Scale features
        X_processed[self.continuous_features] = self.scaler.transform(
            X_processed[self.continuous_features]
        )
        
        return X_processed.values.astype(np.float32)

    def get_dataset_info(self) -> Dict:
        """Get comprehensive dataset information."""
        return {
            'processor_config': {
                'sequence_length': self.sequence_length,
                'normalize_method': self.normalize_method,
                'n_features': len(self.feature_names),
                'n_categorical': len(self.categorical_features),
                'n_continuous': len(self.continuous_features) if self.continuous_features else 0,
            },
            'dataset_stats': self.dataset_stats,
            'feature_info': {
                'feature_names': self.feature_names,
                'categorical_features': self.categorical_features,
                'continuous_features': self.continuous_features,
            }
        }


def create_synthetic_ids_data(n_samples: int = 10000, n_features: int = 41,
                            attack_rate: float = 0.1, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic intrusion detection data for testing and demonstration.

    Args:
        n_samples: Number of samples to generate
        n_features: Number of features (should be 41 for standard IDS)
        attack_rate: Proportion of attack samples
        random_state: Random seed for reproducibility

    Returns:
        X: Synthetic feature matrix
        y: Synthetic labels (0=normal, 1=attack)
    """
    np.random.seed(random_state)

    # Generate base features
    X = np.random.randn(n_samples, n_features).astype(np.float32)

    # Generate labels
    y = np.random.binomial(1, attack_rate, n_samples)

    # Make attack samples distinguishable
    attack_mask = y == 1

    # Modify some features for attack samples
    X[attack_mask, :10] += np.random.normal(2, 0.5, (attack_mask.sum(), 10))  # Higher values
    X[attack_mask, 10:20] *= np.random.uniform(1.5, 3, (attack_mask.sum(), 10))  # Scale up

    # Add some categorical-like features (first 3 features)
    X[:, 1] = np.random.randint(0, 4, n_samples)  # protocol_type
    X[:, 2] = np.random.randint(0, 70, n_samples)  # service
    X[:, 3] = np.random.randint(0, 11, n_samples)  # flag

    # Ensure non-negative values for count-like features
    count_features = [4, 5, 22, 23, 31, 32]  # src_bytes, dst_bytes, count, srv_count, etc.
    for feat_idx in count_features:
        if feat_idx < n_features:
            X[:, feat_idx] = np.abs(X[:, feat_idx])

    # Ensure rate features are between 0 and 1
    rate_features = [24, 25, 26, 27, 28, 29, 30, 33, 34, 35, 36, 37, 38, 39, 40]
    for feat_idx in rate_features:
        if feat_idx < n_features:
            X[:, feat_idx] = np.clip(np.abs(X[:, feat_idx]), 0, 1)

    logger.info(f"Generated synthetic IDS data: {n_samples} samples, "
               f"{n_features} features, attack rate: {attack_rate:.3f}")

    return X, y
