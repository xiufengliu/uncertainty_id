#!/usr/bin/env python3
"""
SWaT Dataset Processing Script
Processes the SWaT (Secure Water Treatment) dataset for intrusion detection experiments.
This script assumes you have manually downloaded the SWaT dataset.
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import logging
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_swat_dataset():
    """Find SWaT dataset in common locations"""
    logger.info("Looking for SWaT dataset...")
    
    # Common file names for SWaT dataset
    possible_names = [
        'Attack2.csv',
        'SWaT_Dataset_Attack_v0.csv',
        'SWaT_Dataset_Normal_v1.csv',
        'swat_attack.csv',
        'swat_normal.csv',
        'SWaT.csv',
        'SWaT_sample.csv'
    ]
    
    # Common locations
    search_dirs = [
        'data/raw',
        'data',
        '.',
        'datasets',
        'SWaT'
    ]
    
    found_files = []
    
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            for filename in possible_names:
                filepath = os.path.join(search_dir, filename)
                if os.path.exists(filepath):
                    found_files.append(filepath)
                    logger.info(f"Found SWaT dataset: {filepath}")
    
    return found_files

def load_and_explore_swat(csv_path):
    """Load and explore the SWaT dataset"""
    logger.info(f"Loading SWaT dataset from: {csv_path}")
    
    try:
        # Try different encodings and separators
        encodings = ['utf-8', 'latin-1', 'cp1252']
        separators = [',', ';', '\t']
        
        df = None
        for encoding in encodings:
            for sep in separators:
                try:
                    df = pd.read_csv(csv_path, encoding=encoding, sep=sep)
                    if df.shape[1] > 1:  # More than one column means successful parsing
                        logger.info(f"Successfully loaded with encoding={encoding}, separator='{sep}'")
                        break
                except:
                    continue
            if df is not None and df.shape[1] > 1:
                break
        
        if df is None or df.shape[1] <= 1:
            logger.error("Could not parse the CSV file with any encoding/separator combination")
            return None
        
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Columns ({len(df.columns)}): {list(df.columns[:10])}{'...' if len(df.columns) > 10 else ''}")
        logger.info(f"Data types:\n{df.dtypes.value_counts()}")
        logger.info(f"Missing values: {df.isnull().sum().sum()}")
        
        # Look for potential target columns
        potential_targets = []
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['attack', 'normal', 'label', 'class', 'target']):
                potential_targets.append(col)
                logger.info(f"Potential target column '{col}': {df[col].value_counts().to_dict()}")
        
        if not potential_targets:
            logger.info("No obvious target column found. Checking for binary columns...")
            for col in df.columns:
                if df[col].nunique() == 2:
                    unique_vals = df[col].unique()
                    logger.info(f"Binary column '{col}': {unique_vals}")
                    potential_targets.append(col)
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return None

def preprocess_swat_dataset(df):
    """Preprocess the SWaT dataset for intrusion detection"""
    logger.info("Preprocessing SWaT dataset...")
    
    try:
        # Make a copy to avoid modifying original
        df_processed = df.copy()
        
        # Identify the target column
        target_col = None
        
        # Look for common SWaT target column names
        swat_target_names = ['Normal/Attack', 'attack', 'label', 'class']
        for col_name in swat_target_names:
            if col_name in df_processed.columns:
                target_col = col_name
                break
        
        # If not found, look for binary columns
        if target_col is None:
            for col in df_processed.columns:
                if df_processed[col].nunique() == 2:
                    unique_vals = df_processed[col].unique()
                    # Check if values suggest attack/normal classification
                    val_str = str(unique_vals).lower()
                    if any(keyword in val_str for keyword in ['normal', 'attack', '0', '1']):
                        target_col = col
                        logger.info(f"Using binary column '{col}' as target")
                        break
        
        if target_col is None:
            logger.error("Could not identify target column for attack/normal classification")
            logger.info("Available columns:")
            for i, col in enumerate(df_processed.columns):
                logger.info(f"  {i}: {col} (unique values: {df_processed[col].nunique()})")
            return None, None, None, None
        
        logger.info(f"Using '{target_col}' as target column")
        
        # Separate features and target
        X = df_processed.drop(columns=[target_col])
        y = df_processed[target_col]
        
        # Handle timestamp columns if present
        timestamp_cols = []
        for col in X.columns:
            if any(keyword in col.lower() for keyword in ['time', 'date', 'timestamp']):
                timestamp_cols.append(col)
            elif X[col].dtype == 'object':
                # Check if it's a datetime string
                try:
                    pd.to_datetime(X[col].head(100), errors='raise')
                    timestamp_cols.append(col)
                except:
                    pass
        
        if timestamp_cols:
            logger.info(f"Dropping timestamp columns: {timestamp_cols}")
            X = X.drop(columns=timestamp_cols)
        
        # Handle categorical columns
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols:
            logger.info(f"Encoding categorical columns: {categorical_cols}")
            le = LabelEncoder()
            for col in categorical_cols:
                X[col] = le.fit_transform(X[col].astype(str))
        
        # Handle missing values
        missing_count = X.isnull().sum().sum()
        if missing_count > 0:
            logger.info(f"Handling {missing_count} missing values...")
            # For numeric columns, use mean
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())
            # For non-numeric columns, use mode
            non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns
            for col in non_numeric_cols:
                X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 0)
        
        # Encode target variable
        if y.dtype == 'object':
            le_target = LabelEncoder()
            y_encoded = le_target.fit_transform(y)
            logger.info(f"Target classes: {dict(zip(le_target.classes_, range(len(le_target.classes_))))}")
            y = y_encoded
        
        # Convert to binary classification (0: Normal, 1: Attack)
        unique_targets = np.unique(y)
        if len(unique_targets) > 2:
            # If more than 2 classes, convert to binary (normal vs any attack)
            y = (y > 0).astype(int)
            logger.info("Converted to binary classification (0: Normal, 1: Attack)")
        elif len(unique_targets) == 2:
            # Ensure 0 is normal and 1 is attack
            if 0 not in unique_targets:
                y = y - y.min()  # Shift to start from 0
        
        logger.info(f"Final feature shape: {X.shape}")
        logger.info(f"Final target distribution: Normal={np.sum(y==0)}, Attack={np.sum(y==1)}")
        
        return X, y, X.columns.tolist(), target_col
        
    except Exception as e:
        logger.error(f"Error preprocessing dataset: {e}")
        return None, None, None, None

def create_train_test_split(X, y, test_size=0.2, random_state=42):
    """Create train/test split for the dataset"""
    logger.info("Creating train/test split...")
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Training set shape: {X_train.shape}")
        logger.info(f"Test set shape: {X_test.shape}")
        logger.info(f"Training set attack ratio: {y_train.sum() / len(y_train):.4f}")
        logger.info(f"Test set attack ratio: {y_test.sum() / len(y_test):.4f}")
        
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        logger.error(f"Error creating train/test split: {e}")
        return None, None, None, None

def normalize_features(X_train, X_test):
    """Normalize features using StandardScaler"""
    logger.info("Normalizing features...")
    
    try:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        logger.info("Feature normalization completed")
        
        return X_train_scaled, X_test_scaled, scaler
        
    except Exception as e:
        logger.error(f"Error normalizing features: {e}")
        return None, None, None

def save_processed_dataset(X_train, X_test, y_train, y_test, feature_names, output_dir='data/processed/swat'):
    """Save the processed dataset"""
    logger.info(f"Saving processed dataset to {output_dir}...")
    
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save arrays
        np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
        np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
        np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
        np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
        
        # Save feature names
        with open(os.path.join(output_dir, 'feature_names.txt'), 'w') as f:
            for name in feature_names:
                f.write(f"{name}\n")
        
        # Save dataset info
        info = {
            'dataset': 'SWaT',
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'features': len(feature_names),
            'train_attack_ratio': float(y_train.sum() / len(y_train)),
            'test_attack_ratio': float(y_test.sum() / len(y_test))
        }
        
        with open(os.path.join(output_dir, 'dataset_info.json'), 'w') as f:
            json.dump(info, f, indent=2)
        
        logger.info("Dataset saved successfully!")
        logger.info(f"Files saved in: {output_dir}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error saving dataset: {e}")
        return False

def main():
    """Main function to process SWaT dataset"""
    logger.info("Starting SWaT dataset processing...")
    
    # Find dataset files
    found_files = find_swat_dataset()
    
    if not found_files:
        logger.error("No SWaT dataset files found!")
        logger.info("Please download the SWaT dataset and place it in one of these locations:")
        logger.info("  - data/raw/Attack2.csv")
        logger.info("  - data/raw/SWaT_Dataset_Attack_v0.csv")
        logger.info("  - data/SWaT.csv")
        logger.info("You can download it from: https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/")
        return False
    
    # Process the first found file
    csv_path = found_files[0]
    logger.info(f"Processing: {csv_path}")
    
    # Load and explore dataset
    df = load_and_explore_swat(csv_path)
    if df is None:
        logger.error("Failed to load dataset")
        return False
    
    # Preprocess dataset
    X, y, feature_names, target_col = preprocess_swat_dataset(df)
    if X is None:
        logger.error("Failed to preprocess dataset")
        return False
    
    # Create train/test split
    X_train, X_test, y_train, y_test = create_train_test_split(X, y)
    if X_train is None:
        logger.error("Failed to create train/test split")
        return False
    
    # Normalize features
    X_train_scaled, X_test_scaled, scaler = normalize_features(X_train, X_test)
    if X_train_scaled is None:
        logger.error("Failed to normalize features")
        return False
    
    # Save processed dataset
    success = save_processed_dataset(
        X_train_scaled, X_test_scaled, y_train, y_test, feature_names
    )
    
    if success:
        logger.info("SWaT dataset processing completed successfully!")
        logger.info("Dataset ready for uncertainty-aware intrusion detection experiments")
        return True
    else:
        logger.error("Failed to save processed dataset")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
