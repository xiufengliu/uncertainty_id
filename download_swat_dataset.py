#!/usr/bin/env python3
"""
SWaT Dataset Download and Processing Script
Downloads and preprocesses the SWaT (Secure Water Treatment) dataset for intrusion detection experiments.
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import zipfile
import requests
from urllib.parse import urlparse
import logging


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_swat_dataset():
    """Download SWaT dataset from Google Drive"""
    logger.info("Starting SWaT dataset download...")

    try:
        # Create data directory
        data_dir = 'data/raw'
        os.makedirs(data_dir, exist_ok=True)

        # File paths
        zip_path = os.path.join(data_dir, 'Attack2.csv.zip')
        csv_path = os.path.join(data_dir, 'Attack2.csv')

        # Check if already downloaded
        if os.path.exists(csv_path):
            logger.info(f"Dataset already exists at: {csv_path}")
            return csv_path

        # Download from Google Drive
        file_id = '1klDpUNwhYp_pbUALdpKMbydBTYupIvkH'
        download_url = f'https://drive.google.com/uc?export=download&id={file_id}'

        logger.info("Downloading SWaT dataset...")
        response = requests.get(download_url, stream=True)
        response.raise_for_status()

        # Save zip file
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info(f"Dataset downloaded to: {zip_path}")

        # Extract zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)

        logger.info(f"CSV extracted to: {csv_path}")

        return csv_path

    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
        return None

def load_and_explore_swat(csv_path):
    """Load and explore the SWaT dataset"""
    logger.info("Loading SWaT dataset...")
    
    try:
        # Load the dataset
        df = pd.read_csv(csv_path)
        
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")
        logger.info(f"Data types:\n{df.dtypes}")
        logger.info(f"Missing values:\n{df.isnull().sum()}")
        
        # Check for attack labels
        if 'Normal/Attack' in df.columns:
            logger.info(f"Attack distribution:\n{df['Normal/Attack'].value_counts()}")
        elif 'label' in df.columns:
            logger.info(f"Label distribution:\n{df['label'].value_counts()}")
        else:
            logger.info("No clear attack/label column found. Available columns:")
            for col in df.columns:
                logger.info(f"  - {col}")
        
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
        if 'Normal/Attack' in df_processed.columns:
            target_col = 'Normal/Attack'
        elif 'label' in df_processed.columns:
            target_col = 'label'
        elif 'attack' in df_processed.columns:
            target_col = 'attack'
        else:
            # Look for binary columns that might be labels
            for col in df_processed.columns:
                if df_processed[col].nunique() == 2:
                    unique_vals = df_processed[col].unique()
                    if any(val in str(unique_vals).lower() for val in ['normal', 'attack', '0', '1']):
                        target_col = col
                        break
        
        if target_col is None:
            logger.error("Could not identify target column for attack/normal classification")
            return None, None, None, None
        
        logger.info(f"Using '{target_col}' as target column")
        
        # Separate features and target
        X = df_processed.drop(columns=[target_col])
        y = df_processed[target_col]
        
        # Handle timestamp columns if present
        timestamp_cols = []
        for col in X.columns:
            if 'time' in col.lower() or 'date' in col.lower() or X[col].dtype == 'object':
                if pd.to_datetime(X[col], errors='coerce').notna().sum() > len(X) * 0.5:
                    timestamp_cols.append(col)
        
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
        if X.isnull().sum().sum() > 0:
            logger.info("Handling missing values...")
            X = X.fillna(X.mean())
        
        # Encode target variable
        if y.dtype == 'object':
            le_target = LabelEncoder()
            y = le_target.fit_transform(y)
            logger.info(f"Target classes: {le_target.classes_}")
        
        # Convert to binary classification (0: Normal, 1: Attack)
        if len(np.unique(y)) > 2:
            # If more than 2 classes, convert to binary (normal vs any attack)
            y = (y > 0).astype(int)
        
        logger.info(f"Final feature shape: {X.shape}")
        logger.info(f"Final target distribution: {np.bincount(y)}")
        
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
        
        import json
        with open(os.path.join(output_dir, 'dataset_info.json'), 'w') as f:
            json.dump(info, f, indent=2)
        
        logger.info("Dataset saved successfully!")
        logger.info(f"Files saved in: {output_dir}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error saving dataset: {e}")
        return False

def main():
    """Main function to download and process SWaT dataset"""
    logger.info("Starting SWaT dataset download and processing...")
    
    # Download dataset
    csv_path = download_swat_dataset()
    if csv_path is None:
        logger.error("Failed to download dataset")
        return False
    
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
