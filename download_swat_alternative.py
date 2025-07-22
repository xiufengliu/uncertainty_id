#!/usr/bin/env python3
"""
Alternative SWaT Dataset Download Script
Uses wget to download the SWaT dataset from Google Drive
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import zipfile
import subprocess
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_swat_with_wget():
    """Download SWaT dataset using wget"""
    logger.info("Starting SWaT dataset download with wget...")
    
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
        
        # Download using wget
        file_id = '1klDpUNwhYp_pbUALdpKMbydBTYupIvkH'
        download_url = f'https://drive.google.com/uc?export=download&id={file_id}'
        
        logger.info("Downloading SWaT dataset with wget...")
        
        # Use wget to download
        cmd = ['wget', '--no-check-certificate', '-O', zip_path, download_url]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"wget failed: {result.stderr}")
            return None
        
        logger.info(f"Dataset downloaded to: {zip_path}")
        
        # Extract zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        
        logger.info(f"CSV extracted to: {csv_path}")
        
        return csv_path
        
    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
        return None

def download_swat_with_curl():
    """Download SWaT dataset using curl"""
    logger.info("Starting SWaT dataset download with curl...")
    
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
        
        # Download using curl
        file_id = '1klDpUNwhYp_pbUALdpKMbydBTYupIvkH'
        download_url = f'https://drive.google.com/uc?export=download&id={file_id}'
        
        logger.info("Downloading SWaT dataset with curl...")
        
        # Use curl to download
        cmd = ['curl', '-L', '-o', zip_path, download_url]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"curl failed: {result.stderr}")
            return None
        
        logger.info(f"Dataset downloaded to: {zip_path}")
        
        # Check if file was downloaded successfully
        if not os.path.exists(zip_path) or os.path.getsize(zip_path) < 1000:
            logger.error("Downloaded file is too small or doesn't exist")
            return None
        
        # Extract zip file
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            logger.info(f"CSV extracted to: {csv_path}")
        except zipfile.BadZipFile:
            logger.error("Downloaded file is not a valid zip file")
            return None
        
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

def main():
    """Main function to download SWaT dataset"""
    logger.info("Starting SWaT dataset download...")
    
    # Try curl first, then wget
    csv_path = download_swat_with_curl()
    
    if csv_path is None:
        logger.info("Curl failed, trying wget...")
        csv_path = download_swat_with_wget()
    
    if csv_path is None:
        logger.error("All download methods failed")
        return False
    
    # Load and explore dataset
    df = load_and_explore_swat(csv_path)
    if df is None:
        logger.error("Failed to load dataset")
        return False
    
    logger.info("SWaT dataset download completed successfully!")
    logger.info(f"Dataset available at: {csv_path}")
    logger.info("Run the full processing script to preprocess the data")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
