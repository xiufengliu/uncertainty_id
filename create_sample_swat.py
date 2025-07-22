#!/usr/bin/env python3
"""
Create Sample SWaT Dataset
Creates a synthetic SWaT-like dataset for testing the processing pipeline.
This is for demonstration purposes only.
"""

import os
import pandas as pd
import numpy as np
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_sample_swat_dataset():
    """Create a sample SWaT-like dataset"""
    logger.info("Creating sample SWaT dataset...")
    
    try:
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Number of samples
        n_samples = 10000
        
        # SWaT typically has sensor readings from water treatment process
        # Create realistic sensor features
        features = {}
        
        # Flow sensors (FIT)
        for i in range(1, 6):
            features[f'FIT_{i:03d}'] = np.random.normal(2.5, 0.5, n_samples)
        
        # Level sensors (LIT)
        for i in range(1, 4):
            features[f'LIT_{i:03d}'] = np.random.normal(50, 10, n_samples)
        
        # Pressure sensors (PIT)
        for i in range(1, 3):
            features[f'PIT_{i:03d}'] = np.random.normal(1.2, 0.2, n_samples)
        
        # Temperature sensors (TIT)
        for i in range(1, 3):
            features[f'TIT_{i:03d}'] = np.random.normal(25, 3, n_samples)
        
        # Actuator status (binary)
        for i in range(1, 8):
            features[f'MV_{i:03d}'] = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
        
        # Pump status (binary)
        for i in range(1, 4):
            features[f'P_{i:03d}'] = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
        
        # Create DataFrame
        df = pd.DataFrame(features)
        
        # Add timestamp
        start_time = pd.Timestamp('2015-12-28 10:00:00')
        df['Timestamp'] = pd.date_range(start=start_time, periods=n_samples, freq='1S')
        
        # Create attack labels
        # Normal operation for first 80% of data
        normal_samples = int(0.8 * n_samples)
        attack_samples = n_samples - normal_samples
        
        labels = ['Normal'] * normal_samples + ['Attack'] * attack_samples
        df['Normal/Attack'] = labels
        
        # Introduce anomalies in attack period
        attack_start = normal_samples
        
        # Modify some sensor readings during attack period
        for col in ['FIT_001', 'LIT_001', 'PIT_001']:
            if col in df.columns:
                # Introduce spikes and unusual values
                attack_indices = range(attack_start, n_samples)
                noise = np.random.normal(0, 2, len(attack_indices))
                df.loc[attack_start:, col] += noise
                
                # Add some extreme outliers
                outlier_indices = np.random.choice(attack_indices, size=int(0.1 * attack_samples), replace=False)
                df.loc[outlier_indices, col] *= np.random.uniform(2, 5, len(outlier_indices))
        
        # Reorder columns to put timestamp first and label last
        cols = ['Timestamp'] + [col for col in df.columns if col not in ['Timestamp', 'Normal/Attack']] + ['Normal/Attack']
        df = df[cols]
        
        logger.info(f"Created sample dataset with shape: {df.shape}")
        logger.info(f"Features: {len(df.columns) - 2}")  # Exclude timestamp and label
        logger.info(f"Normal samples: {(df['Normal/Attack'] == 'Normal').sum()}")
        logger.info(f"Attack samples: {(df['Normal/Attack'] == 'Attack').sum()}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error creating sample dataset: {e}")
        return None

def save_sample_dataset(df, output_path='data/raw/SWaT_sample.csv'):
    """Save the sample dataset"""
    logger.info(f"Saving sample dataset to: {output_path}")
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        
        logger.info("Sample dataset saved successfully!")
        logger.info(f"File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
        
        return True
        
    except Exception as e:
        logger.error(f"Error saving dataset: {e}")
        return False

def main():
    """Main function"""
    logger.info("Creating sample SWaT dataset for testing...")
    
    # Create sample dataset
    df = create_sample_swat_dataset()
    if df is None:
        logger.error("Failed to create sample dataset")
        return False
    
    # Save dataset
    success = save_sample_dataset(df)
    if success:
        logger.info("Sample SWaT dataset created successfully!")
        logger.info("You can now run: python process_swat_dataset.py")
        return True
    else:
        logger.error("Failed to save sample dataset")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
