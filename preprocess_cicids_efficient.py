#!/usr/bin/env python3
"""
Memory-efficient preprocessing for CICIDS2017 dataset
"""

import pandas as pd
import numpy as np
import os
import glob
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import warnings
warnings.filterwarnings('ignore')

def preprocess_cicids2017_efficient():
    """Memory-efficient preprocessing for CICIDS2017"""
    print("Processing CICIDS2017 dataset (memory-efficient)...")
    
    data_dir = 'data'
    csv_files = glob.glob(os.path.join(data_dir, 'TrafficLabelling*', '*.csv'))
    
    # Process files in chunks and sample data to manage memory
    all_samples = []
    max_samples_per_file = 50000  # Limit samples per file
    
    for file in csv_files:
        try:
            print(f"Processing {file}...")
            
            # Read file in chunks
            chunk_size = 10000
            chunks = []
            
            for chunk in pd.read_csv(file, chunksize=chunk_size, encoding='utf-8', on_bad_lines='skip'):
                # Clean column names
                chunk.columns = chunk.columns.str.strip()
                
                # Handle label column
                label_col = chunk.columns[-1]
                if 'Label' in chunk.columns:
                    label_col = 'Label'
                elif ' Label' in chunk.columns:
                    label_col = ' Label'
                
                # Binary classification
                chunk['label'] = chunk[label_col].apply(
                    lambda x: 0 if str(x).upper() == 'BENIGN' else 1
                )
                
                # Remove original label column
                chunk = chunk.drop([label_col], axis=1)
                
                # Select only numeric columns
                numeric_cols = chunk.select_dtypes(include=[np.number]).columns
                chunk = chunk[numeric_cols]
                
                # Handle infinite values
                chunk = chunk.replace([np.inf, -np.inf], np.nan)
                chunk = chunk.fillna(0)
                
                chunks.append(chunk)
                
                # Break if we have enough samples
                if len(chunks) * chunk_size >= max_samples_per_file:
                    break
            
            if chunks:
                file_df = pd.concat(chunks, ignore_index=True)
                
                # Sample if too large
                if len(file_df) > max_samples_per_file:
                    file_df = file_df.sample(n=max_samples_per_file, random_state=42)
                
                all_samples.append(file_df)
                print(f"Processed {file}: {len(file_df)} samples")
                
        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue
    
    # Combine all samples
    if not all_samples:
        raise ValueError("No data could be processed")
    
    print("Combining all samples...")
    combined_df = pd.concat(all_samples, ignore_index=True)
    print(f"Combined dataset size: {len(combined_df)}")
    
    # Further sample if still too large
    max_total_samples = 200000
    if len(combined_df) > max_total_samples:
        combined_df = combined_df.sample(n=max_total_samples, random_state=42)
        print(f"Sampled down to {len(combined_df)} samples")
    
    # Separate features and labels
    X = combined_df.drop('label', axis=1)
    y = combined_df['label']
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    # Save processed data
    output_dir = os.path.join(data_dir, 'processed', 'cicids2017')
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train_scaled.values)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test_scaled.values)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train.values)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test.values)
    
    # Save feature names
    with open(os.path.join(output_dir, 'feature_names.txt'), 'w') as f:
        for feature in X_train_scaled.columns:
            f.write(f"{feature}\n")
    
    # Save scaler
    with open(os.path.join(output_dir, 'preprocessors.pkl'), 'wb') as f:
        pickle.dump({'scaler': scaler}, f)
    
    print(f"Saved processed CICIDS2017 data to {output_dir}")
    print(f"Training set: {X_train_scaled.shape}, Test set: {X_test_scaled.shape}")
    print(f"Class distribution - Train: {np.bincount(y_train)}, Test: {np.bincount(y_test)}")

if __name__ == "__main__":
    preprocess_cicids2017_efficient()
