#!/usr/bin/env python3
"""
Data preprocessing script for uncertainty-aware intrusion detection experiments.
Handles NSL-KDD, CICIDS2017, and UNSW-NB15 datasets according to paper specifications.
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

class DatasetPreprocessor:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.scalers = {}
        self.encoders = {}
        
    def preprocess_nsl_kdd(self):
        """Preprocess NSL-KDD dataset"""
        print("Processing NSL-KDD dataset...")
        
        # Column names for NSL-KDD
        columns = [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
            'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
            'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
            'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
            'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
            'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
            'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
            'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
            'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty'
        ]
        
        # Load training and test data
        train_path = os.path.join(self.data_dir, 'nsl-kdd', 'KDDTrain+.txt')
        test_path = os.path.join(self.data_dir, 'nsl-kdd', 'KDDTest+.txt')
        
        train_df = pd.read_csv(train_path, names=columns, header=None)
        test_df = pd.read_csv(test_path, names=columns, header=None)
        
        # Remove difficulty column
        train_df = train_df.drop('difficulty', axis=1)
        test_df = test_df.drop('difficulty', axis=1)
        
        # Binary classification: normal vs attack
        train_df['label'] = train_df['label'].apply(lambda x: 0 if x == 'normal' else 1)
        test_df['label'] = test_df['label'].apply(lambda x: 0 if x == 'normal' else 1)
        
        # Separate features and labels
        X_train = train_df.drop('label', axis=1)
        y_train = train_df['label']
        X_test = test_df.drop('label', axis=1)
        y_test = test_df['label']
        
        # Process features
        X_train_processed, X_test_processed = self._process_features(
            X_train, X_test, 'nsl_kdd'
        )
        
        return X_train_processed, X_test_processed, y_train, y_test
    
    def preprocess_cicids2017(self):
        """Preprocess CICIDS2017 dataset"""
        print("Processing CICIDS2017 dataset...")
        
        # Load all CSV files from TrafficLabelling directory
        csv_files = glob.glob(os.path.join(self.data_dir, 'TrafficLabelling*', '*.csv'))
        
        dfs = []
        for file in csv_files:
            try:
                df = pd.read_csv(file)
                dfs.append(df)
                print(f"Loaded {file}: {len(df)} samples")
            except Exception as e:
                print(f"Error loading {file}: {e}")
                continue
        
        # Combine all dataframes
        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"Combined dataset size: {len(combined_df)}")
        
        # Clean column names
        combined_df.columns = combined_df.columns.str.strip()
        
        # Handle label column (usually the last column)
        label_col = combined_df.columns[-1]
        if 'Label' in combined_df.columns:
            label_col = 'Label'
        elif ' Label' in combined_df.columns:
            label_col = ' Label'
        
        # Binary classification: BENIGN vs attack
        combined_df['label'] = combined_df[label_col].apply(
            lambda x: 0 if str(x).upper() == 'BENIGN' else 1
        )
        
        # Remove original label column and any non-numeric columns
        X = combined_df.drop([label_col, 'label'], axis=1)
        y = combined_df['label']
        
        # Remove non-numeric columns and handle infinite values
        X = X.select_dtypes(include=[np.number])
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        
        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Process features
        X_train_processed, X_test_processed = self._process_features(
            X_train, X_test, 'cicids2017'
        )
        
        return X_train_processed, X_test_processed, y_train, y_test
    
    def preprocess_unsw_nb15(self):
        """Preprocess UNSW-NB15 dataset"""
        print("Processing UNSW-NB15 dataset...")
        
        # Load training and test sets
        train_path = os.path.join(self.data_dir, 'UNSW_NB15_training-set.csv')
        test_path = os.path.join(self.data_dir, 'UNSW_NB15_testing-set.csv')
        
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        # Remove ID column if present
        if 'id' in train_df.columns:
            train_df = train_df.drop('id', axis=1)
        if 'id' in test_df.columns:
            test_df = test_df.drop('id', axis=1)
        
        # Binary classification using 'label' column (0=normal, 1=attack)
        X_train = train_df.drop(['label', 'attack_cat'], axis=1, errors='ignore')
        y_train = train_df['label']
        X_test = test_df.drop(['label', 'attack_cat'], axis=1, errors='ignore')
        y_test = test_df['label']
        
        # Handle non-numeric columns
        X_train = X_train.select_dtypes(include=[np.number])
        X_test = X_test.select_dtypes(include=[np.number])
        
        # Handle infinite values
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        X_test = X_test.replace([np.inf, -np.inf], np.nan)
        X_train = X_train.fillna(0)
        X_test = X_test.fillna(0)
        
        # Process features
        X_train_processed, X_test_processed = self._process_features(
            X_train, X_test, 'unsw_nb15'
        )
        
        return X_train_processed, X_test_processed, y_train, y_test
    
    def _process_features(self, X_train, X_test, dataset_name):
        """Process features according to paper specifications"""
        
        # Identify categorical and continuous features
        categorical_features = []
        continuous_features = []
        
        for col in X_train.columns:
            if X_train[col].dtype == 'object':
                categorical_features.append(col)
            else:
                continuous_features.append(col)
        
        # Process categorical features with label encoding
        X_train_processed = X_train.copy()
        X_test_processed = X_test.copy()
        
        for col in categorical_features:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                X_train_processed[col] = self.encoders[col].fit_transform(
                    X_train_processed[col].astype(str)
                )
            else:
                X_train_processed[col] = self.encoders[col].transform(
                    X_train_processed[col].astype(str)
                )
            
            # Handle unseen categories in test set
            test_categories = set(X_test_processed[col].astype(str))
            train_categories = set(self.encoders[col].classes_)
            unseen_categories = test_categories - train_categories
            
            if unseen_categories:
                # Map unseen categories to 0 (or most frequent class)
                X_test_processed[col] = X_test_processed[col].astype(str)
                X_test_processed[col] = X_test_processed[col].apply(
                    lambda x: x if x in train_categories else self.encoders[col].classes_[0]
                )
            
            X_test_processed[col] = self.encoders[col].transform(
                X_test_processed[col].astype(str)
            )
        
        # Z-score normalization for continuous features
        if continuous_features:
            scaler_name = f'{dataset_name}_scaler'
            if scaler_name not in self.scalers:
                self.scalers[scaler_name] = StandardScaler()
                X_train_processed[continuous_features] = self.scalers[scaler_name].fit_transform(
                    X_train_processed[continuous_features]
                )
            else:
                X_train_processed[continuous_features] = self.scalers[scaler_name].transform(
                    X_train_processed[continuous_features]
                )
            
            X_test_processed[continuous_features] = self.scalers[scaler_name].transform(
                X_test_processed[continuous_features]
            )
        
        return X_train_processed, X_test_processed
    
    def save_processed_data(self, dataset_name, X_train, X_test, y_train, y_test):
        """Save processed data to files"""
        output_dir = os.path.join(self.data_dir, 'processed', dataset_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save data
        np.save(os.path.join(output_dir, 'X_train.npy'), X_train.values)
        np.save(os.path.join(output_dir, 'X_test.npy'), X_test.values)
        np.save(os.path.join(output_dir, 'y_train.npy'), y_train.values)
        np.save(os.path.join(output_dir, 'y_test.npy'), y_test.values)
        
        # Save feature names
        with open(os.path.join(output_dir, 'feature_names.txt'), 'w') as f:
            for feature in X_train.columns:
                f.write(f"{feature}\n")
        
        # Save preprocessing objects
        with open(os.path.join(output_dir, 'preprocessors.pkl'), 'wb') as f:
            pickle.dump({
                'scalers': self.scalers,
                'encoders': self.encoders
            }, f)
        
        print(f"Saved processed {dataset_name} data to {output_dir}")
        print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        print(f"Class distribution - Train: {np.bincount(y_train)}, Test: {np.bincount(y_test)}")

def main():
    """Main preprocessing function"""
    preprocessor = DatasetPreprocessor()
    
    # Process each dataset
    datasets = [
        ('nsl_kdd', preprocessor.preprocess_nsl_kdd),
        ('cicids2017', preprocessor.preprocess_cicids2017),
        ('unsw_nb15', preprocessor.preprocess_unsw_nb15)
    ]
    
    for dataset_name, preprocess_func in datasets:
        try:
            print(f"\n{'='*50}")
            print(f"Processing {dataset_name.upper()} dataset")
            print(f"{'='*50}")
            
            X_train, X_test, y_train, y_test = preprocess_func()
            preprocessor.save_processed_data(dataset_name, X_train, X_test, y_train, y_test)
            
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
            continue
    
    print(f"\n{'='*50}")
    print("Data preprocessing completed!")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
