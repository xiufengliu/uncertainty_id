#!/bin/bash

# Script to verify processed datasets are available and valid

echo "Verifying processed datasets..."

datasets=("nsl_kdd" "cicids2017" "unsw_nb15" "swat")
required_files=("X_train.npy" "X_test.npy" "y_train.npy" "y_test.npy" "feature_names.txt")

for dataset in "${datasets[@]}"; do
    echo ""
    echo "Checking $dataset..."
    dataset_path="data/processed/$dataset"
    
    if [ ! -d "$dataset_path" ]; then
        echo "  ✗ Directory not found: $dataset_path"
        continue
    fi
    
    echo "  ✓ Directory exists: $dataset_path"
    
    # Check required files
    all_files_exist=true
    for file in "${required_files[@]}"; do
        if [ -f "$dataset_path/$file" ]; then
            size=$(du -h "$dataset_path/$file" | cut -f1)
            echo "  ✓ $file ($size)"
        else
            echo "  ✗ Missing: $file"
            all_files_exist=false
        fi
    done
    
    # Check optional files
    if [ -f "$dataset_path/preprocessors.pkl" ]; then
        size=$(du -h "$dataset_path/preprocessors.pkl" | cut -f1)
        echo "  ✓ preprocessors.pkl ($size)"
    else
        echo "  ⚠ Optional: preprocessors.pkl (not found)"
    fi
    
    if [ -f "$dataset_path/dataset_info.json" ]; then
        size=$(du -h "$dataset_path/dataset_info.json" | cut -f1)
        echo "  ✓ dataset_info.json ($size)"
    else
        echo "  ⚠ Optional: dataset_info.json (not found)"
    fi
    
    # Try to load and check data shapes
    if [ "$all_files_exist" = true ]; then
        echo "  Checking data shapes..."
        python -c "
import numpy as np
import sys
import os

dataset_path = '$dataset_path'
try:
    X_train = np.load(os.path.join(dataset_path, 'X_train.npy'))
    X_test = np.load(os.path.join(dataset_path, 'X_test.npy'))
    y_train = np.load(os.path.join(dataset_path, 'y_train.npy'))
    y_test = np.load(os.path.join(dataset_path, 'y_test.npy'))
    
    print(f'  ✓ X_train shape: {X_train.shape}')
    print(f'  ✓ X_test shape: {X_test.shape}')
    print(f'  ✓ y_train shape: {y_train.shape}')
    print(f'  ✓ y_test shape: {y_test.shape}')
    
    # Check label distribution
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    unique_test, counts_test = np.unique(y_test, return_counts=True)
    
    print(f'  ✓ Train labels: {dict(zip(unique_train, counts_train))}')
    print(f'  ✓ Test labels: {dict(zip(unique_test, counts_test))}')
    
    # Check feature names
    with open(os.path.join(dataset_path, 'feature_names.txt'), 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
    
    print(f'  ✓ Features: {len(feature_names)} (expected: {X_train.shape[1]})')
    
    if len(feature_names) != X_train.shape[1]:
        print(f'  ⚠ Warning: Feature count mismatch!')
    
except Exception as e:
    print(f'  ✗ Error loading data: {e}')
    sys.exit(1)
" 2>/dev/null
        
        if [ $? -eq 0 ]; then
            echo "  ✓ Data validation passed"
        else
            echo "  ✗ Data validation failed"
        fi
    fi
done

echo ""
echo "=========================================="
echo "Processed Data Verification Complete"
echo "=========================================="

# Test loading with our processed_loader
echo ""
echo "Testing processed data loader..."

python -c "
import sys
sys.path.append('.')

try:
    from uncertainty_ids.data.processed_loader import load_processed_dataset, get_dataset_info
    
    datasets = ['nsl_kdd', 'cicids2017', 'unsw_nb15', 'swat']
    
    for dataset in datasets:
        try:
            print(f'Testing {dataset}...')
            train_dataset, test_dataset = load_processed_dataset(dataset)
            info = get_dataset_info(dataset)
            print(f'  ✓ {dataset}: Train={len(train_dataset)}, Test={len(test_dataset)}')
        except Exception as e:
            print(f'  ✗ {dataset}: {e}')
    
    print('✓ Processed data loader test completed')
    
except Exception as e:
    print(f'✗ Failed to import processed_loader: {e}')
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    echo "✓ All processed datasets are ready for experiments!"
    echo ""
    echo "Next steps:"
    echo "  1. Run quick test: ./run_experiments.sh --test"
    echo "  2. Run all experiments: ./run_experiments.sh --all"
    echo "  3. Run single dataset: ./run_experiments.sh --single nsl_kdd"
else
    echo "✗ Some issues found with processed datasets"
    echo "Please check the error messages above"
fi
