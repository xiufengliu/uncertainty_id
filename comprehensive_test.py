#!/usr/bin/env python3
"""
Comprehensive test script to validate all components before cluster submission
Tests all critical functions and identifies potential issues
"""

import sys
import os
import numpy as np
import json
import traceback
from datetime import datetime

def test_imports():
    """Test all required imports"""
    print("Testing imports...")
    
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        print(f"✅ PyTorch {torch.__version__}")
        
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
        
        import pandas as pd
        print(f"✅ Pandas {pd.__version__}")
        
        import sklearn
        print(f"✅ Scikit-learn {sklearn.__version__}")
        
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import confusion_matrix, accuracy_score
        print("✅ Scikit-learn components")
        
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_datasets():
    """Test dataset loading"""
    print("\nTesting dataset loading...")
    
    datasets = ['nsl_kdd', 'cicids2017', 'unsw_nb15', 'swat']
    all_good = True
    
    for dataset in datasets:
        try:
            data_dir = f'data/processed/{dataset}'
            
            if not os.path.exists(data_dir):
                print(f"❌ {dataset}: directory not found")
                all_good = False
                continue
            
            # Test loading
            X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
            X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
            y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
            y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
            
            # Validate data
            assert X_train.shape[1] == X_test.shape[1], "Feature dimension mismatch"
            assert len(X_train) == len(y_train), "Train data length mismatch"
            assert len(X_test) == len(y_test), "Test data length mismatch"
            assert not np.any(np.isnan(X_train)), "NaN values in training data"
            assert not np.any(np.isnan(X_test)), "NaN values in test data"
            
            print(f"✅ {dataset}: {X_train.shape} train, {X_test.shape} test")
            
        except Exception as e:
            print(f"❌ {dataset}: {e}")
            all_good = False
    
    return all_good

def test_model_creation():
    """Test model creation and basic operations"""
    print("\nTesting model creation...")
    
    try:
        # Import the model class
        sys.path.append('.')
        from cluster_experiments import SingleLayerTransformer, BayesianEnsembleTransformer
        
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Test single model
        model = SingleLayerTransformer(input_dim=50).to(device)
        x = torch.randn(32, 50).to(device)
        output = model(x)
        
        assert output.shape == (32,), f"Wrong output shape: {output.shape}"
        assert torch.all(output >= 0) and torch.all(output <= 1), "Output not in [0,1] range"
        
        print(f"✅ SingleLayerTransformer: input {x.shape} -> output {output.shape}")
        
        # Test ensemble
        ensemble = BayesianEnsembleTransformer(input_dim=50, ensemble_size=3, device=device)
        print(f"✅ BayesianEnsembleTransformer: {len(ensemble.models)} models")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        traceback.print_exc()
        return False

def test_metrics_calculation():
    """Test metrics calculation function"""
    print("\nTesting metrics calculation...")
    
    try:
        from cluster_experiments import calculate_metrics
        
        # Test with perfect predictions
        targets = np.array([0, 0, 1, 1, 0, 1])
        predictions = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])
        
        metrics = calculate_metrics(predictions, targets)
        
        required_keys = ['fpr', 'precision', 'recall', 'f1', 'accuracy', 'auc']
        for key in required_keys:
            assert key in metrics, f"Missing metric: {key}"
            assert isinstance(metrics[key], (int, float)), f"Invalid metric type: {key}"
            assert 0 <= metrics[key] <= 1, f"Metric out of range: {key} = {metrics[key]}"
        
        print(f"✅ Metrics calculation: {metrics}")
        
        # Test edge cases
        # All same class
        targets_same = np.array([0, 0, 0, 0])
        predictions_same = np.array([0.1, 0.2, 0.3, 0.4])
        metrics_same = calculate_metrics(predictions_same, targets_same)
        print(f"✅ Edge case (same class): {metrics_same}")
        
        return True
        
    except Exception as e:
        print(f"❌ Metrics calculation failed: {e}")
        traceback.print_exc()
        return False

def test_data_loaders():
    """Test data loader creation"""
    print("\nTesting data loader creation...")
    
    try:
        from cluster_experiments import create_data_loaders
        import torch
        
        # Create sample data
        X_train = np.random.randn(1000, 20)
        X_test = np.random.randn(200, 20)
        y_train = np.random.randint(0, 2, 1000)
        y_test = np.random.randint(0, 2, 200)
        
        train_loader, val_loader, test_loader = create_data_loaders(
            X_train, X_test, y_train, y_test, batch_size=64
        )
        
        # Test one batch
        for batch_x, batch_y in train_loader:
            assert batch_x.shape[1] == 20, "Wrong feature dimension"
            assert len(batch_x) == len(batch_y), "Batch size mismatch"
            break
        
        print(f"✅ Data loaders: Train={len(train_loader)}, Val={len(val_loader)}, Test={len(test_loader)} batches")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loader creation failed: {e}")
        traceback.print_exc()
        return False

def test_baseline_models():
    """Test baseline model training"""
    print("\nTesting baseline models...")
    
    try:
        from cluster_experiments import run_baseline_experiments
        
        # Create sample data
        X_train = np.random.randn(500, 10)
        X_test = np.random.randn(100, 10)
        y_train = np.random.randint(0, 2, 500)
        y_test = np.random.randint(0, 2, 100)
        
        results = run_baseline_experiments(X_train, X_test, y_train, y_test, "test")
        
        expected_methods = ['Random Forest', 'SVM', 'Logistic Regression']
        for method in expected_methods:
            assert method in results, f"Missing method: {method}"
            metrics = results[method]
            assert 'f1' in metrics, f"Missing F1 score for {method}"
        
        print(f"✅ Baseline models: {list(results.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Baseline model testing failed: {e}")
        traceback.print_exc()
        return False

def test_file_operations():
    """Test file I/O operations"""
    print("\nTesting file operations...")
    
    try:
        # Test checkpoint saving
        from cluster_experiments import save_checkpoint
        
        test_results = {
            'test_dataset': {
                'Random Forest': {'f1': 0.85, 'precision': 0.80},
                'SVM': {'f1': 0.82, 'precision': 0.78}
            }
        }
        
        save_checkpoint(test_results, 'test_dataset')
        
        # Check if file was created
        checkpoint_file = 'checkpoints/test_dataset_checkpoint.json'
        assert os.path.exists(checkpoint_file), "Checkpoint file not created"
        
        # Load and verify
        with open(checkpoint_file, 'r') as f:
            loaded_results = json.load(f)
        
        assert loaded_results == test_results, "Checkpoint data mismatch"
        
        # Clean up
        os.remove(checkpoint_file)
        
        print("✅ File operations: checkpoint save/load")
        
        return True
        
    except Exception as e:
        print(f"❌ File operations failed: {e}")
        traceback.print_exc()
        return False

def test_gpu_availability():
    """Test GPU availability and basic operations"""
    print("\nTesting GPU availability...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
            
            # Test basic GPU operations
            x = torch.randn(100, 100).to(device)
            y = torch.randn(100, 100).to(device)
            z = torch.mm(x, y)
            
            assert z.device.type == 'cuda', "GPU computation failed"
            print("✅ GPU computation test passed")
            
            # Test memory management
            torch.cuda.empty_cache()
            print("✅ GPU memory management")
            
        else:
            print("⚠️  CUDA not available, will use CPU")
        
        return True
        
    except Exception as e:
        print(f"❌ GPU testing failed: {e}")
        return False

def main():
    """Run comprehensive tests"""
    print("="*60)
    print("COMPREHENSIVE PRE-SUBMISSION TEST")
    print("="*60)
    print(f"Test started at: {datetime.now()}")
    print()
    
    tests = [
        ("Imports", test_imports),
        ("Datasets", test_datasets),
        ("Model Creation", test_model_creation),
        ("Metrics Calculation", test_metrics_calculation),
        ("Data Loaders", test_data_loaders),
        ("Baseline Models", test_baseline_models),
        ("File Operations", test_file_operations),
        ("GPU Availability", test_gpu_availability)
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20}: {status}")
        if result:
            passed += 1
    
    print("="*60)
    print(f"PASSED: {passed}/{total}")
    
    if passed == total:
        print("🚀 ALL TESTS PASSED - READY FOR CLUSTER SUBMISSION!")
        print("   Run: ./prepare_and_submit.sh")
    else:
        print("⚠️  SOME TESTS FAILED - FIX ISSUES BEFORE SUBMISSION")
        print("   Check the error messages above")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
