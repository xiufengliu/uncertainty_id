#!/usr/bin/env python3
"""
Test script to verify local Python installation works for cluster experiments
"""

import sys
import os
from datetime import datetime

def test_basic_imports():
    """Test basic Python imports"""
    print("Testing basic imports...")
    
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print(f"✅ Pandas {pd.__version__}")
    except ImportError as e:
        print(f"❌ Pandas import failed: {e}")
        return False
    
    try:
        import sklearn
        print(f"✅ Scikit-learn {sklearn.__version__}")
    except ImportError as e:
        print(f"❌ Scikit-learn import failed: {e}")
        return False
    
    return True

def test_pytorch():
    """Test PyTorch installation"""
    print("\nTesting PyTorch...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA version: {torch.version.cuda}")
            print(f"✅ GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        
        return True
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False

def test_dataset_access():
    """Test dataset file access"""
    print("\nTesting dataset access...")
    
    datasets = ['nsl_kdd', 'cicids2017', 'unsw_nb15', 'swat']
    all_good = True
    
    for dataset in datasets:
        data_dir = f'data/processed/{dataset}'
        if os.path.exists(data_dir):
            files = os.listdir(data_dir)
            print(f"✅ {dataset}: {len(files)} files")
        else:
            print(f"❌ {dataset}: directory not found")
            all_good = False
    
    return all_good

def test_gpu_simple():
    """Test simple GPU computation"""
    print("\nTesting GPU computation...")
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("⚠️  CUDA not available, skipping GPU test")
            return True
        
        device = torch.device('cuda')
        x = torch.randn(100, 100).to(device)
        y = torch.randn(100, 100).to(device)
        z = torch.mm(x, y)
        
        print(f"✅ GPU computation successful")
        print(f"   Device: {z.device}")
        print(f"   Result shape: {z.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ GPU computation failed: {e}")
        return False

def main():
    """Main test function"""
    print("="*60)
    print("LOCAL PYTHON ENVIRONMENT TEST")
    print("="*60)
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Test time: {datetime.now()}")
    print("="*60)
    
    # Run tests
    tests = [
        ("Basic Imports", test_basic_imports),
        ("PyTorch", test_pytorch),
        ("Dataset Access", test_dataset_access),
        ("GPU Computation", test_gpu_simple)
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 30)
        results[test_name] = test_func()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_passed = True
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20}: {status}")
        if not result:
            all_passed = False
    
    print("="*60)
    if all_passed:
        print("🚀 LOCAL PYTHON ENVIRONMENT IS READY!")
        print("   You can now submit the cluster job:")
        print("   bsub < submit_cluster_experiments.sh")
    else:
        print("⚠️  SOME TESTS FAILED - FIX BEFORE SUBMITTING JOB")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
