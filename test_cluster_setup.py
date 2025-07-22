#!/usr/bin/env python3
"""
Test script to verify cluster setup and dataset availability
"""

import os
import sys
import torch
import numpy as np
import json
from datetime import datetime

def test_environment():
    """Test the cluster environment"""
    print("="*60)
    print("CLUSTER ENVIRONMENT TEST")
    print("="*60)
    
    # Python version
    print(f"Python version: {sys.version}")
    
    # PyTorch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.current_device()}")
        print(f"GPU name: {torch.cuda.get_device_name()}")
        
        # Test GPU memory
        device = torch.device('cuda')
        print(f"GPU memory allocated: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
        print(f"GPU memory cached: {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB")
    
    print()

def test_datasets():
    """Test dataset availability and integrity"""
    print("="*60)
    print("DATASET AVAILABILITY TEST")
    print("="*60)
    
    datasets = ['nsl_kdd', 'cicids2017', 'unsw_nb15', 'swat']
    all_good = True
    
    for dataset in datasets:
        print(f"\nTesting {dataset}:")
        data_dir = f'data/processed/{dataset}'
        
        if not os.path.exists(data_dir):
            print(f"  ❌ Directory not found: {data_dir}")
            all_good = False
            continue
        
        required_files = ['X_train.npy', 'X_test.npy', 'y_train.npy', 'y_test.npy']
        missing_files = []
        
        for file in required_files:
            file_path = os.path.join(data_dir, file)
            if not os.path.exists(file_path):
                missing_files.append(file)
        
        if missing_files:
            print(f"  ❌ Missing files: {missing_files}")
            all_good = False
        else:
            # Load and check data
            try:
                X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
                X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
                y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
                y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
                
                print(f"  ✅ Dataset loaded successfully")
                print(f"     Train: {X_train.shape}, Test: {X_test.shape}")
                print(f"     Features: {X_train.shape[1]}")
                print(f"     Train classes: {np.bincount(y_train)}")
                print(f"     Test classes: {np.bincount(y_test)}")
                
                # Check for NaN or infinite values
                if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
                    print(f"  ⚠️  Warning: NaN or infinite values in training data")
                if np.any(np.isnan(X_test)) or np.any(np.isinf(X_test)):
                    print(f"  ⚠️  Warning: NaN or infinite values in test data")
                
            except Exception as e:
                print(f"  ❌ Error loading data: {e}")
                all_good = False
    
    return all_good

def test_directories():
    """Test required directories"""
    print("="*60)
    print("DIRECTORY STRUCTURE TEST")
    print("="*60)
    
    required_dirs = [
        'data/processed',
        'experiment_results',
        'figures',
        'logs',
        'checkpoints'
    ]
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path}")
        else:
            print(f"❌ {dir_path} (will be created)")
            os.makedirs(dir_path, exist_ok=True)
            print(f"✅ {dir_path} (created)")

def test_gpu_computation():
    """Test basic GPU computation"""
    print("="*60)
    print("GPU COMPUTATION TEST")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available, skipping GPU test")
        return False
    
    try:
        device = torch.device('cuda')
        
        # Create test tensors
        x = torch.randn(1000, 100).to(device)
        y = torch.randn(100, 50).to(device)
        
        # Test computation
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        result = torch.mm(x, y)
        end_time.record()
        
        torch.cuda.synchronize()
        elapsed_time = start_time.elapsed_time(end_time)
        
        print(f"✅ GPU computation successful")
        print(f"   Matrix multiplication (1000x100 @ 100x50): {elapsed_time:.2f} ms")
        print(f"   Result shape: {result.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ GPU computation failed: {e}")
        return False

def test_model_creation():
    """Test model creation and basic operations"""
    print("="*60)
    print("MODEL CREATION TEST")
    print("="*60)
    
    try:
        # Import the model class
        sys.path.append('.')
        from cluster_experiments import SingleLayerTransformer
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create a test model
        model = SingleLayerTransformer(input_dim=50).to(device)
        
        # Test forward pass
        x = torch.randn(32, 50).to(device)
        output = model(x)
        
        print(f"✅ Model creation successful")
        print(f"   Input shape: {x.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Device: {device}")
        print(f"   Model parameters: {sum(p.numel() for p in model.parameters())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False

def generate_test_report():
    """Generate a comprehensive test report"""
    print("\n" + "="*60)
    print("GENERATING TEST REPORT")
    print("="*60)
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'environment': {
            'python_version': sys.version,
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
        },
        'tests': {}
    }
    
    if torch.cuda.is_available():
        report['environment']['cuda_version'] = torch.version.cuda
        report['environment']['gpu_count'] = torch.cuda.device_count()
        report['environment']['gpu_name'] = torch.cuda.get_device_name()
    
    # Run tests
    print("Running comprehensive tests...")
    
    report['tests']['datasets'] = test_datasets()
    report['tests']['gpu_computation'] = test_gpu_computation()
    report['tests']['model_creation'] = test_model_creation()
    
    # Save report
    os.makedirs('logs', exist_ok=True)
    report_file = 'logs/cluster_test_report.json'
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nTest report saved: {report_file}")
    
    # Summary
    all_tests_passed = all(report['tests'].values())
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, result in report['tests'].items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name.upper():<20}: {status}")
    
    overall_status = "✅ ALL TESTS PASSED" if all_tests_passed else "❌ SOME TESTS FAILED"
    print(f"\nOVERALL STATUS: {overall_status}")
    
    if all_tests_passed:
        print("\n🚀 Cluster is ready for experiments!")
        print("   Run: bsub < submit_cluster_experiments.sh")
    else:
        print("\n⚠️  Please fix the failed tests before running experiments.")
    
    return all_tests_passed

def main():
    """Main test function"""
    print("CLUSTER SETUP VERIFICATION")
    print("=" * 60)
    print(f"Test started at: {datetime.now()}")
    print()
    
    # Run individual tests
    test_environment()
    test_directories()
    
    # Generate comprehensive report
    success = generate_test_report()
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
