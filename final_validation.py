#!/usr/bin/env python3
"""
Final validation script before cluster submission
Comprehensive check of all components and dependencies
"""

import os
import sys
import json
import subprocess
from datetime import datetime

def check_file_permissions():
    """Check that all scripts are executable"""
    print("🔍 Checking file permissions...")
    
    scripts = [
        'submit_cluster_experiments.sh',
        'comprehensive_experiments.py',
        'analyze_comprehensive_results.py',
        'comprehensive_test.py',
        'prepare_and_submit.sh'
    ]
    
    all_good = True
    for script in scripts:
        if os.path.exists(script):
            if os.access(script, os.X_OK):
                print(f"✅ {script} is executable")
            else:
                print(f"❌ {script} is NOT executable")
                all_good = False
        else:
            print(f"❌ {script} does not exist")
            all_good = False
    
    return all_good

def check_datasets():
    """Check all datasets are present and valid"""
    print("\n🔍 Checking datasets...")
    
    datasets = ['nsl_kdd', 'cicids2017', 'unsw_nb15', 'swat']
    required_files = ['X_train.npy', 'X_test.npy', 'y_train.npy', 'y_test.npy']
    
    all_good = True
    for dataset in datasets:
        data_dir = f'data/processed/{dataset}'
        if not os.path.exists(data_dir):
            print(f"❌ {dataset}: directory missing")
            all_good = False
            continue
        
        missing_files = []
        for file in required_files:
            if not os.path.exists(os.path.join(data_dir, file)):
                missing_files.append(file)
        
        if missing_files:
            print(f"❌ {dataset}: missing {missing_files}")
            all_good = False
        else:
            print(f"✅ {dataset}: all files present")
    
    return all_good

def check_directories():
    """Check all required directories exist"""
    print("\n🔍 Checking directories...")
    
    required_dirs = [
        'logs',
        'experiment_results', 
        'figures',
        'checkpoints',
        'data/processed'
    ]
    
    all_good = True
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path}")
        else:
            print(f"❌ {dir_path} missing")
            os.makedirs(dir_path, exist_ok=True)
            print(f"✅ {dir_path} created")
    
    return all_good

def check_python_imports():
    """Check critical Python imports"""
    print("\n🔍 Checking Python imports...")
    
    critical_imports = [
        'torch',
        'numpy', 
        'pandas',
        'sklearn',
        'matplotlib',
        'json',
        'os'
    ]
    
    all_good = True
    for module in critical_imports:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module} not available")
            all_good = False
    
    return all_good

def check_script_syntax():
    """Check Python scripts for syntax errors"""
    print("\n🔍 Checking script syntax...")
    
    python_scripts = [
        'comprehensive_experiments.py',
        'analyze_comprehensive_results.py', 
        'comprehensive_test.py',
        'cluster_experiments.py'
    ]
    
    all_good = True
    for script in python_scripts:
        if os.path.exists(script):
            try:
                result = subprocess.run([sys.executable, '-m', 'py_compile', script], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"✅ {script} syntax OK")
                else:
                    print(f"❌ {script} syntax error: {result.stderr}")
                    all_good = False
            except Exception as e:
                print(f"❌ {script} check failed: {e}")
                all_good = False
        else:
            print(f"❌ {script} not found")
            all_good = False
    
    return all_good

def estimate_runtime():
    """Estimate total runtime for comprehensive experiments"""
    print("\n🔍 Estimating runtime...")
    
    # Runtime estimates based on dataset sizes and experiment complexity (optimized)
    dataset_times = {
        'nsl_kdd': 1.5,      # hours (optimized)
        'cicids2017': 4.0,   # hours (optimized, largest dataset)
        'unsw_nb15': 2.0,    # hours (optimized)
        'swat': 0.5          # hours (smallest dataset)
    }

    experiment_components = {
        'baseline_comparisons': 1.1,    # multiplier (optimized)
        'ablation_studies': 1.8,       # multiplier (reduced experiments)
        'theoretical_validation': 1.2,  # multiplier (optimized)
        'figure_generation': 0.5       # hours
    }
    
    total_time = 0
    for dataset, base_time in dataset_times.items():
        dataset_total = base_time
        for component, multiplier in experiment_components.items():
            if component != 'figure_generation':
                dataset_total *= multiplier
        total_time += dataset_total
    
    total_time += experiment_components['figure_generation']
    
    print(f"📊 Estimated total runtime: {total_time:.1f} hours")
    print(f"📊 Job time limit: 24.0 hours")
    
    if total_time <= 20.0:  # Leave 4 hours buffer
        print("✅ Runtime estimate within time limit")
        return True
    else:
        print("⚠️  Runtime estimate may exceed time limit")
        return False

def check_cluster_compatibility():
    """Check cluster-specific compatibility"""
    print("\n🔍 Checking cluster compatibility...")
    
    # Check job script
    if os.path.exists('submit_cluster_experiments.sh'):
        with open('submit_cluster_experiments.sh', 'r') as f:
            content = f.read()
            
        checks = [
            ('CUDA module', 'cuda/12.8.1' in content),
            ('GPU queue', 'gpua100' in content),
            ('Memory allocation', '32GB' in content),
            ('Time limit', '24:00' in content),
            ('Comprehensive script', 'comprehensive_experiments.py' in content)
        ]
        
        all_good = True
        for check_name, passed in checks:
            if passed:
                print(f"✅ {check_name}")
            else:
                print(f"❌ {check_name}")
                all_good = False
        
        return all_good
    else:
        print("❌ Job submission script not found")
        return False

def generate_submission_summary():
    """Generate final submission summary"""
    print("\n" + "="*60)
    print("FINAL SUBMISSION SUMMARY")
    print("="*60)
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'datasets': 4,
        'experimental_components': [
            'Baseline Comparisons (RF, SVM, LR)',
            'Bayesian Ensemble Transformer',
            'Ablation Studies (ensemble size, dimensions, hyperparameters)',
            'Theoretical Validation (convergence, uncertainty quality)',
            'Figure Generation (5 PDF figures)',
            'Table Generation (6+ LaTeX tables)'
        ],
        'expected_outputs': {
            'figures': [
                'ensemble_size_analysis.pdf',
                'convergence_analysis.pdf', 
                'uncertainty_distribution.pdf',
                'reliability_diagram.pdf',
                'confidence_histogram.pdf'
            ],
            'tables': [
                'baseline_comparison_table.tex',
                'hyperparameters_table.tex',
                'performance_analysis_table.tex',
                'convergence_analysis_table.tex',
                'adversarial_robustness_table.tex'
            ],
            'data': [
                'comprehensive_results.json',
                'comprehensive_experimental_report.txt'
            ]
        },
        'cluster_config': {
            'queue': 'gpua100',
            'gpu': 'NVIDIA A100 (exclusive)',
            'memory': '32GB',
            'time_limit': '24 hours',
            'cuda_version': '12.8.1'
        }
    }
    
    # Save summary
    with open('final_submission_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("📋 Experimental Components:")
    for component in summary['experimental_components']:
        print(f"   • {component}")
    
    print(f"\n📊 Expected Outputs:")
    print(f"   • {len(summary['expected_outputs']['figures'])} PDF figures")
    print(f"   • {len(summary['expected_outputs']['tables'])} LaTeX tables")
    print(f"   • {len(summary['expected_outputs']['data'])} data files")
    
    print(f"\n🖥️  Cluster Configuration:")
    for key, value in summary['cluster_config'].items():
        print(f"   • {key}: {value}")
    
    print(f"\n💾 Summary saved to: final_submission_summary.json")

def main():
    """Main validation function"""
    print("🔍 FINAL VALIDATION BEFORE CLUSTER SUBMISSION")
    print("="*60)
    print(f"Validation started at: {datetime.now()}")
    print()
    
    # Run all checks
    checks = [
        ("File Permissions", check_file_permissions),
        ("Datasets", check_datasets),
        ("Directories", check_directories),
        ("Python Imports", check_python_imports),
        ("Script Syntax", check_script_syntax),
        ("Runtime Estimate", estimate_runtime),
        ("Cluster Compatibility", check_cluster_compatibility)
    ]
    
    results = {}
    for check_name, check_func in checks:
        results[check_name] = check_func()
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    passed = 0
    total = len(checks)
    
    for check_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{check_name:<20}: {status}")
        if result:
            passed += 1
    
    print("="*60)
    print(f"PASSED: {passed}/{total}")
    
    if passed == total:
        print("\n🚀 ALL VALIDATIONS PASSED!")
        print("🎯 READY FOR CLUSTER SUBMISSION!")
        print("\nExecute: ./prepare_and_submit.sh")
        
        # Generate submission summary
        generate_submission_summary()
        
        return True
    else:
        print("\n⚠️  SOME VALIDATIONS FAILED!")
        print("🔧 Please fix the issues before submission")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
