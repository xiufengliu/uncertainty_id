#!/bin/bash
#BSUB -J uncertainty_ids_experiments
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 8
#BSUB -R "rusage[mem=32GB]"
#BSUB -W 24:00
#BSUB -o logs/cluster_experiments_%J.out
#BSUB -e logs/cluster_experiments_%J.err
#BSUB -N

# DTU GPU Cluster Job Submission Script for Uncertainty-Aware Intrusion Detection
# Runs experiments on all 4 datasets: NSL-KDD, CICIDS2017, UNSW-NB15, SWaT

echo "=========================================="
echo "Starting Uncertainty-Aware IDS Experiments"
echo "Job ID: $LSB_JOBID"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "=========================================="

# Set working directory
cd /zhome/bb/9/101964/xiuli/IntrDetection

# Create necessary directories
mkdir -p logs
mkdir -p experiment_results
mkdir -p figures
mkdir -p checkpoints

# Load CUDA module
echo "Loading CUDA module..."
module load cuda/12.8.1

# Use local Python installation
echo "Using local Python installation..."
export PATH="/zhome/bb/9/101964/xiuli/anaconda3/bin:$PATH"

# Try to activate conda environment with error handling
if [ -f "/zhome/bb/9/101964/xiuli/anaconda3/etc/profile.d/conda.sh" ]; then
    echo "Activating conda environment..."
    source /zhome/bb/9/101964/xiuli/anaconda3/etc/profile.d/conda.sh
    conda activate base
    if [ $? -ne 0 ]; then
        echo "Warning: conda activation failed, using system Python"
    fi
else
    echo "Warning: conda not found, using system Python"
fi

# Check GPU availability
echo "GPU Information:"
nvidia-smi
echo ""

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export OMP_NUM_THREADS=8

# Check Python environment
echo "Checking Python environment..."
which python
python --version
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Verify all datasets are available
echo "Verifying datasets..."
for dataset in nsl_kdd cicids2017 unsw_nb15 swat; do
    if [ -d "data/processed/$dataset" ]; then
        echo "✓ Dataset $dataset found"
        ls -la "data/processed/$dataset/"
    else
        echo "✗ Dataset $dataset NOT found"
        exit 1
    fi
done

echo ""
echo "Starting experiments..."
echo "=========================================="

# Run comprehensive experiments (includes baselines + ablations + theoretical validation)
echo "Running comprehensive experiments including all paper components..."
python comprehensive_experiments.py 2>&1 | tee logs/comprehensive_experiment_log_${LSB_JOBID}.txt

# Check if experiments completed successfully
if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "Experiments completed successfully!"
    echo "Results saved to: experiment_results/"
    echo "Figures saved to: figures/"
    echo "=========================================="
    
    # Display summary of results
    echo "Comprehensive Results Summary:"
    if [ -f "experiment_results/comprehensive_results.json" ]; then
        python -c "
import json
with open('experiment_results/comprehensive_results.json', 'r') as f:
    results = json.load(f)
    
print('\\n' + '='*80)
print('COMPREHENSIVE EXPERIMENTAL RESULTS SUMMARY')
print('='*80)

for dataset, dataset_results in results.items():
    print(f'\\n{dataset.upper()} Dataset Results:')
    print('='*60)

    # Baseline results
    if 'baselines' in dataset_results:
        print('\\nBaseline Methods:')
        print('-' * 50)
        print(f'{'Method':<25} {'FPR':<8} {'Precision':<10} {'Recall':<8} {'F1':<8}')
        print('-' * 50)

        for method, metrics in dataset_results['baselines'].items():
            fpr = metrics.get('fpr', 0)
            precision = metrics.get('precision', 0)
            recall = metrics.get('recall', 0)
            f1 = metrics.get('f1', 0)
            print(f'{method:<25} {fpr:<8.4f} {precision:<10.4f} {recall:<8.4f} {f1:<8.4f}')

    # Main method results
    if 'main_method' in dataset_results:
        main = dataset_results['main_method']
        print('\\nMain Method (Bayesian Ensemble Transformer):')
        print('-' * 50)
        print(f'FPR: {main.get(\"fpr\", 0):.4f}, Precision: {main.get(\"precision\", 0):.4f}')
        print(f'Recall: {main.get(\"recall\", 0):.4f}, F1: {main.get(\"f1\", 0):.4f}')
        if 'uncertainty_quality' in main:
            uq = main['uncertainty_quality']
            print(f'ECE: {uq.get(\"ece\", 0):.4f}, AURC: {uq.get(\"aurc\", 0):.4f}')

    # Ablation study highlights
    if 'ensemble_size_analysis' in dataset_results:
        esa = dataset_results['ensemble_size_analysis']
        best_size = max(esa.keys(), key=lambda k: esa[k].get('f1', 0)) if esa else 'N/A'
        print(f'\\nBest Ensemble Size: {best_size}')

    if 'convergence_analysis' in dataset_results:
        conv = dataset_results['convergence_analysis']
        print(f'Convergence Rate: {conv.get(\"empirical_rate\", 0):.6f}')
"
    fi
else
    echo "=========================================="
    echo "Experiments failed! Check error logs."
    echo "=========================================="
    exit 1
fi

echo ""
echo "Job completed at: $(date)"
echo "Total runtime: $SECONDS seconds"
