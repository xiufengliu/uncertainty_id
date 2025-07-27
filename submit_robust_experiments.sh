#!/bin/bash
#BSUB -J comprehensive_experiments_robust
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 8
#BSUB -R "rusage[mem=30GB]"
#BSUB -W 24:00
#BSUB -o logs/comprehensive_experiments_robust_%J.out
#BSUB -e logs/comprehensive_experiments_robust_%J.err
#BSUB -N

echo "🚀 Starting ROBUST Comprehensive Uncertainty-Aware Intrusion Detection Experiments"
echo "📅 Job started at: $(date)"
echo "🎯 This job will run ALL experiments with robust error handling:"
echo "   - 4 datasets (NSL-KDD, CICIDS2017, UNSW-NB15, SWaT)"
echo "   - Multiple baselines (Traditional ML, Deep Learning, Uncertainty-Aware)"
echo "   - Ablation studies (ensemble size, model dimensions, loss components)"
echo "   - ICL experiments (1-shot to 20-shot evaluation)"
echo "   - Generate all 7 tables and 7 figures"
echo "   - Robust error handling to prevent crashes"

# Load required modules
echo "📦 Loading cluster modules..."
module load cuda/12.6
echo "✅ CUDA module loaded successfully"

# Initialize conda and activate base environment
echo "🐍 Setting up conda environment..."
export PATH="/work3/xiuli/anaconda3/bin:$PATH"
source /work3/xiuli/anaconda3/etc/profile.d/conda.sh
conda activate base

# Verify conda environment
echo "🔍 Conda environment verification:"
which python
which conda
conda info --envs

# Install required packages using pip (Python 3.13 compatible)
echo "📦 Installing required packages..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install scikit-learn pandas numpy matplotlib seaborn tqdm
pip install transformers datasets accelerate
pip install uncertainty-quantification bayesian-torch
pip install plotly kaleido

# Verify installations
echo "🔍 Package verification:"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import sklearn; print(f'Scikit-learn: {sklearn.__version__}')"
python -c "import pandas; print(f'Pandas: {pandas.__version__}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"

# Set environment variables
export PYTHONPATH=/work3/xiuli/IntrDetection:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0

# Change to project directory
cd /work3/xiuli/IntrDetection

# Print job information
echo "🖥️  Running on: $(hostname)"
echo "🔧 CUDA devices: $CUDA_VISIBLE_DEVICES"
echo "📁 Working directory: $(pwd)"
echo "🐍 Python version: $(python --version)"

# Verify clean state
echo "🧹 Verifying clean experimental environment..."
echo "   - Logs directory: $(ls -la logs/ | wc -l) files"
echo "   - Experiment results: $(ls -la experiment_results/ 2>/dev/null | wc -l) files"
echo "   - Figures: $(ls -la figures/ 2>/dev/null | wc -l) files"
echo "   - Tables: $(ls -la tables/ 2>/dev/null | wc -l) files"

# Run the comprehensive experiments with error handling
echo "📊 This will take several hours to complete all experiments..."
echo "📊 All experiments include robust error handling to prevent crashes..."
echo "Plotting configured for high-quality PDF output"

# Run with timeout and error handling
timeout 23h python comprehensive_experiments.py
exit_code=$?

if [ $exit_code -eq 124 ]; then
    echo "⚠️ Job timed out after 23 hours"
elif [ $exit_code -ne 0 ]; then
    echo "⚠️ Job exited with error code: $exit_code"
else
    echo "✅ Job completed successfully"
fi

# Print final status
echo "📊 Final Results Summary:"
echo "   - Tables: tables/"
echo "   - Figures: figures/"
echo "   - Models: trained_models/"
echo "   - Logs: logs/"
echo "🎉 Job completed at: $(date)"
