# Experimental Setup and Reproduction Guide

This document provides detailed instructions for reproducing all experiments from our paper "Uncertainty-Aware Intrusion Detection with Bayesian Ensemble Transformers".

## 📋 Overview

Our experimental evaluation covers:

1. **Main Performance Comparison** - Compare against 9 baseline methods
2. **Uncertainty Quality Analysis** - Calibration and correlation metrics
3. **Adversarial Robustness** - Robustness against FGSM, PGD, C&W attacks
4. **Convergence Analysis** - Theoretical vs empirical convergence rates
5. **Ablation Studies** - Ensemble size, hyperparameter sensitivity
6. **In-Context Learning** - Meta-learning evaluation (if implemented)

## 🖥️ Hardware Requirements

### Minimum Requirements
- **GPU**: NVIDIA GPU with 8GB+ VRAM (GTX 1080 Ti, RTX 2080, etc.)
- **RAM**: 16GB system memory
- **Storage**: 50GB free space for datasets and results
- **CPU**: 8+ cores recommended

### Recommended Setup (Used in Paper)
- **GPU**: NVIDIA A100 (40GB VRAM) or V100 (32GB VRAM)
- **RAM**: 64GB+ system memory
- **Storage**: 200GB+ SSD storage
- **CPU**: 16+ cores (Intel Xeon or AMD EPYC)

## 📊 Datasets

### Automatic Download
All datasets are automatically downloaded when running experiments:

```bash
# Downloads and preprocesses all datasets
python main_experiment.py --dataset all --download_only
```

### Manual Download (if needed)

1. **NSL-KDD**
   - Source: [NSL-KDD Dataset](https://www.unb.ca/cic/datasets/nsl.html)
   - Size: ~150MB
   - Classes: Binary (Normal/Attack)

2. **CICIDS2017**
   - Source: [CICIDS2017 Dataset](https://www.unb.ca/cic/datasets/ids-2017.html)
   - Size: ~8GB
   - Classes: Multi-class (8 attack types + Normal)

3. **UNSW-NB15**
   - Source: [UNSW-NB15 Dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
   - Size: ~2GB
   - Classes: Binary (Normal/Attack)

4. **SWaT** (Optional)
   - Source: [SWaT Dataset](https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/)
   - Size: ~1GB
   - Note: Requires registration

## 🚀 Running Experiments

### 1. Complete Experimental Suite

Run all experiments from the paper:

```bash
# Submit all experiments to LSF cluster
./run_experiments.sh --all

# Or run locally (will take much longer)
./run_experiments.sh --all --local
```

### 2. Individual Dataset Experiments

```bash
# Run experiments for specific dataset
./run_experiments.sh --single nsl_kdd
./run_experiments.sh --single cicids2017
./run_experiments.sh --single unsw_nb15
```

### 3. Individual Experiment Types

```bash
# Main performance comparison
python main_experiment.py --dataset nsl_kdd --experiment_type standard

# Adversarial robustness evaluation
python main_experiment.py --dataset nsl_kdd --experiment_type adversarial

# Uncertainty analysis
python main_experiment.py --dataset nsl_kdd --experiment_type uncertainty

# Convergence analysis
python main_experiment.py --dataset nsl_kdd --experiment_type convergence
```

## ⚙️ Configuration

### Default Configuration

The default configuration is in `configs/default.json`:

```json
{
  "model": {
    "d_model": 256,
    "n_heads": 8,
    "n_layers": 6,
    "ensemble_size": 5,
    "dropout": 0.1
  },
  "training": {
    "batch_size": 256,
    "learning_rate": 0.001,
    "epochs": 100,
    "early_stopping_patience": 10
  },
  "uncertainty": {
    "mc_samples": 100,
    "temperature_scaling": true,
    "calibration_method": "temperature"
  }
}
```

### Custom Configuration

Create custom configurations for specific experiments:

```bash
# Copy default config
cp configs/default.json configs/my_experiment.json

# Edit configuration
vim configs/my_experiment.json

# Run with custom config
python main_experiment.py --config configs/my_experiment.json --dataset nsl_kdd
```

## 📈 Expected Results

### Main Performance Results

| Method | NSL-KDD F1 | CICIDS2017 F1 | UNSW-NB15 F1 |
|--------|------------|---------------|---------------|
| Random Forest | 75.2% | 98.1% | 85.3% |
| SVM | 72.8% | 97.5% | 82.1% |
| MLP | 74.1% | 98.3% | 84.7% |
| LSTM | 76.3% | 98.7% | 87.2% |
| CNN | 75.8% | 98.5% | 86.1% |
| MC Dropout | 76.9% | 98.9% | 88.4% |
| Deep Ensemble | 77.5% | 99.2% | 89.1% |
| Single Transformer | 76.1% | 98.8% | 87.8% |
| **Ours (Bayesian Ensemble)** | **77.13%** | **99.88%** | **92.06%** |

### Uncertainty Quality Results

| Dataset | ECE | Mutual Information | AURC |
|---------|-----|-------------------|------|
| NSL-KDD | 0.2046 | 0.34 bits | 0.156 |
| CICIDS2017 | 0.0003 | 0.89 bits | 0.001 |
| UNSW-NB15 | 0.0782 | 0.52 bits | 0.089 |

### Adversarial Robustness Results

| Attack | Clean Acc | Adversarial Acc | Robustness Ratio |
|--------|-----------|-----------------|------------------|
| FGSM | 77.13% | 71.85% | 0.932 |
| PGD | 77.13% | 70.92% | 0.920 |
| C&W | 77.13% | 72.41% | 0.939 |

## 🔧 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size in config
   "batch_size": 128  # or 64
   
   # Or use gradient accumulation
   "gradient_accumulation_steps": 2
   ```

2. **Dataset Download Fails**
   ```bash
   # Manual download and place in data/ directory
   mkdir -p data/raw/
   # Download datasets manually to data/raw/
   ```

3. **LSF Job Submission Fails**
   ```bash
   # Check LSF availability
   which bsub
   
   # Run locally instead
   ./run_experiments.sh --all --local
   ```

4. **Slow Training**
   ```bash
   # Use mixed precision training
   "use_amp": true
   
   # Reduce model size
   "d_model": 128
   "n_layers": 4
   ```

### Performance Optimization

1. **Enable Mixed Precision**
   ```json
   {
     "training": {
       "use_amp": true,
       "gradient_clipping": 1.0
     }
   }
   ```

2. **Optimize Data Loading**
   ```json
   {
     "data": {
       "num_workers": 8,
       "pin_memory": true,
       "prefetch_factor": 2
     }
   }
   ```

3. **Use Compiled Models** (PyTorch 2.0+)
   ```json
   {
     "model": {
       "compile": true,
       "compile_mode": "default"
     }
   }
   ```

## 📊 Result Analysis

### Generated Outputs

After running experiments, you'll find:

```
results/
├── figures/                    # PDF plots for paper
│   ├── nsl_kdd/
│   │   ├── confusion_matrix.pdf
│   │   ├── calibration_diagram.pdf
│   │   ├── uncertainty_distribution.pdf
│   │   └── training_history.pdf
│   └── ...
├── logs/                       # Training logs
├── models/                     # Saved model checkpoints
├── metrics/                    # Detailed metric files
└── tables/                     # LaTeX tables for paper
```

### Analyzing Results

```python
# Load and analyze results
import json
import pandas as pd

# Load experiment results
with open('results/nsl_kdd_results.json', 'r') as f:
    results = json.load(f)

# Print key metrics
print(f"Accuracy: {results['test_results']['accuracy']:.4f}")
print(f"F1 Score: {results['test_results']['f1_score']:.4f}")
print(f"ECE: {results['test_results']['ece']:.4f}")
```

## 🎯 Reproducing Paper Results

To exactly reproduce the results in our paper:

1. **Use the same random seeds**:
   ```json
   {
     "random_seed": 42,
     "torch_seed": 42,
     "numpy_seed": 42
   }
   ```

2. **Use the exact same hyperparameters** (see `configs/paper_reproduction.json`)

3. **Run 5 independent trials** for statistical significance:
   ```bash
   for seed in 42 123 456 789 101; do
     python main_experiment.py --dataset nsl_kdd --seed $seed
   done
   ```

4. **Compute mean and standard deviation** across trials

## 📝 Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{uncertainty_ids_2024,
  title={Uncertainty-Aware Intrusion Detection with Bayesian Ensemble Transformers},
  author={[Authors]},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2024},
  publisher={IEEE}
}
```

## 🆘 Support

For issues with reproduction:

1. Check this documentation first
2. Look at existing [GitHub Issues](https://github.com/your-repo/issues)
3. Create a new issue with:
   - Your system configuration
   - Complete error messages
   - Steps to reproduce the problem
   - Expected vs actual behavior
