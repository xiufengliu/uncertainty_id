# 🏗️ Project Structure

## 📁 **Clean Directory Structure**

```
IntrDetection/
├── 📄 comprehensive_experiments.py    # MAIN EXPERIMENT FILE
├── 📄 experiment_config.json         # Configuration
├── 📄 validate_comprehensive_experiments.py  # Validation
├── 📄 setup.py                       # Package setup
├── 📄 requirements.txt               # Dependencies
├── 📄 README.md                      # Project documentation
│
├── 📁 uncertainty_ids/               # CORE LIBRARY
│   ├── __init__.py
│   ├── data/                         # Data loading & preprocessing
│   ├── models/                       # Model implementations
│   ├── training/                     # Training utilities
│   ├── evaluation/                   # Evaluation metrics
│   └── utils/                        # Utility functions
│
├── 📁 data/
│   └── processed/                    # REAL DATASETS
│       ├── nsl_kdd/
│       ├── cicids2017/
│       ├── unsw_nb15/
│       └── swat/
│
├── 📁 scripts/
│   ├── cluster/
│   │   └── submit_comprehensive_experiments.sh  # ONLY SUBMISSION SCRIPT
│   ├── download_datasets.sh
│   ├── run_quick_test.sh
│   └── verify_processed_data.sh
│
├── 📁 configs/
│   └── default.json
│
├── 📁 examples/
│   └── train_nsl_kdd.py
│
├── 📁 docs/
│   ├── API_REFERENCE.md
│   ├── DEVELOPMENT_GUIDE.md
│   └── EXPERIMENTAL_SETUP.md
│
├── 📁 paper/                         # LaTeX paper
│   ├── paper.tex
│   ├── references.bib
│   └── figures/
│
├── 📁 results/                       # Experiment outputs
├── 📁 figures/                       # Generated figures
├── 📁 tables/                        # Generated tables
├── 📁 trained_models/                # Model checkpoints
├── 📁 logs/                          # Job logs
└── 📁 experiment_results/            # Detailed results
```

## 🎯 **Key Files**

### **Essential Files (DO NOT REMOVE)**
- `comprehensive_experiments.py` - Main experiment runner
- `uncertainty_ids/` - Core library implementation
- `data/processed/` - Real datasets
- `scripts/cluster/submit_comprehensive_experiments.sh` - Job submission

### **Configuration Files**
- `experiment_config.json` - Experiment parameters
- `configs/default.json` - Default model configuration

### **Validation**
- `validate_comprehensive_experiments.py` - Pre-submission validation

## 🧹 **Cleaned Up (REMOVED)**
- ❌ Multiple old submission scripts (submit_lsf_*.sh)
- ❌ Old test files (test_real_data_loading.py, etc.)
- ❌ Cache directories (__pycache__, *.egg-info)
- ❌ Backup results (backup_results/)
- ❌ Duplicate experiment runners (main_experiment.py, run_experiments.sh)

## 🚀 **How to Run Experiments**

1. **Validate setup**: `python validate_comprehensive_experiments.py`
2. **Submit job**: `bsub < scripts/cluster/submit_comprehensive_experiments.sh`
3. **Check status**: `bjobs`
4. **View results**: Check `results/`, `figures/`, `tables/` directories

## 📊 **Expected Outputs**
- `comprehensive_experiment_results.json` - All experimental results
- `figures/*.pdf` - Research figures
- `tables/*.tex` - LaTeX tables for paper
- `trained_models/*.pth` - Model checkpoints
