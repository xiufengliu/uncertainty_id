#!/bin/bash

# Main experiment runner for Uncertainty-Aware Intrusion Detection
# This script provides a unified interface for running all experiments

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

print_header() {
    echo ""
    echo "=========================================="
    echo "$1"
    echo "=========================================="
}

# Check if running on LSF cluster
check_lsf() {
    if command -v bsub &> /dev/null; then
        echo "LSF detected - will use job submission"
        return 0
    else
        echo "No LSF detected - will run locally"
        return 1
    fi
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help              Show this help message"
    echo "  -t, --test              Run quick test with synthetic data"
    echo "  -d, --download          Download and setup datasets"
    echo "  -v, --verify            Verify dataset availability"
    echo "  -a, --all               Run all experiments (requires datasets)"
    echo "  -s, --single DATASET    Run experiments for single dataset"
    echo "  -i, --individual        Submit individual jobs for each dataset"
    echo "  -c, --config CONFIG     Use custom configuration file"
    echo "  --local                 Force local execution (no LSF)"
    echo ""
    echo "Datasets: nsl_kdd, cicids2017, unsw_nb15, swat"
    echo ""
    echo "Examples:"
    echo "  $0 --test                    # Quick test with synthetic data"
    echo "  $0 --download                # Download datasets"
    echo "  $0 --all                     # Run all experiments"
    echo "  $0 --single nsl_kdd          # Run NSL-KDD experiments only"
    echo "  $0 --individual              # Submit separate jobs for each dataset"
}

# Function to run quick test
run_quick_test() {
    print_header "Running Quick Test"
    print_status $BLUE "Testing implementation with synthetic data..."
    
    if [ -f "scripts/run_quick_test.sh" ]; then
        ./scripts/run_quick_test.sh
    else
        print_status $RED "Error: scripts/run_quick_test.sh not found"
        exit 1
    fi
}

# Function to download datasets
download_datasets() {
    print_header "Dataset Download Setup"
    print_status $BLUE "Setting up dataset download instructions..."
    
    if [ -f "scripts/download_datasets.sh" ]; then
        ./scripts/download_datasets.sh
    else
        print_status $RED "Error: scripts/download_datasets.sh not found"
        exit 1
    fi
}

# Function to verify datasets
verify_datasets() {
    print_header "Verifying Datasets"
    
    if [ -f "scripts/verify_datasets.sh" ]; then
        ./scripts/verify_datasets.sh
    else
        print_status $RED "Error: scripts/verify_datasets.sh not found"
        exit 1
    fi
}

# Function to run all experiments
run_all_experiments() {
    local config_file=$1
    local use_slurm=$2
    
    print_header "Running All Experiments"
    
    if [ "$use_lsf" = true ]; then
        print_status $BLUE "Submitting LSF job for all experiments..."
        if [ -f "scripts/submit_experiments.sh" ]; then
            bsub < scripts/submit_experiments.sh
            print_status $GREEN "Job submitted! Monitor with: bjobs -u $USER"
        else
            print_status $RED "Error: scripts/submit_experiments.sh not found"
            exit 1
        fi
    else
        print_status $BLUE "Running experiments locally..."
        # Run experiments locally (not recommended for large datasets)
        print_status $YELLOW "Warning: Running locally may take a very long time"
        read -p "Continue? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            bash scripts/submit_experiments.sh
        else
            print_status $YELLOW "Cancelled by user"
            exit 0
        fi
    fi
}

# Function to run single dataset experiment
run_single_experiment() {
    local dataset=$1
    local config_file=$2
    local use_slurm=$3
    
    print_header "Running $dataset Experiments"
    
    # Define dataset paths (using processed data)
    case $dataset in
        nsl_kdd)
            data_path="data/processed/nsl_kdd"
            ;;
        cicids2017)
            data_path="data/processed/cicids2017"
            ;;
        unsw_nb15)
            data_path="data/processed/unsw_nb15"
            ;;
        swat)
            data_path="data/processed/swat"
            ;;
        *)
            print_status $RED "Error: Unknown dataset '$dataset'"
            print_status $YELLOW "Available datasets: nsl_kdd, cicids2017, unsw_nb15, swat"
            exit 1
            ;;
    esac

    # Check if processed dataset exists
    if [ ! -d "$data_path" ]; then
        print_status $RED "Error: Processed dataset not found at $data_path"
        print_status $YELLOW "Available processed datasets:"
        ls -la data/processed/ 2>/dev/null || echo "  No processed datasets found"
        exit 1
    fi
    
    if [ "$use_lsf" = true ]; then
        print_status $BLUE "Submitting LSF job for $dataset..."
        if [ -f "scripts/individual/run_${dataset}.sh" ]; then
            bsub < "scripts/individual/run_${dataset}.sh"
            print_status $GREEN "Job submitted! Monitor with: bjobs -u $USER"
        else
            print_status $RED "Error: scripts/individual/run_${dataset}.sh not found"
            print_status $YELLOW "Run: ./scripts/submit_individual_experiments.sh to create individual scripts"
            exit 1
        fi
    else
        print_status $BLUE "Running $dataset experiments locally..."
        
        # Run standard experiment
        print_status $BLUE "Running standard experiment..."
        python main_experiment.py \
            --dataset $dataset \
            --data_path $data_path \
            --experiment_type standard \
            --config $config_file \
            --output_dir results/$dataset \
            --device auto \
            --log_level INFO
        
        # Run ICL experiment
        print_status $BLUE "Running ICL experiment..."
        python main_experiment.py \
            --dataset $dataset \
            --data_path $data_path \
            --experiment_type icl \
            --config $config_file \
            --output_dir results/$dataset \
            --device auto \
            --log_level INFO
        
        print_status $GREEN "$dataset experiments completed!"
    fi
}

# Function to submit individual jobs
submit_individual_jobs() {
    print_header "Submitting Individual Jobs"
    
    # Create individual job scripts if they don't exist
    if [ ! -d "scripts/individual" ]; then
        print_status $BLUE "Creating individual job scripts..."
        ./scripts/submit_individual_experiments.sh
    fi
    
    # Submit all individual jobs
    if [ -f "scripts/submit_all_individual.sh" ]; then
        ./scripts/submit_all_individual.sh
    else
        print_status $RED "Error: scripts/submit_all_individual.sh not found"
        exit 1
    fi
}

# Main script logic
main() {
    # Default values
    config_file="configs/default.json"
    use_lsf=true

    # Check for LSF
    if ! check_lsf; then
        use_lsf=false
    fi
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_usage
                exit 0
                ;;
            -t|--test)
                run_quick_test
                exit 0
                ;;
            -d|--download)
                download_datasets
                exit 0
                ;;
            -v|--verify)
                verify_datasets
                exit 0
                ;;
            -a|--all)
                run_all_experiments $config_file $use_lsf
                exit 0
                ;;
            -s|--single)
                if [ -z "$2" ]; then
                    print_status $RED "Error: Dataset name required for --single"
                    show_usage
                    exit 1
                fi
                run_single_experiment $2 $config_file $use_lsf
                exit 0
                ;;
            -i|--individual)
                if [ "$use_lsf" = false ]; then
                    print_status $RED "Error: Individual job submission requires LSF"
                    exit 1
                fi
                submit_individual_jobs
                exit 0
                ;;
            -c|--config)
                if [ -z "$2" ]; then
                    print_status $RED "Error: Configuration file required for --config"
                    exit 1
                fi
                config_file=$2
                shift
                ;;
            --local)
                use_lsf=false
                ;;
            *)
                print_status $RED "Error: Unknown option '$1'"
                show_usage
                exit 1
                ;;
        esac
        shift
    done
    
    # If no arguments provided, show usage
    show_usage
}

# Create necessary directories
mkdir -p logs results data scripts/individual

# Print header
print_header "Uncertainty-Aware Intrusion Detection Experiments"
print_status $BLUE "CUDA Version: 12.9.1"
print_status $BLUE "PyTorch Version: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"
print_status $BLUE "CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'Unknown')"

# Run main function
main "$@"
