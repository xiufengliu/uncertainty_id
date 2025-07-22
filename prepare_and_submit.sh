#!/bin/bash
"""
Preparation and submission script for DTU GPU cluster experiments
This script prepares the environment and submits the job
"""

echo "=========================================="
echo "PREPARING CLUSTER EXPERIMENTS"
echo "=========================================="

# Make scripts executable
chmod +x submit_cluster_experiments.sh
chmod +x test_local_python.py
chmod +x cluster_experiments.py
chmod +x comprehensive_experiments.py
chmod +x analyze_comprehensive_results.py
chmod +x comprehensive_test.py

echo "✅ Made scripts executable"

# Run comprehensive tests first
echo ""
echo "Running comprehensive pre-submission tests..."
python comprehensive_test.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Environment test passed!"
    echo ""
    echo "=========================================="
    echo "SUBMITTING JOB TO CLUSTER"
    echo "=========================================="
    
    # Submit the job
    bsub < submit_cluster_experiments.sh
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Job submitted successfully!"
        echo ""
        echo "Monitor your job with:"
        echo "  bjobs"
        echo "  bpeek <job_id>"
        echo ""
        echo "Check results in:"
        echo "  logs/cluster_experiments_<job_id>.out"
        echo "  logs/cluster_experiments_<job_id>.err"
        echo "  experiment_results/"
        echo ""
    else
        echo "❌ Job submission failed!"
        exit 1
    fi
else
    echo ""
    echo "❌ Environment test failed!"
    echo "Please fix the issues before submitting the job."
    exit 1
fi
