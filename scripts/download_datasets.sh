#!/bin/bash

# Script to download and prepare datasets for uncertainty-aware IDS experiments
# This script downloads the four datasets used in the paper

echo "Downloading and preparing datasets for Uncertainty-Aware IDS experiments..."

# Create data directories
mkdir -p data/{NSL-KDD,CICIDS2017,UNSW-NB15,SWaT}

# Function to download and extract files
download_and_extract() {
    local url=$1
    local output_dir=$2
    local filename=$3
    
    echo "Downloading $filename to $output_dir..."
    
    if [ ! -f "$output_dir/$filename" ]; then
        wget -O "$output_dir/$filename" "$url"
        
        # Extract if it's an archive
        case $filename in
            *.zip)
                echo "Extracting $filename..."
                unzip -o "$output_dir/$filename" -d "$output_dir/"
                ;;
            *.tar.gz)
                echo "Extracting $filename..."
                tar -xzf "$output_dir/$filename" -C "$output_dir/"
                ;;
            *.rar)
                echo "Extracting $filename..."
                unrar x "$output_dir/$filename" "$output_dir/"
                ;;
        esac
        
        echo "✓ $filename downloaded and extracted"
    else
        echo "✓ $filename already exists"
    fi
}

# NSL-KDD Dataset
echo "=========================================="
echo "Downloading NSL-KDD Dataset"
echo "=========================================="

# NSL-KDD is available from multiple sources
NSL_KDD_URLS=(
    "https://www.unb.ca/cic/datasets/nsl.html"
    "https://github.com/defcom17/NSL_KDD"
)

echo "NSL-KDD dataset needs to be downloaded manually from:"
echo "https://www.unb.ca/cic/datasets/nsl.html"
echo ""
echo "Please download the following files and place them in data/NSL-KDD/:"
echo "  - KDDTrain+.txt"
echo "  - KDDTest+.txt"
echo "  - KDDTrain+_20Percent.txt"
echo "  - KDDTest-21.txt"

# CICIDS2017 Dataset
echo "=========================================="
echo "Downloading CICIDS2017 Dataset"
echo "=========================================="

echo "CICIDS2017 dataset needs to be downloaded manually from:"
echo "https://www.unb.ca/cic/datasets/ids-2017.html"
echo ""
echo "Please download and place the CSV files in data/CICIDS2017/:"
echo "  - Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
echo "  - Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
echo "  - Friday-WorkingHours-Morning.pcap_ISCX.csv"
echo "  - Monday-WorkingHours.pcap_ISCX.csv"
echo "  - Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv"
echo "  - Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"
echo "  - Tuesday-WorkingHours.pcap_ISCX.csv"
echo "  - Wednesday-workingHours.pcap_ISCX.csv"

# UNSW-NB15 Dataset
echo "=========================================="
echo "Downloading UNSW-NB15 Dataset"
echo "=========================================="

echo "UNSW-NB15 dataset needs to be downloaded manually from:"
echo "https://research.unsw.edu.au/projects/unsw-nb15-dataset"
echo ""
echo "Please download and place the following files in data/UNSW-NB15/:"
echo "  - UNSW_NB15_training-set.csv"
echo "  - UNSW_NB15_testing-set.csv"
echo "  - UNSW_NB15_features.csv"

# SWaT Dataset
echo "=========================================="
echo "Downloading SWaT Dataset"
echo "=========================================="

echo "SWaT dataset needs to be requested from:"
echo "https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/"
echo ""
echo "Please request access and download the following files to data/SWaT/:"
echo "  - SWaT_Dataset_Normal_v1.csv"
echo "  - SWaT_Dataset_Attack_v0.csv"

# Create sample datasets for testing
echo "=========================================="
echo "Creating Sample Datasets for Testing"
echo "=========================================="

# Create sample NSL-KDD data
cat > data/NSL-KDD/sample_data.txt << 'EOF'
# Sample NSL-KDD data for testing (not real data)
# Format: duration,protocol_type,service,flag,src_bytes,dst_bytes,...,attack_type
0,tcp,http,SF,181,5450,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,8,8,0.00,0.00,0.00,0.00,1.00,0.00,0.00,9,9,1.00,0.00,0.11,0.00,0.00,0.00,0.00,0.00,normal
0,tcp,http,SF,239,486,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,8,8,0.00,0.00,0.00,0.00,1.00,0.00,0.00,19,19,1.00,0.00,0.05,0.00,0.00,0.00,0.00,0.00,normal
0,tcp,http,SF,235,1337,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,8,8,0.00,0.00,0.00,0.00,1.00,0.00,0.00,29,29,1.00,0.00,0.03,0.00,0.00,0.00,0.00,0.00,normal
EOF

echo "✓ Sample datasets created for testing"

# Create dataset verification script
cat > scripts/verify_datasets.sh << 'EOF'
#!/bin/bash

echo "Verifying dataset availability..."

datasets=(
    "data/NSL-KDD/KDDTrain+.txt:NSL-KDD Training Set"
    "data/CICIDS2017/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv:CICIDS2017 DDoS Data"
    "data/UNSW-NB15/UNSW_NB15_training-set.csv:UNSW-NB15 Training Set"
    "data/SWaT/SWaT_Dataset_Normal_v1.csv:SWaT Normal Data"
)

for dataset in "${datasets[@]}"; do
    IFS=':' read -r filepath description <<< "$dataset"
    
    if [ -f "$filepath" ]; then
        size=$(du -h "$filepath" | cut -f1)
        lines=$(wc -l < "$filepath" 2>/dev/null || echo "unknown")
        echo "✓ $description: $filepath ($size, $lines lines)"
    else
        echo "✗ $description: $filepath (not found)"
    fi
done

echo ""
echo "Dataset download instructions:"
echo "1. NSL-KDD: https://www.unb.ca/cic/datasets/nsl.html"
echo "2. CICIDS2017: https://www.unb.ca/cic/datasets/ids-2017.html"
echo "3. UNSW-NB15: https://research.unsw.edu.au/projects/unsw-nb15-dataset"
echo "4. SWaT: https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/"
EOF

chmod +x scripts/verify_datasets.sh

echo "=========================================="
echo "Dataset Download Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Download datasets manually from the provided URLs"
echo "2. Place them in the appropriate data/ subdirectories"
echo "3. Run: ./scripts/verify_datasets.sh to verify"
echo "4. Run experiments with: ./scripts/submit_experiments.sh"
echo ""
echo "Note: Due to licensing restrictions, most datasets require manual download"
echo "Sample datasets have been created for testing purposes"
