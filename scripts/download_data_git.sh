#!/bin/bash
# Download PKU-SafeRLHF dataset using git lfs (alternative method)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

OUTPUT_DIR="data/raw"
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Downloading PKU-SafeRLHF dataset via git"
echo "=========================================="
echo "Output directory: $PROJECT_ROOT/$OUTPUT_DIR"
echo ""

# Check if git lfs is installed
if ! command -v git-lfs &> /dev/null; then
    echo "Error: git-lfs not found."
    echo "Please install git-lfs first:"
    echo "  sudo apt-get install git-lfs  # Ubuntu/Debian"
    echo "  brew install git-lfs           # macOS"
    echo "  conda install -c conda-forge git-lfs  # conda"
    exit 1
fi

# Initialize git lfs if not already done
git lfs install 2>/dev/null || true

# Clone the dataset
DATASET_NAME="PKU-Alignment/PKU-SafeRLHF"
TEMP_DIR="$OUTPUT_DIR/temp_saferlhf"

echo "Cloning dataset from Hugging Face..."
if [ -d "$TEMP_DIR" ]; then
    echo "Directory $TEMP_DIR already exists. Removing..."
    rm -rf "$TEMP_DIR"
fi

# Try to clone the dataset
if git clone "https://huggingface.co/datasets/$DATASET_NAME" "$TEMP_DIR" 2>/dev/null; then
    echo "Dataset cloned successfully."
    
    # Convert to JSONL format
    echo "Converting to JSONL format..."
    
    # Process train split
    if [ -f "$TEMP_DIR/train.arrow" ] || [ -d "$TEMP_DIR/train" ]; then
        echo "Note: Dataset is in Arrow format. You may need to use Python to convert it."
        echo "Please run the Python download script instead, or use:"
        echo "  python3 -c \"from datasets import load_dataset; ds = load_dataset('$DATASET_NAME', split='train'); ds.to_json('$OUTPUT_DIR/pku_saferlhf_train.jsonl')\""
    fi
    
    # Check for JSON/JSONL files
    if find "$TEMP_DIR" -name "*.jsonl" -o -name "*.json" | head -1 | grep -q .; then
        echo "Found JSON/JSONL files, copying..."
        find "$TEMP_DIR" -name "train*.jsonl" -o -name "train*.json" | head -1 | xargs -I {} cp {} "$OUTPUT_DIR/pku_saferlhf_train.jsonl" 2>/dev/null || true
        find "$TEMP_DIR" -name "test*.jsonl" -o -name "test*.json" | head -1 | xargs -I {} cp {} "$OUTPUT_DIR/pku_saferlhf_test.jsonl" 2>/dev/null || true
    fi
    
    # Clean up
    rm -rf "$TEMP_DIR"
    echo "Download completed!"
else
    echo "Git clone failed. This dataset may require authentication or use a different format."
    echo "Please use the Python download script instead:"
    echo "  python3 scripts/download_safe_rlhf_data.py --output_dir $OUTPUT_DIR --splits train test"
    exit 1
fi

echo ""
echo "=========================================="
echo "Files saved to: $PROJECT_ROOT/$OUTPUT_DIR"
echo "=========================================="

