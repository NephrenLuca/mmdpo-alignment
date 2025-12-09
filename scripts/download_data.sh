#!/bin/bash
# Download PKU-SafeRLHF dataset from Hugging Face to data/raw

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

OUTPUT_DIR="data/raw"
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Downloading PKU-SafeRLHF dataset"
echo "=========================================="
echo "Output directory: $PROJECT_ROOT/$OUTPUT_DIR"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found. Please install Python 3.10+"
    exit 1
fi

# Check if datasets library is available
if ! python3 -c "import datasets" 2>/dev/null; then
    echo "Warning: 'datasets' library not found."
    echo "Attempting to install datasets and huggingface_hub..."
    echo ""
    echo "If you're using a virtual environment, please activate it first:"
    echo "  conda activate humanAlignment"
    echo "  # or"
    echo "  source venv/bin/activate"
    echo ""
    echo "Then run: pip install datasets huggingface_hub"
    echo ""
    read -p "Do you want to try installing now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        pip3 install --user datasets huggingface_hub || {
            echo "Installation failed. Please install manually."
            exit 1
        }
    else
        echo "Please install dependencies and run this script again."
        exit 1
    fi
fi

# Run the Python download script
echo "Starting download..."
python3 "$SCRIPT_DIR/download_safe_rlhf_data.py" \
    --output_dir "$OUTPUT_DIR" \
    --splits train test

echo ""
echo "=========================================="
echo "Download completed!"
echo "Files saved to: $PROJECT_ROOT/$OUTPUT_DIR"
echo "=========================================="

