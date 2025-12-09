#!/usr/bin/env python3
"""
Simple script to download PKU-SafeRLHF dataset using huggingface_hub only.
This script doesn't require the datasets library.
"""

import json
import os
import sys
from pathlib import Path

try:
    from huggingface_hub import snapshot_download, hf_hub_download
    from huggingface_hub.utils import HfHubHTTPError
except ImportError:
    print("Error: huggingface_hub not found.")
    print("Please install it with: pip install --user huggingface_hub")
    print("Or activate your virtual environment and run: pip install huggingface_hub")
    sys.exit(1)

def download_dataset_simple(
    dataset_name: str = "PKU-Alignment/PKU-SafeRLHF",
    output_dir: str = "data/raw"
):
    """
    Download dataset files from Hugging Face using huggingface_hub.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading dataset: {dataset_name}")
    print(f"Output directory: {output_path.absolute()}")
    print("")
    
    # Try to download the dataset repository
    try:
        print("Attempting to download dataset repository...")
        local_dir = snapshot_download(
            repo_id=dataset_name,
            repo_type="dataset",
            local_dir=str(output_path / "pku_saferlhf_temp"),
            local_dir_use_symlinks=False
        )
        print(f"Downloaded to: {local_dir}")
        
        # Look for JSON/JSONL files
        temp_path = Path(local_dir)
        jsonl_files = list(temp_path.glob("**/*.jsonl")) + list(temp_path.glob("**/*.json"))
        
        if jsonl_files:
            print(f"\nFound {len(jsonl_files)} JSON/JSONL files")
            for jsonl_file in jsonl_files:
                # Try to identify train/test splits
                filename = jsonl_file.name.lower()
                if "train" in filename:
                    dest = output_path / "pku_saferlhf_train.jsonl"
                    print(f"Copying {jsonl_file.name} -> {dest.name}")
                    import shutil
                    shutil.copy2(jsonl_file, dest)
                elif "test" in filename or "val" in filename or "eval" in filename:
                    dest = output_path / "pku_saferlhf_test.jsonl"
                    print(f"Copying {jsonl_file.name} -> {dest.name}")
                    import shutil
                    shutil.copy2(jsonl_file, dest)
        
        # Clean up temp directory
        import shutil
        shutil.rmtree(temp_path)
        print("\nTemporary files cleaned up.")
        
    except HfHubHTTPError as e:
        print(f"Error downloading dataset: {e}")
        print("\nTrying alternative: using datasets library (requires pip install datasets)")
        print("Please run: python3 scripts/download_safe_rlhf_data.py")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        print("\nNote: This dataset may require the 'datasets' library to load properly.")
        print("Please install it and use the full download script:")
        print("  pip install datasets")
        print("  python3 scripts/download_safe_rlhf_data.py")
        return False
    
    print("\n" + "="*50)
    print("Download completed!")
    print(f"Files saved to: {output_path.absolute()}")
    print("="*50)
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download PKU-SafeRLHF dataset from Hugging Face (simple version)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="PKU-Alignment/PKU-SafeRLHF",
        help="Hugging Face dataset name"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/raw",
        help="Output directory for JSONL files"
    )
    
    args = parser.parse_args()
    
    success = download_dataset_simple(
        dataset_name=args.dataset,
        output_dir=args.output_dir
    )
    
    if not success:
        sys.exit(1)

