#!/usr/bin/env python3
"""
Download PKU-SafeRLHF dataset from Hugging Face and save to data/raw directory.
"""

import json
import os
from pathlib import Path
from datasets import load_dataset

def download_and_save_dataset(
    dataset_name: str = "PKU-Alignment/PKU-SafeRLHF",
    output_dir: str = "data/raw",
    splits: list = ["train", "test"]
):
    """
    Download PKU-SafeRLHF dataset from Hugging Face and save as JSONL files.
    
    Args:
        dataset_name: Hugging Face dataset name
        output_dir: Output directory for JSONL files
        splits: List of splits to download (e.g., ["train", "test"])
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading dataset: {dataset_name}")
    print(f"Output directory: {output_path.absolute()}")
    
    for split in splits:
        print(f"\nDownloading {split} split...")
        try:
            dataset = load_dataset(dataset_name, split=split)
            print(f"Loaded {len(dataset)} samples from {split} split")
            
            # Save as JSONL
            output_file = output_path / f"pku_saferlhf_{split}.jsonl"
            with open(output_file, "w", encoding="utf-8") as f:
                for item in dataset:
                    # Convert to dict and write as JSON line
                    json.dump(item, f, ensure_ascii=False)
                    f.write("\n")
            
            print(f"Saved {len(dataset)} samples to {output_file}")
            
        except Exception as e:
            print(f"Error downloading {split} split: {e}")
            # Try alternative dataset names
            if "PKU-SafeRLHF" in dataset_name:
                alternatives = [
                    "PKU-Alignment/PKU-SafeRLHF-30K",
                    "PKU-Alignment/PKU-SafeRLHF-10K",
                ]
                for alt_name in alternatives:
                    try:
                        print(f"Trying alternative: {alt_name}")
                        dataset = load_dataset(alt_name, split=split)
                        output_file = output_path / f"pku_saferlhf_{split}.jsonl"
                        with open(output_file, "w", encoding="utf-8") as f:
                            for item in dataset:
                                json.dump(item, f, ensure_ascii=False)
                                f.write("\n")
                        print(f"Successfully downloaded from {alt_name}")
                        print(f"Saved {len(dataset)} samples to {output_file}")
                        break
                    except Exception as alt_e:
                        print(f"Alternative {alt_name} also failed: {alt_e}")
                        continue
            raise
    
    print("\n" + "="*50)
    print("Download completed!")
    print(f"Files saved to: {output_path.absolute()}")
    print("="*50)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download PKU-SafeRLHF dataset from Hugging Face"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="PKU-Alignment/PKU-SafeRLHF",
        help="Hugging Face dataset name (default: PKU-Alignment/PKU-SafeRLHF)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/raw",
        help="Output directory for JSONL files (default: data/raw)"
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "test"],
        help="Dataset splits to download (default: train test)"
    )
    
    args = parser.parse_args()
    
    download_and_save_dataset(
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        splits=args.splits
    )

