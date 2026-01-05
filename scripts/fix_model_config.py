#!/usr/bin/env python3
"""
Fix model configuration files that are missing model_type or have incorrect model_type.

This script can fix:
1. Reward models missing model_type in config.json
2. Policy models with incorrect model_type (e.g., 'align' instead of 'mistral')
"""

import argparse
import json
from pathlib import Path
from typing import Optional


def fix_reward_model_config(model_path: Path, base_model_path: Optional[Path] = None) -> bool:
    """Fix config.json for a reward model (sequence classification)."""
    config_path = model_path / "config.json"
    
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return False
    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Check if model_type is missing or incorrect
    model_type = config.get("model_type")
    architectures = config.get("architectures", [])
    
    # Try to infer model_type from architectures
    if not model_type and architectures:
        arch = architectures[0] if architectures else None
        if arch:
            # Map common architectures to model_type
            if "Mistral" in arch:
                model_type = "mistral"
            elif "Llama" in arch:
                model_type = "llama"
            elif "GPT" in arch or "GPT2" in arch:
                model_type = "gpt2"
            elif "Bloom" in arch:
                model_type = "bloom"
            elif "OPT" in arch:
                model_type = "opt"
    
    # If still no model_type, try to load from base model
    if not model_type and base_model_path:
        base_config_path = base_model_path / "config.json"
        if base_config_path.exists():
            with open(base_config_path, 'r') as f:
                base_config = json.load(f)
            model_type = base_config.get("model_type")
            print(f"  Inferred model_type from base model: {model_type}")
    
    if not model_type:
        print(f"❌ Could not determine model_type for {model_path}")
        return False
    
    # Update config
    if config.get("model_type") != model_type:
        config["model_type"] = model_type
        print(f"✓ Fixed model_type: {config.get('model_type', 'missing')} -> {model_type}")
        
        # Backup original config
        backup_path = config_path.with_suffix('.json.bak')
        with open(backup_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"  Backup saved to: {backup_path}")
        
        # Save fixed config
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✓ Updated config.json")
        return True
    else:
        print(f"✓ Config already has correct model_type: {model_type}")
        return False


def fix_policy_model_config(model_path: Path, expected_model_type: Optional[str] = None) -> bool:
    """Fix config.json for a policy model (causal LM)."""
    config_path = model_path / "config.json"
    
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return False
    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    model_type = config.get("model_type")
    architectures = config.get("architectures", [])
    
    # Check if model_type is incorrect (e.g., 'align' for a causal LM)
    if model_type == "align":
        print(f"⚠️  Detected incorrect model_type='align' for causal LM")
        
        # Try to infer from architectures
        if architectures:
            arch = architectures[0]
            if "Mistral" in arch:
                correct_type = "mistral"
            elif "Llama" in arch:
                correct_type = "llama"
            elif "GPT" in arch:
                correct_type = "gpt2"
            else:
                correct_type = expected_model_type or "mistral"  # Default to mistral
        else:
            correct_type = expected_model_type or "mistral"
        
        config["model_type"] = correct_type
        print(f"  Fixed model_type: align -> {correct_type}")
        
        # Backup and save
        backup_path = config_path.with_suffix('.json.bak')
        with open(backup_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"  Backup saved to: {backup_path}")
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✓ Updated config.json")
        return True
    elif not model_type:
        # Missing model_type
        if architectures:
            arch = architectures[0]
            if "Mistral" in arch:
                model_type = "mistral"
            elif "Llama" in arch:
                model_type = "llama"
            else:
                model_type = expected_model_type or "mistral"
        else:
            model_type = expected_model_type or "mistral"
        
        config["model_type"] = model_type
        print(f"✓ Added missing model_type: {model_type}")
        
        backup_path = config_path.with_suffix('.json.bak')
        with open(backup_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✓ Updated config.json")
        return True
    else:
        print(f"✓ Config already has model_type: {model_type}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Fix model configuration files (config.json) that are missing or have incorrect model_type"
    )
    parser.add_argument(
        "model_path",
        type=Path,
        help="Path to the model directory (should contain config.json)"
    )
    parser.add_argument(
        "--type",
        choices=["reward", "policy"],
        default="policy",
        help="Type of model: 'reward' for sequence classification, 'policy' for causal LM (default: policy)"
    )
    parser.add_argument(
        "--base-model-path",
        type=Path,
        help="For reward models: path to base model to infer model_type from"
    )
    parser.add_argument(
        "--expected-model-type",
        type=str,
        help="Expected model_type (e.g., 'mistral', 'llama'). Used as fallback if cannot infer from config"
    )
    
    args = parser.parse_args()
    
    model_path = args.model_path
    if not model_path.exists():
        print(f"❌ Model path does not exist: {model_path}")
        return 1
    
    print(f"Fixing config for {args.type} model: {model_path}")
    print()
    
    if args.type == "reward":
        fixed = fix_reward_model_config(model_path, args.base_model_path)
    else:
        fixed = fix_policy_model_config(model_path, args.expected_model_type)
    
    if fixed:
        print()
        print("✓ Config fixed successfully!")
        return 0
    else:
        print()
        print("ℹ️  No changes needed or fix failed")
        return 0 if not fixed else 1


if __name__ == "__main__":
    exit(main())
