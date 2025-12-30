#!/usr/bin/env python3
"""
Prepare safety benchmark from local raw data.

This script extracts safety-related prompts from PKU-SafeRLHF raw data
to create a comprehensive safety evaluation benchmark.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set


def load_raw_data(raw_file: Path) -> List[Dict]:
    """Load raw PKU-SafeRLHF data."""
    records = []
    with raw_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def extract_safety_prompts(
    records: List[Dict],
    min_responses_per_prompt: int = 1,
    include_safe_prompts: bool = True,
    include_unsafe_prompts: bool = True,
    prioritize_unsafe: bool = True,
) -> List[Dict]:
    """
    Extract unique prompts from raw data, focusing on safety-related ones.
    
    Args:
        records: Raw PKU-SafeRLHF records
        min_responses_per_prompt: Minimum number of responses per prompt to include
        include_safe_prompts: Whether to include prompts with safe responses
        include_unsafe_prompts: Whether to include prompts with unsafe responses
        prioritize_unsafe: Whether to prioritize prompts with unsafe responses (for safety testing)
    
    Returns:
        List of prompt dictionaries with category information
    """
    prompt_info = defaultdict(lambda: {
        "prompt": None,
        "has_safe_response": False,
        "has_unsafe_response": False,
        "response_count": 0,
        "categories": set(),
        "harm_categories": set(),
    })
    
    # Common harm category mappings
    category_mapping = {
        "Violence": "violence",
        "Physical Harm": "violence",
        "Cybercrime": "cybercrime",
        "Privacy Violation": "privacy",
        "Economic Crime": "economic_crime",
        "White-Collar Crime": "economic_crime",
        "Discriminatory Behavior": "discrimination",
        "Mental Manipulation": "manipulation",
        "Psychological Harm": "psychological_harm",
        "Endangering National Security": "national_security",
        "Endangering Public Health": "public_health",
        "Disrupting Public Order": "public_order",
        "Drugs": "drugs",
        "Sexual Content": "sexual_content",
        "Animal Abuse": "animal_abuse",
        "Environmental Damage": "environmental",
    }
    
    for rec in records:
        prompt = rec.get("prompt")
        if not prompt:
            continue
        
        info = prompt_info[prompt]
        info["prompt"] = prompt
        
        # Check responses for safety labels and harm categories
        for i in [0, 1]:
            response_key = f"response_{i}"
            safety_key = f"is_response_{i}_safe"
            harm_category_key = f"response_{i}_harm_category"
            
            if response_key in rec:
                info["response_count"] += 1
                
                # Check safety label
                is_safe = rec.get(safety_key)
                if is_safe is True:
                    info["has_safe_response"] = True
                elif is_safe is False:
                    info["has_unsafe_response"] = True
                    
                    # Extract harm categories
                    harm_cats = rec.get(harm_category_key, {})
                    if isinstance(harm_cats, dict):
                        for harm_cat, is_present in harm_cats.items():
                            if is_present:
                                # Map to simplified category
                                mapped = category_mapping.get(harm_cat, harm_cat.lower().replace(" ", "_"))
                                info["harm_categories"].add(mapped)
                                info["categories"].add(mapped)
        
        # If no harm categories found, try to infer from prompt content
        if not info["categories"]:
            prompt_lower = prompt.lower()
            if any(word in prompt_lower for word in ["violence", "kill", "harm", "attack", "weapon", "bomb"]):
                info["categories"].add("violence")
            elif any(word in prompt_lower for word in ["hack", "privacy", "steal", "password", "credit card", "cyber"]):
                info["categories"].add("cybercrime")
            elif any(word in prompt_lower for word in ["drug", "illegal", "criminal", "fraud"]):
                info["categories"].add("economic_crime")
            elif any(word in prompt_lower for word in ["discrimination", "racist", "sexist", "hate"]):
                info["categories"].add("discrimination")
            elif any(word in prompt_lower for word in ["suicide", "self-harm", "depression"]):
                info["categories"].add("psychological_harm")
            else:
                info["categories"].add("other")
    
    # Filter prompts based on criteria
    selected_prompts = []
    for prompt, info in prompt_info.items():
        # Filter by response count
        if info["response_count"] < min_responses_per_prompt:
            continue
        
        # Filter by safety labels
        if include_safe_prompts and include_unsafe_prompts:
            # Include all prompts
            pass
        elif include_safe_prompts and not include_unsafe_prompts:
            if not info["has_safe_response"]:
                continue
        elif not include_safe_prompts and include_unsafe_prompts:
            if not info["has_unsafe_response"]:
                continue
        else:
            # Both False, skip
            continue
        
        # Create prompt record
        categories = list(info["categories"])
        category = categories[0] if categories else "unknown"
        
        # Priority score: higher for unsafe prompts (for safety testing)
        priority = 1.0
        if prioritize_unsafe and info["has_unsafe_response"]:
            priority = 2.0
        
        selected_prompts.append({
            "prompt": prompt,
            "category": category,
            "has_safe_response": info["has_safe_response"],
            "has_unsafe_response": info["has_unsafe_response"],
            "priority": priority,
            "all_categories": list(info["categories"]),
        })
    
    # Sort by priority (unsafe prompts first)
    if prioritize_unsafe:
        selected_prompts.sort(key=lambda x: (-x["priority"], x["prompt"]))
    
    return selected_prompts


def create_balanced_benchmark(
    prompts: List[Dict],
    max_prompts: int = None,
    balance_by_category: bool = True,
    seed: int = 42,
) -> List[Dict]:
    """
    Create a balanced benchmark from prompts.
    
    Args:
        prompts: List of prompt dictionaries
        max_prompts: Maximum number of prompts to include (None = all)
        balance_by_category: Whether to balance prompts across categories
        seed: Random seed for sampling
    
    Returns:
        Balanced list of prompts
    """
    random.seed(seed)
    
    if not balance_by_category or max_prompts is None:
        # Simple random sampling
        if max_prompts and len(prompts) > max_prompts:
            return random.sample(prompts, max_prompts)
        return prompts
    
    # Balance by category
    category_prompts = defaultdict(list)
    for prompt in prompts:
        category_prompts[prompt["category"]].append(prompt)
    
    # Calculate prompts per category
    num_categories = len(category_prompts)
    prompts_per_category = max_prompts // num_categories if max_prompts else None
    
    balanced = []
    for category, cat_prompts in category_prompts.items():
        if prompts_per_category:
            sampled = random.sample(
                cat_prompts,
                min(prompts_per_category, len(cat_prompts))
            )
        else:
            sampled = cat_prompts
        balanced.extend(sampled)
    
    # If we have room, add more prompts randomly
    if max_prompts and len(balanced) < max_prompts:
        remaining = [p for p in prompts if p not in balanced]
        needed = max_prompts - len(balanced)
        if remaining:
            balanced.extend(random.sample(remaining, min(needed, len(remaining))))
    
    random.shuffle(balanced)
    return balanced


def main():
    parser = argparse.ArgumentParser(
        description="Prepare safety benchmark from local raw data"
    )
    parser.add_argument(
        "--raw_data_paths",
        type=str,
        nargs="+",
        default=["data/raw/pku_saferlhf_test.jsonl", "data/raw/pku_saferlhf_train.jsonl"],
        help="Paths to raw PKU-SafeRLHF data files",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("data/benchmarks/safety_benchmark.jsonl"),
        help="Output path for safety benchmark",
    )
    parser.add_argument(
        "--max_prompts",
        type=int,
        default=None,
        help="Maximum number of prompts to include (None = all)",
    )
    parser.add_argument(
        "--min_responses_per_prompt",
        type=int,
        default=1,
        help="Minimum number of responses per prompt to include",
    )
    parser.add_argument(
        "--include_safe",
        action="store_true",
        default=True,
        help="Include prompts with safe responses",
    )
    parser.add_argument(
        "--include_unsafe",
        action="store_true",
        default=True,
        help="Include prompts with unsafe responses",
    )
    parser.add_argument(
        "--balance_by_category",
        action="store_true",
        default=True,
        help="Balance prompts across categories",
    )
    parser.add_argument(
        "--prioritize_unsafe",
        action="store_true",
        default=True,
        help="Prioritize prompts with unsafe responses (for safety testing)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )
    
    args = parser.parse_args()
    
    # Load raw data
    print("Loading raw data...")
    all_records = []
    for raw_path in args.raw_data_paths:
        path = Path(raw_path)
        if not path.exists():
            print(f"WARNING: File not found: {path}")
            continue
        
        records = load_raw_data(path)
        print(f"  Loaded {len(records)} records from {path}")
        all_records.extend(records)
    
    print(f"Total records: {len(all_records)}")
    
    # Extract safety prompts
    print("\nExtracting safety prompts...")
    prompts = extract_safety_prompts(
        all_records,
        min_responses_per_prompt=args.min_responses_per_prompt,
        include_safe_prompts=args.include_safe,
        include_unsafe_prompts=args.include_unsafe,
        prioritize_unsafe=args.prioritize_unsafe,
    )
    print(f"Extracted {len(prompts)} unique prompts")
    
    # Show category distribution
    category_counts = defaultdict(int)
    for prompt in prompts:
        category_counts[prompt["category"]] += 1
    
    print("\nCategory distribution:")
    for category, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"  {category}: {count}")
    
    # Create balanced benchmark
    print("\nCreating balanced benchmark...")
    benchmark = create_balanced_benchmark(
        prompts,
        max_prompts=args.max_prompts,
        balance_by_category=args.balance_by_category,
        seed=args.seed,
    )
    print(f"Selected {len(benchmark)} prompts for benchmark")
    
    # Save benchmark
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as f:
        for prompt_dict in benchmark:
            # Save prompt, category, and metadata
            output_dict = {
                "prompt": prompt_dict["prompt"],
                "category": prompt_dict["category"],
            }
            # Optionally include additional metadata
            if "all_categories" in prompt_dict and len(prompt_dict["all_categories"]) > 1:
                output_dict["all_categories"] = prompt_dict["all_categories"]
            f.write(json.dumps(output_dict, ensure_ascii=False) + "\n")
    
    print(f"\n✓ Benchmark saved to {args.output_path}")
    print(f"  Total prompts: {len(benchmark)}")
    
    # Show final category distribution
    final_category_counts = defaultdict(int)
    for prompt in benchmark:
        final_category_counts[prompt["category"]] += 1
    
    print("\nFinal category distribution:")
    for category, count in sorted(final_category_counts.items(), key=lambda x: -x[1]):
        print(f"  {category}: {count}")


if __name__ == "__main__":
    main()
