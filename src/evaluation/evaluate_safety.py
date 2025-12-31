"""
Safety evaluation script using BeaverTails Safety Benchmark.

This script evaluates model safety by:
1. Loading the trained policy model
2. Generating responses to safety prompts from BeaverTails dataset
3. Using Harmless RM (Cost Model) to score responses
4. Calculating safety rate (percentage of safe responses)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from src.models.reward_model import RewardModelConfig, load_reward_model
except ImportError:
    # Handle case where evaluation is run from different directory
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.models.reward_model import RewardModelConfig, load_reward_model


def load_safety_benchmark(benchmark_path: Path) -> List[Dict]:
    """
    Load safety benchmark prompts from JSONL file.
    
    Expected format:
    {
        "prompt": "...",
        "category": "...",  # optional
        "expected_safe": true/false  # optional
    }
    """
    prompts = []
    with benchmark_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            prompts.append(json.loads(line))
    return prompts


def generate_response(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    device: torch.device = None,
) -> str:
    """Generate a response from the model given a prompt."""
    if device is None:
        device = next(model.parameters()).device
    
    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode response (excluding prompt)
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return response.strip()


def evaluate_safety(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    harmless_rm,
    prompts: List[Dict],
    device: torch.device,
    max_new_tokens: int = 256,
    safety_threshold: float = 0.0,
    batch_size: int = 8,
) -> Dict:
    """
    Evaluate model safety on a set of prompts.
    
    Args:
        model: Policy model to evaluate
        tokenizer: Tokenizer for the model
        harmless_rm: Harmless Reward Model (Cost Model)
        prompts: List of prompt dictionaries
        device: Device to run evaluation on
        max_new_tokens: Maximum tokens to generate
        safety_threshold: Safety score threshold (responses with score > threshold are considered safe)
        batch_size: Batch size for RM evaluation
    
    Returns:
        Dictionary containing evaluation metrics
    """
    model.eval()
    harmless_rm.eval()
    
    results = []
    total_prompts = len(prompts)
    safe_count = 0
    
    print(f"Evaluating {total_prompts} prompts...")
    
    for i, prompt_dict in enumerate(prompts):
        prompt = prompt_dict["prompt"]
        category = prompt_dict.get("category", "unknown")
        
        # Generate response
        response = generate_response(
            model, tokenizer, prompt, max_new_tokens=max_new_tokens, device=device
        )
        
        # Evaluate safety using Harmless RM
        # Format: prompt + response
        full_text = f"{prompt}\n{response}"
        inputs = tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(device)
        
        with torch.no_grad():
            safety_score = harmless_rm(
                inputs["input_ids"],
                inputs["attention_mask"],
            ).item()
        
        # Determine if response is safe
        # Harmless RM outputs higher scores for safer responses
        # We consider a response safe if safety_score > safety_threshold
        is_safe = safety_score > safety_threshold
        
        if is_safe:
            safe_count += 1
        
        results.append({
            "prompt": prompt,
            "response": response,
            "category": category,
            "safety_score": safety_score,
            "is_safe": is_safe,
        })
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{total_prompts} prompts...")
    
    # Calculate metrics
    safety_rate = safe_count / total_prompts if total_prompts > 0 else 0.0
    avg_safety_score = sum(r["safety_score"] for r in results) / total_prompts if total_prompts > 0 else 0.0
    
    # Category-wise statistics
    category_stats = {}
    for result in results:
        cat = result["category"]
        if cat not in category_stats:
            category_stats[cat] = {"total": 0, "safe": 0, "scores": []}
        category_stats[cat]["total"] += 1
        if result["is_safe"]:
            category_stats[cat]["safe"] += 1
        category_stats[cat]["scores"].append(result["safety_score"])
    
    for cat in category_stats:
        stats = category_stats[cat]
        stats["safety_rate"] = stats["safe"] / stats["total"] if stats["total"] > 0 else 0.0
        stats["avg_score"] = sum(stats["scores"]) / len(stats["scores"]) if stats["scores"] else 0.0
        del stats["scores"]  # Remove raw scores for cleaner output
    
    return {
        "total_prompts": total_prompts,
        "safe_count": safe_count,
        "safety_rate": safety_rate,
        "avg_safety_score": avg_safety_score,
        "category_stats": category_stats,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model safety using BeaverTails Safety Benchmark"
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        required=True,
        help="Path to the trained policy model (e.g., models/aligned/epoch_2)",
    )
    parser.add_argument(
        "--harmless_rm_path",
        type=Path,
        required=True,
        help="Path to the trained Harmless RM (e.g., models/harmless_rm)",
    )
    parser.add_argument(
        "--benchmark_path",
        type=Path,
        required=True,
        help="Path to BeaverTails safety benchmark JSONL file",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        required=True,
        help="Path to save evaluation results (JSON file)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate per response",
    )
    parser.add_argument(
        "--safety_threshold",
        type=float,
        default=0.0,
        help="Safety score threshold (responses with score > threshold are considered safe)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for RM evaluation (not used in current implementation)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run evaluation on",
    )
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    print("=" * 60)
    print("Safety Evaluation using BeaverTails Safety Benchmark")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Harmless RM: {args.harmless_rm_path}")
    print(f"Benchmark: {args.benchmark_path}")
    print(f"Device: {device}")
    print()
    
    # Load model
    print("Loading policy model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with GPU acceleration
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32
    
    print("Loading policy model...")
    print(f"  Model path: {args.model_path}")
    print(f"  Device: {device}")
    print(f"  Dtype: {dtype}")
    
    if device.type == "cuda":
        # For evaluation, use single GPU (device_map="cuda:0") instead of "auto"
        # "auto" can cause issues in evaluation scripts (deadlock, slow loading)
        # If you have multiple GPUs and want to use them, specify device_map explicitly
        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                dtype=dtype,
                device_map="cuda:0",  # Use single GPU for evaluation (more stable)
                # Alternative: device_map="auto" for multi-GPU, but may be slower
            )
            print("  Using device_map='cuda:0' for single GPU")
        except Exception as e:
            print(f"  Warning: device_map failed ({e}), falling back to manual .to(device)")
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                dtype=dtype,
            )
            model = model.to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            dtype=dtype,
        )
        model = model.to(device)
    
    model.eval()
    print("✓ Policy model loaded")
    
    # Load Harmless RM
    print("Loading Harmless RM...")
    print(f"  RM path: {args.harmless_rm_path}")
    rm_cfg = RewardModelConfig(
        base_model_path=str(args.harmless_rm_path),
        tokenizer_name=str(args.harmless_rm_path),
        max_length=512,
    )
    harmless_rm, _ = load_reward_model(rm_cfg, dtype=dtype, use_gradient_checkpointing=False)
    harmless_rm = harmless_rm.to(device)
    harmless_rm.eval()
    print("✓ Harmless RM loaded")
    
    # Load benchmark
    print(f"Loading benchmark from {args.benchmark_path}...")
    prompts = load_safety_benchmark(args.benchmark_path)
    print(f"✓ Loaded {len(prompts)} prompts")
    print()
    
    # Run evaluation
    print("Starting evaluation...")
    results = evaluate_safety(
        model=model,
        tokenizer=tokenizer,
        harmless_rm=harmless_rm,
        prompts=prompts,
        device=device,
        max_new_tokens=args.max_new_tokens,
        safety_threshold=args.safety_threshold,
        batch_size=args.batch_size,
    )
    
    # Print summary
    print()
    print("=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Total Prompts: {results['total_prompts']}")
    print(f"Safe Responses: {results['safe_count']}")
    print(f"Safety Rate: {results['safety_rate']:.2%}")
    print(f"Average Safety Score: {results['avg_safety_score']:.4f}")
    print()
    
    if results["category_stats"]:
        print("Category-wise Statistics:")
        for category, stats in results["category_stats"].items():
            print(f"  {category}:")
            print(f"    Safety Rate: {stats['safety_rate']:.2%} ({stats['safe']}/{stats['total']})")
            print(f"    Avg Score: {stats['avg_score']:.4f}")
        print()
    
    # Save results
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"✓ Results saved to {args.output_path}")
    
    # Save detailed results (all prompts and responses)
    detailed_path = args.output_path.parent / f"{args.output_path.stem}_detailed.json"
    with detailed_path.open("w", encoding="utf-8") as f:
        json.dump(results["results"], f, ensure_ascii=False, indent=2)
    print(f"✓ Detailed results saved to {detailed_path}")


if __name__ == "__main__":
    main()
