#!/usr/bin/env python3
"""
Compare safety evaluation results between baseline model and aligned models.

This script:
1. Evaluates baseline Mistral model
2. Evaluates MMDPO-aligned models (each epoch)
3. Generates a comparison report
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


def run_evaluation(
    model_path: Path,
    harmless_rm_path: Path,
    benchmark_path: Path,
    output_path: Path,
    max_new_tokens: int = 256,
    safety_threshold: float = 0.0,
    device: str = "cuda",
) -> Dict:
    """Run safety evaluation for a single model."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_path}")
    print(f"{'='*60}")
    
    cmd = [
        sys.executable,
        "-m", "src.evaluation.evaluate_safety",
        "--model_path", str(model_path),
        "--harmless_rm_path", str(harmless_rm_path),
        "--benchmark_path", str(benchmark_path),
        "--output_path", str(output_path),
        "--max_new_tokens", str(max_new_tokens),
        "--safety_threshold", str(safety_threshold),
        "--device", device,
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"ERROR: Evaluation failed for {model_path}")
        print(result.stderr)
        return None
    
    # Load and return results
    with output_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def generate_comparison_report(
    results: Dict[str, Dict],
    output_path: Path,
) -> None:
    """Generate a comparison report from evaluation results."""
    
    report = {
        "summary": {},
        "models": {},
        "comparison": {},
    }
    
    # Extract summary for each model
    for model_name, result in results.items():
        if result is None:
            continue
        report["models"][model_name] = {
            "safety_rate": result["safety_rate"],
            "avg_safety_score": result["avg_safety_score"],
            "safe_count": result["safe_count"],
            "total_prompts": result["total_prompts"],
            "category_stats": result.get("category_stats", {}),
        }
    
    # Find baseline model (usually the first one or named "baseline")
    baseline_name = None
    for name in results.keys():
        if "baseline" in name.lower() or "mistral" in name.lower() or "base" in name.lower():
            baseline_name = name
            break
    if not baseline_name and results:
        baseline_name = list(results.keys())[0]
    
    # Calculate improvements
    if baseline_name and baseline_name in report["models"]:
        baseline_rate = report["models"][baseline_name]["safety_rate"]
        baseline_score = report["models"][baseline_name]["avg_safety_score"]
        
        report["comparison"]["baseline"] = baseline_name
        report["comparison"]["baseline_safety_rate"] = baseline_rate
        report["comparison"]["baseline_avg_score"] = baseline_score
        
        improvements = {}
        for model_name, model_data in report["models"].items():
            if model_name == baseline_name:
                continue
            
            rate_improvement = model_data["safety_rate"] - baseline_rate
            score_improvement = model_data["avg_safety_score"] - baseline_score
            
            improvements[model_name] = {
                "safety_rate_improvement": rate_improvement,
                "safety_rate_improvement_pct": (rate_improvement / baseline_rate * 100) if baseline_rate > 0 else 0,
                "avg_score_improvement": score_improvement,
                "relative_improvement": (rate_improvement / (1 - baseline_rate) * 100) if baseline_rate < 1 else 0,
            }
        
        report["comparison"]["improvements"] = improvements
    
    # Overall summary
    if report["models"]:
        best_model = max(
            report["models"].items(),
            key=lambda x: x[1]["safety_rate"]
        )
        report["summary"] = {
            "best_model": best_model[0],
            "best_safety_rate": best_model[1]["safety_rate"],
            "total_models_evaluated": len(report["models"]),
        }
    
    # Save report
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("COMPARISON REPORT")
    print("="*60)
    
    if report["summary"]:
        print(f"\nBest Model: {report['summary']['best_model']}")
        print(f"Best Safety Rate: {report['summary']['best_safety_rate']:.2%}")
    
    if report["comparison"]:
        print(f"\nBaseline: {report['comparison']['baseline']}")
        print(f"Baseline Safety Rate: {report['comparison']['baseline_safety_rate']:.2%}")
        print(f"Baseline Avg Score: {report['comparison']['baseline_avg_score']:.4f}")
        
        if "improvements" in report["comparison"]:
            print("\nImprovements over baseline:")
            for model_name, improvement in report["comparison"]["improvements"].items():
                print(f"\n  {model_name}:")
                print(f"    Safety Rate Improvement: {improvement['safety_rate_improvement']:+.2%}")
                print(f"    Relative Improvement: {improvement['relative_improvement']:+.2f}%")
                print(f"    Avg Score Improvement: {improvement['avg_score_improvement']:+.4f}")
    
    print("\n" + "="*60)
    print(f"Full report saved to: {output_path}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Compare safety evaluation results between baseline and aligned models"
    )
    parser.add_argument(
        "--baseline_model_path",
        type=Path,
        required=True,
        help="Path to baseline Mistral model (e.g., models/base/Mistral-7B-v0.1)",
    )
    parser.add_argument(
        "--aligned_model_paths",
        type=str,
        nargs="+",
        required=True,
        help="Paths to aligned models (e.g., models/aligned/epoch_1 models/aligned/epoch_2)",
    )
    parser.add_argument(
        "--harmless_rm_path",
        type=Path,
        required=True,
        help="Path to trained Harmless RM (e.g., models/harmless_rm)",
    )
    parser.add_argument(
        "--benchmark_path",
        type=Path,
        required=True,
        help="Path to safety benchmark JSONL file",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/safety_comparison"),
        help="Directory to save evaluation results and comparison report",
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
        help="Safety score threshold",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if Path("/dev/nvidia0").exists() else "cpu",
        help="Device to run evaluation on",
    )
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare model list
    models_to_evaluate = {
        "baseline": args.baseline_model_path,
    }
    
    for i, aligned_path in enumerate(args.aligned_model_paths, start=1):
        path = Path(aligned_path)
        if path.exists():
            # Try to infer model name from path
            if "epoch" in path.name:
                model_name = f"epoch_{path.name.split('_')[-1]}"
            else:
                model_name = f"aligned_{i}"
            models_to_evaluate[model_name] = path
        else:
            print(f"WARNING: Model path does not exist: {aligned_path}")
    
    print("="*60)
    print("Safety Evaluation Comparison")
    print("="*60)
    print(f"Baseline Model: {args.baseline_model_path}")
    print(f"Aligned Models: {len([m for k, m in models_to_evaluate.items() if k != 'baseline'])}")
    print(f"Benchmark: {args.benchmark_path}")
    print(f"Output Directory: {args.output_dir}")
    print()
    
    # Evaluate each model
    results = {}
    for model_name, model_path in models_to_evaluate.items():
        output_path = args.output_dir / f"{model_name}_evaluation.json"
        
        result = run_evaluation(
            model_path=model_path,
            harmless_rm_path=args.harmless_rm_path,
            benchmark_path=args.benchmark_path,
            output_path=output_path,
            max_new_tokens=args.max_new_tokens,
            safety_threshold=args.safety_threshold,
            device=args.device,
        )
        
        if result:
            results[model_name] = result
    
    # Generate comparison report
    if len(results) > 1:
        comparison_path = args.output_dir / "comparison_report.json"
        generate_comparison_report(results, comparison_path)
    else:
        print("WARNING: Need at least 2 models for comparison")
    
    print("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()
