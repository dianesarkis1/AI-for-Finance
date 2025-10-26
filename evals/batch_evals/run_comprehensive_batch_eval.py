#!/usr/bin/env python3
"""
Comprehensive Batch Evaluation Script

This script creates a comprehensive evaluation dataset by combining:
1. The sampled indices from baseline_sampled_indices_seed42.json
2. The first 3 indices from train.jsonl (indices 0, 1, 2)
3. A random subset of 37 additional inputs from train.jsonl

Then runs batch evaluations using:
- Model: Claude Sonnet 4
- Evaluators: GPT-5, Claude Sonnet 4, Gemini 2.5 Pro

Outputs comprehensive results similar to baseline_benchmark_results_complete.json
(used as a template) with mean, median, and individual scores.
Results are saved in the batch_evals folder.

Usage:
    python run_comprehensive_batch_eval.py
"""

import json
import random
from pathlib import Path
import sys
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evals.batch_evals.batch_evaluate import run_batch_benchmark

# Configuration
TRAIN_FILE = Path("data/train.jsonl")
BASELINE_SAMPLED_INDICES_FILE = Path("evals/benchmark/baseline_sampled_indices_seed42.json")
OUTPUT_DIR = Path("evals/batch_evals")
OUTPUT_DIR.mkdir(exist_ok=True)

# Random seed for reproducibility
RANDOM_SEED = 42

# Model configuration
MODEL_TO_EVALUATE = "claude-sonnet-4-20250514"
EVALUATOR_MODELS = ["gpt-5", "claude-sonnet-4-20250514", "gemini-2.5-pro"]


def load_baseline_sampled_indices(file_path: Path) -> list[int]:
    """Load the baseline sampled indices from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data['sampled_indices']


def count_train_samples(train_file: Path) -> int:
    """Count total number of samples in train.jsonl."""
    with open(train_file, 'r') as f:
        return sum(1 for _ in f)


def create_comprehensive_sample(
    baseline_indices: list[int],
    first_n: int = 3,
    random_sample_size: int = 37,
    total_samples: int = 484,
    seed: int = 42
) -> dict:
    """
    Create comprehensive sample combining baseline, first N, and random samples.

    Args:
        baseline_indices: Indices from baseline sampling
        first_n: Number of first indices to include (0, 1, 2, ...)
        random_sample_size: Number of random indices to sample
        total_samples: Total number of samples in dataset
        seed: Random seed for reproducibility

    Returns:
        Dict with sampling information and combined indices
    """
    # Start with baseline indices
    combined_indices = set(baseline_indices)

    # Add first N indices
    first_indices = list(range(first_n))
    combined_indices.update(first_indices)

    # Sample random indices (excluding already selected ones)
    random.seed(seed)
    available_indices = [i for i in range(total_samples) if i not in combined_indices]
    random_indices = random.sample(available_indices, random_sample_size)
    combined_indices.update(random_indices)

    # Convert to sorted list
    all_indices = sorted(list(combined_indices))

    # Create detailed breakdown
    sampling_info = {
        "random_seed": seed,
        "total_sampled": len(all_indices),
        "sampling_breakdown": {
            "from_baseline_sampled_indices_seed42": {
                "count": len(baseline_indices),
                "indices": sorted(baseline_indices)
            },
            "first_n_indices": {
                "count": len(first_indices),
                "indices": first_indices
            },
            "additional_random_sample": {
                "count": len(random_indices),
                "indices": sorted(random_indices)
            }
        },
        "all_sampled_indices": all_indices,
        "total_inputs_in_dataset": total_samples,
        "created_at": datetime.now().isoformat()
    }

    return sampling_info


def save_results(results: dict, sampling_info: dict, output_dir: Path):
    """
    Save results in the format similar to baseline_benchmark_results_complete.json template.

    Args:
        results: Results from run_batch_benchmark
        sampling_info: Information about sampling
        output_dir: Directory to save results (batch_evals/)
    """
    # Create comprehensive results structure matching the template format
    comprehensive_results = {
        "model": results["model_evaluated"],
        "evaluator_models": results["evaluator_models"],
        "random_seed": sampling_info["random_seed"],
        "sample_size": sampling_info["total_sampled"],
        "sampling_breakdown": sampling_info["sampling_breakdown"],
        "sampled_indices": sampling_info["all_sampled_indices"],

        # Summary statistics (matching template)
        "mean_score": results["summary_statistics"]["mean_score"],
        "median_score": results["summary_statistics"]["median_score"],
        "worst_score": results["summary_statistics"]["worst_score"],
        "best_score": results["summary_statistics"]["best_score"],
        "std_dev": results["summary_statistics"]["std_dev"],
        "score_range": results["summary_statistics"]["score_range"],

        "total_inputs_in_dataset": sampling_info["total_inputs_in_dataset"],
        "successful_evals": results["summary_statistics"]["successful_evals"],
        "failed_evals": results["summary_statistics"]["failed_evals"],

        # Per-metric statistics
        "metric_statistics": results["metric_statistics"],

        # Individual results - simplified format matching baseline template
        "all_results": [
            {
                "input_index": r["input_index"],
                "source_url": r["source_url"],
                "score": r["summary_score"],
                "error": r["error"]
            }
            for r in results["detailed_results"]
        ],

        # Store detailed results with full metric breakdowns
        "detailed_results": results["detailed_results"],

        "created_at": sampling_info["created_at"]
    }

    # Save main results to batch_evals folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"comprehensive_batch_eval_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(comprehensive_results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"RESULTS SAVED")
    print(f"{'='*70}")
    print(f"Results file: {results_file}")

    # Also save just the sampling info for reference
    sampling_file = output_dir / f"comprehensive_sampled_indices_{timestamp}.json"
    with open(sampling_file, 'w') as f:
        json.dump(sampling_info, f, indent=2)

    print(f"Sampling info: {sampling_file}")
    print(f"{'='*70}\n")

    return results_file, sampling_file


def main():
    """Main execution function."""
    print(f"\n{'='*70}")
    print(f"COMPREHENSIVE BATCH EVALUATION")
    print(f"{'='*70}\n")

    # Load baseline sampled indices
    print(f"Loading baseline sampled indices from {BASELINE_SAMPLED_INDICES_FILE}...")
    baseline_indices = load_baseline_sampled_indices(BASELINE_SAMPLED_INDICES_FILE)
    print(f"  Loaded {len(baseline_indices)} baseline indices: {baseline_indices}")

    # Count total samples
    print(f"\nCounting samples in {TRAIN_FILE}...")
    total_samples = count_train_samples(TRAIN_FILE)
    print(f"  Total samples in dataset: {total_samples}")

    # Create comprehensive sample
    print(f"\nCreating comprehensive sample...")
    print(f"  - Including {len(baseline_indices)} baseline indices")
    print(f"  - Including first 3 indices (0, 1, 2)")
    print(f"  - Randomly sampling 37 additional indices")
    print(f"  - Using random seed: {RANDOM_SEED}")

    sampling_info = create_comprehensive_sample(
        baseline_indices=baseline_indices,
        first_n=3,
        random_sample_size=37,
        total_samples=total_samples,
        seed=RANDOM_SEED
    )

    print(f"\n  Created comprehensive sample with {sampling_info['total_sampled']} total indices")
    print(f"  Breakdown:")
    for key, value in sampling_info['sampling_breakdown'].items():
        print(f"    - {key}: {value['count']} indices")

    # Run batch benchmark
    print(f"\n{'='*70}")
    print(f"RUNNING BATCH BENCHMARK")
    print(f"{'='*70}")
    print(f"Model to evaluate: {MODEL_TO_EVALUATE}")
    print(f"Evaluator models: {', '.join(EVALUATOR_MODELS)}")
    print(f"Total inputs: {sampling_info['total_sampled']}")
    print(f"{'='*70}\n")

    results = run_batch_benchmark(
        model=MODEL_TO_EVALUATE,
        train_file=TRAIN_FILE,
        indices=sampling_info['all_sampled_indices'],
        evaluator_models=EVALUATOR_MODELS,
        poll_interval=60,
        delay_between_inputs=5.0,
        save_results=False  # We'll handle saving ourselves
    )

    # Save results to batch_evals folder
    print(f"\nSaving comprehensive results to batch_evals folder...")
    results_file, sampling_file = save_results(results, sampling_info, OUTPUT_DIR)

    # Print summary
    print(f"\n{'='*70}")
    print(f"COMPREHENSIVE BATCH EVALUATION COMPLETE")
    print(f"{'='*70}")
    print(f"Model evaluated:      {MODEL_TO_EVALUATE}")
    print(f"Evaluator models:     {', '.join(EVALUATOR_MODELS)}")
    print(f"Total inputs:         {sampling_info['total_sampled']}")
    print(f"Successful evals:     {results['summary_statistics']['successful_evals']}")
    print(f"Failed evals:         {results['summary_statistics']['failed_evals']}")
    print(f"")
    print(f"SUMMARY STATISTICS:")
    print(f"  Mean Score:         {results['summary_statistics']['mean_score']:.2f}/100")
    print(f"  Median Score:       {results['summary_statistics']['median_score']:.2f}/100")
    print(f"  Worst Score:        {results['summary_statistics']['worst_score']:.2f}/100")
    print(f"  Best Score:         {results['summary_statistics']['best_score']:.2f}/100")
    print(f"  Std Dev:            {results['summary_statistics']['std_dev']:.2f}")
    print(f"  Score Range:        {results['summary_statistics']['score_range']:.2f}")
    print(f"")
    print(f"Results saved to: {results_file}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
