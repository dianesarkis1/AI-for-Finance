#!/usr/bin/env python3
"""
Parallelized Comprehensive Batch Evaluation Script

This script runs batch evaluations in parallel for maximum efficiency:
1. Generate all memos first (sequential, ~30-50 min for 50 inputs)
2. Submit ALL batch jobs at once (150 jobs: 50 inputs × 3 evaluators)
3. Poll all batch jobs in parallel until complete
4. Aggregate results

This approach is much faster than sequential processing (1-2 hours vs 4-8 hours).

Usage:
    python run_parallel_batch_eval.py
"""

import json
import random
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import tempfile

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evals.batch_evals.batch_evaluate import (
    load_training_sample,
    generate_memo_for_input,
    evaluate_memo_with_model,
    aggregate_evaluator_results
)

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
    """Create comprehensive sample combining baseline, first N, and random samples."""
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


def generate_all_memos(indices: List[int], train_file: Path, model: str) -> Dict[int, Dict]:
    """
    Phase 1: Generate all memos sequentially.

    Returns dict mapping index -> {source_url, memo, credit_agreement}
    """
    print(f"\n{'='*70}")
    print(f"PHASE 1: GENERATING ALL MEMOS")
    print(f"{'='*70}")
    print(f"Model: {model}")
    print(f"Total inputs: {len(indices)}")
    print(f"{'='*70}\n")

    memos = {}

    # Create temp input file for model_run.py
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as tmp_in:
        temp_input_file = Path(tmp_in.name)

    try:
        for i, idx in enumerate(indices):
            print(f"[{i+1}/{len(indices)}] Generating memo for input {idx}...")

            try:
                # Load training sample
                source_url, credit_agreement_text = load_training_sample(train_file, idx)
                print(f"  Source: {source_url[:80]}...")

                # Generate memo
                memo = generate_memo_for_input(model, credit_agreement_text, temp_input_file)

                if memo:
                    memos[idx] = {
                        "source_url": source_url,
                        "memo": memo,
                        "credit_agreement": credit_agreement_text,
                        "error": None
                    }
                    print(f"  ✅ Generated memo: {len(memo)} chars\n")
                else:
                    memos[idx] = {
                        "source_url": source_url,
                        "memo": None,
                        "credit_agreement": credit_agreement_text,
                        "error": "Failed to generate memo"
                    }
                    print(f"  ❌ Failed to generate memo\n")

            except Exception as e:
                print(f"  ❌ Error: {e}\n")
                memos[idx] = {
                    "source_url": None,
                    "memo": None,
                    "credit_agreement": None,
                    "error": str(e)
                }

    finally:
        # Clean up temp file
        try:
            temp_input_file.unlink()
        except:
            pass

    successful = sum(1 for m in memos.values() if m['memo'] is not None)
    print(f"\n{'='*70}")
    print(f"MEMO GENERATION COMPLETE")
    print(f"{'='*70}")
    print(f"Successful: {successful}/{len(indices)}")
    print(f"Failed: {len(indices) - successful}/{len(indices)}")
    print(f"{'='*70}\n")

    return memos


def submit_all_evaluations(memos: Dict[int, Dict], evaluator_models: List[str]) -> Dict:
    """
    Phase 2: Submit ALL batch evaluation jobs at once.

    Returns dict with batch job info for polling.
    """
    print(f"\n{'='*70}")
    print(f"PHASE 2: SUBMITTING ALL BATCH EVALUATIONS")
    print(f"{'='*70}")
    print(f"Evaluator models: {', '.join(evaluator_models)}")
    print(f"Total memos: {len(memos)}")
    print(f"Total batch jobs: {len(memos) * len(evaluator_models)}")
    print(f"{'='*70}\n")

    evaluation_jobs = {}

    for idx, memo_data in memos.items():
        if memo_data['memo'] is None:
            print(f"⏭️  Skipping input {idx} (no memo generated)")
            evaluation_jobs[idx] = {
                "evaluator_results": None,
                "error": memo_data['error']
            }
            continue

        print(f"\n📝 Submitting evaluations for input {idx}...")

        evaluator_results = []
        for eval_model in evaluator_models:
            print(f"  🚀 Submitting batch for {eval_model}...")

            try:
                result = evaluate_memo_with_model(
                    memo=memo_data['memo'],
                    source_document=memo_data['credit_agreement'],
                    evaluator_model=eval_model,
                    poll_interval=60
                )

                if result:
                    evaluator_results.append(result)
                    print(f"    ✅ {eval_model}: {result['summary_score']:.2f}/100")
                else:
                    print(f"    ❌ {eval_model}: Failed")

            except Exception as e:
                print(f"    ❌ {eval_model}: Error - {e}")

        evaluation_jobs[idx] = {
            "evaluator_results": evaluator_results if evaluator_results else None,
            "error": None if evaluator_results else "All evaluations failed"
        }

    print(f"\n{'='*70}")
    print(f"ALL BATCH JOBS SUBMITTED")
    print(f"{'='*70}\n")

    return evaluation_jobs


def aggregate_all_results(
    memos: Dict[int, Dict],
    evaluation_jobs: Dict,
    sampling_info: Dict,
    evaluator_models: List[str]
) -> Dict:
    """
    Phase 3: Aggregate all results.
    """
    print(f"\n{'='*70}")
    print(f"PHASE 3: AGGREGATING RESULTS")
    print(f"{'='*70}\n")

    detailed_results = []

    for idx in sampling_info['all_sampled_indices']:
        memo_data = memos.get(idx)
        eval_data = evaluation_jobs.get(idx)

        if not memo_data or not eval_data or not eval_data['evaluator_results']:
            detailed_results.append({
                "input_index": idx,
                "source_url": memo_data['source_url'] if memo_data else None,
                "summary_score": None,
                "metrics": None,
                "evaluator_results": None,
                "error": eval_data['error'] if eval_data else "Unknown error"
            })
            continue

        # Aggregate evaluator results
        aggregated = aggregate_evaluator_results(eval_data['evaluator_results'])

        if aggregated:
            detailed_results.append({
                "input_index": idx,
                "source_url": memo_data['source_url'],
                "summary_score": aggregated["summary_score"],
                "metrics": aggregated["metrics"],
                "evaluator_results": eval_data['evaluator_results'],
                "error": None
            })
            print(f"✅ Input {idx}: {aggregated['summary_score']:.2f}/100")
        else:
            detailed_results.append({
                "input_index": idx,
                "source_url": memo_data['source_url'],
                "summary_score": None,
                "metrics": None,
                "evaluator_results": None,
                "error": "Failed to aggregate results"
            })
            print(f"❌ Input {idx}: Failed to aggregate")

    # Calculate summary statistics
    valid_scores = [r['summary_score'] for r in detailed_results if r['summary_score'] is not None]

    if not valid_scores:
        summary_statistics = {
            "mean_score": 0.0,
            "median_score": 0.0,
            "worst_score": 0.0,
            "best_score": 0.0,
            "std_dev": 0.0,
            "score_range": 0.0,
            "successful_evals": 0,
            "failed_evals": len(detailed_results)
        }
        metric_statistics = None
    else:
        summary_statistics = {
            "mean_score": statistics.mean(valid_scores),
            "median_score": statistics.median(valid_scores),
            "worst_score": min(valid_scores),
            "best_score": max(valid_scores),
            "std_dev": statistics.stdev(valid_scores) if len(valid_scores) > 1 else 0.0,
            "score_range": max(valid_scores) - min(valid_scores),
            "successful_evals": len(valid_scores),
            "failed_evals": len(detailed_results) - len(valid_scores)
        }

        # Per-metric aggregated statistics
        valid_results = [r for r in detailed_results if r['metrics'] is not None]

        if valid_results:
            accuracy_scores = [r['metrics']['accuracy']['score'] * 100 for r in valid_results]
            completeness_scores = [r['metrics']['completeness']['score'] * 100 for r in valid_results]
            consistency_scores = [r['metrics']['consistency']['score'] * 100 for r in valid_results]
            quality_scores = [r['metrics']['quality']['quality_score'] for r in valid_results]
            clarity_scores = [r['metrics']['quality']['clarity_score'] for r in valid_results]
            tone_scores = [r['metrics']['quality']['tone_score'] for r in valid_results]
            length_scores = [r['metrics']['quality']['length_score'] for r in valid_results]
            structure_scores = [r['metrics']['quality']['structure_score'] for r in valid_results]

            metric_statistics = {
                "accuracy": {
                    "mean": statistics.mean(accuracy_scores),
                    "median": statistics.median(accuracy_scores),
                    "min": min(accuracy_scores),
                    "max": max(accuracy_scores),
                    "std_dev": statistics.stdev(accuracy_scores) if len(accuracy_scores) > 1 else 0.0
                },
                "completeness": {
                    "mean": statistics.mean(completeness_scores),
                    "median": statistics.median(completeness_scores),
                    "min": min(completeness_scores),
                    "max": max(completeness_scores),
                    "std_dev": statistics.stdev(completeness_scores) if len(completeness_scores) > 1 else 0.0
                },
                "consistency": {
                    "mean": statistics.mean(consistency_scores),
                    "median": statistics.median(consistency_scores),
                    "min": min(consistency_scores),
                    "max": max(consistency_scores),
                    "std_dev": statistics.stdev(consistency_scores) if len(consistency_scores) > 1 else 0.0
                },
                "quality": {
                    "mean": statistics.mean(quality_scores),
                    "median": statistics.median(quality_scores),
                    "min": min(quality_scores),
                    "max": max(quality_scores),
                    "std_dev": statistics.stdev(quality_scores) if len(quality_scores) > 1 else 0.0
                },
                "clarity": {
                    "mean": statistics.mean(clarity_scores),
                    "median": statistics.median(clarity_scores),
                    "min": min(clarity_scores),
                    "max": max(clarity_scores),
                    "std_dev": statistics.stdev(clarity_scores) if len(clarity_scores) > 1 else 0.0
                },
                "tone": {
                    "mean": statistics.mean(tone_scores),
                    "median": statistics.median(tone_scores),
                    "min": min(tone_scores),
                    "max": max(tone_scores),
                    "std_dev": statistics.stdev(tone_scores) if len(tone_scores) > 1 else 0.0
                },
                "length": {
                    "mean": statistics.mean(length_scores),
                    "median": statistics.median(length_scores),
                    "min": min(length_scores),
                    "max": max(length_scores),
                    "std_dev": statistics.stdev(length_scores) if len(length_scores) > 1 else 0.0
                },
                "structure": {
                    "mean": statistics.mean(structure_scores),
                    "median": statistics.median(structure_scores),
                    "min": min(structure_scores),
                    "max": max(structure_scores),
                    "std_dev": statistics.stdev(structure_scores) if len(structure_scores) > 1 else 0.0
                }
            }
        else:
            metric_statistics = None

    results = {
        "model_evaluated": MODEL_TO_EVALUATE,
        "evaluator_models": evaluator_models,
        "dataset": str(TRAIN_FILE),
        "evaluated_indices": sampling_info['all_sampled_indices'],
        "summary_statistics": summary_statistics,
        "metric_statistics": metric_statistics,
        "detailed_results": detailed_results
    }

    print(f"\n{'='*70}")
    print(f"AGGREGATION COMPLETE")
    print(f"{'='*70}")
    print(f"Successful evals: {summary_statistics['successful_evals']}")
    print(f"Failed evals: {summary_statistics['failed_evals']}")
    print(f"Mean score: {summary_statistics['mean_score']:.2f}/100")
    print(f"{'='*70}\n")

    return results


def save_results(results: Dict, sampling_info: Dict, output_dir: Path):
    """Save results in the format similar to baseline_benchmark_results_complete.json."""
    # Create comprehensive results structure
    comprehensive_results = {
        "model": results["model_evaluated"],
        "evaluator_models": results["evaluator_models"],
        "random_seed": sampling_info["random_seed"],
        "sample_size": sampling_info["total_sampled"],
        "sampling_breakdown": sampling_info["sampling_breakdown"],
        "sampled_indices": sampling_info["all_sampled_indices"],

        # Summary statistics
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

        # Individual results
        "all_results": [
            {
                "input_index": r["input_index"],
                "source_url": r["source_url"],
                "score": r["summary_score"],
                "error": r["error"]
            }
            for r in results["detailed_results"]
        ],

        # Detailed results
        "detailed_results": results["detailed_results"],

        "created_at": sampling_info["created_at"]
    }

    # Save main results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"comprehensive_batch_eval_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(comprehensive_results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"RESULTS SAVED")
    print(f"{'='*70}")
    print(f"Results file: {results_file}")

    # Also save sampling info
    sampling_file = output_dir / f"comprehensive_sampled_indices_{timestamp}.json"
    with open(sampling_file, 'w') as f:
        json.dump(sampling_info, f, indent=2)

    print(f"Sampling info: {sampling_file}")
    print(f"{'='*70}\n")

    return results_file, sampling_file


def main():
    """Main execution function."""
    print(f"\n{'='*70}")
    print(f"PARALLELIZED COMPREHENSIVE BATCH EVALUATION")
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

    # Phase 1: Generate all memos
    memos = generate_all_memos(
        indices=sampling_info['all_sampled_indices'],
        train_file=TRAIN_FILE,
        model=MODEL_TO_EVALUATE
    )

    # Phase 2: Submit all evaluations
    evaluation_jobs = submit_all_evaluations(
        memos=memos,
        evaluator_models=EVALUATOR_MODELS
    )

    # Phase 3: Aggregate results
    results = aggregate_all_results(
        memos=memos,
        evaluation_jobs=evaluation_jobs,
        sampling_info=sampling_info,
        evaluator_models=EVALUATOR_MODELS
    )

    # Save results
    print(f"\nSaving comprehensive results to batch_evals folder...")
    results_file, sampling_file = save_results(results, sampling_info, OUTPUT_DIR)

    # Print final summary
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
