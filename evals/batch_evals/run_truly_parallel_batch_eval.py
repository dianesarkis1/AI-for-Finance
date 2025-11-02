#!/usr/bin/env python3
"""
Truly Parallelized Batch Evaluation Script

This script runs batch evaluations in TRUE parallel:
1. Generate all memos first (sequential, ~30-50 min for 50 inputs)
2. Submit ALL 150 batch jobs at once WITHOUT waiting (50 inputs × 3 evaluators)
3. Poll all 150 batch jobs in parallel until complete
4. Aggregate results

This is MUCH faster than sequential processing.

Usage:
    # Use default comprehensive sample (50 indices)
    python run_truly_parallel_batch_eval.py

    # Test with specific indices
    python run_truly_parallel_batch_eval.py --indices 0 1 2 6 12

    # Test with just one index
    python run_truly_parallel_batch_eval.py --indices 128
"""

import argparse
import json
import os
import random
import statistics
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evals.batch_evals.batch_evaluate import (
    load_training_sample,
    generate_memo_for_input,
    aggregate_evaluator_results
)
from evals.batch_evals.batch_metrics import (
    create_batch_requests_for_memo,
    parse_batch_results,
    create_claude_batch_requests_for_memo,
    parse_claude_batch_results,
    create_gemini_batch_requests_for_memo,
    parse_gemini_batch_results
)
from evals.batch_evals.batch_utils import (
    upload_batch_file,
    create_batch_job,
    check_batch_status,
    download_batch_results,
    load_batch_results,
    create_claude_batch,
    check_claude_batch_status,
    download_claude_batch_results,
    create_gemini_batch,
    check_gemini_batch_status,
    extract_gemini_batch_results
)

# Configuration
TRAIN_FILE = Path("data/train.jsonl")
BASELINE_SAMPLED_INDICES_FILE = Path("evals/benchmark/baseline_sampled_indices_seed42.json")
OUTPUT_DIR = Path("evals/batch_evals")
# Use batch_temp_2 to avoid overwriting existing files for debugging
BATCH_TEMP_DIR = OUTPUT_DIR / "batch_temp_2"
BATCH_TEMP_DIR.mkdir(parents=True, exist_ok=True)
print(f"✓ Using directory: {BATCH_TEMP_DIR}")

# Random seed for reproducibility
RANDOM_SEED = 42

# Model configuration
MODEL_TO_EVALUATE = "claude-sonnet-4-20250514"
EVALUATOR_MODELS = ["gpt-5", "claude-sonnet-4-20250514", "gemini-2.5-pro"]

# Prompt configuration
# Set to None to use default (prompts/baseline.txt)
# Or specify a path like: Path("prompts/my_custom_prompt.txt")
PROMPT_FILE = None


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
    """Phase 1: Generate all memos sequentially."""
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
                memo = generate_memo_for_input(model, credit_agreement_text, temp_input_file, prompt_file=PROMPT_FILE)

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


def submit_all_batch_jobs(memos: Dict[int, Dict], evaluator_models: List[str]) -> List[Dict]:
    """
    Phase 2: Submit ALL batch jobs at once WITHOUT waiting.

    Returns list of batch job info dicts.
    """
    print(f"\n{'='*70}")
    print(f"PHASE 2: SUBMITTING ALL BATCH JOBS (NO WAITING)")
    print(f"{'='*70}")
    print(f"Evaluator models: {', '.join(evaluator_models)}")
    print(f"Total memos: {len([m for m in memos.values() if m['memo'] is not None])}")
    print(f"Total batch jobs to submit: {len([m for m in memos.values() if m['memo'] is not None]) * len(evaluator_models)}")
    print(f"{'='*70}\n")

    batch_jobs = []

    # Get API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")

    for idx, memo_data in memos.items():
        if memo_data['memo'] is None:
            print(f"⏭️  Skipping input {idx} (no memo generated)")
            continue

        print(f"\n📝 Submitting batch jobs for input {idx}...")

        for eval_model in evaluator_models:
            try:
                print(f"  🚀 Submitting {eval_model} batch job...", end=" ")

                if eval_model.startswith("gpt"):
                    # GPT batch
                    requests = create_batch_requests_for_memo(
                        memo=memo_data['memo'],
                        source_document=memo_data['credit_agreement'],
                        template=None,
                        model=eval_model
                    )
                    file_id = upload_batch_file(requests, BATCH_TEMP_DIR, openai_key, input_index=idx)
                    batch_id = create_batch_job(file_id, openai_key, f"Eval {idx} with {eval_model}")

                    batch_jobs.append({
                        "input_index": idx,
                        "evaluator_model": eval_model,
                        "provider": "openai",
                        "batch_id": batch_id,
                        "file_id": file_id
                    })
                    print(f"✅ Job ID: {batch_id}")

                elif "claude" in eval_model.lower():
                    # Claude batch
                    requests = create_claude_batch_requests_for_memo(
                        memo=memo_data['memo'],
                        source_document=memo_data['credit_agreement'],
                        template=None,
                        model=eval_model
                    )
                    batch_id = create_claude_batch(requests, anthropic_key)

                    batch_jobs.append({
                        "input_index": idx,
                        "evaluator_model": eval_model,
                        "provider": "anthropic",
                        "batch_id": batch_id
                    })
                    print(f"✅ Job ID: {batch_id}")

                elif "gemini" in eval_model.lower():
                    # Gemini batch
                    requests = create_gemini_batch_requests_for_memo(
                        memo=memo_data['memo'],
                        source_document=memo_data['credit_agreement'],
                        template=None,
                        model=eval_model
                    )
                    batch_id = create_gemini_batch(requests, gemini_key, eval_model)

                    batch_jobs.append({
                        "input_index": idx,
                        "evaluator_model": eval_model,
                        "provider": "gemini",
                        "batch_id": batch_id
                    })
                    print(f"✅ Job ID: {batch_id}")

            except Exception as e:
                print(f"❌ Error: {e}")
                batch_jobs.append({
                    "input_index": idx,
                    "evaluator_model": eval_model,
                    "provider": None,
                    "batch_id": None,
                    "error": str(e)
                })

    print(f"\n{'='*70}")
    print(f"ALL BATCH JOBS SUBMITTED!")
    print(f"{'='*70}")
    print(f"Total jobs submitted: {len([j for j in batch_jobs if j.get('batch_id')])}")
    print(f"Failed submissions: {len([j for j in batch_jobs if not j.get('batch_id')])}")
    print(f"{'='*70}\n")

    return batch_jobs


def poll_all_batch_jobs(batch_jobs: List[Dict], poll_interval: int = 60) -> Dict[str, Dict]:
    """
    Phase 3: Poll all batch jobs until complete.

    Returns dict mapping batch_id -> results.
    """
    print(f"\n{'='*70}")
    print(f"PHASE 3: POLLING ALL BATCH JOBS")
    print(f"{'='*70}")
    print(f"Total jobs to poll: {len([j for j in batch_jobs if j.get('batch_id')])}")
    print(f"Poll interval: {poll_interval} seconds")
    print(f"{'='*70}\n")

    # Get API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")

    results = {}
    completed = set()
    failed = set()

    jobs_to_poll = {j["batch_id"]: j for j in batch_jobs if j.get("batch_id")}

    start_time = time.time()
    check_count = 0

    while len(completed) + len(failed) < len(jobs_to_poll):
        check_count += 1
        elapsed = int(time.time() - start_time)

        print(f"\n[Check #{check_count}, {elapsed}s elapsed]")
        print(f"  Completed: {len(completed)}/{len(jobs_to_poll)}")
        print(f"  Failed: {len(failed)}/{len(jobs_to_poll)}")
        print(f"  Still running: {len(jobs_to_poll) - len(completed) - len(failed)}")

        for batch_id, job_info in jobs_to_poll.items():
            if batch_id in completed or batch_id in failed:
                continue

            try:
                provider = job_info["provider"]

                if provider == "openai":
                    status_data = check_batch_status(batch_id, openai_key)
                    status = status_data.get("status")

                    if status == "completed":
                        output_file_id = status_data.get("output_file_id")
                        output_path = download_batch_results(output_file_id, BATCH_TEMP_DIR, openai_key, input_index=job_info['input_index'])
                        batch_results = load_batch_results(output_path)
                        parsed = parse_batch_results(batch_results)
                        results[batch_id] = parsed
                        completed.add(batch_id)
                        print(f"  ✅ {job_info['evaluator_model']} for input {job_info['input_index']}: COMPLETE")

                    elif status in ["failed", "expired", "cancelled"]:
                        failed.add(batch_id)
                        print(f"  ❌ {job_info['evaluator_model']} for input {job_info['input_index']}: {status}")

                elif provider == "anthropic":
                    status_data = check_claude_batch_status(batch_id, anthropic_key)
                    processing_status = status_data.get("processing_status")

                    if processing_status == "ended":
                        results_url = status_data.get("results_url")
                        output_path = download_claude_batch_results(results_url, BATCH_TEMP_DIR, anthropic_key, input_index=job_info['input_index'])
                        with open(output_path, 'r') as f:
                            batch_results = [json.loads(line) for line in f]
                        parsed = parse_claude_batch_results(batch_results)
                        results[batch_id] = parsed
                        completed.add(batch_id)
                        print(f"  ✅ {job_info['evaluator_model']} for input {job_info['input_index']}: COMPLETE")

                    elif processing_status in ["failed", "expired", "cancelled"]:
                        failed.add(batch_id)
                        print(f"  ❌ {job_info['evaluator_model']} for input {job_info['input_index']}: {processing_status}")

                elif provider == "gemini":
                    status_data = check_gemini_batch_status(batch_id, gemini_key)
                    state = status_data.get("state")

                    if state == "STATE_SUCCEEDED":
                        output_path = extract_gemini_batch_results(status_data, BATCH_TEMP_DIR, input_index=job_info['input_index'])
                        with open(output_path, 'r') as f:
                            batch_results = [json.loads(line) for line in f]
                        parsed = parse_gemini_batch_results(batch_results)
                        results[batch_id] = parsed
                        completed.add(batch_id)
                        print(f"  ✅ {job_info['evaluator_model']} for input {job_info['input_index']}: COMPLETE")

                    elif state in ["STATE_FAILED", "STATE_CANCELLED"]:
                        failed.add(batch_id)
                        print(f"  ❌ {job_info['evaluator_model']} for input {job_info['input_index']}: {state}")

            except Exception as e:
                print(f"  ⚠️  Error checking {batch_id}: {e}")

        if len(completed) + len(failed) < len(jobs_to_poll):
            print(f"\n  ⏳ Waiting {poll_interval}s before next check...")
            time.sleep(poll_interval)

    print(f"\n{'='*70}")
    print(f"POLLING COMPLETE")
    print(f"{'='*70}")
    print(f"Completed: {len(completed)}/{len(jobs_to_poll)}")
    print(f"Failed: {len(failed)}/{len(jobs_to_poll)}")
    print(f"Total time: {int(time.time() - start_time)}s")
    print(f"{'='*70}\n")

    return results, jobs_to_poll


def aggregate_all_results(
    memos: Dict[int, Dict],
    batch_results: Dict[str, Dict],
    batch_jobs: List[Dict],
    indices: List[int],
    evaluator_models: List[str]
) -> Dict:
    """Phase 4: Aggregate all results."""
    print(f"\n{'='*70}")
    print(f"PHASE 4: AGGREGATING RESULTS")
    print(f"{'='*70}\n")

    # Map batch_id -> job_info
    batch_id_to_job = {j["batch_id"]: j for j in batch_jobs if j.get("batch_id")}

    # Group results by input_index
    results_by_input = {}
    for batch_id, parsed_results in batch_results.items():
        job_info = batch_id_to_job.get(batch_id)
        if not job_info:
            continue

        input_idx = job_info["input_index"]
        if input_idx not in results_by_input:
            results_by_input[input_idx] = []

        # Add evaluator model info to results
        evaluator_result = {
            "evaluator_model": job_info["evaluator_model"],
            "summary_score": None,
            "metrics": parsed_results
        }

        # Calculate summary score from metrics
        from evals.metrics import calculate_summary_score
        summary_result = calculate_summary_score(
            accuracy_result=parsed_results["accuracy_result"],
            completeness_result=parsed_results["completeness_result"],
            consistency_result=parsed_results["consistency_result"],
            quality_result=parsed_results["quality_result"]
        )
        evaluator_result["summary_score"] = summary_result["summary_score"]
        evaluator_result["metrics"] = {
            "accuracy": parsed_results["accuracy_result"],
            "completeness": parsed_results["completeness_result"],
            "consistency": parsed_results["consistency_result"],
            "quality": parsed_results["quality_result"]
        }

        results_by_input[input_idx].append(evaluator_result)

    # Aggregate per input
    detailed_results = []
    for idx in indices:
        memo_data = memos.get(idx)
        evaluator_results = results_by_input.get(idx, [])

        if not memo_data or not evaluator_results:
            detailed_results.append({
                "input_index": idx,
                "source_url": memo_data['source_url'] if memo_data else None,
                "summary_score": None,
                "metrics": None,
                "evaluator_results": None,
                "error": memo_data.get('error') if memo_data else "Unknown error"
            })
            continue

        # Aggregate evaluator results
        aggregated = aggregate_evaluator_results(evaluator_results)

        if aggregated:
            detailed_results.append({
                "input_index": idx,
                "source_url": memo_data['source_url'],
                "summary_score": aggregated["summary_score"],
                "metrics": aggregated["metrics"],
                "evaluator_results": evaluator_results,
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
        "evaluated_indices": indices,
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
    """Save results."""
    comprehensive_results = {
        "model": results["model_evaluated"],
        "evaluator_models": results["evaluator_models"],
        "random_seed": sampling_info["random_seed"],
        "sample_size": sampling_info["total_sampled"],
        "sampling_breakdown": sampling_info["sampling_breakdown"],
        "sampled_indices": sampling_info["all_sampled_indices"],

        "mean_score": results["summary_statistics"]["mean_score"],
        "median_score": results["summary_statistics"]["median_score"],
        "worst_score": results["summary_statistics"]["worst_score"],
        "best_score": results["summary_statistics"]["best_score"],
        "std_dev": results["summary_statistics"]["std_dev"],
        "score_range": results["summary_statistics"]["score_range"],

        "total_inputs_in_dataset": sampling_info["total_inputs_in_dataset"],
        "successful_evals": results["summary_statistics"]["successful_evals"],
        "failed_evals": results["summary_statistics"]["failed_evals"],

        "metric_statistics": results["metric_statistics"],

        "all_results": [
            {
                "input_index": r["input_index"],
                "source_url": r["source_url"],
                "score": r["summary_score"],
                "error": r["error"]
            }
            for r in results["detailed_results"]
        ],

        "detailed_results": results["detailed_results"],
        "created_at": sampling_info["created_at"]
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"comprehensive_batch_eval_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(comprehensive_results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"RESULTS SAVED")
    print(f"{'='*70}")
    print(f"Results file: {results_file}")

    sampling_file = output_dir / f"comprehensive_sampled_indices_{timestamp}.json"
    with open(sampling_file, 'w') as f:
        json.dump(sampling_info, f, indent=2)

    print(f"Sampling info: {sampling_file}")
    print(f"{'='*70}\n")

    return results_file, sampling_file


def main():
    """Main execution function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run truly parallelized batch evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default comprehensive sample (50 indices)
  python run_truly_parallel_batch_eval.py

  # Test with specific indices
  python run_truly_parallel_batch_eval.py --indices 0 1 2 6 12

  # Test with just one index
  python run_truly_parallel_batch_eval.py --indices 128
        """
    )
    parser.add_argument(
        '--indices',
        type=int,
        nargs='+',
        help='Custom indices to evaluate (space-separated). If not provided, uses default comprehensive sample.'
    )
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"TRULY PARALLELIZED COMPREHENSIVE BATCH EVALUATION")
    print(f"{'='*70}\n")

    # Determine which indices to use
    if args.indices:
        # Use custom indices from command line
        indices_to_evaluate = args.indices
        sampling_info = {
            'all_sampled_indices': indices_to_evaluate,
            'total_sampled': len(indices_to_evaluate),
            'source': 'command_line_custom',
            'baseline_count': 0,
            'first_n_count': 0,
            'random_sample_count': 0
        }
        print(f"Using custom indices from command line: {indices_to_evaluate}")
        print(f"  Total indices: {len(indices_to_evaluate)}\n")
    else:
        # Use default comprehensive sample
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
        sampling_info = create_comprehensive_sample(
            baseline_indices=baseline_indices,
            first_n=3,
            random_sample_size=37,
            total_samples=total_samples,
            seed=RANDOM_SEED
        )

        indices_to_evaluate = sampling_info['all_sampled_indices']
        print(f"\n  Created comprehensive sample with {sampling_info['total_sampled']} total indices")

    # Phase 1: Generate all memos
    memos = generate_all_memos(
        indices=indices_to_evaluate,
        train_file=TRAIN_FILE,
        model=MODEL_TO_EVALUATE
    )

    # Phase 2: Submit all batch jobs (NO WAITING)
    batch_jobs = submit_all_batch_jobs(
        memos=memos,
        evaluator_models=EVALUATOR_MODELS
    )

    # Phase 3: Poll all batch jobs until complete
    batch_results, jobs_info = poll_all_batch_jobs(
        batch_jobs=batch_jobs,
        poll_interval=60
    )

    # Phase 4: Aggregate results
    # COMMENTED OUT: This aggregation creates comprehensive_batch_eval_results_{timestamp}.json
    # results = aggregate_all_results(
    #     memos=memos,
    #     batch_results=batch_results,
    #     batch_jobs=batch_jobs,
    #     indices=sampling_info['all_sampled_indices'],
    #     evaluator_models=EVALUATOR_MODELS
    # )

    # # Save results
    # print(f"\nSaving comprehensive results to batch_evals folder...")
    # results_file, sampling_file = save_results(results, sampling_info, OUTPUT_DIR)

    # Print final summary
    print(f"\n{'='*70}")
    print(f"COMPREHENSIVE BATCH EVALUATION COMPLETE")
    print(f"{'='*70}")
    print(f"Model evaluated:      {MODEL_TO_EVALUATE}")
    print(f"Evaluator models:     {', '.join(EVALUATOR_MODELS)}")
    print(f"Total inputs:         {sampling_info['total_sampled']}")
    # COMMENTED OUT: Summary statistics depend on aggregation results
    # print(f"Successful evals:     {results['summary_statistics']['successful_evals']}")
    # print(f"Failed evals:         {results['summary_statistics']['failed_evals']}")
    # print(f"")
    # print(f"SUMMARY STATISTICS:")
    # print(f"  Mean Score:         {results['summary_statistics']['mean_score']:.2f}/100")
    # print(f"  Median Score:       {results['summary_statistics']['median_score']:.2f}/100")
    # print(f"  Worst Score:        {results['summary_statistics']['worst_score']:.2f}/100")
    # print(f"  Best Score:         {results['summary_statistics']['best_score']:.2f}/100")
    # print(f"  Std Dev:            {results['summary_statistics']['std_dev']:.2f}")
    # print(f"  Score Range:        {results['summary_statistics']['score_range']:.2f}")
    # print(f"")
    # print(f"Results saved to: {results_file}")
    print(f"")
    print(f"Batch evaluation jobs completed. Results are in batch_temp/ folder.")
    print(f"Run generate_final_results.py to aggregate the results.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
