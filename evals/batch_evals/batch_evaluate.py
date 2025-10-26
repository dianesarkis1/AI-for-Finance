#!/usr/bin/env python3
"""
Batch Evaluation Script

Evaluates a model's memos using batch evaluation APIs for faster processing.
Takes a list of input indices from the training set, generates memos using the
specified model, then evaluates them using batch APIs with multiple evaluator models.

Outputs a comprehensive results dictionary with:
- Summary statistics (mean, median, worst, best scores)
- Per-metric aggregated statistics
- Detailed per-input results with full metric breakdowns

Usage:
    python batch_evaluate.py --indices 12 52 57 71 --model claude-sonnet-4-20250514
    python batch_evaluate.py --indices-file sampled_indices.json --model gpt-5
"""

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evals.batch_evals.batch_utils import (
    evaluate_memo_batch,
    submit_and_wait_for_batch
)
from evals.batch_evals.batch_metrics import parse_batch_results
import os

# Directory for temporary batch files
BATCH_TEMP_DIR = Path(__file__).parent / "batch_temp"
BATCH_TEMP_DIR.mkdir(exist_ok=True)


def load_training_sample(train_file: Path, index: int) -> tuple[str, str]:
    """
    Load a single training sample by index.

    Args:
        train_file: Path to train.jsonl
        index: 0-based index of the sample to load

    Returns:
        Tuple of (source_url, credit_agreement_text)
    """
    with open(train_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == index:
                data = json.loads(line)
                return data['source_url'], data['text']

    raise ValueError(f"Index {index} not found in {train_file}")


def generate_memo_for_input(
    model: str,
    credit_agreement_text: str,
    temp_input_file: Path
) -> Optional[str]:
    """
    Generate memo using model_run.py for a given credit agreement.

    Args:
        model: Model identifier (e.g., 'claude-sonnet-4-20250514')
        credit_agreement_text: Credit agreement text
        temp_input_file: Temporary file to write credit agreement to

    Returns:
        Generated memo text, or None if generation failed
    """
    # Write credit agreement to temp JSONL file for model_run.py
    with open(temp_input_file, 'w', encoding='utf-8') as f:
        json.dump({"text": credit_agreement_text}, f)

    # Create temp output file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as tmp_out:
        output_path = tmp_out.name

    try:
        # Call model_run.py
        cmd = [
            "python",
            "latest project scripts/model_run.py",
            "--model", model,
            "--input-file", str(temp_input_file),
            "--output", output_path
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        if result.returncode != 0:
            print(f"  ❌ Error generating memo: {result.stderr}", file=sys.stderr)
            return None

        # Read generated memo
        with open(output_path, 'r', encoding='utf-8') as f:
            memo = f.read()

        return memo

    except subprocess.TimeoutExpired:
        print(f"  ⏰ Timeout generating memo", file=sys.stderr)
        return None
    except Exception as e:
        print(f"  💥 Error: {e}", file=sys.stderr)
        return None
    finally:
        # Clean up output file
        try:
            Path(output_path).unlink()
        except:
            pass


def evaluate_memo_with_model(
    memo: str,
    source_document: str,
    evaluator_model: str,
    poll_interval: int = 60
) -> Dict:
    """
    Evaluate a memo using a specific evaluator model via batch API.

    Args:
        memo: Generated memo text
        source_document: Original credit agreement
        evaluator_model: Model to use for evaluation (gpt-5, claude, gemini)
        poll_interval: Seconds between batch status checks

    Returns:
        Dict with detailed metric results, or None if evaluation failed
    """
    try:
        if evaluator_model.startswith("gpt"):
            # GPT-5 batch evaluation
            from evals.batch_evals.batch_metrics import create_batch_requests_for_memo

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment")

            requests = create_batch_requests_for_memo(
                memo=memo,
                source_document=source_document,
                template=None,
                model=evaluator_model
            )

            results = submit_and_wait_for_batch(
                requests=requests,
                api_key=api_key,
                temp_dir=BATCH_TEMP_DIR,
                description=f"Evaluation with {evaluator_model}",
                poll_interval=poll_interval
            )

            parsed = parse_batch_results(results)

        elif "claude" in evaluator_model.lower():
            # Claude batch evaluation
            from evals.batch_evals.batch_metrics import create_claude_batch_requests_for_memo, parse_claude_batch_results
            from evals.batch_evals.batch_utils import submit_and_wait_for_claude_batch

            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not found in environment")

            requests = create_claude_batch_requests_for_memo(
                memo=memo,
                source_document=source_document,
                template=None,
                model=evaluator_model
            )

            results = submit_and_wait_for_claude_batch(
                requests=requests,
                api_key=api_key,
                temp_dir=BATCH_TEMP_DIR,
                poll_interval=poll_interval
            )

            parsed = parse_claude_batch_results(results)

        elif "gemini" in evaluator_model.lower():
            # Gemini batch evaluation
            from evals.batch_evals.batch_metrics import create_gemini_batch_requests_for_memo, parse_gemini_batch_results
            from evals.batch_evals.batch_utils import submit_and_wait_for_gemini_batch

            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found in environment")

            requests = create_gemini_batch_requests_for_memo(
                memo=memo,
                source_document=source_document,
                template=None,
                model=evaluator_model
            )

            results = submit_and_wait_for_gemini_batch(
                requests=requests,
                api_key=api_key,
                temp_dir=BATCH_TEMP_DIR,
                model=evaluator_model,
                poll_interval=poll_interval
            )

            parsed = parse_gemini_batch_results(results)

        else:
            raise ValueError(f"Unknown evaluator model: {evaluator_model}")

        # Extract metric details (common for all implemented models)
        accuracy_result = parsed["accuracy_result"]
        completeness_result = parsed["completeness_result"]
        consistency_result = parsed["consistency_result"]
        quality_result = parsed["quality_result"]

        # Calculate summary score
        from evals.metrics import calculate_summary_score
        summary_result = calculate_summary_score(
            accuracy_result=accuracy_result,
            completeness_result=completeness_result,
            consistency_result=consistency_result,
            quality_result=quality_result
        )

        return {
            "evaluator_model": evaluator_model,
            "summary_score": summary_result["summary_score"],
            "metrics": {
                "accuracy": accuracy_result,
                "completeness": completeness_result,
                "consistency": consistency_result,
                "quality": quality_result
            }
        }

    except Exception as e:
        print(f"  ❌ Evaluation failed with {evaluator_model}: {e}")
        return None


def aggregate_evaluator_results(evaluator_results: List[Dict]) -> Dict:
    """
    Aggregate results from multiple evaluator models into a single score.

    Uses ensemble averaging: takes the mean of all evaluator scores.

    Args:
        evaluator_results: List of evaluation results from different models

    Returns:
        Dict with aggregated metrics and summary score
    """
    # Filter out failed evaluations
    valid_results = [r for r in evaluator_results if r is not None]

    if not valid_results:
        return None

    # Average summary scores
    summary_scores = [r["summary_score"] for r in valid_results]
    aggregated_summary_score = statistics.mean(summary_scores)

    # Average individual metrics
    accuracy_scores = [r["metrics"]["accuracy"]["score"] for r in valid_results]
    completeness_scores = [r["metrics"]["completeness"]["score"] for r in valid_results]
    consistency_scores = [r["metrics"]["consistency"]["score"] for r in valid_results]
    quality_scores = [r["metrics"]["quality"]["quality_score"] for r in valid_results]
    clarity_scores = [r["metrics"]["quality"]["clarity_score"] for r in valid_results]
    tone_scores = [r["metrics"]["quality"]["tone_score"] for r in valid_results]
    length_scores = [r["metrics"]["quality"]["length_score"] for r in valid_results]
    structure_scores = [r["metrics"]["quality"]["structure_score"] for r in valid_results]

    return {
        "summary_score": aggregated_summary_score,
        "evaluator_models_used": [r["evaluator_model"] for r in valid_results],
        "individual_evaluator_scores": {
            r["evaluator_model"]: r["summary_score"] for r in valid_results
        },
        "metrics": {
            "accuracy": {
                "score": statistics.mean(accuracy_scores),
                "min": min(accuracy_scores),
                "max": max(accuracy_scores)
            },
            "completeness": {
                "score": statistics.mean(completeness_scores),
                "min": min(completeness_scores),
                "max": max(completeness_scores)
            },
            "consistency": {
                "score": statistics.mean(consistency_scores),
                "min": min(consistency_scores),
                "max": max(consistency_scores)
            },
            "quality": {
                "quality_score": statistics.mean(quality_scores),
                "clarity_score": statistics.mean(clarity_scores),
                "tone_score": statistics.mean(tone_scores),
                "length_score": statistics.mean(length_scores),
                "structure_score": statistics.mean(structure_scores)
            }
        }
    }


def run_batch_benchmark(
    model: str,
    train_file: Path,
    indices: List[int],
    evaluator_models: List[str] = None,
    poll_interval: int = 60,
    delay_between_inputs: float = 5.0,
    save_results: bool = True
) -> Dict:
    """
    Run batch benchmark on specified training set indices.

    Args:
        model: Model to evaluate (e.g., 'claude-sonnet-4-20250514')
        train_file: Path to train.jsonl
        indices: List of 0-based indices to evaluate
        evaluator_models: Models to use for evaluation (default: gpt-5, claude, gemini)
        poll_interval: Seconds between batch status checks (default: 60)
        delay_between_inputs: Seconds to wait between processing inputs (default: 5.0)
        save_results: Whether to save results to JSON file (default: True)

    Returns:
        Dict with comprehensive benchmark results:
            - summary_statistics: Aggregated stats across all inputs
            - metric_statistics: Per-metric aggregated stats
            - detailed_results: Per-input detailed breakdowns
    """
    if evaluator_models is None:
        evaluator_models = ["gpt-5", "claude-sonnet-4-20250514", "gemini-2.5-pro"]

    print(f"\n{'='*70}")
    print(f"BATCH BENCHMARK")
    print(f"{'='*70}")
    print(f"Model to evaluate: {model}")
    print(f"Evaluator models: {', '.join(evaluator_models)}")
    print(f"Training set: {train_file}")
    print(f"Indices to evaluate: {indices}")
    print(f"Total inputs: {len(indices)}")
    print(f"{'='*70}\n")

    # Results storage
    detailed_results = []

    # Create temp input file for model_run.py
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as tmp_in:
        temp_input_file = Path(tmp_in.name)

    try:
        for i, idx in enumerate(indices):
            print(f"\n[{i+1}/{len(indices)}] Processing input {idx}...")

            # Load training sample
            try:
                source_url, credit_agreement_text = load_training_sample(train_file, idx)
                print(f"  Source: {source_url[:80]}...")
            except Exception as e:
                print(f"  ❌ Failed to load input {idx}: {e}")
                detailed_results.append({
                    "input_index": idx,
                    "source_url": None,
                    "summary_score": None,
                    "metrics": None,
                    "evaluator_results": None,
                    "error": f"Failed to load input: {str(e)}"
                })
                continue

            # Generate memo
            print(f"  Generating memo with {model}...")
            memo = generate_memo_for_input(model, credit_agreement_text, temp_input_file)

            if memo is None:
                detailed_results.append({
                    "input_index": idx,
                    "source_url": source_url,
                    "summary_score": None,
                    "metrics": None,
                    "evaluator_results": None,
                    "error": "Failed to generate memo"
                })
                continue

            print(f"  Generated memo: {len(memo)} chars")

            # Evaluate with all evaluator models
            print(f"  Evaluating with {len(evaluator_models)} evaluator models...")
            evaluator_results = []

            for eval_model in evaluator_models:
                print(f"    Evaluating with {eval_model}...")
                result = evaluate_memo_with_model(
                    memo=memo,
                    source_document=credit_agreement_text,
                    evaluator_model=eval_model,
                    poll_interval=poll_interval
                )

                if result is not None:
                    evaluator_results.append(result)
                    print(f"    ✅ {eval_model}: {result['summary_score']:.2f}/100")

            # Aggregate results from all evaluators
            if evaluator_results:
                aggregated = aggregate_evaluator_results(evaluator_results)
                print(f"  📊 Aggregated score: {aggregated['summary_score']:.2f}/100")

                detailed_results.append({
                    "input_index": idx,
                    "source_url": source_url,
                    "summary_score": aggregated["summary_score"],
                    "metrics": aggregated["metrics"],
                    "evaluator_results": evaluator_results,
                    "error": None
                })
            else:
                print(f"  ❌ All evaluations failed")
                detailed_results.append({
                    "input_index": idx,
                    "source_url": source_url,
                    "summary_score": None,
                    "metrics": None,
                    "evaluator_results": None,
                    "error": "All evaluations failed"
                })

            # Wait between inputs (except on last iteration)
            if i < len(indices) - 1:
                print(f"  ⏳ Waiting {delay_between_inputs}s before next input...")
                time.sleep(delay_between_inputs)

    finally:
        # Clean up temp input file
        try:
            temp_input_file.unlink()
        except:
            pass

    # Calculate summary statistics
    valid_scores = [r['summary_score'] for r in detailed_results if r['summary_score'] is not None]

    if not valid_scores:
        print("\n❌ No valid scores obtained!")
        return {
            "model_evaluated": model,
            "evaluator_models": evaluator_models,
            "dataset": str(train_file),
            "evaluated_indices": indices,
            "summary_statistics": {
                "mean_score": 0.0,
                "median_score": 0.0,
                "worst_score": 0.0,
                "best_score": 0.0,
                "std_dev": 0.0,
                "score_range": 0.0,
                "successful_evals": 0,
                "failed_evals": len(indices)
            },
            "metric_statistics": None,
            "detailed_results": detailed_results
        }

    # Summary statistics
    summary_statistics = {
        "mean_score": statistics.mean(valid_scores),
        "median_score": statistics.median(valid_scores),
        "worst_score": min(valid_scores),
        "best_score": max(valid_scores),
        "std_dev": statistics.stdev(valid_scores) if len(valid_scores) > 1 else 0.0,
        "score_range": max(valid_scores) - min(valid_scores),
        "successful_evals": len(valid_scores),
        "failed_evals": len(indices) - len(valid_scores)
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

    # Print results
    print(f"\n{'='*70}")
    print(f"BENCHMARK RESULTS")
    print(f"{'='*70}")
    print(f"Model evaluated:      {model}")
    print(f"Evaluator models:     {', '.join(evaluator_models)}")
    print(f"Total inputs:         {len(indices)}")
    print(f"Successful evals:     {summary_statistics['successful_evals']}")
    print(f"Failed evals:         {summary_statistics['failed_evals']}")
    print(f"")
    print(f"SUMMARY STATISTICS:")
    print(f"  Mean Score:         {summary_statistics['mean_score']:.2f}/100")
    print(f"  Median Score:       {summary_statistics['median_score']:.2f}/100")
    print(f"  Worst Score:        {summary_statistics['worst_score']:.2f}/100")
    print(f"  Best Score:         {summary_statistics['best_score']:.2f}/100")
    print(f"  Std Dev:            {summary_statistics['std_dev']:.2f}")
    print(f"  Score Range:        {summary_statistics['score_range']:.2f}")

    if metric_statistics:
        print(f"\nPER-METRIC STATISTICS:")
        print(f"  Accuracy:       {metric_statistics['accuracy']['mean']:.2f}/100")
        print(f"  Completeness:   {metric_statistics['completeness']['mean']:.2f}/100")
        print(f"  Consistency:    {metric_statistics['consistency']['mean']:.2f}/100")
        print(f"  Quality:        {metric_statistics['quality']['mean']:.2f}/100")
        print(f"    - Clarity:    {metric_statistics['clarity']['mean']:.2f}/100")
        print(f"    - Tone:       {metric_statistics['tone']['mean']:.2f}/100")
        print(f"    - Length:     {metric_statistics['length']['mean']:.2f}/100")
        print(f"    - Structure:  {metric_statistics['structure']['mean']:.2f}/100")

    print(f"{'='*70}\n")

    # Compile full results
    results = {
        "model_evaluated": model,
        "evaluator_models": evaluator_models,
        "dataset": str(train_file),
        "evaluated_indices": indices,
        "summary_statistics": summary_statistics,
        "metric_statistics": metric_statistics,
        "detailed_results": detailed_results
    }

    # Save results
    if save_results:
        output_file = f"batch_benchmark_results_{model.replace('/', '_')}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Saved complete results to {output_file}\n")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run batch benchmark on training set with specified indices"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-sonnet-4-20250514",
        help="Model to evaluate (default: claude-sonnet-4-20250514)"
    )
    parser.add_argument(
        "--train-file",
        type=str,
        default="data/train.jsonl",
        help="Path to training JSONL file (default: data/train.jsonl)"
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        help="List of 0-based indices to evaluate (e.g., --indices 12 52 57 71)"
    )
    parser.add_argument(
        "--indices-file",
        type=str,
        help="JSON file containing list of indices (e.g., sampled_indices.json)"
    )
    parser.add_argument(
        "--evaluator-models",
        type=str,
        nargs="+",
        default=["gpt-5", "claude-sonnet-4-20250514", "gemini-2.5-pro"],
        help="Models to use for evaluation (default: gpt-5 claude-sonnet-4-20250514 gemini-2.5-pro)"
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=60,
        help="Seconds between batch status checks (default: 60)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=5.0,
        help="Seconds to wait between processing inputs (default: 5.0)"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to file"
    )

    args = parser.parse_args()

    # Load indices
    if args.indices:
        indices = args.indices
    elif args.indices_file:
        with open(args.indices_file, 'r') as f:
            data = json.load(f)
            if "sampled_indices" in data:
                indices = data["sampled_indices"]
            else:
                indices = data
    else:
        print("ERROR: Must specify either --indices or --indices-file", file=sys.stderr)
        sys.exit(1)

    train_file = Path(args.train_file)
    if not train_file.exists():
        print(f"ERROR: Training file not found: {train_file}", file=sys.stderr)
        sys.exit(1)

    results = run_batch_benchmark(
        model=args.model,
        train_file=train_file,
        indices=indices,
        evaluator_models=args.evaluator_models,
        poll_interval=args.poll_interval,
        delay_between_inputs=args.delay,
        save_results=not args.no_save
    )

    print(f"\n✅ Batch benchmark complete!")
    print(f"   Mean score: {results['summary_statistics']['mean_score']:.2f}/100")


if __name__ == "__main__":
    main()
