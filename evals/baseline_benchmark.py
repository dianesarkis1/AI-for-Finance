#!/usr/bin/env python3
"""
Baseline Benchmark Script

Runs the baseline model (Claude Sonnet 4) on the entire training set
and computes summary statistics including:
- Mean summary score (primary benchmark)
- Worst score across all inputs (worst input, not worst run)
- Best score
- Standard deviation
- Distribution statistics

Usage:
    python baseline_benchmark.py --model claude-sonnet-4-20250514 --train-file data/train.jsonl
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

from evals.evaluator import evaluate_memo


def extract_credit_agreement_text(jsonl_line: str) -> tuple[str, str]:
    """
    Extract credit agreement text and source URL from JSONL line.

    Args:
        jsonl_line: JSON string from train.jsonl

    Returns:
        Tuple of (source_url, text)
    """
    data = json.loads(jsonl_line)
    return data['source_url'], data['text']


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


def run_baseline_benchmark(
    model: str,
    train_file: Path,
    delay_between_runs: float = 35.0,
    save_results: bool = True,
    resume_from: Optional[int] = None
) -> Dict:
    """
    Run baseline model on entire training set and compute benchmark statistics.

    Args:
        model: Model to use for baseline (e.g., 'claude-sonnet-4-20250514')
        train_file: Path to train.jsonl
        delay_between_runs: Seconds to wait between API calls (default: 35.0)
        save_results: Whether to save detailed results to JSON file (default: True)
        resume_from: Resume from a specific line number (0-indexed)

    Returns:
        Dict with benchmark statistics:
            - mean_score: Average summary score across all inputs
            - worst_score: Minimum score (worst input)
            - best_score: Maximum score
            - std_dev: Standard deviation
            - median_score: Median score
            - all_scores: List of all scores with metadata
    """
    print(f"\n{'='*70}")
    print(f"BASELINE BENCHMARK: {model}")
    print(f"Training set: {train_file}")
    print(f"{'='*70}\n")

    # Load training data
    with open(train_file, 'r', encoding='utf-8') as f:
        train_lines = f.readlines()

    total_inputs = len(train_lines)
    print(f"Total training inputs: {total_inputs}\n")

    # Results storage
    all_results = []
    start_idx = resume_from if resume_from is not None else 0

    # Create temp input file for model_run.py
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as tmp_in:
        temp_input_file = Path(tmp_in.name)

    try:
        for idx in range(start_idx, total_inputs):
            line = train_lines[idx].strip()
            if not line:
                continue

            print(f"[{idx+1}/{total_inputs}] Processing input...")

            # Extract credit agreement
            try:
                source_url, credit_agreement_text = extract_credit_agreement_text(line)
                print(f"  Source: {source_url[:80]}...")
            except Exception as e:
                print(f"  ❌ Failed to parse line {idx}: {e}")
                all_results.append({
                    "input_index": idx,
                    "source_url": None,
                    "score": None,
                    "error": str(e)
                })
                continue

            # Generate memo
            print(f"  Generating memo...")
            memo = generate_memo_for_input(model, credit_agreement_text, temp_input_file)

            if memo is None:
                all_results.append({
                    "input_index": idx,
                    "source_url": source_url,
                    "score": None,
                    "error": "Failed to generate memo"
                })
                continue

            # Evaluate memo
            print(f"  Evaluating memo...")
            try:
                score = evaluate_memo(
                    memo=memo,
                    source_document=credit_agreement_text,
                    template=None,  # No template for now
                    eval_models=None,  # Use default models
                    weights=None  # Use default weights
                )

                print(f"  ✅ Score: {score:.2f}/100\n")

                all_results.append({
                    "input_index": idx,
                    "source_url": source_url,
                    "score": score,
                    "error": None
                })

            except Exception as e:
                print(f"  ❌ Evaluation failed: {e}\n")
                all_results.append({
                    "input_index": idx,
                    "source_url": source_url,
                    "score": None,
                    "error": f"Evaluation failed: {str(e)}"
                })
                continue

            # Wait between runs (except on last iteration)
            if idx < total_inputs - 1:
                print(f"  ⏳ Waiting {delay_between_runs}s for API quota reset...\n")
                time.sleep(delay_between_runs)

            # Save intermediate results every 10 inputs
            if save_results and (idx + 1) % 10 == 0:
                intermediate_file = f"baseline_results_intermediate_{idx+1}.json"
                with open(intermediate_file, 'w') as f:
                    json.dump(all_results, f, indent=2)
                print(f"  💾 Saved intermediate results to {intermediate_file}\n")

    finally:
        # Clean up temp input file
        try:
            temp_input_file.unlink()
        except:
            pass

    # Calculate statistics
    valid_scores = [r['score'] for r in all_results if r['score'] is not None]

    if not valid_scores:
        print("❌ No valid scores obtained!")
        return {
            "mean_score": 0.0,
            "worst_score": 0.0,
            "best_score": 0.0,
            "std_dev": 0.0,
            "median_score": 0.0,
            "total_inputs": total_inputs,
            "successful_evals": 0,
            "failed_evals": total_inputs,
            "all_results": all_results
        }

    mean_score = statistics.mean(valid_scores)
    worst_score = min(valid_scores)
    best_score = max(valid_scores)
    std_dev = statistics.stdev(valid_scores) if len(valid_scores) > 1 else 0.0
    median_score = statistics.median(valid_scores)

    # Print results
    print(f"\n{'='*70}")
    print(f"BASELINE BENCHMARK RESULTS")
    print(f"{'='*70}")
    print(f"Total inputs:       {total_inputs}")
    print(f"Successful evals:   {len(valid_scores)}")
    print(f"Failed evals:       {total_inputs - len(valid_scores)}")
    print(f"")
    print(f"Mean Score:         {mean_score:.2f}/100  ← PRIMARY BENCHMARK")
    print(f"Median Score:       {median_score:.2f}/100")
    print(f"Worst Score:        {worst_score:.2f}/100  (worst input)")
    print(f"Best Score:         {best_score:.2f}/100")
    print(f"Std Dev:            {std_dev:.2f}")
    print(f"Score Range:        {best_score - worst_score:.2f}")
    print(f"{'='*70}\n")

    results = {
        "model": model,
        "mean_score": mean_score,
        "median_score": median_score,
        "worst_score": worst_score,
        "best_score": best_score,
        "std_dev": std_dev,
        "score_range": best_score - worst_score,
        "total_inputs": total_inputs,
        "successful_evals": len(valid_scores),
        "failed_evals": total_inputs - len(valid_scores),
        "all_results": all_results
    }

    # Save final results
    if save_results:
        output_file = f"baseline_benchmark_results_{model.replace('/', '_')}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Saved complete results to {output_file}\n")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run baseline benchmark on training set"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-sonnet-4-20250514",
        help="Model to use for baseline (default: claude-sonnet-4-20250514)"
    )
    parser.add_argument(
        "--train-file",
        type=str,
        default="data/train.jsonl",
        help="Path to training JSONL file (default: data/train.jsonl)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=35.0,
        help="Seconds to wait between API calls (default: 35.0)"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to file"
    )
    parser.add_argument(
        "--resume-from",
        type=int,
        help="Resume from a specific line number (0-indexed)"
    )

    args = parser.parse_args()

    train_file = Path(args.train_file)
    if not train_file.exists():
        print(f"ERROR: Training file not found: {train_file}", file=sys.stderr)
        sys.exit(1)

    results = run_baseline_benchmark(
        model=args.model,
        train_file=train_file,
        delay_between_runs=args.delay,
        save_results=not args.no_save,
        resume_from=args.resume_from
    )

    print(f"\n✅ Baseline benchmark complete!")
    print(f"   Mean score: {results['mean_score']:.2f}/100")


if __name__ == "__main__":
    main()
