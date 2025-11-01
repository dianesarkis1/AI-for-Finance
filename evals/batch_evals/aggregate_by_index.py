#!/usr/bin/env python3
"""
Aggregate batch evaluation results properly mapped to dataset indices.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict
import statistics
import re

# Paths
BATCH_TEMP_DIR = Path(__file__).parent / "batch_temp"
RESULTS_DIR = Path(__file__).parent

# Dataset indices that were evaluated (from console output)
EVALUATED_INDICES = [0, 1, 2, 6, 12, 16, 17, 19, 20, 48, 51, 52, 57, 58, 63, 71, 78, 108, 114, 119, 120, 122, 125, 128, 134, 140, 150, 152, 224, 226, 239, 268, 289, 297, 311, 312, 318, 327, 338, 343, 357, 370, 377, 378, 379, 390, 392, 427, 458, 469]


def extract_score_from_content(content: str, metric: str) -> Any:
    """Extract score or result from evaluation content."""
    if metric in ["quality_clarity", "quality_tone", "quality_length", "quality_structure"]:
        # Extract numeric score - prioritize finding "SCORE: XX" pattern first
        score_match = re.search(r'SCORE:\s*(\d+)', content, re.IGNORECASE)

        # If no "SCORE:" found, look for last number in text (more likely to be the actual score)
        if not score_match:
            score_match = re.search(r'(\d+)(?!.*\d)', content)

        if score_match:
            try:
                return {"score": int(score_match.group(1))}
            except:
                pass
        return {"raw": content}

    elif metric == "accuracy":
        # Parse accuracy
        answer_match = re.search(r'ANSWER:\s*(YES|NO)', content, re.IGNORECASE)
        answer = answer_match.group(1).upper() if answer_match else None

        hall_match = re.search(r'HALLUCINATIONS:\s*(.+?)(?:\n|$)', content, re.DOTALL)
        hallucinations = hall_match.group(1).strip() if hall_match else None

        return {
            "answer": answer,
            "hallucinations": hallucinations,
            "has_hallucinations": answer == "YES" if answer else None
        }

    elif metric == "completeness":
        # Parse completeness
        answer_match = re.search(r'ANSWER:\s*(YES|NO)', content, re.IGNORECASE)
        answer = answer_match.group(1).upper() if answer_match else None

        missing_match = re.search(r'MISSING_TERMS:\s*(.+?)(?:\n\n|$)', content, re.DOTALL)
        missing = missing_match.group(1).strip() if missing_match else None

        return {
            "answer": answer,
            "missing_terms": missing,
            "is_incomplete": answer == "YES" if answer else None
        }

    elif metric == "consistency":
        # Try to parse JSON
        try:
            json_match = re.search(r'\{[^}]+\}', content, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
                return parsed
        except:
            pass
        return {"raw": content}

    return {"raw": content}


def parse_openai_output(file_path: Path) -> Dict[str, Any]:
    """Parse OpenAI batch output file and return metrics."""
    metrics = {}
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                result = json.loads(line)
                custom_id = result.get("custom_id", "")
                response_body = result.get("response", {}).get("body", {})

                if response_body.get("choices"):
                    content = response_body["choices"][0]["message"]["content"]
                    parsed = extract_score_from_content(content, custom_id)
                    parsed["metric"] = custom_id
                    metrics[custom_id] = parsed
    return metrics


def parse_claude_output(file_path: Path) -> Dict[str, Any]:
    """Parse Claude batch output file and return metrics."""
    metrics = {}
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                result = json.loads(line)
                custom_id = result.get("custom_id", "")

                if result.get("result", {}).get("type") == "succeeded":
                    content_blocks = result["result"]["message"]["content"]
                    if content_blocks:
                        content = content_blocks[0]["text"]
                        parsed = extract_score_from_content(content, custom_id)
                        parsed["metric"] = custom_id
                        metrics[custom_id] = parsed
    return metrics


def aggregate_results():
    """Aggregate all available batch results mapped to dataset indices."""
    print("="*70)
    print("AGGREGATING BATCH RESULTS BY DATASET INDEX")
    print("="*70)

    # Get all output files
    openai_outputs = sorted(BATCH_TEMP_DIR.glob("batch_output_*.jsonl"))
    claude_outputs = sorted(BATCH_TEMP_DIR.glob("claude_batch_output_*.jsonl"))

    print(f"\nFound {len(openai_outputs)} OpenAI output files")
    print(f"Found {len(claude_outputs)} Claude output files")

    # Map each output file to dataset index (in order of creation)
    results_by_index = {}

    # Process OpenAI outputs (one per memo, in order)
    for i, output_file in enumerate(openai_outputs):
        if i < len(EVALUATED_INDICES):
            idx = EVALUATED_INDICES[i]
            try:
                metrics = parse_openai_output(output_file)
                if idx not in results_by_index:
                    results_by_index[idx] = {}
                results_by_index[idx]["gpt-5"] = metrics
            except Exception as e:
                print(f"  Error processing {output_file.name}: {e}")

    # Process Claude outputs (one per memo, in order)
    for i, output_file in enumerate(claude_outputs):
        if i < len(EVALUATED_INDICES):
            idx = EVALUATED_INDICES[i]
            try:
                metrics = parse_claude_output(output_file)
                if idx not in results_by_index:
                    results_by_index[idx] = {}
                results_by_index[idx]["claude-sonnet-4-20250514"] = metrics
            except Exception as e:
                print(f"  Error processing {output_file.name}: {e}")

    print(f"\nTotal unique dataset indices evaluated: {len(results_by_index)}")
    print(f"Dataset indices: {sorted(results_by_index.keys())}")

    # Calculate aggregate statistics
    all_quality_scores = []
    score_by_evaluator = defaultdict(list)
    score_by_metric = defaultdict(list)

    for idx in sorted(results_by_index.keys()):
        evaluators = results_by_index[idx]
        for evaluator, metrics in evaluators.items():
            for metric, data in metrics.items():
                if "score" in data:
                    score = data["score"]
                    all_quality_scores.append(score)
                    score_by_evaluator[evaluator].append(score)
                    score_by_metric[metric].append(score)

    print(f"Total quality scores collected: {len(all_quality_scores)}")

    # Calculate summary statistics
    summary = {
        "total_memos_evaluated": len(results_by_index),
        "dataset_indices_evaluated": sorted(results_by_index.keys()),
        "total_quality_scores": len(all_quality_scores),
        "mean_score": round(statistics.mean(all_quality_scores), 2) if all_quality_scores else None,
        "median_score": round(statistics.median(all_quality_scores), 1) if all_quality_scores else None,
        "min_score": min(all_quality_scores) if all_quality_scores else None,
        "max_score": max(all_quality_scores) if all_quality_scores else None,
        "stdev_score": round(statistics.stdev(all_quality_scores), 2) if len(all_quality_scores) > 1 else None,
        "evaluators": {
            evaluator: {
                "count": len(scores),
                "mean": round(statistics.mean(scores), 2) if scores else None,
                "median": round(statistics.median(scores), 1) if scores else None
            }
            for evaluator, scores in score_by_evaluator.items()
        },
        "metrics": {
            metric: {
                "count": len(scores),
                "mean": round(statistics.mean(scores), 2) if scores else None,
                "median": round(statistics.median(scores), 1) if scores else None
            }
            for metric, scores in score_by_metric.items()
        }
    }

    # Prepare output
    output_data = {
        "summary": summary,
        "results_by_index": results_by_index,
        "metadata": {
            "note": "Results from 50 memos evaluated by GPT-5 and Claude Sonnet 4",
            "openai_files_processed": len(openai_outputs),
            "claude_files_processed": len(claude_outputs),
            "gemini_evaluations": "Not completed - 50 missing",
            "total_evaluations": f"{len(results_by_index)} memos × 2 evaluators = {len(results_by_index) * 2}"
        }
    }

    # Save results
    output_path = RESULTS_DIR / "comprehensive_eval_results_fixed.json"
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n{'='*70}")
    print(f"RESULTS SAVED TO: {output_path}")
    print(f"{'='*70}")
    print(f"\nSummary Statistics:")
    print(f"  Total Memos Evaluated: {summary['total_memos_evaluated']}")
    print(f"  Mean Score: {summary['mean_score']:.2f}" if summary['mean_score'] else "  Mean Score: N/A")
    print(f"  Median Score: {summary['median_score']:.1f}" if summary['median_score'] else "  Median Score: N/A")
    print(f"  Score Range: {summary['min_score']}-{summary['max_score']}" if summary['min_score'] else "  Score Range: N/A")
    print(f"  Std Dev: {summary['stdev_score']:.2f}" if summary['stdev_score'] else "  Std Dev: N/A")

    print(f"\n  By Evaluator:")
    for evaluator, stats in summary['evaluators'].items():
        print(f"    {evaluator}: mean={stats['mean']:.2f}, n={stats['count']}")

    print(f"\n  By Metric:")
    for metric, stats in summary['metrics'].items():
        print(f"    {metric}: mean={stats['mean']:.2f}, n={stats['count']}")

    return output_data


if __name__ == "__main__":
    results = aggregate_results()
