#!/usr/bin/env python3
"""
Aggregate existing batch evaluation results from completed jobs.
This script salvages results from the incomplete parallel batch run.
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


def parse_openai_output(file_path: Path) -> List[Dict]:
    """Parse OpenAI batch output file."""
    results = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                result = json.loads(line)
                results.append(result)
    return results


def parse_claude_output(file_path: Path) -> List[Dict]:
    """Parse Claude batch output file."""
    results = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                result = json.loads(line)
                results.append(result)
    return results


def extract_score_from_content(content: str, metric: str) -> Any:
    """Extract score or result from evaluation content."""
    if metric in ["quality_clarity", "quality_tone", "quality_table", "quality_structure"]:
        # Extract numeric score
        score_match = re.search(r'(?:SCORE:\s*)?(\d+)', content)
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
            # Look for JSON block
            json_match = re.search(r'\{[^}]+\}', content, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
                return parsed
        except:
            pass
        return {"raw": content}

    return {"raw": content}


def extract_scores_from_openai_result(result: Dict) -> Dict[str, Any]:
    """Extract evaluation scores from OpenAI batch result."""
    custom_id = result.get("custom_id", "")
    response_body = result.get("response", {}).get("body", {})

    if response_body.get("choices"):
        content = response_body["choices"][0]["message"]["content"]
        parsed = extract_score_from_content(content, custom_id)
        parsed["metric"] = custom_id
        return parsed

    return {"metric": custom_id, "error": "Could not parse"}


def extract_scores_from_claude_result(result: Dict) -> Dict[str, Any]:
    """Extract evaluation scores from Claude batch result."""
    custom_id = result.get("custom_id", "")

    if result.get("result", {}).get("type") == "succeeded":
        content_blocks = result["result"]["message"]["content"]
        if content_blocks:
            content = content_blocks[0]["text"]
            parsed = extract_score_from_content(content, custom_id)
            parsed["metric"] = custom_id
            return parsed

    return {"metric": custom_id, "error": "Could not parse"}


def aggregate_results():
    """Aggregate all available batch results."""
    print("="*70)
    print("AGGREGATING EXISTING BATCH RESULTS")
    print("="*70)

    # Organize by timestamp (proxy for input)
    evaluations_by_timestamp = defaultdict(lambda: {
        "gpt-5": {},
        "claude-sonnet-4-20250514": {}
    })

    # Process OpenAI outputs
    openai_outputs = sorted(BATCH_TEMP_DIR.glob("batch_output_*.jsonl"))
    print(f"\nFound {len(openai_outputs)} OpenAI output files")

    for output_file in openai_outputs:
        try:
            results = parse_openai_output(output_file)
            file_timestamp = output_file.stem.split("_")[-1]

            for result in results:
                scores = extract_scores_from_openai_result(result)
                metric = scores.get("metric")
                evaluations_by_timestamp[file_timestamp]["gpt-5"][metric] = scores
        except Exception as e:
            print(f"  Error processing {output_file.name}: {e}")

    # Process Claude outputs
    claude_outputs = sorted(BATCH_TEMP_DIR.glob("claude_batch_output_*.jsonl"))
    print(f"Found {len(claude_outputs)} Claude output files")

    for output_file in claude_outputs:
        try:
            results = parse_claude_output(output_file)
            file_timestamp = output_file.stem.split("_")[-1]

            for result in results:
                scores = extract_scores_from_claude_result(result)
                metric = scores.get("metric")
                evaluations_by_timestamp[file_timestamp]["claude-sonnet-4-20250514"][metric] = scores
        except Exception as e:
            print(f"  Error processing {output_file.name}: {e}")

    print(f"\nTotal unique evaluations: {len(evaluations_by_timestamp)}")

    # Calculate aggregate statistics
    all_quality_scores = []
    score_by_evaluator = defaultdict(list)
    score_by_metric = defaultdict(list)

    for timestamp, evaluators in evaluations_by_timestamp.items():
        for evaluator, metrics in evaluators.items():
            if not metrics:
                continue

            for metric, data in metrics.items():
                if "score" in data:
                    score = data["score"]
                    all_quality_scores.append(score)
                    score_by_evaluator[evaluator].append(score)
                    score_by_metric[metric].append(score)

    print(f"Total quality scores collected: {len(all_quality_scores)}")

    # Calculate summary statistics
    summary = {
        "total_evaluations": len(evaluations_by_timestamp),
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

    # Convert defaultdict to regular dict for JSON serialization
    evaluations_serializable = {
        ts: {
            evaluator: dict(metrics)
            for evaluator, metrics in evals.items()
        }
        for ts, evals in evaluations_by_timestamp.items()
    }

    # Prepare output
    output_data = {
        "summary": summary,
        "evaluations_by_timestamp": evaluations_serializable,
        "metadata": {
            "note": "Partial results aggregated from incomplete parallel batch run",
            "openai_files_processed": len(openai_outputs),
            "claude_files_processed": len(claude_outputs),
            "gemini_files_processed": 0,
            "missing_evaluations": "Gemini evaluations (51 memos) not completed"
        }
    }

    # Save results
    output_path = RESULTS_DIR / "comprehensive_eval_partial_results.json"
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n{'='*70}")
    print(f"RESULTS SAVED TO: {output_path}")
    print(f"{'='*70}")
    print(f"\nSummary Statistics:")
    print(f"  Total Evaluations: {summary['total_evaluations']} memos")
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
