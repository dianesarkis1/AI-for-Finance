#!/usr/bin/env python3
"""
Compile final comprehensive evaluation results from all 3 evaluators:
- GPT-5
- Claude Sonnet 4
- Gemini 2.5 Pro

This script loads the existing comprehensive_eval_results.json (GPT-5 + Claude)
and adds Gemini results, then recalculates all statistics across all evaluators.

Output: final_comprehensive_eval_results.json
"""

import json
import re
import statistics
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

# Paths
BATCH_TEMP_DIR = Path(__file__).parent / "batch_temp"
EXISTING_RESULTS = Path(__file__).parent / "comprehensive_eval_results_fixed.json"
OUTPUT_FILE = Path(__file__).parent / "final_comprehensive_eval_results2.json"


def parse_gemini_output_files() -> Dict[int, Dict]:
    """Parse all Gemini output JSONL files and organize by index."""
    gemini_results = defaultdict(dict)

    # Find all Gemini output files
    gemini_files = list(BATCH_TEMP_DIR.glob("gemini_batch_output_*.jsonl"))

    print(f"Found {len(gemini_files)} Gemini output files")

    for file_path in gemini_files:
        # Extract index from filename: gemini_batch_output_{index}_{timestamp}.jsonl
        filename = file_path.stem
        parts = filename.split('_')
        try:
            index = int(parts[3])  # gemini_batch_output_INDEX_timestamp
        except (IndexError, ValueError):
            print(f"  ⚠️  Could not parse index from {file_path.name}, skipping")
            continue

        # Parse JSONL file
        with open(file_path, 'r') as f:
            for line in f:
                result = json.loads(line)
                custom_id = result['custom_id']
                metric = custom_id.split('_')[0]  # e.g., "accuracy_0" -> "accuracy"

                content = result['response']['body']['choices'][0]['message']['content']

                # Parse based on metric type
                parsed = parse_gemini_content(content, metric)
                gemini_results[index][metric] = parsed

    return dict(gemini_results)


def parse_gemini_content(content: str, metric: str) -> Dict:
    """Parse Gemini evaluation content based on metric type."""

    result = {"metric": metric}

    if metric == "accuracy":
        # Extract ANSWER and HALLUCINATIONS
        answer_match = re.search(r'ANSWER:\s*(YES|NO)', content, re.IGNORECASE)
        result["answer"] = answer_match.group(1).upper() if answer_match else "UNKNOWN"

        hallucinations_match = re.search(r'HALLUCINATIONS:(.*)', content, re.DOTALL)
        result["hallucinations"] = hallucinations_match.group(1).strip() if hallucinations_match else ""
        result["has_hallucinations"] = result["answer"] == "YES"

    elif metric == "completeness":
        # Extract ANSWER and MISSING_TERMS
        answer_match = re.search(r'ANSWER:\s*(YES|NO)', content, re.IGNORECASE)
        result["answer"] = answer_match.group(1).upper() if answer_match else "UNKNOWN"

        missing_match = re.search(r'MISSING_TERMS:(.*)', content, re.DOTALL)
        result["missing_terms"] = missing_match.group(1).strip() if missing_match else ""
        result["is_incomplete"] = result["answer"] == "YES"

    elif metric == "consistency":
        # Extract JSON
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
        if json_match:
            consistency_data = json.loads(json_match.group(1))
            result["has_issues"] = consistency_data.get("has_issues", False)
            result["issues"] = consistency_data.get("issues", [])
        else:
            result["has_issues"] = False
            result["issues"] = []

    elif metric.startswith("quality_"):
        # Extract numeric score
        score_match = re.search(r'(?:SCORE:\s*)?(\d+)', content)
        if score_match:
            result["score"] = int(score_match.group(1))
        else:
            result["score"] = None

    return result


def calculate_statistics(scores: List[int]) -> Dict:
    """Calculate mean, median, min, max, stdev for a list of scores."""
    if not scores:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "min": 0,
            "max": 0,
            "stdev": 0.0
        }

    return {
        "count": len(scores),
        "mean": round(statistics.mean(scores), 2),
        "median": statistics.median(scores),
        "min": min(scores),
        "max": max(scores),
        "stdev": round(statistics.stdev(scores), 2) if len(scores) > 1 else 0.0
    }


def main():
    print(f"\n{'='*70}")
    print(f"COMPILING FINAL COMPREHENSIVE EVALUATION RESULTS")
    print(f"{'='*70}\n")

    # Load existing results (GPT-5 + Claude)
    print(f"Loading existing results from {EXISTING_RESULTS.name}...")
    with open(EXISTING_RESULTS, 'r') as f:
        results = json.load(f)

    existing_indices = results['summary']['dataset_indices_evaluated']
    print(f"  Found results for {len(existing_indices)} indices")
    print(f"  Evaluators: {', '.join(results['summary']['evaluators'].keys())}\n")

    # Parse Gemini results
    print("Parsing Gemini results...")
    gemini_results = parse_gemini_output_files()
    print(f"  Parsed Gemini results for {len(gemini_results)} indices\n")

    # Add Gemini results to each index
    print("Adding Gemini results to dataset...")
    gemini_added = 0

    for index_str, index_data in results['results_by_index'].items():
        index = int(index_str)

        if index in gemini_results:
            index_data['gemini-2.5-pro'] = gemini_results[index]
            gemini_added += 1

    print(f"  ✅ Added Gemini results for {gemini_added} indices\n")

    # Calculate and add summary scores for each memo and evaluator
    print("Calculating summary scores for each memo and evaluator...")
    quality_metrics = ['quality_clarity', 'quality_tone', 'quality_length', 'quality_structure']

    for index_str, index_data in results['results_by_index'].items():
        evaluator_summaries = []

        for evaluator, eval_results in index_data.items():
            if evaluator == 'summary_score':  # Skip if already exists
                continue

            # Calculate summary score for this evaluator (average of 4 quality metrics)
            quality_scores = []
            for metric in quality_metrics:
                if metric in eval_results and eval_results[metric].get('score') is not None:
                    quality_scores.append(eval_results[metric]['score'])

            if quality_scores:
                evaluator_summary = round(statistics.mean(quality_scores), 2)
                eval_results['summary_score'] = evaluator_summary
                evaluator_summaries.append(evaluator_summary)

        # Calculate overall summary score for this memo (average across evaluators)
        if evaluator_summaries:
            index_data['summary_score'] = round(statistics.mean(evaluator_summaries), 2)

    print(f"  ✅ Summary scores calculated\n")

    # Recalculate summary statistics across all 3 evaluators
    print("Recalculating summary statistics...")

    memo_level_scores = []  # Average score per memo+evaluator
    evaluator_memo_scores = defaultdict(list)  # Memo-level scores by evaluator
    metric_scores = defaultdict(list)  # Individual metric scores for by-metric stats

    quality_metrics = ['quality_clarity', 'quality_tone', 'quality_length', 'quality_structure']

    # Debug: track invalid scores
    invalid_scores = []

    for index_str, index_data in results['results_by_index'].items():
        for evaluator, eval_results in index_data.items():
            if evaluator == 'summary_score':  # Skip summary_score key
                continue

            # Calculate memo-level average (average of 4 quality metrics)
            memo_quality_scores = []

            for metric in quality_metrics:
                if metric in eval_results and eval_results[metric].get('score') is not None:
                    score = eval_results[metric]['score']

                    # Debug: check for invalid scores
                    if score > 100:
                        invalid_scores.append({
                            'index': index_str,
                            'evaluator': evaluator,
                            'metric': metric,
                            'score': score
                        })

                    memo_quality_scores.append(score)
                    metric_scores[metric].append(score)

            # Calculate average for this memo+evaluator
            if memo_quality_scores:
                memo_avg = statistics.mean(memo_quality_scores)
                memo_level_scores.append(memo_avg)
                evaluator_memo_scores[evaluator].append(memo_avg)

    # Report invalid scores if found
    if invalid_scores:
        print(f"\n  ⚠️  WARNING: Found {len(invalid_scores)} scores > 100:")
        for item in invalid_scores[:10]:  # Show first 10
            print(f"     Index {item['index']}, {item['evaluator']}, {item['metric']}: {item['score']}")
        if len(invalid_scores) > 10:
            print(f"     ... and {len(invalid_scores) - 10} more")
        print()

    # Update summary with memo-level statistics
    total_evaluations = len(memo_level_scores)  # Should be 50 memos × 3 evaluators = 150

    results['summary']['total_evaluations'] = total_evaluations
    results['summary']['total_quality_scores'] = sum(len(scores) for scores in metric_scores.values())
    results['summary']['mean_score'] = round(statistics.mean(memo_level_scores), 2)
    results['summary']['median_score'] = round(statistics.median(memo_level_scores), 2)
    results['summary']['min_score'] = round(min(memo_level_scores), 2)
    results['summary']['max_score'] = round(max(memo_level_scores), 2)
    results['summary']['stdev_score'] = round(statistics.stdev(memo_level_scores), 2) if len(memo_level_scores) > 1 else 0.0

    # Update evaluator statistics (memo-level averages)
    results['summary']['evaluators'] = {}
    for evaluator, scores in evaluator_memo_scores.items():
        results['summary']['evaluators'][evaluator] = {
            'count': len(scores),
            'mean': round(statistics.mean(scores), 2),
            'median': round(statistics.median(scores), 2)
        }

    # Update metric statistics (individual metric scores)
    results['summary']['metrics'] = {}
    for metric, scores in metric_scores.items():
        results['summary']['metrics'][metric] = {
            'count': len(scores),
            'mean': round(statistics.mean(scores), 2),
            'median': round(statistics.median(scores), 2)
        }

    print(f"  ✅ Statistics calculated\n")

    # Save to new file
    print(f"Saving final results to {OUTPUT_FILE.name}...")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"  ✅ Saved\n")

    # Print summary
    print(f"{'='*70}")
    print(f"FINAL RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"Total memos evaluated: {results['summary']['total_memos_evaluated']}")
    print(f"Total evaluations: {results['summary']['total_evaluations']}")
    print(f"Total quality scores: {results['summary']['total_quality_scores']}")
    print(f"\nOverall Quality:")
    print(f"  Mean:   {results['summary']['mean_score']}")
    print(f"  Median: {results['summary']['median_score']}")
    print(f"  Range:  {results['summary']['min_score']} - {results['summary']['max_score']}")
    print(f"  Stdev:  {results['summary']['stdev_score']}")
    print(f"\nBy Evaluator:")
    for evaluator, stats in results['summary']['evaluators'].items():
        print(f"  {evaluator}:")
        print(f"    Count:  {stats['count']}")
        print(f"    Mean:   {stats['mean']}")
        print(f"    Median: {stats['median']}")
    print(f"\nBy Metric:")
    for metric, stats in results['summary']['metrics'].items():
        print(f"  {metric}:")
        print(f"    Count:  {stats['count']}")
        print(f"    Mean:   {stats['mean']}")
        print(f"    Median: {stats['median']}")
    print()


if __name__ == "__main__":
    main()
