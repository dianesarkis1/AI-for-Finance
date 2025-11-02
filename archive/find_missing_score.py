#!/usr/bin/env python3
"""
Find which index/evaluator combination is missing quality_structure score.
"""

import json
from pathlib import Path

def find_missing_quality_structure():
    """Find missing quality_structure scores in final results."""

    results_file = Path("results_benchmark_3/final_comprehensive_eval_results.json")

    with open(results_file, 'r') as f:
        data = json.load(f)

    evaluators = ["gpt-5", "claude-sonnet-4-20250514", "gemini-2.5-pro"]

    print("Checking for missing quality_structure scores...\n")

    missing_count = 0

    # Get the results_by_index section
    results_by_index = data.get("results_by_index", {})

    for index_key, index_data in results_by_index.items():
        index = int(index_key)

        for evaluator in evaluators:
            if evaluator not in index_data:
                print(f"⚠️  Index {index}: {evaluator} - ENTIRE EVALUATOR MISSING")
                missing_count += 1
                continue

            evaluator_data = index_data[evaluator]

            if "quality_structure" not in evaluator_data:
                print(f"❌ Index {index}: {evaluator} - quality_structure MISSING")
                missing_count += 1
            elif evaluator_data["quality_structure"].get("score") is None:
                print(f"⚠️  Index {index}: {evaluator} - quality_structure has no score")
                missing_count += 1

    print(f"\n{'='*60}")
    print(f"Total missing quality_structure scores: {missing_count}")
    print(f"Expected: 150 (50 indices × 3 evaluators)")
    print(f"Found: {150 - missing_count}")

    if missing_count == 0:
        print("\n✅ All quality_structure scores are present!")
    elif missing_count == 1:
        print("\n✅ Found the 1 missing score!")
    else:
        print(f"\n⚠️  Found {missing_count} missing scores (expected 1)")

if __name__ == "__main__":
    find_missing_quality_structure()
