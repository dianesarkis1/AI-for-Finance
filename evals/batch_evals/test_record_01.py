"""
Test batch evaluator on record_01 (GPT-5 memo) to compare with previous results.

Outputs results to JSON files matching the format of test_accuracy_results.json
and test_quality_results.json for easy comparison.
"""

import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evals.batch_evals.evaluator_batch import BATCH_TEMP_DIR
from evals.batch_evals.batch_utils import submit_and_wait_for_batch
from evals.batch_evals.metrics_batch import create_batch_requests_for_memo, parse_batch_results
from evals.metrics import calculate_summary_score
import os


def main():
    print("=" * 70)
    print("BATCH EVALUATION TEST: record_01_gpt_5_memo.md")
    print("=" * 70)

    # Load source document
    source_file = Path(__file__).parent.parent.parent / "data" / "exploratory_outputs" / "record_01_httpswwwsecgovArchivesedgardata1838126000119312524.jsonl"
    memo_file = Path(__file__).parent.parent.parent / "data" / "exploratory_outputs" / "record_01_gpt_5_memo.md"

    print(f"Source: {source_file}")
    print(f"Memo: {memo_file}")

    # Load source document
    with open(source_file, "r") as f:
        data = json.loads(f.readline())
        source_document = data["text"]

    # Load memo
    with open(memo_file, "r") as f:
        memo = f.read()

    print(f"\nSource document: {len(source_document)} chars")
    print(f"Memo: {len(memo)} chars")
    print("=" * 70)

    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment")

    # Create batch requests
    print("\n📦 Creating batch requests...")
    requests = create_batch_requests_for_memo(
        memo=memo,
        source_document=source_document,
        template=None,
        model="gpt-5"
    )
    print(f"   Created {len(requests)} batch requests")

    # Submit and wait
    print("\n🚀 Submitting batch job...")
    results = submit_and_wait_for_batch(
        requests=requests,
        api_key=api_key,
        temp_dir=BATCH_TEMP_DIR,
        description=f"record_01 test evaluation",
        poll_interval=60
    )

    # Parse results
    print("\n📊 Parsing results...")
    parsed = parse_batch_results(results)

    accuracy_result = parsed["accuracy_result"]
    completeness_result = parsed["completeness_result"]
    consistency_result = parsed["consistency_result"]
    quality_result = parsed["quality_result"]

    # Calculate summary score
    summary_result = calculate_summary_score(
        accuracy_result=accuracy_result,
        completeness_result=completeness_result,
        consistency_result=consistency_result,
        quality_result=quality_result
    )

    # Save results to JSON files
    output_dir = Path(__file__).parent / "test_outputs"
    output_dir.mkdir(exist_ok=True)

    accuracy_output = output_dir / "batch_accuracy_results.json"
    completeness_output = output_dir / "batch_completeness_results.json"
    consistency_output = output_dir / "batch_consistency_results.json"
    quality_output = output_dir / "batch_quality_results.json"
    summary_output = output_dir / "batch_summary_results.json"

    with open(accuracy_output, "w") as f:
        json.dump(accuracy_result, f, indent=2)

    with open(completeness_output, "w") as f:
        json.dump(completeness_result, f, indent=2)

    with open(consistency_output, "w") as f:
        json.dump(consistency_result, f, indent=2)

    with open(quality_output, "w") as f:
        json.dump(quality_result, f, indent=2)

    with open(summary_output, "w") as f:
        json.dump(summary_result, f, indent=2)

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print("\n📄 ACCURACY:")
    print(json.dumps(accuracy_result, indent=2))

    print("\n📄 COMPLETENESS:")
    print(json.dumps(completeness_result, indent=2))

    print("\n📄 CONSISTENCY:")
    print(json.dumps(consistency_result, indent=2))

    print("\n📄 QUALITY:")
    print(json.dumps(quality_result, indent=2))

    print("\n📄 SUMMARY:")
    print(json.dumps(summary_result, indent=2))

    print("\n" + "=" * 70)
    print("✅ Results saved to:")
    print(f"   {accuracy_output}")
    print(f"   {completeness_output}")
    print(f"   {consistency_output}")
    print(f"   {quality_output}")
    print(f"   {summary_output}")
    print("=" * 70)

    print(f"\n{'=' * 70}")
    print(f"✅ FINAL SCORE: {summary_result['summary_score']:.2f}/100")
    print(f"{'=' * 70}\n")

    print("Compare these JSON files with your previous test results.")


if __name__ == "__main__":
    main()
