"""
Resume the test batch job and save results to JSON files.
"""

import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evals.batch_evals import resume_batch_evaluation
from evals.batch_evals.metrics_batch import parse_batch_results
from evals.metrics import calculate_summary_score

# The batch job that's currently running
BATCH_ID = "batch_68fadd72d3ec8190a522a745ec714f08"


def main():
    print("=" * 70)
    print(f"RESUMING BATCH JOB: {BATCH_ID}")
    print("=" * 70)

    # Resume and wait for results
    results = resume_batch_evaluation(
        batch_id=BATCH_ID,
        poll_interval=60
    )

    print(f"\n✅ Received {len(results)} results")

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


if __name__ == "__main__":
    main()
