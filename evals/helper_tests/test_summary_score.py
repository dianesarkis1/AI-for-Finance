#!/usr/bin/env python3
"""
Test script for summary score aggregation.

NOTE: This is an interim test script used to verify the summary score calculation works properly.
It runs all 4 metrics on a memo and aggregates them into a single score.

Usage:
    python evals/helper_tests/test_summary_score.py
"""

import json
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add project root directory to path to import from evals
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evals.metrics import (
    evaluate_accuracy,
    evaluate_completeness,
    evaluate_consistency,
    evaluate_quality,
    calculate_summary_score
)


def load_memo(memo_path: str) -> str:
    """Load memo from file."""
    with open(memo_path, 'r', encoding='utf-8') as f:
        return f.read()


def load_source_document_from_jsonl(jsonl_path: str, record_index: int = 0) -> str:
    """Load source document from JSONL file."""
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == record_index:
                data = json.loads(line)
                return data.get('text', '')
    raise ValueError(f"Record {record_index} not found in {jsonl_path}")


def main():
    # Test with first GPT-5 memo
    memo_path = "data/exploratory_outputs/record_01_gpt_5_memo.md"
    source_jsonl = "data/exploratory_outputs/record_01_httpswwwsecgovArchivesedgardata1838126000119312524.jsonl"
    record_index = 0

    print("Loading memo and source document...")
    memo = load_memo(memo_path)
    source_doc = load_source_document_from_jsonl(source_jsonl, record_index)

    print(f"Memo length: {len(memo)} characters")
    print(f"Source doc length: {len(source_doc)} characters")

    # Define template for quality evaluation
    template = """1. Executive Summary/Overview
2. Transaction/Company Details
3. Financial Terms
4. Investment Strengths/Highlights
5. Risks and Concerns
6. Recommendation/Conclusion"""

    # Check which API keys are available
    import os
    import time
    available_models = []
    if os.getenv("OPENAI_API_KEY"):
        available_models.append("gpt-5")
    if os.getenv("ANTHROPIC_API_KEY"):
        available_models.append("claude-sonnet-4-20250514")
    if os.getenv("GEMINI_API_KEY"):
        available_models.append("gemini-2.5-pro")

    if not available_models:
        print("\n❌ No API keys found! Please set at least one API key.")
        return

    print(f"\n✅ Available models: {', '.join(available_models)}")
    print("\n" + "=" * 60)
    print("RUNNING ALL METRICS")
    print("=" * 60)

    # Run all 4 metrics with delays between them to respect rate limits
    print("\n1. Evaluating accuracy...")
    accuracy_result = evaluate_accuracy(
        memo=memo,
        source_document=source_doc,
        models=available_models,
        consensus_threshold=0.5
    )
    print(f"   Accurate: {accuracy_result['accurate']}, Score: {accuracy_result['score']:.2%}")

    # Delay to respect rate limits (especially Gemini: 2 req/min)
    if len(available_models) >= 3:
        print("   Waiting 35 seconds to respect rate limits...")
        time.sleep(35)

    print("\n2. Evaluating completeness...")
    completeness_result = evaluate_completeness(
        memo=memo,
        source_document=source_doc,
        models=available_models,
        consensus_threshold=0.5
    )
    print(f"   Complete: {completeness_result['complete']}, Score: {completeness_result['score']:.2%}")

    # Delay to respect rate limits
    if len(available_models) >= 3:
        print("   Waiting 35 seconds to respect rate limits...")
        time.sleep(35)

    print("\n3. Evaluating consistency...")
    consistency_result = evaluate_consistency(
        memo=memo,
        models=available_models,
        consensus_threshold=0.5
    )
    print(f"   Consistent: {consistency_result['consistent']}, Score: {consistency_result['score']:.2%}")

    # Delay to respect rate limits
    if len(available_models) >= 3:
        print("   Waiting 35 seconds to respect rate limits...")
        time.sleep(35)

    print("\n4. Evaluating quality...")
    quality_result = evaluate_quality(
        memo=memo,
        template=template,
        models=available_models,
        consensus_threshold=0.5
    )
    print(f"   Quality Score: {quality_result['quality_score']:.2f}/100")
    print(f"      Clarity:    {quality_result['clarity_score']:.2f}/100")
    print(f"      Tone:       {quality_result['tone_score']:.2f}/100")
    print(f"      Length:     {quality_result['length_score']:.2f}/100")
    print(f"      Structure:  {quality_result['structure_score']:.2f}/100")

    # Calculate summary score
    print("\n" + "=" * 60)
    print("CALCULATING SUMMARY SCORE")
    print("=" * 60)

    summary_result = calculate_summary_score(
        accuracy_result=accuracy_result,
        completeness_result=completeness_result,
        consistency_result=consistency_result,
        quality_result=quality_result
    )

    print(f"\n🎯 OVERALL SUMMARY SCORE: {summary_result['summary_score']:.2f}/100")
    print("\nNormalized Component Scores (0-100):")
    for metric, score in summary_result['normalized_scores'].items():
        weight = summary_result['weights_used'][metric]
        print(f"  {metric.capitalize():15} {score:6.2f}/100  (weight: {weight:.2%})")

    if summary_result['missing_metrics']:
        print(f"\n⚠️  Missing metrics: {', '.join(summary_result['missing_metrics'])}")

    print("\n" + "=" * 60)

    # Save results to JSON file
    output_dir = Path(__file__).parent
    output_file = output_dir / "test_summary_score_results.json"

    full_results = {
        "accuracy": accuracy_result,
        "completeness": completeness_result,
        "consistency": consistency_result,
        "quality": quality_result,
        "summary": summary_result
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(full_results, f, indent=2)

    print(f"\n💾 Results saved to: {output_file}")


if __name__ == "__main__":
    main()
