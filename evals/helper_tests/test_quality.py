#!/usr/bin/env python3
"""
Test script for quality of presentation metric.

NOTE: This is an interim test script used to verify the quality evaluation function
works properly. It runs on exploratory outputs as a sanity check before full evaluation.

Usage:
    python evals/helper_tests/test_quality.py
"""

import json
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add project root directory to path to import from evals
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evals.metrics import evaluate_quality


def load_memo(memo_path: str) -> str:
    """Load memo from file."""
    with open(memo_path, 'r', encoding='utf-8') as f:
        return f.read()


def main():
    # Test with first GPT-5 memo
    memo_path = "data/exploratory_outputs/record_01_gpt_5_memo.md"

    print("Loading memo...")
    memo = load_memo(memo_path)

    print(f"Memo length: {len(memo)} characters")

    # Define a simple template for structure evaluation
    template = """1. Executive Summary/Overview
2. Transaction/Company Details
3. Financial Terms
4. Investment Strengths/Highlights
5. Risks and Concerns
6. Recommendation/Conclusion"""

    # Check which API keys are available
    import os
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
    print("\nEvaluating quality of presentation with LLM consensus...")
    print(f"This will call {len(available_models)} model(s) for each of 4 dimensions (clarity, tone, length, structure)...")
    print(f"Total API calls: {len(available_models) * 4}\n")

    # Run evaluation with available models
    result = evaluate_quality(
        memo=memo,
        template=template,
        models=available_models,
        consensus_threshold=0.5  # Lower threshold for fewer models
    )

    print("=" * 60)
    print("QUALITY OF PRESENTATION EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nOverall Quality Score: {result['quality_score']:.2f}/100")
    print("\nDimension Scores:")
    print(f"  Clarity:    {result['clarity_score']:.2f}/100")
    print(f"  Tone:       {result['tone_score']:.2f}/100")
    print(f"  Length:     {result['length_score']:.2f}/100")
    print(f"  Structure:  {result['structure_score']:.2f}/100")

    print("\n" + "=" * 60)
    print("DETAILED VOTES BY MODEL")
    print("=" * 60)
    for model, scores in result['votes'].items():
        print(f"\n{model}:")
        print(f"  Clarity:    {scores['clarity']:.1f}" if scores['clarity'] is not None else "  Clarity:    ERROR")
        print(f"  Tone:       {scores['tone']:.1f}" if scores['tone'] is not None else "  Tone:       ERROR")
        print(f"  Length:     {scores['length']:.1f}" if scores['length'] is not None else "  Length:     ERROR")
        print(f"  Structure:  {scores['structure']:.1f}" if scores['structure'] is not None else "  Structure:  ERROR")

    print("\n" + "=" * 60)

    # Save results to JSON file in helper_tests directory
    output_dir = Path(__file__).parent  # Save in same directory as test script (helper_tests)

    output_file = output_dir / "test_quality_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)

    print(f"\n💾 Results saved to: {output_file}")


if __name__ == "__main__":
    main()