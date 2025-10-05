#!/usr/bin/env python3
"""
Test script for intra-memo consistency metric.

NOTE: This is an interim test script used to verify the consistency evaluation function
works properly. It runs on exploratory outputs as a sanity check before full evaluation.

Usage:
    python evals/helper_tests/test_consistency.py
"""

import json
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add project root directory to path to import from evals
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evals.metrics import evaluate_consistency


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
    print("\nEvaluating intra-memo consistency with LLM consensus...")
    print(f"This will call {len(available_models)} model(s)...\n")

    # Run evaluation with available models
    result = evaluate_consistency(
        memo=memo,
        models=available_models,
        consensus_threshold=0.5  # Lower threshold for fewer models
    )

    print("=" * 60)
    print("CONSISTENCY EVALUATION RESULTS")
    print("=" * 60)
    print(f"Consistent: {result['consistent']}")
    print(f"Score: {result['score']:.2%}")
    print(f"Consensus reached: {result['consensus_reached']}")
    print(f"Has issues votes: {result['has_issues_votes']}")
    print(f"No issues votes: {result['no_issues_votes']}")
    print("\nVotes by model:")
    for model, vote_data in result['votes'].items():
        print(f"  {model}:")
        print(f"    Has issues: {vote_data['has_issues']}")
        print(f"    Parse error: {vote_data.get('parse_error', False)}")
        if vote_data['issues']:
            print(f"    Issues found:")
            for issue in vote_data['issues']:
                print(f"      - {issue}")
        else:
            print(f"    Issues found: None")
    print("=" * 60)

    # Save results to JSON file in helper_tests directory
    output_dir = Path(__file__).parent  # Save in same directory as test script (helper_tests)

    output_file = output_dir / "test_consistency_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)

    print(f"\n💾 Results saved to: {output_file}")


if __name__ == "__main__":
    main()
