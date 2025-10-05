#!/usr/bin/env python3
"""
Test script for completeness metric.

NOTE: This is an interim test script used to verify the completeness evaluation function
works properly. It runs on exploratory outputs as a sanity check before full evaluation.

Usage:
    python evals/helper_tests/test_completeness.py
"""

import json
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add parent directory to path to import from evals
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evals.metrics import evaluate_completeness


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
    record_index = 0  # First record

    print("Loading memo and source document...")
    memo = load_memo(memo_path)
    source_doc = load_source_document_from_jsonl(source_jsonl, record_index)

    print(f"Memo length: {len(memo)} characters")
    print(f"Source doc length: {len(source_doc)} characters")

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
    print("\nEvaluating completeness with LLM consensus...")
    print(f"This will call {len(available_models)} model(s)...\n")

    # Run evaluation with available models
    result = evaluate_completeness(
        memo=memo,
        source_document=source_doc,
        models=available_models,
        consensus_threshold=0.5  # Lower threshold for fewer models
    )

    print("=" * 60)
    print("COMPLETENESS EVALUATION RESULTS")
    print("=" * 60)
    print(f"Complete: {result['complete']}")
    print(f"Score: {result['score']:.2%}")
    print(f"Consensus reached: {result['consensus_reached']}")
    print(f"YES votes (missing terms detected): {result['yes_votes']}")
    print(f"NO votes (all terms present): {result['no_votes']}")
    print("\nVotes by model:")
    for model, vote_data in result['votes'].items():
        print(f"  {model}: {vote_data['vote']}")
        print(f"    Missing terms: {vote_data['missing_terms']}")
    print("=" * 60)

    # Save results to JSON file in helper_tests directory
    import json
    from pathlib import Path
    output_dir = Path(__file__).parent  # Save in same directory as test script (helper_tests)

    output_file = output_dir / "test_completeness_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)

    print(f"\n💾 Results saved to: {output_file}")


if __name__ == "__main__":
    main()
