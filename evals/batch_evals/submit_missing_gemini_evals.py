#!/usr/bin/env python3
"""
Submit missing Gemini evaluations for the 50 memos.
Extracts memos from existing batch input files and submits to Gemini.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evals.batch_evals.batch_metrics import (
    create_gemini_batch_requests_for_memo,
)
from evals.batch_evals.batch_utils import (
    create_gemini_batch,
)

# Paths
BATCH_TEMP_DIR = Path(__file__).parent / "batch_temp"

# Dataset indices that were evaluated
EVALUATED_INDICES = [0, 1, 2, 6, 12, 16, 17, 19, 20, 48, 51, 52, 57, 58, 63, 71, 78,
                     108, 114, 119, 120, 122, 125, 128, 134, 140, 150, 152, 224, 226,
                     239, 268, 289, 297, 311, 312, 318, 327, 338, 343, 357, 370, 377,
                     378, 379, 390, 392, 427, 458, 469]


def extract_memo_from_batch_input(input_file: Path) -> Dict[str, str]:
    """Extract memo and source document from batch input file."""
    with open(input_file, 'r') as f:
        # Read first line (any evaluation request contains the full memo)
        first_line = f.readline()
        data = json.loads(first_line)

        # Extract from the prompt content
        content = data['body']['messages'][0]['content']

        # Parse out SOURCE DOCUMENT and GENERATED MEMO
        parts = content.split('SOURCE DOCUMENT:')
        if len(parts) < 2:
            return None

        after_source = parts[1]

        # Split on GENERATED MEMO
        if 'GENERATED MEMO:' in after_source:
            source_and_memo = after_source.split('GENERATED MEMO:')
            source_document = source_and_memo[0].strip()

            # Get memo (before the evaluation question)
            memo_section = source_and_memo[1]

            # Find where the evaluation question starts
            # (usually starts with "Does the memo contain..." or similar)
            if '\n\nDoes ' in memo_section:
                memo = memo_section.split('\n\nDoes ')[0].strip()
            elif '\n\nYou are evaluating' in memo_section:
                memo = memo_section.split('\n\nYou are evaluating')[0].strip()
            else:
                # Just take everything as memo
                memo = memo_section.strip()

            return {
                'source_document': source_document,
                'memo': memo
            }

    return None


def submit_gemini_evaluations():
    """Submit Gemini batch evaluations for all 50 memos."""
    print("="*70)
    print("SUBMITTING MISSING GEMINI EVALUATIONS")
    print("="*70)
    print(f"Total memos to evaluate: {len(EVALUATED_INDICES)}")
    print("="*70 + "\n")

    # Get API key
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        print("ERROR: GEMINI_API_KEY not found in environment")
        sys.exit(1)

    # Get all batch input files (sorted by timestamp to match order)
    input_files = sorted(BATCH_TEMP_DIR.glob("batch_input_*.jsonl"))

    print(f"Found {len(input_files)} input files\n")

    if len(input_files) < len(EVALUATED_INDICES):
        print(f"WARNING: Only {len(input_files)} input files but {len(EVALUATED_INDICES)} indices")

    submitted_jobs = []
    failed_jobs = []

    for i, input_file in enumerate(input_files[:len(EVALUATED_INDICES)]):
        idx = EVALUATED_INDICES[i]

        print(f"[{i+1}/{len(EVALUATED_INDICES)}] Processing input {idx}...")

        try:
            # Extract memo and source document
            data = extract_memo_from_batch_input(input_file)

            if not data:
                print(f"  ❌ Failed to extract memo from {input_file.name}")
                failed_jobs.append(idx)
                continue

            print(f"  📄 Extracted memo: {len(data['memo'])} chars")

            # Create Gemini batch requests
            requests = create_gemini_batch_requests_for_memo(
                memo=data['memo'],
                source_document=data['source_document'],
                template=None,
                model="gemini-2.5-pro"
            )

            # Submit batch
            batch_id = create_gemini_batch(requests, gemini_key, "gemini-2.5-pro")

            submitted_jobs.append({
                "input_index": idx,
                "batch_id": batch_id
            })

            print(f"  ✅ Submitted Gemini batch: {batch_id}\n")

        except Exception as e:
            print(f"  ❌ Error: {e}\n")
            failed_jobs.append(idx)

    # Summary
    print("="*70)
    print("SUBMISSION COMPLETE")
    print("="*70)
    print(f"Successfully submitted: {len(submitted_jobs)}/{len(EVALUATED_INDICES)}")
    print(f"Failed: {len(failed_jobs)}/{len(EVALUATED_INDICES)}")

    if failed_jobs:
        print(f"\nFailed indices: {failed_jobs}")

    # Save job tracking info
    tracking_file = BATCH_TEMP_DIR.parent / "gemini_batch_jobs.json"
    with open(tracking_file, 'w') as f:
        json.dump({
            "submitted_jobs": submitted_jobs,
            "failed_jobs": failed_jobs,
            "total_submitted": len(submitted_jobs),
            "total_failed": len(failed_jobs)
        }, f, indent=2)

    print(f"\nJob tracking saved to: {tracking_file}")

    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("Gemini batch jobs typically take 1-24 hours to complete.")
    print("Check status with: ls -lh evals/batch_evals/batch_temp/gemini_batch_*.json")
    print("Once complete, re-run the aggregation script to get final results.")
    print("="*70)


if __name__ == "__main__":
    submit_gemini_evaluations()
