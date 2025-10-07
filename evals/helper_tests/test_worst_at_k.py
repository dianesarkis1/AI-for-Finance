#!/usr/bin/env python3
"""
Test worst_at_k evaluation on record 01 with GPT-5. Using k=3 due to limited access to API.
"""

import json
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evals.evaluator import worst_at_k


def load_source_document(jsonl_path: str) -> str:
    """Load source document from JSONL file."""
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        line = f.readline().strip()
        data = json.loads(line)
        return data.get('text', '')


def main():
    # Paths
    input_file = "data/exploratory_outputs/record_01_httpswwwsecgovArchivesedgardata1838126000119312524.jsonl"

    # Load source document
    print("Loading source document...")
    source_document = load_source_document(input_file)
    print(f"Source document length: {len(source_document):,} characters\n")

    # Define template (from model_run.py)
    template = """1. Executive Summary: Provide a concise overview that includes:
    * Date of the agreement
    * Borrower / Company overview
    * Brief description of the transaction (type, structure, counterparties)
    * Purpose of the financing
    * Brief company background and context
2. Investment Highlights & Risks: Present clear, bullet-pointed analysis from the perspective of an investor:
    * Key strengths / credit positives
    * Principal risks and mitigating factors
3. Key Deal Terms Table: Include a well-formatted table listing:
    * Deal size
    * Deal price
    * Interest rate (and type, if applicable)
    * Maturity date
    * Payment frequency
    * Key covenants or financial maintenance terms"""

    # Run worst_at_k
    result = worst_at_k(
        model="gpt-5",
        input_file=input_file,
        source_document=source_document,
        k=3,
        template=template,
        delay_between_runs=35.0,
        fail_fast=True
    )

    # Print results
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()