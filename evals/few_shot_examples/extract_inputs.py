#!/usr/bin/env python3
"""
Helper script to extract input examples from train.jsonl for few-shot learning.
Extracts the credit agreement text for specified indices and saves them to the few-shot examples folder.

Usage:
    python extract_inputs.py
"""

import json
from pathlib import Path

# Configuration
TRAIN_FILE = Path(__file__).parent.parent.parent / "data" / "train.jsonl"
OUTPUT_DIR = Path(__file__).parent
INDICES_TO_EXTRACT = [103, 281, 446]


def extract_input_example(train_file: Path, index: int) -> dict:
    """
    Extract a single training sample by index.

    Args:
        train_file: Path to train.jsonl
        index: 0-based index of the sample to load

    Returns:
        Dict with source_url and text
    """
    with open(train_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == index:
                data = json.loads(line)
                return {
                    'source_url': data['source_url'],
                    'text': data['text']
                }

    raise ValueError(f"Index {index} not found in {train_file}")


def main():
    print(f"Extracting input examples from {TRAIN_FILE}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    for idx in INDICES_TO_EXTRACT:
        print(f"Extracting index {idx}...")
        data = extract_input_example(TRAIN_FILE, idx)

        # Save the input to a text file
        output_file = OUTPUT_DIR / f"input_{idx}.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Source URL: {data['source_url']}\n\n")
            f.write(data['text'])

        print(f"  Saved to {output_file}")
        print(f"  Text length: {len(data['text']):,} characters")
        print()

    print("Done!")


if __name__ == "__main__":
    main()
