#!/usr/bin/env python3
"""
Preview the sampling that will be used for comprehensive batch evaluation.
This doesn't run any evaluations - just shows what indices will be sampled.

NOTE ON TRAINING DATA SAMPLING:
    This script uses baseline_sampled_indices_seed42.json to generate a 50-index
    comprehensive sample (10 baseline + 3 first indices + 37 random). This produces
    IDENTICAL data to data/train_final.jsonl, which explicitly contains the same 50
    training samples. The sampling approach is maintained here for backward compatibility
    with existing run naming conventions and historical workflows.
"""

import json
import random
from pathlib import Path

# Configuration
BASELINE_SAMPLED_INDICES_FILE = Path("evals/benchmark/baseline_sampled_indices_seed42.json")
TRAIN_FILE = Path("data/train.jsonl")
RANDOM_SEED = 42

def load_baseline_sampled_indices(file_path: Path) -> list[int]:
    """Load the baseline sampled indices from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data['sampled_indices']

def count_train_samples(train_file: Path) -> int:
    """Count total number of samples in train.jsonl."""
    with open(train_file, 'r') as f:
        return sum(1 for _ in f)

def main():
    print("\n" + "="*70)
    print("SAMPLING PREVIEW")
    print("="*70 + "\n")

    # Load baseline indices
    baseline_indices = load_baseline_sampled_indices(BASELINE_SAMPLED_INDICES_FILE)
    print(f"1. Baseline sampled indices (from baseline_sampled_indices_seed42.json):")
    print(f"   Count: {len(baseline_indices)}")
    print(f"   Indices: {baseline_indices}\n")

    # First 3 indices
    first_indices = [0, 1, 2]
    print(f"2. First 3 indices from train.jsonl:")
    print(f"   Count: {len(first_indices)}")
    print(f"   Indices: {first_indices}\n")

    # Check for overlaps between baseline and first 3
    overlap_baseline_first = set(baseline_indices) & set(first_indices)
    print(f"   Note: Overlap between baseline and first 3: {overlap_baseline_first if overlap_baseline_first else 'None'}\n")

    # Random sample
    total_samples = count_train_samples(TRAIN_FILE)
    combined_so_far = set(baseline_indices) | set(first_indices)

    random.seed(RANDOM_SEED)
    available_indices = [i for i in range(total_samples) if i not in combined_so_far]
    random_indices = sorted(random.sample(available_indices, 37))

    print(f"3. Additional random sample (seed={RANDOM_SEED}):")
    print(f"   Count: {len(random_indices)}")
    print(f"   Indices: {random_indices}\n")

    # Final combined list
    all_indices = sorted(list(combined_so_far | set(random_indices)))

    print("="*70)
    print("FINAL COMBINED SAMPLE")
    print("="*70)
    print(f"Total unique indices: {len(all_indices)}")
    print(f"Total inputs in dataset: {total_samples}")
    print(f"\nBreakdown:")
    print(f"  - From baseline_sampled_indices_seed42: {len(baseline_indices)} indices")
    print(f"  - First 3 indices (0, 1, 2): {len(first_indices)} indices")
    print(f"  - Additional random sample: {len(random_indices)} indices")
    print(f"  - Overlaps removed: {len(baseline_indices) + len(first_indices) + len(random_indices) - len(all_indices)}")
    print(f"\nAll sampled indices (sorted):")
    print(f"{all_indices}")
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    main()
