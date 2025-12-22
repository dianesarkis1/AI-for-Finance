#!/usr/bin/env python3
"""
Compare the outputs from data_cleaning.py test run against the existing data files.
Verifies that the pipeline can accurately reproduce all data files.
"""

import json
from pathlib import Path
from collections import Counter

def load_jsonl(file_path):
    """Load JSONL file and return list of records."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

def compare_jsonl_files(original_path, test_path, file_name):
    """Compare two JSONL files and report differences."""
    print(f"\n{'='*70}")
    print(f"Comparing: {file_name}")
    print(f"{'='*70}")

    if not original_path.exists():
        print(f"❌ Original file missing: {original_path}")
        return False

    if not test_path.exists():
        print(f"❌ Test file missing: {test_path}")
        return False

    # Load files
    original = load_jsonl(original_path)
    test = load_jsonl(test_path)

    # Compare counts
    print(f"Original count: {len(original)}")
    print(f"Test count:     {len(test)}")

    if len(original) != len(test):
        print(f"❌ Count mismatch!")
        return False

    # Extract URLs for comparison
    original_urls = [record['source_url'] for record in original]
    test_urls = [record['source_url'] for record in test]

    # Check if URLs match (order matters for train/test splits)
    if original_urls != test_urls:
        print(f"❌ URL mismatch!")

        # Find differences
        original_set = set(original_urls)
        test_set = set(test_urls)

        only_in_original = original_set - test_set
        only_in_test = test_set - original_set

        if only_in_original:
            print(f"  URLs only in original: {len(only_in_original)}")
            for url in list(only_in_original)[:3]:
                print(f"    - {url}")

        if only_in_test:
            print(f"  URLs only in test: {len(only_in_test)}")
            for url in list(only_in_test)[:3]:
                print(f"    - {url}")

        # Check if it's just an order issue
        if original_set == test_set:
            print(f"  ⚠️  Same URLs but different order")
            # Find first index where they differ
            for i, (o_url, t_url) in enumerate(zip(original_urls, test_urls)):
                if o_url != t_url:
                    print(f"  First difference at index {i}:")
                    print(f"    Original: {o_url}")
                    print(f"    Test:     {t_url}")
                    break

        return False

    # Compare actual text content for a sample
    mismatches = 0
    for i, (orig_record, test_record) in enumerate(zip(original, test)):
        if orig_record['text'] != test_record['text']:
            mismatches += 1
            if mismatches == 1:
                print(f"❌ Text content mismatch at index {i}")
                print(f"  URL: {orig_record['source_url']}")
                print(f"  Original text length: {len(orig_record['text'])}")
                print(f"  Test text length: {len(test_record['text'])}")

    if mismatches > 0:
        print(f"❌ Total text mismatches: {mismatches}")
        return False

    print(f"✅ Files match perfectly!")
    return True

def main():
    print("\n" + "="*70)
    print("DATA PIPELINE VERIFICATION")
    print("="*70)
    print("\nComparing data_test/ outputs against data/ originals...\n")

    original_dir = Path("data")
    test_dir = Path("data_test")

    files_to_compare = [
        "train.jsonl",
        "test.jsonl",
        "train_final.jsonl"
    ]

    results = {}

    for file_name in files_to_compare:
        original_path = original_dir / file_name
        test_path = test_dir / file_name
        results[file_name] = compare_jsonl_files(original_path, test_path, file_name)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    all_passed = all(results.values())

    for file_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {file_name}")

    if all_passed:
        print(f"\n🎉 All files match! data_cleaning.py correctly reproduces all outputs.")
    else:
        print(f"\n❌ Some files don't match. Review the differences above.")

    print()

    return all_passed

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
