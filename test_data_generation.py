#!/usr/bin/env python3
"""
Test the data generation logic WITHOUT fetching from SEC.
Uses existing cleaned_data.jsonl to generate the split files.
"""

import json
import random
from pathlib import Path

def get_train_final_indices():
    """
    Return the exact 50 indices from cleaned_data.jsonl that comprise train_final.jsonl.

    Historical context: This 50-sample set evolved organically during development:
    - Started by manually reviewing first 3 indices (0, 1, 2)
    - Then sampled 10 baseline indices for initial evals
    - Later needed 37 more for a comprehensive 50-sample benchmark

    These indices were determined from the original train_final.jsonl file
    and correspond to positions in cleaned_data.jsonl (all 499 URLs).

    Returns:
        List of 50 indices from cleaned_data.jsonl
    """
    return [0, 1, 2, 6, 12, 16, 17, 19, 20, 48, 51, 52, 57, 58, 63, 71, 78,
            109, 116, 121, 122, 124, 127, 130, 136, 143, 153, 155, 230, 232,
            246, 277, 298, 306, 321, 322, 328, 337, 348, 353, 367, 380, 387,
            388, 389, 400, 403, 438, 473, 484]

if __name__ == "__main__":
    # Use existing cleaned_data.jsonl
    data_dir = Path("data")
    test_dir = Path("data_test")
    test_dir.mkdir(parents=True, exist_ok=True)

    all_file = data_dir / "cleaned_data.jsonl"
    train_file = test_dir / "train.jsonl"
    test_file = test_dir / "test.jsonl"
    train_final_file = test_dir / "train_final.jsonl"

    print(f"\n{'='*70}")
    print(f"TEST DATA GENERATION (using existing cleaned_data.jsonl)")
    print(f"{'='*70}\n")

    # Load all data from cleaned_data.jsonl
    with all_file.open("r", encoding="utf-8") as f:
        all_data = [json.loads(line) for line in f]

    total_samples = len(all_data)
    print(f"Total samples in cleaned_data.jsonl: {total_samples}")

    # Define the 15 URLs to exclude from train.jsonl (these were the original eval set)
    EXCLUDED_URLS = {
        "https://www.sec.gov/Archives/edgar/data/1261249/000155837022016628/agrx-20220930xex10d1.htm",
        "https://www.sec.gov/Archives/edgar/data/1396440/000139644023000102/exhibit101-responsetocommi.htm",
        "https://www.sec.gov/Archives/edgar/data/1501134/000150113421000046/invitae-ex101xperceptiveam.htm",
        "https://www.sec.gov/Archives/edgar/data/1617553/000161755322000045/ex101amendmentno3tocredita.htm",
        "https://www.sec.gov/Archives/edgar/data/1637459/000163745920000145/kraftheinz-increaseame.htm",
        "https://www.sec.gov/Archives/edgar/data/1643615/000119312520265580/d42408dex101.htm",
        "https://www.sec.gov/Archives/edgar/data/1687932/000119312520232098/d51878dex101.htm",
        "https://www.sec.gov/Archives/edgar/data/1718405/000171840522000003/waiverandamendmentjanuar.htm",
        "https://www.sec.gov/Archives/edgar/data/1750/000141057825000003/air-20241130xex10d1.htm",
        "https://www.sec.gov/Archives/edgar/data/1806201/000119312520315018/d98682dex101.htm",
        "https://www.sec.gov/Archives/edgar/data/318833/000119312522282609/d418227dex101.htm",
        "https://www.sec.gov/Archives/edgar/data/748790/000162828025003091/gceh-20250130xamendmentno19.htm",
        "https://www.sec.gov/Archives/edgar/data/864270/000119312521084124/d112325dex101.htm",
        "https://www.sec.gov/Archives/edgar/data/864270/000119312525062512/d927840dex101.htm",
        "https://www.sec.gov/Archives/edgar/data/886835/000119312521182447/d425766dex101.htm",
    }

    # Create train.jsonl (all data EXCEPT the 15 excluded URLs, preserving order)
    train_data = [entry for entry in all_data if entry['source_url'] not in EXCLUDED_URLS]
    with train_file.open("w", encoding="utf-8") as f:
        for entry in train_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"✓ Created {train_file.name} ({len(train_data)} entries)")
    print(f"  = cleaned_data ({total_samples}) - excluded ({len(EXCLUDED_URLS)})")

    # Get the 50 train_final indices from cleaned_data.jsonl (ALL 499 URLs)
    train_final_indices = get_train_final_indices()
    print(f"\nTrain_final indices (from cleaned_data): {train_final_indices[:10]}... (showing first 10)")

    # Create train_final.jsonl (50 indices from all_data)
    train_final_data = [all_data[i] for i in train_final_indices]
    with train_final_file.open("w", encoding="utf-8") as f:
        for entry in train_final_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"✓ Created {train_final_file.name} ({len(train_final_data)} entries)")

    # Create test.jsonl with the original structure:
    # - First 15: excluded URLs (not in train.jsonl)
    # - Remaining 434: URLs from train.jsonl that are not in train_final.jsonl
    train_final_url_set = {e['source_url'] for e in train_final_data}
    train_url_set = {e['source_url'] for e in train_data}

    # Part 1: 15 excluded URLs (in cleaned_data order)
    excluded_entries = [entry for entry in all_data if entry['source_url'] not in train_url_set]

    # Part 2: URLs from train.jsonl minus train_final (in train.jsonl order, which preserves cleaned_data order)
    train_minus_final = [entry for entry in train_data if entry['source_url'] not in train_final_url_set]

    # Combine: excluded first, then train_minus_final
    test_data = excluded_entries + train_minus_final

    with test_file.open("w", encoding="utf-8") as f:
        for entry in test_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"✓ Created {test_file.name} ({len(test_data)} entries)")
    print(f"  = {len(excluded_entries)} excluded + {len(train_minus_final)} (train - train_final)")

    # Verify no overlap
    train_final_urls = {e['source_url'] for e in train_final_data}
    test_urls = {e['source_url'] for e in test_data}
    overlap = train_final_urls & test_urls

    print(f"\n✓ Verification: {'No overlap' if not overlap else f'ERROR: {len(overlap)} overlapping URLs'}")
    print(f"  train: {len(train_data)} URLs")
    print(f"  train_final: {len(train_final_urls)} URLs")
    print(f"  test: {len(test_urls)} URLs")
    print(f"\n✅ Data generation complete in {test_dir}/")
