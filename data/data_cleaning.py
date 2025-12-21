#!/usr/bin/env python3
"""
SEC Credit Agreement Data Cleaning Pipeline

This script fetches and cleans credit agreement documents from the SEC EDGAR database,
producing a fully reproducible set of data files for model evaluation.

Output Files (4 JSONL files):
    - cleaned_data.jsonl: All 499 cleaned credit agreements
    - train.jsonl: 484 documents (cleaned_data minus 15 excluded URLs)
    - train_final.jsonl: 50 documents (specific indices from cleaned_data)
    - test.jsonl: 449 documents (15 excluded + train minus train_final)

File Structure & Relationships:
    cleaned_data.jsonl (499 URLs)
        ├── train.jsonl (484 URLs)
        │   └── Excludes 15 specific URLs (hardcoded in EXCLUDED_URLS)
        │
        ├── train_final.jsonl (50 URLs)
        │   └── Specific indices: [0, 1, 2, 6, 12, 16, ...] from cleaned_data
        │       Corresponds to eval_urls.txt
        │
        └── test.jsonl (449 URLs)
            └── Structure: 15 excluded URLs + (train - train_final)

Reproducibility Guarantees:
    - All splits are deterministic using hardcoded indices and URL lists
    - The 15 excluded URLs are specified in EXCLUDED_URLS constant (line ~142)
    - The 50 train_final indices are in get_train_final_indices() (line ~88)
    - No random sampling is used; all selections are fixed
    - Order is preserved from cleaned_data.jsonl throughout

Historical Context:
    - The 15 excluded URLs were selected with random seed during initial exploration
    - The 50 train_final samples evolved organically:
        * First 3 indices (0,1,2) manually reviewed
        * Then 10 baseline indices sampled for initial evals
        * Then 37 more for comprehensive 50-sample benchmark
    - test.jsonl structure (15 excluded first, then remainder) was established
      early and maintained for backward compatibility

Usage:
    # Default: outputs to data/
    python data/data_cleaning.py

    # Custom output directory (e.g., for testing without overwriting)
    python data/data_cleaning.py data_test

Notes:
    - Fetches from SEC servers: 499 documents × 0.2s delay = ~2 minutes
    - Always reads URLs from data/urls.txt
    - All JSONL files use format: {"source_url": "...", "text": "..."}
    - DO NOT modify hardcoded indices unless regenerating all evaluation results

Requirements:
    pip install requests beautifulsoup4 lxml chardet
"""

import requests
from bs4 import BeautifulSoup
import chardet
import json
import re
import unicodedata
import time
import hashlib
import random
from pathlib import Path

def clean_html_to_text(html):
    """Convert SEC exhibit HTML to clean plain text with headings preserved."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()
    for br in soup.find_all("br"):
        br.replace_with("\n")

    blocks = soup.find_all(["h1", "h2", "h3", "p", "li", "div"])
    if blocks:
        parts = []
        for b in blocks:
            text = b.get_text(" ", strip=True)
            if text:
                parts.append(text)
        text = "\n".join(parts)
    else:
        text = soup.get_text("\n", strip=True)

    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00A0", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text.strip()

def detect_decode(data: bytes) -> str:
    enc = (chardet.detect(data).get("encoding") if data else None) or "utf-8"
    try:
        return data.decode(enc, errors="replace")
    except Exception:
        return data.decode("utf-8", errors="replace")

def fetch_and_clean(url: str) -> str:
    headers = {
        "User-Agent": "DianeSarkis-FDEDataCleaning/0.1 (dianesarkis@gmail.com)",
        "Accept-Encoding": "gzip, deflate",
        "Host": "www.sec.gov",
    }
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    html = detect_decode(resp.content)
    time.sleep(0.2)  # polite to SEC
    return clean_html_to_text(html)

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
    import sys

    # Allow optional output directory argument
    if len(sys.argv) > 1:
        data_dir = Path(sys.argv[1])
        print(f"Using output directory: {data_dir}")
    else:
        data_dir = Path("data")

    data_dir.mkdir(parents=True, exist_ok=True)

    input_file = Path("data") / "urls.txt"  # Always read from data/
    all_file = data_dir / "cleaned_data.jsonl"
    train_file = data_dir / "train.jsonl"
    test_file = data_dir / "test.jsonl"
    train_final_file = data_dir / "train_final.jsonl"

    # Load URLs
    with input_file.open("r", encoding="utf-8") as f:
        urls = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]

    # Process all URLs and write to cleaned_data.jsonl only
    with all_file.open("w", encoding="utf-8") as all_f:
        for url in urls:
            try:
                text = fetch_and_clean(url)
                record = {"source_url": url, "text": text}
                all_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(f"Processed: {url}")
            except Exception as e:
                print(f"Error processing {url}: {e}")

    # Generate train.jsonl, test.jsonl and train_final.jsonl from cleaned_data.jsonl
    print("\n" + "="*70)
    print("Generating train.jsonl, test.jsonl, and train_final.jsonl...")
    print("="*70)

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
    print("\n✅ Data pipeline complete!")

