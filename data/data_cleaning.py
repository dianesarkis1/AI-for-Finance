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

def deterministic_top_k(urls, k=15):
    """Pick top-k URLs by md5 hash (deterministic)."""
    return [u for u, _ in sorted(
        ((u, hashlib.md5(u.encode("utf-8")).hexdigest()) for u in urls),
        key=lambda t: t[1]
    )[:k]]

def generate_train_final_indices(total_train_samples, seed=42):
    """
    Generate the 50 training indices used for train_final.jsonl.

    Historical context: This 50-index sample evolved organically during development:
    - Started by manually reviewing first 3 indices (0, 1, 2)
    - Then sampled 10 baseline indices for initial evals
    - Later needed 37 more for a comprehensive 50-sample benchmark

    All sampling uses seed 42 for reproducibility.

    Returns:
        Sorted list of 50 unique indices from train.jsonl
    """
    random.seed(seed)

    # 1. First 3 indices (originally manually reviewed)
    comprehensive_indices = {0, 1, 2}

    # 2. Sample 10 baseline indices
    available = [i for i in range(total_train_samples) if i not in comprehensive_indices]
    baseline_10 = random.sample(available, 10)
    comprehensive_indices.update(baseline_10)

    # 3. Sample 37 more random indices to reach 50 total
    available = [i for i in range(total_train_samples) if i not in comprehensive_indices]
    additional_37 = random.sample(available, 37)
    comprehensive_indices.update(additional_37)

    return sorted(list(comprehensive_indices))

if __name__ == "__main__":
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    input_file = data_dir / "urls.txt"
    all_file = data_dir / "cleaned_data.jsonl"
    train_file = data_dir / "train.jsonl"
    eval_file = data_dir / "eval.jsonl"
    test_file = data_dir / "test.jsonl"
    train_final_file = data_dir / "train_final.jsonl"
    eval_urls_path = data_dir / "eval_urls.txt"  # locked list of 15 eval URLs

    # Load URLs
    with input_file.open("r", encoding="utf-8") as f:
        urls = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]

    # Ensure deterministic, stable 15-eval selection stored on disk
    if eval_urls_path.exists():
        with eval_urls_path.open("r", encoding="utf-8") as f:
            saved_eval = [ln.strip() for ln in f if ln.strip()]
        # Keep only those still present
        saved_eval = [u for u in saved_eval if u in urls]
        # If fewer than 15 remain, top up deterministically from remaining
        if len(saved_eval) < 15:
            remaining = [u for u in urls if u not in saved_eval]
            top_up = deterministic_top_k(remaining, 15 - len(saved_eval))
            saved_eval = saved_eval + top_up
    else:
        saved_eval = deterministic_top_k(urls, 15)

    # Write back (locks the set for future runs)
    with eval_urls_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(saved_eval) + "\n")

    eval_set = set(saved_eval)

    # Open outputs fresh each run
    with all_file.open("w", encoding="utf-8") as all_f, \
         train_file.open("w", encoding="utf-8") as train_f, \
         eval_file.open("w", encoding="utf-8") as eval_f:

        for url in urls:
            try:
                text = fetch_and_clean(url)
                record = {"source_url": url, "text": text}

                # Combined file (optional; keep if you like having everything in one)
                all_f.write(json.dumps(record, ensure_ascii=False) + "\n")

                # Fixed 15-doc eval; rest goes to train
                if url in eval_set:
                    eval_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    split = "eval"
                else:
                    train_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    split = "train"

                print(f"Processed: {url}  →  {split}")
            except Exception as e:
                print(f"Error processing {url}: {e}")

    # Generate test.jsonl and train_final.jsonl
    print("\n" + "="*70)
    print("Generating test.jsonl and train_final.jsonl...")
    print("="*70)

    # Load train.jsonl
    with train_file.open("r", encoding="utf-8") as f:
        train_data = [json.loads(line) for line in f]

    # Load eval.jsonl
    with eval_file.open("r", encoding="utf-8") as f:
        eval_data = [json.loads(line) for line in f]

    total_train = len(train_data)
    print(f"Train samples: {total_train}")
    print(f"Eval samples: {len(eval_data)}")

    # Generate the 50 train_final indices
    train_final_indices = generate_train_final_indices(total_train, seed=42)
    print(f"Train_final indices (50 total): {train_final_indices[:10]}... (showing first 10)")

    # Create train_final.jsonl (50 indices from train)
    train_final_data = [train_data[i] for i in train_final_indices]
    with train_final_file.open("w", encoding="utf-8") as f:
        for entry in train_final_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"✓ Created {train_final_file.name} ({len(train_final_data)} entries)")

    # Create test.jsonl (eval + all train EXCEPT the 50 train_final indices)
    train_final_set = set(train_final_indices)
    test_from_train = [train_data[i] for i in range(total_train) if i not in train_final_set]
    test_data = eval_data + test_from_train

    with test_file.open("w", encoding="utf-8") as f:
        for entry in test_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"✓ Created {test_file.name} ({len(test_data)} entries)")
    print(f"  = {len(eval_data)} eval + {len(test_from_train)} train (excluding train_final)")

    # Verify no overlap
    train_final_urls = {e['source_url'] for e in train_final_data}
    test_urls = {e['source_url'] for e in test_data}
    overlap = train_final_urls & test_urls

    print(f"\n✓ Verification: {'No overlap' if not overlap else f'ERROR: {len(overlap)} overlapping URLs'}")
    print(f"  train_final: {len(train_final_urls)} URLs")
    print(f"  test: {len(test_urls)} URLs")
    print("\n✅ Data pipeline complete!")

