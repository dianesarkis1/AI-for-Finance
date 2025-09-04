import requests
from bs4 import BeautifulSoup
import chardet
import json
import re
import unicodedata
import time
import hashlib
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

if __name__ == "__main__":
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    input_file = data_dir / "urls.txt"
    all_file = data_dir / "cleaned_data.jsonl"
    train_file = data_dir / "train.jsonl"
    eval_file = data_dir / "eval.jsonl"
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

