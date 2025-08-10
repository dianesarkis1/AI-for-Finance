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

    # Keep headings + paragraphs
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
    text = re.sub(r"\n{3,}", "\n\n", text)   # collapse extra blank lines
    text = re.sub(r"[ \t]{2,}", " ", text)   # collapse extra spaces
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
    time.sleep(0.2)  # be polite to SEC
    return clean_html_to_text(html)

def assign_split(url: str, train_ratio: float = 0.8) -> str:
    """Deterministically map URL to 'train' or 'eval' using a stable hash."""
    h = hashlib.md5(url.encode("utf-8")).hexdigest()
    val = int(h, 16) % 1000 / 1000.0  # 0.000 ... 0.999
    return "train" if val < train_ratio else "eval"

if __name__ == "__main__":
    input_file = Path("data/urls.txt")
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    all_file = data_dir / "cleaned_data.jsonl"
    train_file = data_dir / "train.jsonl"
    eval_file = data_dir / "eval.jsonl"

    with input_file.open("r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    # Open all three outputs fresh each run
    with all_file.open("w", encoding="utf-8") as all_f, \
         train_file.open("w", encoding="utf-8") as train_f, \
         eval_file.open("w", encoding="utf-8") as eval_f:

        for url in urls:
            try:
                text = fetch_and_clean(url)
                record = {"source_url": url, "text": text}

                # Write to combined file
                all_f.write(json.dumps(record, ensure_ascii=False) + "\n")

                # Deterministic split
                split = assign_split(url, train_ratio=0.8)
                if split == "train":
                    train_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                else:
                    eval_f.write(json.dumps(record, ensure_ascii=False) + "\n")

                print(f"Processed: {url}  →  {split}")
            except Exception as e:
                print(f"Error processing {url}: {e}")

