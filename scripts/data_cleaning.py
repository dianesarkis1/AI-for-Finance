import requests
from bs4 import BeautifulSoup
import chardet
import json
import re
import unicodedata
import time

def clean_html_to_text(html):
    """Convert SEC exhibit HTML to clean plain text with headings preserved."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()
    for br in soup.find_all("br"):
        br.replace_with("\n")

    # Keep headings + paragraphs
    blocks = soup.find_all(["h1","h2","h3","p","li","div"])
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
    text = re.sub(r"\n{3,}", "\n\n", text)  # collapse extra blank lines
    text = re.sub(r"[ \t]{2,}", " ", text)  # collapse extra spaces
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text.strip()

def detect_decode(data):
    enc = chardet.detect(data).get("encoding") or "utf-8"
    try:
        return data.decode(enc, errors="replace")
    except:
        return data.decode("utf-8", errors="replace")

def fetch_and_clean(url):
    headers = {
        "User-Agent": "DianeSarkis-FDEDataCleaning/0.1 (dianesarkis@gmail.com)",
        "Accept-Encoding": "gzip, deflate",
        "Host": "www.sec.gov"
    }
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    html = detect_decode(resp.content)
    time.sleep(0.2)  # polite pause so SEC doesn’t rate-limit
    return clean_html_to_text(html)

if __name__ == "__main__":
    input_file = "data/urls.txt"
    output_file = "data/cleaned_data.jsonl"

    with open(input_file, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    with open(output_file, "w", encoding="utf-8") as out:
        for url in urls:
            try:
                text = fetch_and_clean(url)
                out.write(json.dumps({"source_url": url, "text": text}, ensure_ascii=False) + "\n")
                print(f"Processed: {url}")
            except Exception as e:
                print(f"Error processing {url}: {e}")

