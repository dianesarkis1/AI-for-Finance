#!/usr/bin/env python3
"""
Extract AMG Credit Agreement from JSONL
"""

import json
import re

def clean_text(text):
    """Clean the extracted text by removing extra whitespace and formatting."""
    # Remove extra whitespace and newlines
    text = re.sub(r'\s+', ' ', text)
    # Remove the "Exhibit 10.1" header that appears multiple times
    text = re.sub(r'Exhibit 10\.1\s+', '', text)
    # Remove page numbers and formatting artifacts
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\[Remainder\s+of\s+Page\s+Intentionally\s+Left\s+Blank\]', '', text)
    # Clean up the text
    text = text.strip()
    return text

def extract_amg_credit_agreement():
    """Extract the AMG credit agreement from the JSONL file."""
    
    # Read the raw JSONL file
    with open('amg_raw.jsonl', 'r', encoding='utf-8') as f:
        line = f.readline().strip()
    
    # Parse the JSON
    data = json.loads(line)
    
    # Extract the text
    raw_text = data['text']
    
    # Clean the text
    cleaned_text = clean_text(raw_text)
    
    # Write the cleaned text to a file
    with open('amg_credit_agreement_clean.txt', 'w', encoding='utf-8') as f:
        f.write(cleaned_text)
    
    print(f"Original text length: {len(raw_text):,} characters")
    print(f"Cleaned text length: {len(cleaned_text):,} characters")
    print(f"Cleaned text saved to: amg_credit_agreement_clean.txt")
    
    return cleaned_text

if __name__ == "__main__":
    extract_amg_credit_agreement() 