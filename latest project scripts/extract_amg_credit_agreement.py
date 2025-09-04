#!/usr/bin/env python3
"""
Better AMG Credit Agreement Extractor
"""

import json
import re

def clean_and_format_text(text):
    """Clean and format the extracted text properly."""
    # Remove extra whitespace and newlines
    text = re.sub(r'\s+', ' ', text)
    
    # Remove duplicate sections by finding unique content
    sections = []
    current_section = ""
    
    # Split by common section markers
    parts = re.split(r'(Section \d+\.|WHEREAS|NOW THEREFORE|IN WITNESS WHEREOF|ANNEX I)', text)
    
    for i, part in enumerate(parts):
        if part.strip():
            if part.startswith(('Section', 'WHEREAS', 'NOW THEREFORE', 'IN WITNESS', 'ANNEX')):
                if current_section.strip():
                    sections.append(current_section.strip())
                current_section = part
            else:
                current_section += part
    
    if current_section.strip():
        sections.append(current_section.strip())
    
    # Remove duplicates while preserving order
    seen = set()
    unique_sections = []
    for section in sections:
        # Create a key for deduplication (first 50 chars)
        key = section[:50]
        if key not in seen:
            seen.add(key)
            unique_sections.append(section)
    
    # Format the final text
    formatted_text = "\n\n".join(unique_sections)
    
    # Clean up specific artifacts
    formatted_text = re.sub(r'Exhibit 10\.1\s+', '', formatted_text)
    formatted_text = re.sub(r'EXECUTION VERSION\s+', '', formatted_text)
    formatted_text = re.sub(r'\[\d+\]', '', formatted_text)
    formatted_text = re.sub(r'\[Remainder\s+of\s+Page\s+Intentionally\s+Left\s+Blank\]', '', formatted_text)
    formatted_text = re.sub(r'£', '', formatted_text)  # Remove pound symbol
    
    return formatted_text.strip()

def extract_amg_credit_agreement():
    """Extract the AMG credit agreement from the JSONL file."""
    
    # Read the raw JSONL file
    with open('amg_raw.jsonl', 'r', encoding='utf-8') as f:
        line = f.readline().strip()
    
    # Parse the JSON
    data = json.loads(line)
    
    # Extract the text
    raw_text = data['text']
    
    # Clean and format the text
    formatted_text = clean_and_format_text(raw_text)
    
    # Write the formatted text to a file
    with open('amg_credit_agreement_final.txt', 'w', encoding='utf-8') as f:
        f.write(formatted_text)
    
    print(f"Original text length: {len(raw_text):,} characters")
    print(f"Formatted text length: {len(formatted_text):,} characters")
    print(f"Formatted text saved to: amg_credit_agreement_final.txt")
    
    return formatted_text

if __name__ == "__main__":
    extract_amg_credit_agreement() 