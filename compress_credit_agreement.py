#!/usr/bin/env python3
"""
Credit Agreement Compressor

This script compresses credit agreement HTML files by extracting key information
and removing HTML formatting, repetitive content, and unnecessary details.
"""

import re
import html
from pathlib import Path
from bs4 import BeautifulSoup
import argparse

def clean_text(text):
    """Clean and normalize text content."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove HTML entities
    text = html.unescape(text)
    # Remove special characters
    text = re.sub(r'[^\w\s\-.,$%()]', '', text)
    return text.strip()

def extract_key_sections(html_content):
    """Extract key sections from credit agreement HTML."""
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()
    
    # Get text content
    text = soup.get_text()
    
    # Define key sections to extract
    key_sections = []
    
    # Extract title and basic info
    title_match = re.search(r'CREDIT AGREEMENT', text, re.IGNORECASE)
    if title_match:
        key_sections.append("CREDIT AGREEMENT")
    
    # Extract company names
    company_patterns = [
        r'between\s+([^,\n]+?)\s+and\s+([^,\n]+?)(?:\s+dated|\s+as\s+of)',
        r'([A-Z][A-Z\s&]+(?:INC|CORP|LLC|LTD|BANK|TRUST)[A-Z\s&]*)',
        r'Borrower[:\s]+([^,\n]+)',
        r'Lender[:\s]+([^,\n]+)'
    ]
    
    for pattern in company_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):
                key_sections.extend([clean_text(m) for m in match if clean_text(m)])
            else:
                cleaned = clean_text(match)
                if cleaned and len(cleaned) > 5:
                    key_sections.append(cleaned)
    
    # Extract key financial terms
    financial_patterns = [
        r'(\$[\d,]+(?:\.\d{2})?)',  # Dollar amounts
        r'(\d+(?:\.\d+)?\s*%)',     # Percentages
        r'(?:interest rate|rate)[:\s]+([^%\n]+)',  # Interest rate
        r'(?:maturity|term)[:\s]+([^,\n]+)',      # Maturity date
        r'(?:commitment|facility)[:\s]+([^,\n]+)', # Commitment amount
        r'(?:loan|credit)[:\s]+([^,\n]+)'         # Loan type
    ]
    
    for pattern in financial_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            cleaned = clean_text(match)
            if cleaned and len(cleaned) > 2:
                key_sections.append(cleaned)
    
    # Extract key covenants and conditions
    covenant_patterns = [
        r'(?:covenant|covenants)[:\s]+([^,\n]+)',
        r'(?:condition|conditions)[:\s]+([^,\n]+)',
        r'(?:warranty|warranties)[:\s]+([^,\n]+)',
        r'(?:default|defaults)[:\s]+([^,\n]+)'
    ]
    
    for pattern in covenant_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            cleaned = clean_text(match)
            if cleaned and len(cleaned) > 5:
                key_sections.append(cleaned)
    
    # Extract date information
    date_patterns = [
        r'(?:dated|as of)\s+([^,\n]+)',
        r'(\w+\s+\d{1,2},?\s+\d{4})',
        r'(\d{1,2}/\d{1,2}/\d{4})',
        r'(\d{4}-\d{2}-\d{2})'
    ]
    
    for pattern in date_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            cleaned = clean_text(match)
            if cleaned and len(cleaned) > 5:
                key_sections.append(cleaned)
    
    # Remove duplicates and sort
    key_sections = list(set(key_sections))
    key_sections.sort()
    
    return key_sections

def compress_credit_agreement(input_file, output_file):
    """Compress a credit agreement HTML file."""
    print(f"Compressing {input_file}...")
    
    # Read the HTML file
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
        html_content = f.read()
    
    # Extract key sections
    key_sections = extract_key_sections(html_content)
    
    # Create compressed content
    key_info = chr(10).join(f"- {section}" for section in key_sections)
    compressed_content = f"""COMPRESSED CREDIT AGREEMENT
Generated from: {Path(input_file).name}

KEY INFORMATION EXTRACTED:
{key_info}

ORIGINAL FILE SIZE: {len(html_content):,} characters
"""
    
    # Calculate compression ratio after creating content
    compression_ratio = len(html_content) / len(compressed_content) if len(compressed_content) > 0 else 0
    compressed_content += f"COMPRESSED SIZE: {len(compressed_content):,} characters\n"
    compressed_content += f"COMPRESSION RATIO: {compression_ratio:.1f}x\n"
    
    # Write compressed file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(compressed_content)
    
    print(f"Compressed file saved to: {output_file}")
    print(f"Original size: {len(html_content):,} characters")
    print(f"Compressed size: {len(compressed_content):,} characters")
    print(f"Compression ratio: {compression_ratio:.1f}x")
    
    return output_file

def main():
    parser = argparse.ArgumentParser(description="Compress credit agreement HTML files")
    parser.add_argument("input_file", help="Input HTML credit agreement file")
    parser.add_argument("--output", "-o", help="Output compressed file (default: input_compressed.txt)")
    
    args = parser.parse_args()
    
    if not args.output:
        input_path = Path(args.input_file)
        args.output = input_path.stem + "_compressed.txt"
    
    try:
        compress_credit_agreement(args.input_file, args.output)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 