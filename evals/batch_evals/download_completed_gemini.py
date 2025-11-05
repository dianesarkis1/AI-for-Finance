#!/usr/bin/env python3
"""
Download completed Gemini batches using batch_job_mappings.json.
"""

import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent))

from batch_utils import check_gemini_batch_status, extract_gemini_batch_results

# Load environment
load_dotenv()
gemini_key = os.getenv("GEMINI_API_KEY")

if not gemini_key:
    print("Error: GEMINI_API_KEY not found in environment")
    sys.exit(1)

# Configuration
batch_temp_dir = Path("batch_temp_anthropic_prompt_gen")
mappings_file = batch_temp_dir / "batch_job_mappings.json"

print(f"\n{'='*70}")
print(f"DOWNLOADING COMPLETED GEMINI BATCHES")
print(f"{'='*70}\n")

# Load mappings
with open(mappings_file, 'r') as f:
    mappings = json.load(f)

# Get all Gemini jobs (filter by provider='gemini')
# This ensures we ONLY process Gemini batches, not OpenAI or Claude
gemini_jobs = [j for j in mappings['jobs'] if j['provider'] == 'gemini']
print(f"Found {len(gemini_jobs)} Gemini batch jobs in mappings file")

# Validate that these are actually Gemini batch IDs (start with 'batches/')
for job in gemini_jobs:
    if not job['batch_id'].startswith('batches/'):
        print(f"⚠️  Warning: Job with provider='gemini' has unexpected batch_id format: {job['batch_id']}")

print(f"\nNote: This script only processes Gemini batches.")
print(f"OpenAI and Claude batches are unaffected.\n")

completed_count = 0
failed_count = 0
already_exist_count = 0

for i, job in enumerate(gemini_jobs, 1):
    index = job['input_index']
    batch_id = job['batch_id']

    # Check if output file already exists
    existing_files = list(batch_temp_dir.glob(f"gemini_batch_output_{index}_*.jsonl"))
    if existing_files:
        print(f"[{i}/{len(gemini_jobs)}] Index {index}: Already downloaded (skipping)")
        already_exist_count += 1
        continue

    print(f"[{i}/{len(gemini_jobs)}] Index {index}: Checking {batch_id}...", end=" ")

    try:
        # Check status
        status_data = check_gemini_batch_status(batch_id, gemini_key)
        state = status_data.get("metadata", {}).get("state")

        if state == "BATCH_STATE_SUCCEEDED":
            # Download results
            output_file = extract_gemini_batch_results(status_data, batch_temp_dir, input_index=index)
            print(f"✅ Downloaded to {output_file.name}")
            completed_count += 1
        else:
            print(f"⚠️  State: {state} (not succeeded)")
            failed_count += 1

    except Exception as e:
        print(f"❌ Error: {e}")
        failed_count += 1

print(f"\n{'='*70}")
print(f"DOWNLOAD COMPLETE")
print(f"{'='*70}")
print(f"  ✅ Downloaded: {completed_count}")
print(f"  📁 Already existed: {already_exist_count}")
print(f"  ❌ Failed: {failed_count}")
print(f"  📊 Total: {len(gemini_jobs)}")
print()
