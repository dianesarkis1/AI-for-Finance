#!/usr/bin/env python3
"""
Download Gemini batch results from API and save as JSONL files.
This script fetches completed Gemini batch results and saves them to batch_temp/
in the same format as OpenAI/Claude outputs for consistency.
"""

import json
import os
import time
from pathlib import Path
import urllib.request

# Paths
BATCH_TEMP_DIR = Path(__file__).parent / "batch_temp"
GEMINI_JOBS_FILE = Path(__file__).parent / "gemini_batch_jobs.json"

def get_gemini_api_key():
    """Load Gemini API key from environment."""
    # Try environment variable first
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        # Try loading from .env file
        env_file = Path(__file__).parent.parent.parent / ".env"
        if env_file.exists():
            with open(env_file, 'r') as f:
                for line in f:
                    if line.startswith('GEMINI_API_KEY='):
                        api_key = line.strip().split('=', 1)[1].strip('"').strip("'")
                        break

    return api_key


def fetch_batch_results(batch_id: str, api_key: str) -> dict:
    """Fetch batch results from Gemini API."""
    url = f"https://generativelanguage.googleapis.com/v1beta/{batch_id}?key={api_key}"

    with urllib.request.urlopen(url) as response:
        data = json.loads(response.read())

    return data


def parse_gemini_batch_to_jsonl(batch_data: dict, input_index: int) -> list:
    """
    Parse Gemini batch API response and convert to JSONL format
    matching OpenAI/Claude output structure.

    Returns list of evaluation results.
    """
    results = []

    # Get the inline responses
    responses = batch_data.get('response', {}).get('inlinedResponses', {}).get('inlinedResponses', [])

    for response_obj in responses:
        metric_key = response_obj.get('metadata', {}).get('key', 'unknown')
        response = response_obj.get('response', {})

        # Extract text content
        candidates = response.get('candidates', [])
        if candidates:
            text = candidates[0].get('content', {}).get('parts', [{}])[0].get('text', '')

            # Create result object
            result = {
                'custom_id': f'{metric_key}_{input_index}',
                'response': {
                    'body': {
                        'choices': [{
                            'message': {
                                'content': text
                            }
                        }]
                    }
                }
            }
            results.append(result)

    return results


def main():
    print(f"\n{'='*70}")
    print(f"DOWNLOADING GEMINI BATCH RESULTS")
    print(f"{'='*70}\n")

    # Load API key
    api_key = get_gemini_api_key()
    if not api_key:
        print("❌ ERROR: GEMINI_API_KEY not found in environment or .env file")
        return

    # Load batch jobs
    print(f"Loading batch jobs from {GEMINI_JOBS_FILE}...")
    with open(GEMINI_JOBS_FILE, 'r') as f:
        jobs_data = json.load(f)

    submitted_jobs = jobs_data['submitted_jobs']
    print(f"  Found {len(submitted_jobs)} batch jobs to download\n")

    # Download and save each batch
    successful = 0
    failed = 0

    for i, job in enumerate(submitted_jobs, 1):
        batch_id = job['batch_id']
        input_index = job['input_index']

        print(f"[{i}/{len(submitted_jobs)}] Downloading batch for index {input_index}...")
        print(f"  Batch ID: {batch_id}")

        try:
            # Fetch from API
            batch_data = fetch_batch_results(batch_id, api_key)

            # Check status
            state = batch_data.get('metadata', {}).get('state', 'UNKNOWN')
            print(f"  Status: {state}")

            if state != 'BATCH_STATE_SUCCEEDED':
                print(f"  ⚠️  Skipping - batch not completed")
                failed += 1
                continue

            # Parse to JSONL format
            results = parse_gemini_batch_to_jsonl(batch_data, input_index)

            # Save to JSONL file
            timestamp = int(time.time())
            output_file = BATCH_TEMP_DIR / f"gemini_batch_output_{input_index}_{timestamp}.jsonl"

            with open(output_file, 'w') as f:
                for result in results:
                    f.write(json.dumps(result) + '\n')

            print(f"  ✅ Saved to: {output_file.name}")
            print(f"     {len(results)} evaluation results")
            successful += 1

        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            failed += 1

        print()

    print(f"{'='*70}")
    print(f"DOWNLOAD COMPLETE")
    print(f"{'='*70}")
    print(f"Successfully downloaded: {successful}/{len(submitted_jobs)}")
    print(f"Failed/Skipped: {failed}/{len(submitted_jobs)}")
    print()


if __name__ == "__main__":
    main()
