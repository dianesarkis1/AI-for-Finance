#!/usr/bin/env python3
"""
Download Gemini batch results for the current run.
Uses the same logic as run_truly_parallel_batch_eval.py polling section.
Maps batches to indices based on creation order.
"""

import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evals.batch_evals.batch_utils import (
    check_gemini_batch_status,
    extract_gemini_batch_results
)
from evals.batch_evals.batch_metrics import (
    parse_gemini_batch_results
)


def load_api_key_from_env(key_name: str):
    """Load API key from environment or .env file."""
    import os

    api_key = os.getenv(key_name)
    if not api_key:
        env_file = Path(__file__).parent.parent.parent / ".env"
        if env_file.exists():
            with open(env_file, 'r') as f:
                for line in f:
                    if line.strip().startswith(key_name):
                        api_key = line.strip().split('=', 1)[1].strip().strip('"').strip("'")
                        break
    return api_key


def get_todays_gemini_batches(api_key: str):
    """Get all Gemini batches from today, sorted by creation time."""
    import subprocess

    # Use curl like batch_utils does to avoid SSL issues
    cmd = [
        "curl",
        "-sS",
        "-X",
        "GET",
        f"https://generativelanguage.googleapis.com/v1beta/batches?key={api_key}",
        "-H",
        f"x-goog-api-key: {api_key}"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)

    batches = []
    if 'operations' in data:
        for op in data['operations']:
            meta = op.get('metadata', {})
            created = meta.get('createTime', '')

            # Filter for today's batches
            if '2025-11-02' in created and 'memo-evaluation-176210' in meta.get('displayName', ''):
                batches.append({
                    'name': op['name'],
                    'display_name': meta.get('displayName', ''),
                    'created': created,
                    'state': meta.get('state', ''),
                })

    # Sort by creation time
    batches.sort(key=lambda x: x['created'])
    return batches


def main():
    """Download Gemini results with correct index mapping."""

    BATCH_TEMP_DIR = Path(__file__).parent / "batch_temp_2"

    print("=" * 70)
    print("DOWNLOADING GEMINI BATCH RESULTS")
    print("=" * 70)
    print(f"Output directory: {BATCH_TEMP_DIR}")
    print()

    # Load API key
    gemini_key = load_api_key_from_env("GEMINI_API_KEY")
    if not gemini_key:
        print("ERROR: GEMINI_API_KEY not found in environment or .env file")
        sys.exit(1)

    # Check if mapping file exists (created by run_truly_parallel_batch_eval.py)
    mapping_file = BATCH_TEMP_DIR / "batch_job_mappings.json"
    batch_to_index = {}

    if mapping_file.exists():
        print(f"Found batch job mappings file: {mapping_file.name}")
        with open(mapping_file, 'r') as f:
            mappings = json.load(f)

        # Extract Gemini batch mappings
        for job in mappings.get("jobs", []):
            if job.get("provider") == "gemini":
                batch_to_index[job["batch_id"]] = job["input_index"]

        if batch_to_index:
            print(f"Loaded {len(batch_to_index)} Gemini batch mappings from file")
            print()
        else:
            print("No Gemini batches found in mapping file, will prompt for manual input")
            mapping_file = None

    if not batch_to_index:
        # Fallback to manual input
        print("Please provide the Gemini batch IDs from your terminal output.")
        print("They should appear as lines like: '✅ Job ID: batches/xxxxx'")
        print("They were printed in the order the indices were processed.")
        print()
        print("Paste the batch IDs (one per line), then press Enter on an empty line:")
        print()

        batch_ids = []
        while True:
            line = input().strip()
            if not line:
                break
            # Extract just the batch ID if they paste the full line
            if "batches/" in line:
                batch_id = line.split("batches/")[1].split()[0]
                batch_ids.append(f"batches/{batch_id}")
            else:
                print(f"  Warning: '{line}' doesn't look like a batch ID, skipping")

        if not batch_ids:
            print("ERROR: No batch IDs provided")
            sys.exit(1)

        print()
        print(f"Got {len(batch_ids)} batch IDs")
        print()

        # Get the indices from batch_input files
        print("Reading batch_input files to determine indices...")
        import re
        indices = []
        for input_file in sorted(BATCH_TEMP_DIR.glob("batch_input_*.jsonl")):
            match = re.match(r'batch_input_(\d+)_(\d+)\.jsonl', input_file.name)
            if match:
                indices.append(int(match.group(1)))

        print(f"Found indices: {indices}")
        print()

        # Verify counts match
        if len(batch_ids) != len(indices):
            print(f"ERROR: Number of batch IDs ({len(batch_ids)}) doesn't match number of indices ({len(indices)})")
            print("Make sure you pasted all Gemini batch IDs from your terminal")
            sys.exit(1)

        # Map batches to indices by order
        for i, batch_id in enumerate(batch_ids):
            batch_to_index[batch_id] = indices[i]

    print("Mapping batches to indices:")
    for batch_id, idx in sorted(batch_to_index.items(), key=lambda x: x[1]):
        print(f"  Index {idx}: {batch_id}")
    print()

    # Download each batch using the EXACT same logic as run_truly_parallel_batch_eval.py
    downloaded = 0
    for batch_name, input_index in batch_to_index.items():
        try:
            # Check if already downloaded
            existing = list(BATCH_TEMP_DIR.glob(f"gemini_batch_output_{input_index}_*.jsonl"))
            if existing:
                print(f"⏭️  Index {input_index}: Already exists ({existing[0].name})")
                continue

            print(f"Downloading index {input_index}...")

            # This is the EXACT same logic as in run_truly_parallel_batch_eval.py lines 582-597
            # (with the fixed state check)
            status_data = check_gemini_batch_status(batch_name, gemini_key)

            # State is nested in metadata for Gemini API
            state = status_data.get("metadata", {}).get("state")

            if state == "BATCH_STATE_SUCCEEDED":
                # Use the same extract function with input_index parameter
                output_path = extract_gemini_batch_results(
                    status_data,
                    BATCH_TEMP_DIR,
                    input_index=input_index
                )

                # Parse results (same as in script)
                with open(output_path, 'r') as f:
                    batch_results = [json.loads(line) for line in f]
                parsed = parse_gemini_batch_results(batch_results)

                print(f"  ✅ Index {input_index}: Downloaded to {output_path.name}")
                print(f"     Results: {len(batch_results)} responses")
                downloaded += 1

            else:
                print(f"  ⚠️  Index {input_index}: Batch state is {state} (not SUCCEEDED)")

        except Exception as e:
            print(f"  ❌ Index {input_index}: Error - {e}")
            import traceback
            traceback.print_exc()

    print()
    print("=" * 70)
    print(f"DOWNLOAD COMPLETE: {downloaded}/{len(batch_to_index)} batches")
    print("=" * 70)

    if downloaded > 0:
        print()
        print("Files created in batch_temp_2/:")
        for f in sorted(BATCH_TEMP_DIR.glob("gemini_batch_output_*.jsonl")):
            print(f"  {f.name}")


if __name__ == "__main__":
    main()
