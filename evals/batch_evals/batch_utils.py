"""
Utilities for OpenAI Batch API operations.

Provides functions to:
- Submit batch jobs
- Monitor batch job status
- Retrieve batch results
- Resume monitoring interrupted jobs
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional
import subprocess


def run_curl(args: List[str], stdin_bytes: Optional[bytes] = None) -> str:
    """Execute curl command and return stdout."""
    result = subprocess.run(
        args,
        input=stdin_bytes,
        capture_output=True,
        check=False,
        text=False,
    )
    if result.returncode != 0:
        stderr_text = result.stderr.decode("utf-8", errors="ignore")
        stdout_text = result.stdout.decode("utf-8", errors="ignore")
        raise RuntimeError(
            f"Command failed: {' '.join(args)}\nSTDERR:\n{stderr_text}\nSTDOUT:\n{stdout_text}"
        )
    return result.stdout.decode("utf-8", errors="ignore")


def upload_batch_file(requests: List[Dict], temp_dir: Path, api_key: str) -> str:
    """
    Upload a batch request file to OpenAI.

    Args:
        requests: List of batch request objects
        temp_dir: Directory to store temporary files
        api_key: OpenAI API key

    Returns:
        File ID from OpenAI
    """
    # Create JSONL file
    timestamp = int(time.time())
    input_file = temp_dir / f"batch_input_{timestamp}.jsonl"

    with open(input_file, "w") as f:
        for request in requests:
            f.write(json.dumps(request) + "\n")

    print(f"📄 Created batch input file: {input_file}")
    print(f"   Total requests: {len(requests)}")

    # Upload file to OpenAI
    cmd = [
        "curl",
        "-sS",
        "-X",
        "POST",
        "https://api.openai.com/v1/files",
        "-H",
        f"Authorization: Bearer {api_key}",
        "-F",
        f"file=@{input_file}",
        "-F",
        "purpose=batch"
    ]

    response = run_curl(cmd)
    file_data = json.loads(response)

    if "id" not in file_data:
        raise RuntimeError(f"Failed to upload file: {response}")

    file_id = file_data["id"]
    print(f"✅ Uploaded to OpenAI: {file_id}")

    return file_id


def create_batch_job(file_id: str, api_key: str, description: str = None) -> str:
    """
    Create a batch job on OpenAI.

    Args:
        file_id: ID of uploaded input file
        api_key: OpenAI API key
        description: Optional description for the batch job

    Returns:
        Batch job ID
    """
    payload = {
        "input_file_id": file_id,
        "endpoint": "/v1/chat/completions",
        "completion_window": "24h"
    }

    if description:
        payload["metadata"] = {"description": description}

    cmd = [
        "curl",
        "-sS",
        "-X",
        "POST",
        "https://api.openai.com/v1/batches",
        "-H",
        f"Authorization: Bearer {api_key}",
        "-H",
        "Content-Type: application/json",
        "--data-binary",
        "@-"
    ]

    response = run_curl(cmd, stdin_bytes=json.dumps(payload).encode("utf-8"))
    batch_data = json.loads(response)

    if "id" not in batch_data:
        raise RuntimeError(f"Failed to create batch: {response}")

    batch_id = batch_data["id"]
    print(f"🚀 Batch job created: {batch_id}")
    print(f"   Status: {batch_data.get('status', 'unknown')}")

    return batch_id


def check_batch_status(batch_id: str, api_key: str) -> Dict:
    """
    Check the status of a batch job.

    Args:
        batch_id: Batch job ID
        api_key: OpenAI API key

    Returns:
        Batch status information
    """
    cmd = [
        "curl",
        "-sS",
        "-X",
        "GET",
        f"https://api.openai.com/v1/batches/{batch_id}",
        "-H",
        f"Authorization: Bearer {api_key}"
    ]

    response = run_curl(cmd)
    return json.loads(response)


def download_batch_results(output_file_id: str, temp_dir: Path, api_key: str) -> Path:
    """
    Download batch results from OpenAI.

    Args:
        output_file_id: ID of output file
        temp_dir: Directory to store downloaded file
        api_key: OpenAI API key

    Returns:
        Path to downloaded results file
    """
    cmd = [
        "curl",
        "-sS",
        "-X",
        "GET",
        f"https://api.openai.com/v1/files/{output_file_id}/content",
        "-H",
        f"Authorization: Bearer {api_key}"
    ]

    response = run_curl(cmd)

    # Save to file
    timestamp = int(time.time())
    output_file = temp_dir / f"batch_output_{timestamp}.jsonl"

    with open(output_file, "w") as f:
        f.write(response)

    print(f"📥 Downloaded results to: {output_file}")

    return output_file


def load_batch_results(output_file: Path) -> List[Dict]:
    """
    Load batch results from JSONL file.

    Args:
        output_file: Path to results file

    Returns:
        List of result objects
    """
    results = []
    with open(output_file, "r") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def poll_batch_until_complete(
    batch_id: str,
    api_key: str,
    temp_dir: Path,
    poll_interval: int = 60,
    max_wait_time: int = 86400  # 24 hours
) -> List[Dict]:
    """
    Poll a batch job until completion and return results.

    Args:
        batch_id: Batch job ID
        api_key: OpenAI API key
        temp_dir: Directory for temporary files
        poll_interval: Seconds between status checks (default: 60)
        max_wait_time: Maximum seconds to wait (default: 24 hours)

    Returns:
        List of result objects

    Raises:
        TimeoutError: If max_wait_time exceeded
        RuntimeError: If batch job fails
    """
    print(f"\n⏳ Polling batch job: {batch_id}")
    print(f"   Checking every {poll_interval} seconds")
    print(f"   You can safely close this and resume later with resume_batch_job('{batch_id}')\n")

    # Save batch ID for resuming
    batch_id_file = temp_dir / f"batch_{batch_id}.json"
    with open(batch_id_file, "w") as f:
        json.dump({
            "batch_id": batch_id,
            "created_at": time.time(),
            "status": "polling"
        }, f, indent=2)

    start_time = time.time()
    last_status = None

    while True:
        elapsed = time.time() - start_time

        if elapsed > max_wait_time:
            raise TimeoutError(f"Batch job exceeded max wait time of {max_wait_time}s")

        # Check status
        status_data = check_batch_status(batch_id, api_key)
        current_status = status_data.get("status")

        # Print status update if changed
        if current_status != last_status:
            print(f"[{int(elapsed)}s] Status: {current_status}")

            # Show progress if available
            request_counts = status_data.get("request_counts", {})
            if request_counts:
                total = request_counts.get("total", 0)
                completed = request_counts.get("completed", 0)
                failed = request_counts.get("failed", 0)
                if total > 0:
                    print(f"         Progress: {completed}/{total} completed, {failed} failed")

            last_status = current_status

        # Check if completed
        if current_status == "completed":
            print(f"✅ Batch job completed in {int(elapsed)}s!")

            # Download results
            output_file_id = status_data.get("output_file_id")
            if not output_file_id:
                raise RuntimeError("Batch completed but no output_file_id found")

            output_file = download_batch_results(output_file_id, temp_dir, api_key)
            results = load_batch_results(output_file)

            # Update batch ID file
            with open(batch_id_file, "w") as f:
                json.dump({
                    "batch_id": batch_id,
                    "created_at": start_time,
                    "completed_at": time.time(),
                    "status": "completed",
                    "output_file": str(output_file)
                }, f, indent=2)

            return results

        elif current_status == "failed":
            error_file_id = status_data.get("error_file_id")
            error_msg = f"Batch job failed. Error file ID: {error_file_id}"

            if error_file_id:
                try:
                    error_file = download_batch_results(error_file_id, temp_dir, api_key)
                    error_msg += f"\nError file saved to: {error_file}"
                except Exception as e:
                    error_msg += f"\nCould not download error file: {e}"

            raise RuntimeError(error_msg)

        elif current_status == "cancelled":
            raise RuntimeError("Batch job was cancelled")

        elif current_status == "expired":
            raise RuntimeError("Batch job expired")

        # Wait before next check
        time.sleep(poll_interval)


def resume_batch_job(batch_id: str, api_key: str, temp_dir: Path, poll_interval: int = 60) -> List[Dict]:
    """
    Resume monitoring a batch job that was previously started.

    Useful if you closed your computer or the script was interrupted.

    Args:
        batch_id: Batch job ID to resume
        api_key: OpenAI API key
        temp_dir: Directory for temporary files
        poll_interval: Seconds between status checks (default: 60)

    Returns:
        List of result objects
    """
    print(f"🔄 Resuming batch job: {batch_id}")

    # Check if already completed
    batch_id_file = temp_dir / f"batch_{batch_id}.json"
    if batch_id_file.exists():
        with open(batch_id_file, "r") as f:
            batch_info = json.load(f)

        if batch_info.get("status") == "completed":
            output_file = Path(batch_info.get("output_file"))
            if output_file.exists():
                print(f"✅ Batch already completed! Loading results from: {output_file}")
                return load_batch_results(output_file)

    # Otherwise, poll until complete
    return poll_batch_until_complete(batch_id, api_key, temp_dir, poll_interval)


def submit_and_wait_for_batch(
    requests: List[Dict],
    api_key: str,
    temp_dir: Path,
    description: str = None,
    poll_interval: int = 60
) -> List[Dict]:
    """
    Complete batch workflow: upload, create job, poll, and return results.

    This is the main convenience function that handles the entire batch process.

    Args:
        requests: List of batch request objects
        api_key: OpenAI API key
        temp_dir: Directory for temporary files
        description: Optional description for the batch job
        poll_interval: Seconds between status checks (default: 60)

    Returns:
        List of result objects
    """
    print("=" * 70)
    print("BATCH API EVALUATION")
    print("=" * 70)

    # Upload file
    file_id = upload_batch_file(requests, temp_dir, api_key)

    # Create batch job
    batch_id = create_batch_job(file_id, api_key, description)

    # Poll until complete
    results = poll_batch_until_complete(batch_id, api_key, temp_dir, poll_interval)

    print("=" * 70)
    print(f"BATCH COMPLETE: {len(results)} results received")
    print("=" * 70)

    return results
