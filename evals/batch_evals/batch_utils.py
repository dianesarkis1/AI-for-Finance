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


def upload_batch_file(requests: List[Dict], temp_dir: Path, api_key: str, input_index: Optional[int] = None) -> str:
    """
    Upload a batch request file to OpenAI.

    Args:
        requests: List of batch request objects
        temp_dir: Directory to store temporary files
        api_key: OpenAI API key
        input_index: Optional index to include in filename for interpretability

    Returns:
        File ID from OpenAI
    """
    # Create JSONL file
    timestamp = int(time.time())
    if input_index is not None:
        input_file = temp_dir / f"batch_input_{input_index}_{timestamp}.jsonl"
    else:
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


def download_batch_results(output_file_id: str, temp_dir: Path, api_key: str, input_index: Optional[int] = None) -> Path:
    """
    Download batch results from OpenAI.

    Args:
        output_file_id: ID of output file
        temp_dir: Directory to store downloaded file
        api_key: OpenAI API key
        input_index: Optional dataset index to include in filename for correct mapping

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

    # Save to file with index in filename if provided
    timestamp = int(time.time())
    if input_index is not None:
        output_file = temp_dir / f"batch_output_{input_index}_{timestamp}.jsonl"
    else:
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


def process_existing_batch_results(batch_output_file: Path, output_dir: Path = None) -> Dict:
    """
    Process existing batch results into metric format.

    This is useful for re-processing batch results without re-running evaluations.
    Originally from process_existing_results.py.

    Args:
        batch_output_file: Path to the batch output JSONL file
        output_dir: Optional directory to save individual metric JSON files

    Returns:
        Dict with parsed metrics and summary score
    """
    from evals.batch_evals.batch_metrics import parse_batch_results
    from evals.metrics import calculate_summary_score

    print("=" * 70)
    print("PROCESSING EXISTING BATCH RESULTS")
    print("=" * 70)
    print(f"\nLoading results from: {batch_output_file}")

    results = load_batch_results(batch_output_file)
    print(f"✅ Loaded {len(results)} results")

    # Parse results
    print("\n📊 Parsing results...")
    parsed = parse_batch_results(results)

    accuracy_result = parsed["accuracy_result"]
    completeness_result = parsed["completeness_result"]
    consistency_result = parsed["consistency_result"]
    quality_result = parsed["quality_result"]

    # Calculate summary score
    summary_result = calculate_summary_score(
        accuracy_result=accuracy_result,
        completeness_result=completeness_result,
        consistency_result=consistency_result,
        quality_result=quality_result
    )

    # Save results to JSON files if output_dir provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        import json

        accuracy_output = output_dir / "batch_accuracy_results.json"
        completeness_output = output_dir / "batch_completeness_results.json"
        consistency_output = output_dir / "batch_consistency_results.json"
        quality_output = output_dir / "batch_quality_results.json"
        summary_output = output_dir / "batch_summary_results.json"

        with open(accuracy_output, "w") as f:
            json.dump(accuracy_result, f, indent=2)

        with open(completeness_output, "w") as f:
            json.dump(completeness_result, f, indent=2)

        with open(consistency_output, "w") as f:
            json.dump(consistency_result, f, indent=2)

        with open(quality_output, "w") as f:
            json.dump(quality_result, f, indent=2)

        with open(summary_output, "w") as f:
            json.dump(summary_result, f, indent=2)

        print("\n" + "=" * 70)
        print("✅ Results saved to:")
        print(f"   {accuracy_output}")
        print(f"   {completeness_output}")
        print(f"   {consistency_output}")
        print(f"   {quality_output}")
        print(f"   {summary_output}")
        print("=" * 70)

    print(f"\n{'=' * 70}")
    print(f"✅ FINAL SCORE: {summary_result['summary_score']:.2f}/100")
    print(f"{'=' * 70}\n")

    return {
        "accuracy": accuracy_result,
        "completeness": completeness_result,
        "consistency": consistency_result,
        "quality": quality_result,
        "summary": summary_result
    }


def evaluate_memo_batch(
    memo: str,
    source_document: str,
    template: str = None,
    model: str = "gpt-5",
    weights: Dict[str, float] = None,
    poll_interval: int = 60,
    api_key: str = None,
    temp_dir: Path = None
) -> float:
    """
    Evaluate a single memo using Batch API for faster processing.

    Supports GPT-5 (OpenAI) and Claude (Anthropic) models.
    Originally from evaluator_batch.py - moved here for consolidation.

    Args:
        memo: Generated investment memo text
        source_document: Original credit agreement text
        template: Optional template for structure evaluation
        model: Model identifier (e.g., 'gpt-5', 'claude-sonnet-4-20250514')
        weights: Optional weights for summary score (default: equal 0.25 each)
        poll_interval: Seconds between status checks (default: 60)
        api_key: Optional API key (defaults to OPENAI_API_KEY or ANTHROPIC_API_KEY)
        temp_dir: Optional temp directory (defaults to batch_temp in module dir)

    Returns:
        float: Summary score (0-100)
    """
    from evals.batch_evals.batch_metrics import (
        create_batch_requests_for_memo,
        create_claude_batch_requests_for_memo,
        create_gemini_batch_requests_for_memo,
        parse_batch_results,
        parse_claude_batch_results,
        parse_gemini_batch_results
    )
    from evals.metrics import calculate_summary_score

    # Determine provider
    is_claude = "claude" in model.lower()
    is_gpt = model.startswith("gpt")
    is_gemini = "gemini" in model.lower()

    if not (is_claude or is_gpt or is_gemini):
        raise ValueError(
            f"Unsupported model: {model}\n"
            "Supported: gpt-5, claude-sonnet-4-20250514, gemini-2.0-flash-exp, etc."
        )

    # Get API key
    if api_key is None:
        if is_gpt:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment")
        elif is_claude:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not found in environment")
        elif is_gemini:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found in environment")

    # Set temp dir
    if temp_dir is None:
        temp_dir = Path(__file__).parent / "batch_temp"
        temp_dir.mkdir(exist_ok=True)

    print(f"\n{'='*70}")
    print(f"BATCH EVALUATION: {model}")
    print(f"{'='*70}")
    print(f"Memo length: {len(memo)} chars")
    print(f"Source doc length: {len(source_document)} chars")
    print(f"Template provided: {'Yes' if template else 'No'}")
    print(f"{'='*70}\n")

    # Create batch requests based on provider
    if is_gpt:
        requests = create_batch_requests_for_memo(
            memo=memo,
            source_document=source_document,
            template=template,
            model=model
        )
    elif is_claude:
        requests = create_claude_batch_requests_for_memo(
            memo=memo,
            source_document=source_document,
            template=template,
            model=model
        )
    else:  # Gemini
        requests = create_gemini_batch_requests_for_memo(
            memo=memo,
            source_document=source_document,
            template=template,
            model=model
        )

    print(f"📦 Created {len(requests)} batch requests:")
    print(f"   - 1 accuracy evaluation")
    print(f"   - 1 completeness evaluation")
    print(f"   - 1 consistency evaluation")
    print(f"   - 4 quality sub-metrics (clarity, tone, length, structure)")
    print()

    # Submit batch and wait for results based on provider
    if is_gpt:
        results = submit_and_wait_for_batch(
            requests=requests,
            api_key=api_key,
            temp_dir=temp_dir,
            description=f"Memo evaluation - {len(memo)} chars",
            poll_interval=poll_interval
        )
    elif is_claude:
        results = submit_and_wait_for_claude_batch(
            requests=requests,
            api_key=api_key,
            temp_dir=temp_dir,
            poll_interval=poll_interval
        )
    else:  # Gemini
        results = submit_and_wait_for_gemini_batch(
            requests=requests,
            api_key=api_key,
            temp_dir=temp_dir,
            model=model,
            poll_interval=poll_interval
        )

    # Parse results into metric format based on provider
    print("\n📊 Parsing results...")
    if is_gpt:
        parsed = parse_batch_results(results)
    elif is_claude:
        parsed = parse_claude_batch_results(results)
    else:  # Gemini
        parsed = parse_gemini_batch_results(results)

    accuracy_result = parsed["accuracy_result"]
    completeness_result = parsed["completeness_result"]
    consistency_result = parsed["consistency_result"]
    quality_result = parsed["quality_result"]

    # Print metric summaries
    print(f"\n{'='*70}")
    print("METRIC RESULTS")
    print(f"{'='*70}")
    print(f"✓ Accuracy:     {accuracy_result['score']*100:.1f}/100 "
          f"({'No hallucinations' if accuracy_result['accurate'] else 'Hallucinations detected'})")
    print(f"✓ Completeness: {completeness_result['score']*100:.1f}/100 "
          f"({'Complete' if completeness_result['complete'] else 'Missing terms'})")
    print(f"✓ Consistency:  {consistency_result['score']*100:.1f}/100 "
          f"({'Consistent' if consistency_result['consistent'] else 'Has contradictions'})")
    print(f"✓ Quality:      {quality_result['quality_score']:.1f}/100")
    print(f"  - Clarity:    {quality_result['clarity_score']:.1f}/100")
    print(f"  - Tone:       {quality_result['tone_score']:.1f}/100")
    print(f"  - Length:     {quality_result['length_score']:.1f}/100")
    print(f"  - Structure:  {quality_result['structure_score']:.1f}/100")
    print(f"{'='*70}\n")

    # Calculate summary score
    summary_result = calculate_summary_score(
        accuracy_result=accuracy_result,
        completeness_result=completeness_result,
        consistency_result=consistency_result,
        quality_result=quality_result,
        weights=weights
    )

    summary_score = summary_result["summary_score"]

    print(f"{'='*70}")
    print(f"SUMMARY SCORE: {summary_score:.2f}/100")
    print(f"{'='*70}\n")

    return summary_score


def evaluate_memo_batch_with_all_models(
    memo: str,
    source_document: str,
    template: str = None,
    weights: Dict[str, float] = None,
    poll_interval: int = 60,
    parallel: bool = True
) -> Dict[str, float]:
    """
    Evaluate a memo using all available models (GPT-5, Claude, Gemini) via batch APIs.

    All three providers are now fully implemented.
    Originally from evaluator_batch.py - moved here for consolidation.

    Args:
        memo: Generated investment memo text
        source_document: Original credit agreement text
        template: Optional template for structure evaluation
        weights: Optional weights for summary score (default: equal 0.25 each)
        poll_interval: Seconds between status checks (default: 60)
        parallel: If True, submit all batches then poll them in parallel (default: True)

    Returns:
        Dict mapping model name to summary score
    """
    from evals.batch_evals.batch_metrics import (
        create_batch_requests_for_memo,
        create_claude_batch_requests_for_memo,
        create_gemini_batch_requests_for_memo,
        parse_batch_results,
        parse_claude_batch_results,
        parse_gemini_batch_results
    )
    from evals.metrics import calculate_summary_score
    import concurrent.futures

    results = {}

    if not parallel:
        # Sequential execution (original behavior)
        # GPT-5
        print("\n" + "="*70)
        print("EVALUATING WITH: GPT-5 (Batch API)")
        print("="*70)
        try:
            results["gpt-5"] = evaluate_memo_batch(
                memo=memo,
                source_document=source_document,
                template=template,
                model="gpt-5",
                weights=weights,
                poll_interval=poll_interval
            )
        except Exception as e:
            print(f"⚠️  GPT-5 evaluation failed: {e}")

        # Claude
        print("\n" + "="*70)
        print("EVALUATING WITH: Claude Sonnet 4 (Batch API)")
        print("="*70)
        try:
            results["claude-sonnet-4-20250514"] = evaluate_memo_batch(
                memo=memo,
                source_document=source_document,
                template=template,
                model="claude-sonnet-4-20250514",
                weights=weights,
                poll_interval=poll_interval
            )
        except Exception as e:
            print(f"⚠️  Claude evaluation failed: {e}")

        # Gemini
        print("\n" + "="*70)
        print("EVALUATING WITH: Gemini 2.5 Pro (Batch API)")
        print("="*70)
        try:
            results["gemini-2.5-pro"] = evaluate_memo_batch(
                memo=memo,
                source_document=source_document,
                template=template,
                model="gemini-2.5-pro",
                weights=weights,
                poll_interval=poll_interval
            )
        except Exception as e:
            print(f"⚠️  Gemini evaluation failed: {e}")
    else:
        # Parallel execution - submit all batches first, then wait
        print("\n" + "="*70)
        print("PARALLEL BATCH EVALUATION")
        print("="*70)
        print("Submitting all batch jobs simultaneously...")
        print()

        batch_jobs = {}
        temp_dir = Path(__file__).parent / "batch_temp"
        temp_dir.mkdir(exist_ok=True)

        # Submit GPT-5 batch
        try:
            gpt_api_key = os.getenv("OPENAI_API_KEY")
            if gpt_api_key:
                print("📤 Submitting GPT-5 batch...")
                gpt_requests = create_batch_requests_for_memo(memo, source_document, template, "gpt-5")
                gpt_batch_id = create_batch(gpt_requests, gpt_api_key, temp_dir, "GPT-5 evaluation")
                batch_jobs["gpt-5"] = {
                    "provider": "openai",
                    "batch_id": gpt_batch_id,
                    "api_key": gpt_api_key,
                    "parser": parse_batch_results
                }
                print(f"   ✓ GPT-5 batch created: {gpt_batch_id}")
        except Exception as e:
            print(f"   ✗ GPT-5 batch submission failed: {e}")

        # Submit Claude batch
        try:
            claude_api_key = os.getenv("ANTHROPIC_API_KEY")
            if claude_api_key:
                print("📤 Submitting Claude batch...")
                claude_requests = create_claude_batch_requests_for_memo(memo, source_document, template, "claude-sonnet-4-20250514")
                claude_batch_id = create_claude_batch(claude_requests, claude_api_key)
                batch_jobs["claude-sonnet-4-20250514"] = {
                    "provider": "anthropic",
                    "batch_id": claude_batch_id,
                    "api_key": claude_api_key,
                    "parser": parse_claude_batch_results
                }
                print(f"   ✓ Claude batch created: {claude_batch_id}")
        except Exception as e:
            print(f"   ✗ Claude batch submission failed: {e}")

        # Submit Gemini batch
        try:
            gemini_api_key = os.getenv("GEMINI_API_KEY")
            if gemini_api_key:
                print("📤 Submitting Gemini batch...")
                gemini_requests = create_gemini_batch_requests_for_memo(memo, source_document, template, "gemini-2.5-pro")
                gemini_batch_name = create_gemini_batch(gemini_requests, gemini_api_key, "gemini-2.5-pro")
                batch_jobs["gemini-2.5-pro"] = {
                    "provider": "google",
                    "batch_id": gemini_batch_name,
                    "api_key": gemini_api_key,
                    "parser": parse_gemini_batch_results
                }
                print(f"   ✓ Gemini batch created: {gemini_batch_name}")
        except Exception as e:
            print(f"   ✗ Gemini batch submission failed: {e}")

        print(f"\n✓ Submitted {len(batch_jobs)} batch jobs")
        print("⏳ Now polling all batches until completion...")
        print()

        # Define polling function for each provider
        def poll_batch(model_name, job_info):
            try:
                if job_info["provider"] == "openai":
                    print(f"[{model_name}] Polling OpenAI batch...")
                    raw_results = poll_batch_until_complete(
                        job_info["batch_id"],
                        job_info["api_key"],
                        temp_dir,
                        poll_interval
                    )
                elif job_info["provider"] == "anthropic":
                    print(f"[{model_name}] Polling Claude batch...")
                    raw_results = poll_claude_batch_until_complete(
                        job_info["batch_id"],
                        job_info["api_key"],
                        temp_dir,
                        poll_interval
                    )
                else:  # google
                    print(f"[{model_name}] Polling Gemini batch...")
                    raw_results = poll_gemini_batch_until_complete(
                        job_info["batch_id"],
                        job_info["api_key"],
                        temp_dir,
                        poll_interval
                    )

                # Parse results
                parsed = job_info["parser"](raw_results)

                # Calculate summary score
                summary_result = calculate_summary_score(
                    accuracy_result=parsed["accuracy_result"],
                    completeness_result=parsed["completeness_result"],
                    consistency_result=parsed["consistency_result"],
                    quality_result=parsed["quality_result"],
                    weights=weights
                )

                score = summary_result["summary_score"]
                print(f"[{model_name}] ✓ Complete! Score: {score:.2f}/100")
                return (model_name, score)

            except Exception as e:
                print(f"[{model_name}] ✗ Failed: {e}")
                return (model_name, None)

        # Poll all batches in parallel using ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_to_model = {
                executor.submit(poll_batch, model, info): model
                for model, info in batch_jobs.items()
            }

            for future in concurrent.futures.as_completed(future_to_model):
                model_name, score = future.result()
                if score is not None:
                    results[model_name] = score

    # Print summary
    print("\n" + "="*70)
    print("EVALUATION COMPLETE - ALL MODELS")
    print("="*70)
    for model, score in results.items():
        print(f"{model:40s} {score:.2f}/100")
    print("="*70 + "\n")

    return results


def resume_batch_evaluation(batch_id: str, poll_interval: int = 60, api_key: str = None, temp_dir: Path = None) -> List[Dict]:
    """
    Resume monitoring a batch evaluation that was interrupted.

    Use this if you closed your computer or stopped the script while a batch
    was running. The batch continues running on OpenAI's servers, and this
    function will check its status and download results when ready.

    Originally from evaluator_batch.py - moved here for consolidation.

    Args:
        batch_id: Batch job ID (printed when batch was started)
        poll_interval: Seconds between status checks (default: 60)
        api_key: Optional OpenAI API key (defaults to OPENAI_API_KEY env var)
        temp_dir: Optional temp directory (defaults to batch_temp in module dir)

    Returns:
        List of raw batch results

    Example:
        >>> # If you see "Batch job created: batch_abc123" in logs
        >>> results = resume_batch_evaluation("batch_abc123")
        >>> # Process results with parse_batch_results() if needed
    """
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment")

    if temp_dir is None:
        temp_dir = Path(__file__).parent / "batch_temp"
        temp_dir.mkdir(exist_ok=True)

    return resume_batch_job(
        batch_id=batch_id,
        api_key=api_key,
        temp_dir=temp_dir,
        poll_interval=poll_interval
    )


# ============================================================================
# CLAUDE (ANTHROPIC) BATCH API FUNCTIONS
# ============================================================================

def create_claude_batch(requests: List[Dict], api_key: str) -> str:
    """
    Create a batch job on Anthropic API.

    Args:
        requests: List of batch request objects in Anthropic format
        api_key: Anthropic API key

    Returns:
        Batch job ID
    """
    payload = {"requests": requests}

    cmd = [
        "curl",
        "-sS",
        "-X",
        "POST",
        "https://api.anthropic.com/v1/messages/batches",
        "-H",
        f"x-api-key: {api_key}",
        "-H",
        "anthropic-version: 2023-06-01",
        "-H",
        "Content-Type: application/json",
        "--data-binary",
        "@-"
    ]

    response = run_curl(cmd, stdin_bytes=json.dumps(payload).encode("utf-8"))
    batch_data = json.loads(response)

    if "id" not in batch_data:
        raise RuntimeError(f"Failed to create Claude batch: {response}")

    batch_id = batch_data["id"]
    print(f"🚀 Claude batch job created: {batch_id}")
    print(f"   Status: {batch_data.get('processing_status', 'unknown')}")
    print(f"   Requests: {len(requests)}")

    return batch_id


def check_claude_batch_status(batch_id: str, api_key: str) -> Dict:
    """
    Check the status of a Claude batch job.

    Args:
        batch_id: Batch job ID
        api_key: Anthropic API key

    Returns:
        Batch status information
    """
    cmd = [
        "curl",
        "-sS",
        "-X",
        "GET",
        f"https://api.anthropic.com/v1/messages/batches/{batch_id}",
        "-H",
        f"x-api-key: {api_key}",
        "-H",
        "anthropic-version: 2023-06-01"
    ]

    response = run_curl(cmd)
    return json.loads(response)


def download_claude_batch_results(results_url: str, temp_dir: Path, api_key: str, input_index: Optional[int] = None) -> Path:
    """
    Download Claude batch results from results URL.

    Args:
        results_url: URL to download results from
        temp_dir: Directory to store downloaded file
        api_key: Anthropic API key
        input_index: Optional dataset index to include in filename for correct mapping

    Returns:
        Path to downloaded results file
    """
    cmd = [
        "curl",
        "-sS",
        "-X",
        "GET",
        results_url,
        "-H",
        f"x-api-key: {api_key}",
        "-H",
        "anthropic-version: 2023-06-01"
    ]

    response = run_curl(cmd)

    # Save to file with index in filename if provided
    timestamp = int(time.time())
    if input_index is not None:
        output_file = temp_dir / f"claude_batch_output_{input_index}_{timestamp}.jsonl"
    else:
        output_file = temp_dir / f"claude_batch_output_{timestamp}.jsonl"

    with open(output_file, "w") as f:
        f.write(response)

    print(f"📥 Downloaded Claude results to: {output_file}")

    return output_file


def poll_claude_batch_until_complete(
    batch_id: str,
    api_key: str,
    temp_dir: Path,
    poll_interval: int = 60,
    max_wait_time: int = 86400  # 24 hours
) -> List[Dict]:
    """
    Poll a Claude batch job until completion and return results.

    Args:
        batch_id: Batch job ID
        api_key: Anthropic API key
        temp_dir: Directory for temporary files
        poll_interval: Seconds between status checks (default: 60)
        max_wait_time: Maximum seconds to wait (default: 24 hours)

    Returns:
        List of result objects

    Raises:
        TimeoutError: If max_wait_time exceeded
        RuntimeError: If batch job fails
    """
    print(f"\n⏳ Polling Claude batch job: {batch_id}")
    print(f"   Checking every {poll_interval} seconds")
    print(f"   You can safely close this and resume later\n")

    # Save batch ID for resuming
    batch_id_file = temp_dir / f"claude_batch_{batch_id}.json"
    with open(batch_id_file, "w") as f:
        json.dump({
            "batch_id": batch_id,
            "created_at": time.time(),
            "status": "polling",
            "provider": "anthropic"
        }, f, indent=2)

    start_time = time.time()
    last_status = None

    while True:
        elapsed = time.time() - start_time

        if elapsed > max_wait_time:
            raise TimeoutError(f"Claude batch job exceeded max wait time of {max_wait_time}s")

        # Check status
        status_data = check_claude_batch_status(batch_id, api_key)
        current_status = status_data.get("processing_status")

        # Print status update if changed
        if current_status != last_status:
            print(f"[{int(elapsed)}s] Status: {current_status}")

            # Show progress if available
            request_counts = status_data.get("request_counts", {})
            if request_counts:
                processing = request_counts.get("processing", 0)
                succeeded = request_counts.get("succeeded", 0)
                errored = request_counts.get("errored", 0)
                total = processing + succeeded + errored
                if total > 0:
                    print(f"         Progress: {succeeded}/{total} succeeded, {errored} errored")

            last_status = current_status

        # Check if completed
        if current_status == "ended":
            print(f"✅ Claude batch job completed in {int(elapsed)}s!")

            # Download results
            results_url = status_data.get("results_url")
            if not results_url:
                raise RuntimeError("Claude batch completed but no results_url found")

            output_file = download_claude_batch_results(results_url, temp_dir, api_key)
            results = load_batch_results(output_file)

            # Update batch ID file
            with open(batch_id_file, "w") as f:
                json.dump({
                    "batch_id": batch_id,
                    "created_at": start_time,
                    "completed_at": time.time(),
                    "status": "completed",
                    "output_file": str(output_file),
                    "provider": "anthropic"
                }, f, indent=2)

            return results

        # Wait before next check
        time.sleep(poll_interval)


def submit_and_wait_for_claude_batch(
    requests: List[Dict],
    api_key: str,
    temp_dir: Path,
    poll_interval: int = 60
) -> List[Dict]:
    """
    Complete Claude batch workflow: create job, poll, and return results.

    Args:
        requests: List of batch request objects in Anthropic format
        api_key: Anthropic API key
        temp_dir: Directory for temporary files
        poll_interval: Seconds between status checks (default: 60)

    Returns:
        List of result objects
    """
    print("=" * 70)
    print("CLAUDE BATCH API EVALUATION")
    print("=" * 70)
    print(f"   Total requests: {len(requests)}")

    # Create batch job
    batch_id = create_claude_batch(requests, api_key)

    # Poll until complete
    results = poll_claude_batch_until_complete(batch_id, api_key, temp_dir, poll_interval)

    print("=" * 70)
    print(f"CLAUDE BATCH COMPLETE: {len(results)} results received")
    print("=" * 70)

    return results


def resume_claude_batch_job(batch_id: str, api_key: str, temp_dir: Path, poll_interval: int = 60) -> List[Dict]:
    """
    Resume monitoring a Claude batch job that was previously started.

    Args:
        batch_id: Batch job ID to resume
        api_key: Anthropic API key
        temp_dir: Directory for temporary files
        poll_interval: Seconds between status checks (default: 60)

    Returns:
        List of result objects
    """
    print(f"🔄 Resuming Claude batch job: {batch_id}")

    # Check if already completed
    batch_id_file = temp_dir / f"claude_batch_{batch_id}.json"
    if batch_id_file.exists():
        with open(batch_id_file, "r") as f:
            batch_info = json.load(f)

        if batch_info.get("status") == "completed":
            output_file = Path(batch_info.get("output_file"))
            if output_file.exists():
                print(f"✅ Claude batch already completed! Loading results from: {output_file}")
                return load_batch_results(output_file)

    # Otherwise, poll until complete
    return poll_claude_batch_until_complete(batch_id, api_key, temp_dir, poll_interval)


# ============================================================================
# GEMINI (GOOGLE) BATCH API FUNCTIONS
# ============================================================================

def create_gemini_batch(requests: List[Dict], api_key: str, model: str = "gemini-2.0-flash-exp") -> str:
    """
    Create a batch job on Google Gemini API.

    Args:
        requests: List of batch request objects in Gemini format
        api_key: Google Gemini API key
        model: Model name (default: gemini-2.0-flash-exp)

    Returns:
        Batch job name (full resource name)
    """
    # Format requests for inline submission
    formatted_requests = []
    for req in requests:
        formatted_requests.append({
            "request": req["request"],
            "metadata": {"key": req["custom_id"]}
        })

    payload = {
        "batch": {
            "display_name": f"memo-evaluation-{int(time.time())}",
            "input_config": {
                "requests": {
                    "requests": formatted_requests
                }
            }
        }
    }

    cmd = [
        "curl",
        "-sS",
        "-X",
        "POST",
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:batchGenerateContent",
        "-H",
        f"x-goog-api-key: {api_key}",
        "-H",
        "Content-Type: application/json",
        "--data-binary",
        "@-"
    ]

    response = run_curl(cmd, stdin_bytes=json.dumps(payload).encode("utf-8"))
    batch_data = json.loads(response)

    if "name" not in batch_data:
        raise RuntimeError(f"Failed to create Gemini batch: {response}")

    batch_name = batch_data["name"]
    print(f"🚀 Gemini batch job created: {batch_name}")
    print(f"   State: {batch_data.get('state', 'unknown')}")
    print(f"   Requests: {len(requests)}")

    return batch_name


def check_gemini_batch_status(batch_name: str, api_key: str) -> Dict:
    """
    Check the status of a Gemini batch job.

    Args:
        batch_name: Batch job name (full resource name)
        api_key: Google Gemini API key

    Returns:
        Batch status information
    """
    cmd = [
        "curl",
        "-sS",
        "-X",
        "GET",
        f"https://generativelanguage.googleapis.com/v1beta/{batch_name}",
        "-H",
        f"x-goog-api-key: {api_key}"
    ]

    response = run_curl(cmd)
    return json.loads(response)


def extract_gemini_batch_results(status_data: Dict, temp_dir: Path, input_index: Optional[int] = None) -> Path:
    """
    Extract Gemini batch results from status response.

    For inline requests, results are in the status response itself.

    Args:
        status_data: Status response from Gemini API
        temp_dir: Directory to store extracted results
        input_index: Optional dataset index to include in filename for correct mapping

    Returns:
        Path to extracted results file in JSONL format
    """
    # Extract results from inline responses
    # Gemini format: metadata.output.inlinedResponses.inlinedResponses
    metadata = status_data.get("metadata", {})
    output = metadata.get("output", {})
    inlined_container = output.get("inlinedResponses", {})
    inlined_responses = inlined_container.get("inlinedResponses", [])

    # Fallback to old format if new format not found
    if not inlined_responses:
        dest = status_data.get("dest", {})
        inlined_responses = dest.get("inlinedResponses", [])

    if not inlined_responses:
        raise RuntimeError("No results found in batch response")

    # Convert to JSONL format matching our parser expectations
    timestamp = int(time.time())
    if input_index is not None:
        output_file = temp_dir / f"gemini_batch_output_{input_index}_{timestamp}.jsonl"
    else:
        output_file = temp_dir / f"gemini_batch_output_{timestamp}.jsonl"

    with open(output_file, "w") as f:
        for response_item in inlined_responses:
            # Extract metadata and response
            metadata = response_item.get("metadata", {})
            custom_id = metadata.get("key", "unknown")

            response_data = response_item.get("response", {})

            # Format as JSONL entry
            result_entry = {
                "custom_id": custom_id,
                "response": response_data
            }
            f.write(json.dumps(result_entry) + "\n")

    print(f"📥 Extracted Gemini results to: {output_file}")

    return output_file


def poll_gemini_batch_until_complete(
    batch_name: str,
    api_key: str,
    temp_dir: Path,
    poll_interval: int = 60,
    max_wait_time: int = 86400  # 24 hours
) -> List[Dict]:
    """
    Poll a Gemini batch job until completion and return results.

    Args:
        batch_name: Batch job name (full resource name)
        api_key: Google Gemini API key
        temp_dir: Directory for temporary files
        poll_interval: Seconds between status checks (default: 60)
        max_wait_time: Maximum seconds to wait (default: 24 hours)

    Returns:
        List of result objects

    Raises:
        TimeoutError: If max_wait_time exceeded
        RuntimeError: If batch job fails
    """
    print(f"\n⏳ Polling Gemini batch job: {batch_name}")
    print(f"   Checking every {poll_interval} seconds")
    print(f"   You can safely close this and resume later\n")

    # Save batch name for resuming
    batch_id_file = temp_dir / f"gemini_batch_{batch_name.split('/')[-1]}.json"
    with open(batch_id_file, "w") as f:
        json.dump({
            "batch_name": batch_name,
            "created_at": time.time(),
            "status": "polling",
            "provider": "google"
        }, f, indent=2)

    start_time = time.time()
    last_state = None

    while True:
        elapsed = time.time() - start_time

        if elapsed > max_wait_time:
            raise TimeoutError(f"Gemini batch job exceeded max wait time of {max_wait_time}s")

        # Check status
        status_data = check_gemini_batch_status(batch_name, api_key)
        current_state = status_data.get("state")

        # Print status update if changed
        if current_state != last_state:
            print(f"[{int(elapsed)}s] State: {current_state}")

            # Show progress if available
            if "completedCount" in status_data:
                completed = status_data.get("completedCount", 0)
                total = status_data.get("requestCount", 0)
                if total > 0:
                    print(f"         Progress: {completed}/{total} completed")

            last_state = current_state

        # Check if completed
        if current_state == "JOB_STATE_SUCCEEDED":
            print(f"✅ Gemini batch job completed in {int(elapsed)}s!")

            # Extract results
            output_file = extract_gemini_batch_results(status_data, temp_dir)
            results = load_batch_results(output_file)

            # Update batch ID file
            with open(batch_id_file, "w") as f:
                json.dump({
                    "batch_name": batch_name,
                    "created_at": start_time,
                    "completed_at": time.time(),
                    "status": "completed",
                    "output_file": str(output_file),
                    "provider": "google"
                }, f, indent=2)

            return results

        elif current_state == "JOB_STATE_FAILED":
            error_msg = status_data.get("error", {}).get("message", "Unknown error")
            raise RuntimeError(f"Gemini batch job failed: {error_msg}")

        elif current_state == "JOB_STATE_CANCELLED":
            raise RuntimeError("Gemini batch job was cancelled")

        elif current_state == "JOB_STATE_EXPIRED":
            raise RuntimeError("Gemini batch job expired (24 hour limit reached)")

        # Wait before next check
        time.sleep(poll_interval)


def submit_and_wait_for_gemini_batch(
    requests: List[Dict],
    api_key: str,
    temp_dir: Path,
    model: str = "gemini-2.0-flash-exp",
    poll_interval: int = 60
) -> List[Dict]:
    """
    Complete Gemini batch workflow: create job, poll, and return results.

    Args:
        requests: List of batch request objects in Gemini format
        api_key: Google Gemini API key
        temp_dir: Directory for temporary files
        model: Model name (default: gemini-2.0-flash-exp)
        poll_interval: Seconds between status checks (default: 60)

    Returns:
        List of result objects
    """
    print("=" * 70)
    print("GEMINI BATCH API EVALUATION")
    print("=" * 70)
    print(f"   Total requests: {len(requests)}")
    print(f"   Model: {model}")

    # Create batch job
    batch_name = create_gemini_batch(requests, api_key, model)

    # Poll until complete
    results = poll_gemini_batch_until_complete(batch_name, api_key, temp_dir, poll_interval)

    print("=" * 70)
    print(f"GEMINI BATCH COMPLETE: {len(results)} results received")
    print("=" * 70)

    return results


def resume_gemini_batch_job(batch_name: str, api_key: str, temp_dir: Path, poll_interval: int = 60) -> List[Dict]:
    """
    Resume monitoring a Gemini batch job that was previously started.

    Args:
        batch_name: Batch job name (full resource name) to resume
        api_key: Google Gemini API key
        temp_dir: Directory for temporary files
        poll_interval: Seconds between status checks (default: 60)

    Returns:
        List of result objects
    """
    print(f"🔄 Resuming Gemini batch job: {batch_name}")

    # Check if already completed
    batch_id_file = temp_dir / f"gemini_batch_{batch_name.split('/')[-1]}.json"
    if batch_id_file.exists():
        with open(batch_id_file, "r") as f:
            batch_info = json.load(f)

        if batch_info.get("status") == "completed":
            output_file = Path(batch_info.get("output_file"))
            if output_file.exists():
                print(f"✅ Gemini batch already completed! Loading results from: {output_file}")
                return load_batch_results(output_file)

    # Otherwise, poll until complete
    return poll_gemini_batch_until_complete(batch_name, api_key, temp_dir, poll_interval)
