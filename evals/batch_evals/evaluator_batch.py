"""
Batch-enabled evaluation harness for investment memo generation.

This module provides batch API versions of the evaluation functions,
optimized to process multiple metrics in a single batch job for faster
evaluation with GPT-5.

Key differences from original evaluator.py:
- Uses OpenAI Batch API instead of sequential API calls
- Combines all metrics (accuracy, completeness, consistency, quality) into one batch
- Supports resuming interrupted batch jobs
- Only implements GPT-5 evaluation (Claude and Gemini to be added later)
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

from evals.batch_evals.batch_utils import (
    submit_and_wait_for_batch,
    resume_batch_job
)
from evals.batch_evals.metrics_batch import (
    create_batch_requests_for_memo,
    parse_batch_results
)
from evals.metrics import calculate_summary_score


# Directory for temporary batch files
BATCH_TEMP_DIR = Path(__file__).parent / "batch_temp"
BATCH_TEMP_DIR.mkdir(exist_ok=True)


def evaluate_memo_batch(
    memo: str,
    source_document: str,
    template: str = None,
    model: str = "gpt-5",
    weights: Dict[str, float] = None,
    poll_interval: int = 60
) -> float:
    """
    Evaluate a single memo using OpenAI Batch API for faster processing.

    This function works identically to evaluate_memo() but uses the Batch API
    to combine all metric evaluations into a single batch job, reducing
    total evaluation time from ~2-3 minutes to whatever the batch queue time is.

    Key benefits:
    - All metrics evaluated in parallel on OpenAI's servers
    - Can close computer during processing - batch runs on OpenAI side
    - Resume monitoring with resume_batch_job() if interrupted
    - Same output format as original evaluate_memo()

    Args:
        memo: Generated investment memo text
        source_document: Original credit agreement text
        template: Optional template for structure evaluation
        model: Model identifier (default: gpt-5, currently only GPT-5 supported)
        weights: Optional weights for summary score (default: equal 0.25 each)
        poll_interval: Seconds between status checks (default: 60)

    Returns:
        float: Summary score (0-100)

    Raises:
        ValueError: If model is not gpt-5 (other models not yet implemented)
        RuntimeError: If batch job fails
        TimeoutError: If batch exceeds 24 hour window
    """
    # Currently only GPT-5 is implemented for batch processing
    if not model.startswith("gpt-5"):
        raise ValueError(
            f"Batch evaluation currently only supports gpt-5, got: {model}\n"
            "Claude and Gemini batch support coming soon."
        )

    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment")

    print(f"\n{'='*70}")
    print(f"BATCH EVALUATION: {model}")
    print(f"{'='*70}")
    print(f"Memo length: {len(memo)} chars")
    print(f"Source doc length: {len(source_document)} chars")
    print(f"Template provided: {'Yes' if template else 'No'}")
    print(f"{'='*70}\n")

    # Create batch requests (7 total: accuracy, completeness, consistency, + 4 quality sub-metrics)
    requests = create_batch_requests_for_memo(
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

    # Submit batch and wait for results
    results = submit_and_wait_for_batch(
        requests=requests,
        api_key=api_key,
        temp_dir=BATCH_TEMP_DIR,
        description=f"Memo evaluation - {len(memo)} chars",
        poll_interval=poll_interval
    )

    # Parse results into metric format
    print("\n📊 Parsing results...")
    parsed = parse_batch_results(results)

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
    poll_interval: int = 60
) -> Dict[str, float]:
    """
    Evaluate a memo using all available models (GPT-5, Claude, Gemini) via batch APIs.

    NOTE: Currently only GPT-5 is implemented. Claude and Gemini coming soon.

    Args:
        memo: Generated investment memo text
        source_document: Original credit agreement text
        template: Optional template for structure evaluation
        weights: Optional weights for summary score (default: equal 0.25 each)
        poll_interval: Seconds between status checks (default: 60)

    Returns:
        Dict mapping model name to summary score
    """
    results = {}

    # GPT-5 (implemented)
    print("\n" + "="*70)
    print("EVALUATING WITH: GPT-5 (Batch API)")
    print("="*70)
    results["gpt-5"] = evaluate_memo_batch(
        memo=memo,
        source_document=source_document,
        template=template,
        model="gpt-5",
        weights=weights,
        poll_interval=poll_interval
    )

    # Claude (not yet implemented)
    print("\n" + "="*70)
    print("EVALUATING WITH: Claude (Batch API)")
    print("="*70)
    print("⚠️  Claude batch API not yet implemented - skipping")
    # results["claude-sonnet-4-20250514"] = evaluate_memo_batch(
    #     memo=memo,
    #     source_document=source_document,
    #     template=template,
    #     model="claude-sonnet-4-20250514",
    #     weights=weights,
    #     poll_interval=poll_interval
    # )

    # Gemini (not yet implemented)
    print("\n" + "="*70)
    print("EVALUATING WITH: Gemini (Batch API)")
    print("="*70)
    print("⚠️  Gemini batch API not yet implemented - skipping")
    # results["gemini-2.5-pro"] = evaluate_memo_batch(
    #     memo=memo,
    #     source_document=source_document,
    #     template=template,
    #     model="gemini-2.5-pro",
    #     weights=weights,
    #     poll_interval=poll_interval
    # )

    # Print summary
    print("\n" + "="*70)
    print("EVALUATION COMPLETE - ALL MODELS")
    print("="*70)
    for model, score in results.items():
        print(f"{model:40s} {score:.2f}/100")
    print("="*70 + "\n")

    return results


def resume_batch_evaluation(batch_id: str, poll_interval: int = 60) -> List[Dict]:
    """
    Resume monitoring a batch evaluation that was interrupted.

    Use this if you closed your computer or stopped the script while a batch
    was running. The batch continues running on OpenAI's servers, and this
    function will check its status and download results when ready.

    Args:
        batch_id: Batch job ID (printed when batch was started)
        poll_interval: Seconds between status checks (default: 60)

    Returns:
        List of raw batch results

    Example:
        >>> # If you see "Batch job created: batch_abc123" in logs
        >>> results = resume_batch_evaluation("batch_abc123")
        >>> # Process results with parse_batch_results() if needed
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment")

    return resume_batch_job(
        batch_id=batch_id,
        api_key=api_key,
        temp_dir=BATCH_TEMP_DIR,
        poll_interval=poll_interval
    )


# Placeholder for future batch implementations
def evaluate_memo_batch_claude(
    memo: str,
    source_document: str,
    template: str = None,
    weights: Dict[str, float] = None
) -> float:
    """
    Evaluate a memo using Claude Batch API.
    TODO: Implement Claude batch API integration
    """
    raise NotImplementedError("Claude batch evaluation not yet implemented")


def evaluate_memo_batch_gemini(
    memo: str,
    source_document: str,
    template: str = None,
    weights: Dict[str, float] = None
) -> float:
    """
    Evaluate a memo using Gemini Batch API.
    TODO: Implement Gemini batch API integration
    """
    raise NotImplementedError("Gemini batch evaluation not yet implemented")
