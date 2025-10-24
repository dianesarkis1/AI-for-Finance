"""
Batch-enabled evaluation metrics for investment memo generation.

This module provides batch API versions of the core evaluation metrics,
optimized to combine multiple metric evaluations into single batch jobs
for faster processing with GPT-5.

Metrics:
- Accuracy: No hallucinated terms
- Completeness: All key terms captured
- Quality: Appropriate length, tone, structure
- Consistency (intra-memo): No internal contradictions
"""

from typing import Dict, List, Tuple
import json


# Import prompt templates from original metrics
from evals.metrics import (
    ACCURACY_PROMPT_TEMPLATE,
    COMPLETENESS_PROMPT_TEMPLATE,
    CONSISTENCY_PROMPT_TEMPLATE,
    CLARITY_PROMPT_TEMPLATE,
    TONE_PROMPT_TEMPLATE,
    LENGTH_PROMPT_TEMPLATE,
    STRUCTURE_PROMPT_TEMPLATE,
    _parse_accuracy_response,
    _parse_completeness_response,
    _parse_consistency_response,
    _parse_quality_score
)


def create_batch_requests_for_memo(
    memo: str,
    source_document: str,
    template: str = None,
    model: str = "gpt-5"
) -> List[Dict]:
    """
    Create all batch API requests for evaluating a single memo.

    This creates one request per metric/sub-metric, allowing all evaluations
    to be processed in a single batch job instead of sequential API calls.

    Args:
        memo: Generated investment memo text
        source_document: Original credit agreement text
        template: Optional template for structure evaluation
        model: Model identifier (default: gpt-5)

    Returns:
        List of batch request objects in OpenAI Batch API format
    """
    requests = []

    # 1. Accuracy request
    accuracy_prompt = ACCURACY_PROMPT_TEMPLATE.format(
        source_document=source_document,
        memo=memo
    )
    requests.append({
        "custom_id": "accuracy",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": [{"role": "user", "content": accuracy_prompt}]
        }
    })

    # 2. Completeness request
    completeness_prompt = COMPLETENESS_PROMPT_TEMPLATE.format(
        source_document=source_document,
        memo=memo
    )
    requests.append({
        "custom_id": "completeness",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": [{"role": "user", "content": completeness_prompt}]
        }
    })

    # 3. Consistency request
    consistency_prompt = CONSISTENCY_PROMPT_TEMPLATE.format(memo=memo)
    requests.append({
        "custom_id": "consistency",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": [{"role": "user", "content": consistency_prompt}]
        }
    })

    # 4. Quality sub-metrics (4 requests: clarity, tone, length, structure)
    clarity_prompt = CLARITY_PROMPT_TEMPLATE.format(memo=memo)
    requests.append({
        "custom_id": "quality_clarity",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": [{"role": "user", "content": clarity_prompt}]
        }
    })

    tone_prompt = TONE_PROMPT_TEMPLATE.format(memo=memo)
    requests.append({
        "custom_id": "quality_tone",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": [{"role": "user", "content": tone_prompt}]
        }
    })

    length_prompt = LENGTH_PROMPT_TEMPLATE.format(memo=memo)
    requests.append({
        "custom_id": "quality_length",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": [{"role": "user", "content": length_prompt}]
        }
    })

    # Structure prompt requires template
    if template:
        structure_prompt = STRUCTURE_PROMPT_TEMPLATE.format(template=template, memo=memo)
    else:
        # Use default template if none provided
        default_template = """1. Executive Summary/Overview
2. Transaction/Company Details
3. Financial Terms
4. Investment Strengths/Highlights
5. Risks and Concerns
6. Recommendation/Conclusion"""
        structure_prompt = STRUCTURE_PROMPT_TEMPLATE.format(template=default_template, memo=memo)

    requests.append({
        "custom_id": "quality_structure",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": [{"role": "user", "content": structure_prompt}]
        }
    })

    return requests


def parse_batch_results(results: List[Dict]) -> Dict:
    """
    Parse batch API results and aggregate into metric scores.

    Args:
        results: List of batch response objects from OpenAI Batch API

    Returns:
        Dict with aggregated metric results matching original evaluator.py format
    """
    # Organize results by custom_id
    results_by_id = {}
    for result in results:
        custom_id = result.get("custom_id")
        response = result.get("response", {})
        body = response.get("body", {})

        # Extract content from response
        choices = body.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            content = message.get("content", "")
            results_by_id[custom_id] = content

    # Parse accuracy
    accuracy_response = results_by_id.get("accuracy", "")
    accuracy_vote, accuracy_hallucinations = _parse_accuracy_response(accuracy_response)

    accuracy_result = {
        "accurate": accuracy_vote == "NO",
        "score": 1.0 if accuracy_vote == "NO" else 0.0,  # Single model, so binary
        "votes": {
            "gpt-5": {
                "vote": accuracy_vote,
                "hallucinations": accuracy_hallucinations
            }
        },
        "consensus_reached": True,  # Single model always has consensus
        "yes_votes": 1 if accuracy_vote == "YES" else 0,
        "no_votes": 1 if accuracy_vote == "NO" else 0
    }

    # Parse completeness
    completeness_response = results_by_id.get("completeness", "")
    completeness_vote, completeness_missing = _parse_completeness_response(completeness_response)

    completeness_result = {
        "complete": completeness_vote == "NO",
        "score": 1.0 if completeness_vote == "NO" else 0.0,
        "votes": {
            "gpt-5": {
                "vote": completeness_vote,
                "missing_terms": completeness_missing
            }
        },
        "consensus_reached": True,
        "yes_votes": 1 if completeness_vote == "YES" else 0,
        "no_votes": 1 if completeness_vote == "NO" else 0
    }

    # Parse consistency
    consistency_response = results_by_id.get("consistency", "")
    consistency_parsed = _parse_consistency_response(consistency_response)

    consistency_result = {
        "consistent": not consistency_parsed["has_issues"],
        "score": 0.0 if consistency_parsed["has_issues"] else 1.0,
        "votes": {
            "gpt-5": consistency_parsed
        },
        "consensus_reached": True,
        "has_issues_votes": 1 if consistency_parsed["has_issues"] else 0,
        "no_issues_votes": 0 if consistency_parsed["has_issues"] else 1
    }

    # Parse quality sub-metrics
    clarity_score = _parse_quality_score(results_by_id.get("quality_clarity", ""))
    tone_score = _parse_quality_score(results_by_id.get("quality_tone", ""))
    length_score = _parse_quality_score(results_by_id.get("quality_length", ""))
    structure_score = _parse_quality_score(results_by_id.get("quality_structure", ""))

    # Calculate overall quality score
    valid_scores = [s for s in [clarity_score, tone_score, length_score, structure_score] if s is not None]
    quality_avg = sum(valid_scores) / len(valid_scores) if valid_scores else 0

    quality_result = {
        "quality_score": quality_avg,
        "clarity_score": clarity_score if clarity_score is not None else 0,
        "tone_score": tone_score if tone_score is not None else 0,
        "length_score": length_score if length_score is not None else 0,
        "structure_score": structure_score if structure_score is not None else 0,
        "votes": {
            "gpt-5": {
                "clarity": clarity_score,
                "tone": tone_score,
                "length": length_score,
                "structure": structure_score
            }
        }
    }

    return {
        "accuracy_result": accuracy_result,
        "completeness_result": completeness_result,
        "consistency_result": consistency_result,
        "quality_result": quality_result
    }


# Placeholder for other models (Claude, Gemini) - to be implemented later
def evaluate_accuracy_batch_claude(memo: str, source_document: str) -> Dict:
    """
    Batch evaluation for accuracy using Claude.
    TODO: Implement Claude batch API integration
    """
    raise NotImplementedError("Claude batch evaluation not yet implemented")


def evaluate_completeness_batch_claude(memo: str, source_document: str) -> Dict:
    """
    Batch evaluation for completeness using Claude.
    TODO: Implement Claude batch API integration
    """
    raise NotImplementedError("Claude batch evaluation not yet implemented")


def evaluate_consistency_batch_claude(memo: str) -> Dict:
    """
    Batch evaluation for consistency using Claude.
    TODO: Implement Claude batch API integration
    """
    raise NotImplementedError("Claude batch evaluation not yet implemented")


def evaluate_quality_batch_claude(memo: str, template: str = None) -> Dict:
    """
    Batch evaluation for quality using Claude.
    TODO: Implement Claude batch API integration
    """
    raise NotImplementedError("Claude batch evaluation not yet implemented")


def evaluate_accuracy_batch_gemini(memo: str, source_document: str) -> Dict:
    """
    Batch evaluation for accuracy using Gemini.
    TODO: Implement Gemini batch API integration
    """
    raise NotImplementedError("Gemini batch evaluation not yet implemented")


def evaluate_completeness_batch_gemini(memo: str, source_document: str) -> Dict:
    """
    Batch evaluation for completeness using Gemini.
    TODO: Implement Gemini batch API integration
    """
    raise NotImplementedError("Gemini batch evaluation not yet implemented")


def evaluate_consistency_batch_gemini(memo: str) -> Dict:
    """
    Batch evaluation for consistency using Gemini.
    TODO: Implement Gemini batch API integration
    """
    raise NotImplementedError("Gemini batch evaluation not yet implemented")


def evaluate_quality_batch_gemini(memo: str, template: str = None) -> Dict:
    """
    Batch evaluation for quality using Gemini.
    TODO: Implement Gemini batch API integration
    """
    raise NotImplementedError("Gemini batch evaluation not yet implemented")
