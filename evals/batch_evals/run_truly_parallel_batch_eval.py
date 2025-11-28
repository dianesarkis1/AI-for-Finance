#!/usr/bin/env python3
"""
Truly Parallelized Batch Evaluation Script

This script runs batch evaluations in TRUE parallel:
1. Generate all memos first (sequential, ~30-50 min for 50 inputs)
2. Submit ALL 150 batch jobs at once WITHOUT waiting (50 inputs × 3 evaluators)
3. Poll all 150 batch jobs in parallel until complete
4. Aggregate results

This is MUCH faster than sequential processing.

Usage:
    # Use default comprehensive sample (50 indices)
    python run_truly_parallel_batch_eval.py

    # Test with specific indices (sequential memo generation)
    python run_truly_parallel_batch_eval.py --indices 0 1 2 6 12

    # Test with specific indices (parallel memo generation - FASTER!)
    python run_truly_parallel_batch_eval.py --indices 0 1 2 6 12 --parallel-memos

    # Test with just one index
    python run_truly_parallel_batch_eval.py --indices 128 --parallel-memos
"""

import argparse
import json
import os
import random
import statistics
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evals.batch_evals.batch_evaluate import (
    load_training_sample,
    generate_memo_for_input,
    aggregate_evaluator_results
)
from evals.batch_evals.batch_metrics import (
    create_batch_requests_for_memo,
    parse_batch_results,
    create_claude_batch_requests_for_memo,
    parse_claude_batch_results,
    create_gemini_batch_requests_for_memo,
    parse_gemini_batch_results
)
from evals.batch_evals.batch_utils import (
    upload_batch_file,
    create_batch_job,
    check_batch_status,
    download_batch_results,
    load_batch_results,
    create_claude_batch,
    check_claude_batch_status,
    download_claude_batch_results,
    create_gemini_batch,
    check_gemini_batch_status,
    extract_gemini_batch_results
)

# Configuration
TRAIN_FILE = Path("data/train.jsonl")
BASELINE_SAMPLED_INDICES_FILE = Path("evals/benchmark/baseline_sampled_indices_seed42.json")
OUTPUT_DIR = Path("evals/batch_evals")

# Random seed for reproducibility
RANDOM_SEED = 42

# Model configuration
MODEL_TO_EVALUATE = "claude-sonnet-4-20250514"
EVALUATOR_MODELS = ["gpt-5", "claude-sonnet-4-20250514", "gemini-2.5-pro"]

# These will be overridden by command-line arguments if provided
DEFAULT_RUN_NAME = "batch_temp_3"
DEFAULT_PROMPT_FILE = None  # Uses prompts/baseline.txt if None

# Global variables set by main() based on command-line arguments
BATCH_TEMP_DIR = None  # Set dynamically in main()
PROMPT_FILE = None  # Set dynamically in main()


# =============================================================================
# QUOTA/CREDIT ERROR HANDLING
# =============================================================================

def is_quota_error(error_message: str) -> tuple[bool, Optional[str]]:
    """
    Detect if an error is due to quota/credit limits.

    Returns (is_quota_error, provider_name)
    """
    error_lower = error_message.lower()

    # OpenAI quota errors
    if any(phrase in error_lower for phrase in [
        "insufficient_quota",
        "quota exceeded",
        "rate_limit_exceeded",
        "you exceeded your current quota",
        "billing hard limit"
    ]):
        return True, "OpenAI"

    # Anthropic quota errors
    if any(phrase in error_lower for phrase in [
        "insufficient credits",
        "credit limit",
        "billing",
        "overloaded_error"
    ]):
        return True, "Anthropic/Claude"

    # Gemini quota errors
    if any(phrase in error_lower for phrase in [
        "quota exceeded",
        "resource exhausted",
        "429"
    ]):
        return True, "Gemini"

    return False, None


def prompt_user_for_credits(provider: str, error_message: str) -> bool:
    """
    Prompt user to add credits and wait for confirmation.

    Returns True if user wants to retry, False to cancel.
    """
    print(f"\n{'='*70}")
    print(f"⚠️  QUOTA/CREDIT LIMIT REACHED - {provider}")
    print(f"{'='*70}")
    print(f"\nError: {error_message}")
    print(f"\n🔴 The script has paused because {provider} credits/quota have been exhausted.")
    print(f"\nPlease:")
    print(f"  1. Go to your {provider} account")
    print(f"  2. Add credits or increase quota limits")
    print(f"  3. Wait a few minutes for the changes to take effect")
    print(f"\nOnce you've added credits, type 'retry' to continue.")
    print(f"Type 'cancel' to stop the evaluation run.")
    print(f"{'='*70}\n")

    while True:
        response = input("Enter 'retry' or 'cancel': ").strip().lower()
        if response == 'retry':
            print(f"\n✅ Retrying {provider} operations...\n")
            return True
        elif response == 'cancel':
            print(f"\n❌ Cancelling evaluation run.\n")
            return False
        else:
            print("Invalid input. Please type 'retry' or 'cancel'.")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_baseline_sampled_indices(file_path: Path) -> list[int]:
    """Load the baseline sampled indices from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data['sampled_indices']


def count_train_samples(train_file: Path) -> int:
    """Count total number of samples in train.jsonl."""
    with open(train_file, 'r') as f:
        return sum(1 for _ in f)


def create_comprehensive_sample(
    baseline_indices: list[int],
    first_n: int = 3,
    random_sample_size: int = 37,
    total_samples: int = 484,
    seed: int = 42
) -> dict:
    """Create comprehensive sample combining baseline, first N, and random samples."""
    # Start with baseline indices
    combined_indices = set(baseline_indices)

    # Add first N indices
    first_indices = list(range(first_n))
    combined_indices.update(first_indices)

    # Sample random indices (excluding already selected ones)
    random.seed(seed)
    available_indices = [i for i in range(total_samples) if i not in combined_indices]
    random_indices = random.sample(available_indices, random_sample_size)
    combined_indices.update(random_indices)

    # Convert to sorted list
    all_indices = sorted(list(combined_indices))

    # Create detailed breakdown
    sampling_info = {
        "random_seed": seed,
        "total_sampled": len(all_indices),
        "sampling_breakdown": {
            "from_baseline_sampled_indices_seed42": {
                "count": len(baseline_indices),
                "indices": sorted(baseline_indices)
            },
            "first_n_indices": {
                "count": len(first_indices),
                "indices": first_indices
            },
            "additional_random_sample": {
                "count": len(random_indices),
                "indices": sorted(random_indices)
            }
        },
        "all_sampled_indices": all_indices,
        "total_inputs_in_dataset": total_samples,
        "created_at": datetime.now().isoformat()
    }

    return sampling_info


def load_api_key_from_env(key_name: str) -> Optional[str]:
    """Load API key from environment or .env file."""
    # Try environment variable first
    api_key = os.getenv(key_name)

    if not api_key:
        # Try loading from .env file
        env_file = Path(__file__).parent.parent.parent / ".env"
        if env_file.exists():
            with open(env_file, 'r') as f:
                for line in f:
                    if line.strip().startswith(key_name):
                        # Handle formats: KEY=value or KEY="value" or KEY='value'
                        api_key = line.strip().split('=', 1)[1].strip().strip('"').strip("'")
                        break

    return api_key


def load_few_shot_examples(few_shot_dir: Path) -> List[Dict[str, str]]:
    """
    Load few-shot examples from a directory containing input-output pairs.

    Expected directory structure:
        few_shot_examples/
            input_103.txt
            example_103.md
            input_281.txt
            example_281.md
            ...

    Args:
        few_shot_dir: Path to directory containing few-shot examples

    Returns:
        List of dicts with 'input' and 'output' keys
    """
    if not few_shot_dir.exists():
        print(f"Warning: Few-shot directory not found: {few_shot_dir}", file=sys.stderr)
        return []

    examples = []

    # Find all input files (format: input_*.txt)
    input_files = sorted(few_shot_dir.glob("input_*.txt"))

    if not input_files:
        print(f"Warning: No input files found in {few_shot_dir}", file=sys.stderr)
        return []

    for input_file in input_files:
        # Extract the index from the filename (e.g., "input_103.txt" -> "103")
        index = input_file.stem.replace("input_", "")

        # Find corresponding output file (format: example_*.md)
        output_file = few_shot_dir / f"example_{index}.md"

        if not output_file.exists():
            print(f"Warning: Output file not found for {input_file}: {output_file}", file=sys.stderr)
            continue

        # Read input and output
        with open(input_file, 'r', encoding='utf-8') as f:
            input_text = f.read()

        with open(output_file, 'r', encoding='utf-8') as f:
            output_text = f.read()

        examples.append({
            'input': input_text,
            'output': output_text
        })

        print(f"  Loaded few-shot example {index}: {len(input_text):,} chars input, {len(output_text):,} chars output")

    print(f"Loaded {len(examples)} few-shot examples from {few_shot_dir}")
    return examples


def generate_all_memos_parallel(indices: List[int], train_file: Path, model: str, api_key: str, few_shot_examples: Optional[List[Dict[str, str]]] = None, use_system_parameter: bool = False, use_xml_tags: bool = False) -> Dict[int, Dict]:
    """
    Phase 1: Generate all memos in parallel using Claude Batch API.
    Much faster than sequential generation.

    Args:
        use_system_parameter: If True, use Claude's native system parameter. If False, use old behavior (everything in user message).
        use_xml_tags: If True, wrap inputs in XML tags for better structure.
    """
    print(f"\n{'='*70}")
    print(f"PHASE 1: GENERATING ALL MEMOS (PARALLEL)")
    print(f"{'='*70}")
    print(f"Model: {model}")
    print(f"Total inputs: {len(indices)}")
    print(f"Method: Claude Batch API (parallel)")
    print(f"{'='*70}\n")

    # Load prompt from file or use default
    if PROMPT_FILE:
        prompt_path = PROMPT_FILE
    else:
        prompt_path = Path(__file__).parent.parent.parent / "prompts" / "baseline.txt"

    with open(prompt_path, 'r') as f:
        prompt_text = f.read()

    # Prepend few-shot examples to prompt if provided
    if few_shot_examples:
        if use_xml_tags:
            # Format with XML tags
            few_shot_section = "\n\n<examples>\n"
            for i, example in enumerate(few_shot_examples, 1):
                few_shot_section += f"<example>\n"
                few_shot_section += f"<input>\n{example['input']}\n</input>\n\n"
                few_shot_section += f"<output>\n{example['output']}\n</output>\n"
                few_shot_section += f"</example>\n\n"
            few_shot_section += "</examples>\n\n"
        else:
            # Original format without XML
            few_shot_section = "\n\n# Few-Shot Examples\n\n"
            few_shot_section += "Here are example credit agreements with their corresponding high-quality investment memos for reference:\n\n"

            for i, example in enumerate(few_shot_examples, 1):
                few_shot_section += f"## Example {i}\n\n"
                few_shot_section += f"### Input Credit Agreement:\n```\n{example['input']}\n```\n\n"
                few_shot_section += f"### Expected Output Memo:\n{example['output']}\n\n"
                few_shot_section += "---\n\n"

        prompt_text = few_shot_section + prompt_text
        print(f"✅ Prepended {len(few_shot_examples)} few-shot examples to prompt (XML tags: {use_xml_tags})\n")

    # Build batch requests for memo generation
    batch_requests = []
    index_to_source = {}  # Track which index corresponds to which source

    print("Loading source documents and building batch requests...")
    for idx in indices:
        try:
            source_url, credit_agreement_text = load_training_sample(train_file, idx)
            index_to_source[idx] = {
                "source_url": source_url,
                "credit_agreement": credit_agreement_text
            }

            # Create batch request for this memo
            # Wrap credit agreement in XML tags if requested
            if use_xml_tags:
                formatted_credit_agreement = f"<credit_agreement>\n{credit_agreement_text}\n</credit_agreement>"
            else:
                formatted_credit_agreement = credit_agreement_text

            if use_system_parameter:
                # NEW: Use Claude's native system parameter for better performance
                request = {
                    "custom_id": f"memo_generation_{idx}",
                    "params": {
                        "model": model,
                        "max_tokens": 8000,
                        "system": prompt_text,  # System instructions
                        "messages": [
                            {
                                "role": "user",
                                "content": formatted_credit_agreement  # Credit agreement (optionally XML-wrapped)
                            }
                        ]
                    }
                }
            else:
                # OLD: Everything in user message (default behavior)
                request = {
                    "custom_id": f"memo_generation_{idx}",
                    "params": {
                        "model": model,
                        "max_tokens": 8000,
                        "messages": [
                            {
                                "role": "user",
                                "content": f"{prompt_text}\n\n{formatted_credit_agreement}"
                            }
                        ]
                    }
                }
            batch_requests.append(request)
            print(f"  ✓ Prepared request for index {idx}")

        except Exception as e:
            print(f"  ✗ Error loading index {idx}: {e}")
            index_to_source[idx] = {
                "source_url": None,
                "credit_agreement": None,
                "error": str(e)
            }

    print(f"\n✓ Prepared {len(batch_requests)} batch requests\n")

    # Submit batch to Claude
    print("Submitting batch job to Claude API...")
    batch_id = create_claude_batch(batch_requests, api_key)
    print(f"✓ Batch submitted: {batch_id}\n")

    # Poll until complete
    print("Polling batch job (checks every 60 seconds)...")
    start_time = time.time()
    poll_interval = 60

    while True:
        status_data = check_claude_batch_status(batch_id, api_key)
        processing_status = status_data.get("processing_status")
        request_counts = status_data.get("request_counts", {})

        elapsed = int(time.time() - start_time)
        print(f"  [{elapsed}s] Status: {processing_status}")
        print(f"    Processing: {request_counts.get('processing', 0)}")
        print(f"    Succeeded:  {request_counts.get('succeeded', 0)}")
        print(f"    Errored:    {request_counts.get('errored', 0)}")

        if processing_status == "ended":
            print(f"\n✓ Batch completed in {elapsed}s\n")
            break
        elif processing_status in ["failed", "expired", "cancelled"]:
            print(f"\n✗ Batch {processing_status}\n")
            raise RuntimeError(f"Batch memo generation {processing_status}")

        time.sleep(poll_interval)

    # Download and parse results
    print("Downloading batch results...")
    results_url = status_data.get("results_url")
    output_path = download_claude_batch_results(results_url, BATCH_TEMP_DIR, api_key, input_index=None)

    # Parse results and match back to indices
    memos = {}
    with open(output_path, 'r') as f:
        for line in f:
            result = json.loads(line)
            custom_id = result.get("custom_id", "")

            # Extract index from custom_id (e.g., "memo_generation_128" -> 128)
            if custom_id.startswith("memo_generation_"):
                idx = int(custom_id.split("_")[-1])

                if result.get("result", {}).get("type") == "succeeded":
                    content_blocks = result["result"]["message"]["content"]
                    memo = content_blocks[0]["text"] if content_blocks else None

                    memos[idx] = {
                        "source_url": index_to_source[idx]["source_url"],
                        "memo": memo,
                        "credit_agreement": index_to_source[idx]["credit_agreement"],
                        "error": None
                    }
                    print(f"  ✓ Index {idx}: {len(memo) if memo else 0} chars")
                else:
                    error_msg = result.get("result", {}).get("error", {}).get("message", "Unknown error")
                    memos[idx] = {
                        "source_url": index_to_source[idx]["source_url"],
                        "memo": None,
                        "credit_agreement": index_to_source[idx]["credit_agreement"],
                        "error": error_msg
                    }
                    print(f"  ✗ Index {idx}: {error_msg}")

    # Add any indices that weren't in results (failed to load source)
    for idx in indices:
        if idx not in memos:
            memos[idx] = {
                "source_url": index_to_source[idx].get("source_url"),
                "memo": None,
                "credit_agreement": index_to_source[idx].get("credit_agreement"),
                "error": index_to_source[idx].get("error", "No result returned")
            }

    successful = sum(1 for m in memos.values() if m['memo'] is not None)
    print(f"\n{'='*70}")
    print(f"PARALLEL MEMO GENERATION COMPLETE")
    print(f"{'='*70}")
    print(f"Successful: {successful}/{len(indices)}")
    print(f"Failed: {len(indices) - successful}/{len(indices)}")
    print(f"Total time: {int(time.time() - start_time)}s")
    print(f"{'='*70}\n")

    return memos


def generate_all_memos(indices: List[int], train_file: Path, model: str, few_shot_examples: Optional[List[Dict[str, str]]] = None, use_system_parameter: bool = False, use_xml_tags: bool = False) -> Dict[int, Dict]:
    """Phase 1: Generate all memos sequentially (legacy method).

    Args:
        use_system_parameter: If True, use Claude's native system parameter. If False, use old behavior (everything in user message).
        use_xml_tags: If True, wrap inputs in XML tags for better structure.
    """
    print(f"\n{'='*70}")
    print(f"PHASE 1: GENERATING ALL MEMOS (SEQUENTIAL)")
    print(f"{'='*70}")
    print(f"Model: {model}")
    print(f"Total inputs: {len(indices)}")
    print(f"Method: Sequential (slower)")
    if few_shot_examples:
        print(f"Few-shot examples: {len(few_shot_examples)}")
    print(f"XML tags: {use_xml_tags}")
    print(f"{'='*70}\n")

    memos = {}

    # Create temp input file for model_run.py
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as tmp_in:
        temp_input_file = Path(tmp_in.name)

    try:
        for i, idx in enumerate(indices):
            print(f"[{i+1}/{len(indices)}] Generating memo for input {idx}...")

            try:
                # Load training sample
                source_url, credit_agreement_text = load_training_sample(train_file, idx)
                print(f"  Source: {source_url[:80]}...")

                # Generate memo
                memo = generate_memo_for_input(model, credit_agreement_text, temp_input_file, prompt_file=PROMPT_FILE, few_shot_examples=few_shot_examples, use_system_parameter=use_system_parameter, use_xml_tags=use_xml_tags)

                if memo:
                    memos[idx] = {
                        "source_url": source_url,
                        "memo": memo,
                        "credit_agreement": credit_agreement_text,
                        "error": None
                    }
                    print(f"  ✅ Generated memo: {len(memo)} chars\n")
                else:
                    memos[idx] = {
                        "source_url": source_url,
                        "memo": None,
                        "credit_agreement": credit_agreement_text,
                        "error": "Failed to generate memo"
                    }
                    print(f"  ❌ Failed to generate memo\n")

            except Exception as e:
                print(f"  ❌ Error: {e}\n")
                memos[idx] = {
                    "source_url": None,
                    "memo": None,
                    "credit_agreement": None,
                    "error": str(e)
                }

    finally:
        # Clean up temp file
        try:
            temp_input_file.unlink()
        except:
            pass

    successful = sum(1 for m in memos.values() if m['memo'] is not None)
    print(f"\n{'='*70}")
    print(f"MEMO GENERATION COMPLETE")
    print(f"{'='*70}")
    print(f"Successful: {successful}/{len(indices)}")
    print(f"Failed: {len(indices) - successful}/{len(indices)}")
    print(f"{'='*70}\n")

    return memos


def aggregate_feedback_from_evaluators(
    eval_results_list: List[Dict],
    evaluator_models: List[str]
) -> Dict:
    """
    Aggregate evaluation feedback from multiple evaluators into a single structure.

    This combines:
    - All hallucinations/issues from all evaluators
    - All missing terms from all evaluators
    - All consistency issues from all evaluators
    - Average quality scores across evaluators

    Args:
        eval_results_list: List of evaluation result dicts (one per evaluator)
        evaluator_models: List of evaluator model names (for attribution)

    Returns:
        Aggregated feedback dict with combined results
    """
    aggregated = {
        'accuracy_result': {
            'score': 0.0,
            'accurate': 'YES',
            'combined_issues': []  # List of (evaluator, issues) tuples
        },
        'completeness_result': {
            'score': 0.0,
            'complete': 'YES',
            'combined_missing_terms': []  # List of (evaluator, missing_terms) tuples
        },
        'consistency_result': {
            'score': 0.0,
            'consistent': 'YES',
            'combined_issues': []  # List of (evaluator, issues) tuples
        },
        'quality_result': {
            'quality_score': 0.0,
            'clarity_score': 0.0,
            'tone_score': 0.0,
            'length_score': 0.0,
            'structure_score': 0.0
        }
    }

    num_evaluators = len(eval_results_list)
    if num_evaluators == 0:
        return aggregated

    # Aggregate accuracy
    acc_scores = []
    for eval_result, evaluator_name in zip(eval_results_list, evaluator_models):
        acc = eval_result.get('accuracy_result', {})
        acc_scores.append(acc.get('score', 0))

        # If any evaluator says NO, aggregated should be NO
        if acc.get('accurate') == 'NO':
            aggregated['accuracy_result']['accurate'] = 'NO'

        # Collect issues with attribution
        if acc.get('votes'):
            for voter, vote_data in acc['votes'].items():
                issues = vote_data.get('issues', [])
                if issues:
                    aggregated['accuracy_result']['combined_issues'].append({
                        'evaluator': evaluator_name,
                        'issues': issues
                    })

    aggregated['accuracy_result']['score'] = sum(acc_scores) / len(acc_scores) if acc_scores else 0

    # Aggregate completeness
    comp_scores = []
    for eval_result, evaluator_name in zip(eval_results_list, evaluator_models):
        comp = eval_result.get('completeness_result', {})
        comp_scores.append(comp.get('score', 0))

        # If any evaluator says NO, aggregated should be NO
        if comp.get('complete') == 'NO':
            aggregated['completeness_result']['complete'] = 'NO'

        # Collect missing terms with attribution
        if comp.get('votes'):
            for voter, vote_data in comp['votes'].items():
                missing_terms = vote_data.get('missing_terms', [])
                if missing_terms:
                    aggregated['completeness_result']['combined_missing_terms'].append({
                        'evaluator': evaluator_name,
                        'missing_terms': missing_terms
                    })

    aggregated['completeness_result']['score'] = sum(comp_scores) / len(comp_scores) if comp_scores else 0

    # Aggregate consistency
    cons_scores = []
    for eval_result, evaluator_name in zip(eval_results_list, evaluator_models):
        cons = eval_result.get('consistency_result', {})
        cons_scores.append(cons.get('score', 0))

        # If any evaluator says NO, aggregated should be NO
        if cons.get('consistent') == 'NO':
            aggregated['consistency_result']['consistent'] = 'NO'

        # Collect consistency issues with attribution
        if cons.get('votes'):
            for voter, vote_data in cons['votes'].items():
                if vote_data.get('has_issues'):
                    issues = vote_data.get('issues', [])
                    if issues:
                        aggregated['consistency_result']['combined_issues'].append({
                            'evaluator': evaluator_name,
                            'issues': issues
                        })

    aggregated['consistency_result']['score'] = sum(cons_scores) / len(cons_scores) if cons_scores else 0

    # Aggregate quality scores (simple average across all evaluators)
    clarity_scores = []
    tone_scores = []
    length_scores = []
    structure_scores = []
    quality_scores = []

    for eval_result in eval_results_list:
        qual = eval_result.get('quality_result', {})
        clarity_scores.append(qual.get('clarity_score', 0))
        tone_scores.append(qual.get('tone_score', 0))
        length_scores.append(qual.get('length_score', 0))
        structure_scores.append(qual.get('structure_score', 0))
        quality_scores.append(qual.get('quality_score', 0))

    aggregated['quality_result']['clarity_score'] = sum(clarity_scores) / len(clarity_scores) if clarity_scores else 0
    aggregated['quality_result']['tone_score'] = sum(tone_scores) / len(tone_scores) if tone_scores else 0
    aggregated['quality_result']['length_score'] = sum(length_scores) / len(length_scores) if length_scores else 0
    aggregated['quality_result']['structure_score'] = sum(structure_scores) / len(structure_scores) if structure_scores else 0
    aggregated['quality_result']['quality_score'] = sum(quality_scores) / len(quality_scores) if quality_scores else 0

    return aggregated


def refine_memo_based_on_feedback(
    source_document: str,
    original_prompt: str,
    current_memo: str,
    evaluation_feedback: Dict,
    api_key: str,
    model: str = "claude-sonnet-4-20250514"
) -> Optional[str]:
    """
    Refine a memo based on evaluation feedback from an evaluator.

    Args:
        source_document: Original credit agreement text
        original_prompt: Original instructions for memo generation
        current_memo: Current version of the memo
        evaluation_feedback: Dict with evaluation results (accuracy, completeness, consistency, quality)
        api_key: Anthropic API key
        model: Claude model to use for refinement

    Returns:
        Refined memo text, or None if refinement failed
    """
    # Check if this is combined feedback (from multiple evaluators) or single evaluator feedback
    acc_result = evaluation_feedback.get('accuracy_result', {})
    comp_result = evaluation_feedback.get('completeness_result', {})
    cons_result = evaluation_feedback.get('consistency_result', {})

    # Format accuracy issues
    if 'combined_issues' in acc_result:
        # Combined feedback format
        acc_issues_text = ""
        for issue_info in acc_result['combined_issues']:
            evaluator = issue_info['evaluator']
            issues = issue_info['issues']
            if issues:
                acc_issues_text += f"\n   {evaluator}: {issues}"
        if not acc_issues_text:
            acc_issues_text = "None"
    else:
        # Single evaluator format
        acc_issues_text = []
        for voter, vote_data in acc_result.get('votes', {}).items():
            issues = vote_data.get('issues', [])
            if issues:
                acc_issues_text.append(str(issues))
        acc_issues_text = ', '.join(acc_issues_text) if acc_issues_text else "None"

    # Format completeness missing terms
    if 'combined_missing_terms' in comp_result:
        # Combined feedback format
        comp_missing_text = ""
        for missing_info in comp_result['combined_missing_terms']:
            evaluator = missing_info['evaluator']
            missing_terms = missing_info['missing_terms']
            if missing_terms:
                comp_missing_text += f"\n   {evaluator}: {missing_terms}"
        if not comp_missing_text:
            comp_missing_text = "None"
    else:
        # Single evaluator format
        comp_missing_text = []
        for voter, vote_data in comp_result.get('votes', {}).items():
            missing = vote_data.get('missing_terms', [])
            if missing:
                comp_missing_text.append(str(missing))
        comp_missing_text = ', '.join(comp_missing_text) if comp_missing_text else "None"

    # Format consistency issues
    if 'combined_issues' in cons_result:
        # Combined feedback format
        cons_issues_text = ""
        for issue_info in cons_result['combined_issues']:
            evaluator = issue_info['evaluator']
            issues = issue_info['issues']
            if issues:
                cons_issues_text += f"\n   {evaluator}: {issues}"
        if not cons_issues_text:
            cons_issues_text = "None"
    else:
        # Single evaluator format
        cons_issues_text = []
        for voter, vote_data in cons_result.get('votes', {}).items():
            if vote_data.get('has_issues'):
                issues = vote_data.get('issues', [])
                if issues:
                    cons_issues_text.append(str(issues))
        cons_issues_text = ', '.join(cons_issues_text) if cons_issues_text else "None"

    # Construct refinement prompt
    refinement_prompt = f"""You are refining an investment memo based on evaluation feedback from 1 to 3 evaluators.

<original_instructions>
{original_prompt}
</original_instructions>

<source_credit_agreement>
{source_document}
</source_credit_agreement>

<current_memo>
{current_memo}
</current_memo>

<evaluation_feedback>
The memo was evaluated on four dimensions:

1. ACCURACY: {evaluation_feedback.get('accuracy_result', {}).get('accurate', 'N/A')}
   Score: {evaluation_feedback.get('accuracy_result', {}).get('score', 0):.2f}
   Issues: {acc_issues_text}

2. COMPLETENESS: {evaluation_feedback.get('completeness_result', {}).get('complete', 'N/A')}
   Score: {evaluation_feedback.get('completeness_result', {}).get('score', 0):.2f}
   Missing terms: {comp_missing_text}

3. CONSISTENCY: {evaluation_feedback.get('consistency_result', {}).get('consistent', 'N/A')}
   Score: {evaluation_feedback.get('consistency_result', {}).get('score', 0):.2f}
   Issues: {cons_issues_text}

4. QUALITY: Score: {evaluation_feedback.get('quality_result', {}).get('quality_score', 0):.2f}
   - Clarity: {evaluation_feedback.get('quality_result', {}).get('clarity_score', 0):.2f}
   - Tone: {evaluation_feedback.get('quality_result', {}).get('tone_score', 0):.2f}
   - Length: {evaluation_feedback.get('quality_result', {}).get('length_score', 0):.2f}
   - Structure: {evaluation_feedback.get('quality_result', {}).get('structure_score', 0):.2f}
</evaluation_feedback>

<task>
Based on the evaluation feedback above, improve the memo to address any identified issues. Focus especially on areas with low scores or identified problems.

Return ONLY the improved memo text (no preamble, no explanations). The improved memo should:
- Fix any accuracy issues by ensuring all facts match the source document
- Add any missing key terms identified in the completeness evaluation
- Resolve any consistency issues or contradictions
- Improve quality in areas with low scores (clarity, tone, length, structure)

Maintain the same overall structure and format as the original memo, but improve the content based on the feedback.
</task>"""

    # Call Claude API to refine
    try:
        payload = {
            "model": model,
            "max_tokens": 16000,
            "messages": [
                {
                    "role": "user",
                    "content": refinement_prompt
                }
            ]
        }

        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(**payload)

        # Extract refined memo
        if response.content and len(response.content) > 0:
            refined_memo = response.content[0].text
            return refined_memo
        else:
            print(f"  ⚠️  No content in refinement response")
            return None

    except Exception as e:
        print(f"  ❌ Error refining memo: {e}")
        return None


def save_memo_to_disk(memo_text: str, idx: int, round_num: int, batch_temp_dir: Path):
    """
    Save memo text to disk with round number in filename.

    Args:
        memo_text: The memo content
        idx: Input index
        round_num: Refinement round number (0 = initial, 1 = first refinement, etc.)
        batch_temp_dir: Directory to save to
    """
    timestamp = int(time.time())
    output_file = batch_temp_dir / f"memo_{idx}_round{round_num}_{timestamp}.txt"

    with open(output_file, 'w') as f:
        f.write(memo_text)

    print(f"    💾 Saved memo to: {output_file.name}")


def save_single_round_eval_to_disk(eval_results: Dict, idx: int, evaluator: str, round_num: int, batch_temp_dir: Path):
    """
    Save evaluation results for a single round to disk in batch API JSONL format.

    Args:
        eval_results: Evaluation results dict
        idx: Input index
        evaluator: Evaluator model name
        round_num: Refinement round number (0 = initial, 1 = first refinement, etc.)
        batch_temp_dir: Directory to save to
    """
    # Determine file prefix based on evaluator
    if "claude" in evaluator.lower():
        file_prefix = "claude_batch_output"
    elif "gemini" in evaluator.lower():
        file_prefix = "gemini_batch_output"
    else:
        file_prefix = "batch_output"

    # Create filename with index, round, and timestamp
    timestamp = int(time.time())
    output_file = batch_temp_dir / f"{file_prefix}_{idx}_round{round_num}_{timestamp}.jsonl"

    # Convert evaluation results to batch API format (same as save_refinement_results_to_disk)
    batch_entries = []

    # Add accuracy metric
    if 'accuracy_result' in eval_results:
        acc = eval_results['accuracy_result']
        vote = list(acc.get('votes', {}).values())[0].get('vote', 'NO') if acc.get('votes') else 'NO'
        hallucinations = list(acc.get('votes', {}).values())[0].get('hallucinations', '') if acc.get('votes') else ''

        content = f"ANSWER: {vote}\n\nHALLUCINATIONS:\n{hallucinations}"

        if "claude" in evaluator.lower():
            entry = {
                "custom_id": "accuracy",
                "result": {
                    "type": "succeeded",
                    "message": {
                        "model": evaluator,
                        "content": [{"type": "text", "text": content}]
                    }
                }
            }
        else:
            entry = {
                "custom_id": "accuracy",
                "response": {
                    "body": {
                        "choices": [{
                            "message": {
                                "content": content
                            }
                        }]
                    }
                }
            }
        batch_entries.append(entry)

    # Add completeness metric
    if 'completeness_result' in eval_results:
        comp = eval_results['completeness_result']
        vote = list(comp.get('votes', {}).values())[0].get('vote', 'NO') if comp.get('votes') else 'NO'
        missing = list(comp.get('votes', {}).values())[0].get('missing_terms', '') if comp.get('votes') else ''

        content = f"ANSWER: {vote}\n\nMISSING_TERMS:\n{missing}"

        if "claude" in evaluator.lower():
            entry = {
                "custom_id": "completeness",
                "result": {
                    "type": "succeeded",
                    "message": {
                        "model": evaluator,
                        "content": [{"type": "text", "text": content}]
                    }
                }
            }
        else:
            entry = {
                "custom_id": "completeness",
                "response": {
                    "body": {
                        "choices": [{
                            "message": {
                                "content": content
                            }
                        }]
                    }
                }
            }
        batch_entries.append(entry)

    # Add consistency metric
    if 'consistency_result' in eval_results:
        cons = eval_results['consistency_result']
        content = json.dumps({
            "has_issues": cons.get('has_issues', False),
            "issues": cons.get('issues', [])
        })

        if "claude" in evaluator.lower():
            entry = {
                "custom_id": "consistency",
                "result": {
                    "type": "succeeded",
                    "message": {
                        "model": evaluator,
                        "content": [{"type": "text", "text": content}]
                    }
                }
            }
        else:
            entry = {
                "custom_id": "consistency",
                "response": {
                    "body": {
                        "choices": [{
                            "message": {
                                "content": content
                            }
                        }]
                    }
                }
            }
        batch_entries.append(entry)

    # Add quality metrics (4 sub-metrics)
    if 'quality_result' in eval_results:
        qual = eval_results['quality_result']
        quality_metrics = {
            'quality_clarity': qual.get('clarity_score', 0),
            'quality_tone': qual.get('tone_score', 0),
            'quality_length': qual.get('length_score', 0),
            'quality_structure': qual.get('structure_score', 0)
        }

        for metric_name, score in quality_metrics.items():
            content = f"SCORE: {score}"

            if "claude" in evaluator.lower():
                entry = {
                    "custom_id": metric_name,
                    "result": {
                        "type": "succeeded",
                        "message": {
                            "model": evaluator,
                            "content": [{"type": "text", "text": content}]
                        }
                    }
                }
            else:
                entry = {
                    "custom_id": metric_name,
                    "response": {
                        "body": {
                            "choices": [{
                                "message": {
                                    "content": content
                                }
                            }]
                        }
                    }
                }
            batch_entries.append(entry)

    # Write to JSONL file
    with open(output_file, 'w') as f:
        for entry in batch_entries:
            f.write(json.dumps(entry) + '\n')


def save_refinement_results_to_disk(refinement_results: Dict, batch_temp_dir: Path):
    """
    Save refinement evaluation results to disk in batch API JSONL format.
    This allows generate_final_results.py to parse them like regular batch results.

    Args:
        refinement_results: Dict mapping (index, evaluator) -> evaluation results
        batch_temp_dir: Directory to save results to
    """
    print(f"\n{'='*70}")
    print(f"SAVING REFINEMENT RESULTS TO DISK")
    print(f"{'='*70}\n")

    # Group results by (index, evaluator)
    results_by_index_evaluator = {}
    for (idx, evaluator), eval_results in refinement_results.items():
        key = (idx, evaluator)
        results_by_index_evaluator[key] = eval_results

    # For each unique (index, evaluator), create a batch output file
    for (idx, evaluator), eval_results in results_by_index_evaluator.items():
        # Determine file prefix based on evaluator
        # NOTE: Gemini uses OpenAI JSON format internally, but needs gemini_batch_output prefix for parser
        if "claude" in evaluator.lower():
            file_prefix = "claude_batch_output"
        elif "gemini" in evaluator.lower():
            file_prefix = "gemini_batch_output"
        else:
            file_prefix = "batch_output"

        # Create filename with index and timestamp
        timestamp = int(time.time())
        output_file = batch_temp_dir / f"{file_prefix}_{idx}_{timestamp}.jsonl"

        # Convert evaluation results to batch API format
        batch_entries = []

        # Add accuracy metric
        if 'accuracy_result' in eval_results:
            acc = eval_results['accuracy_result']
            vote = list(acc.get('votes', {}).values())[0].get('vote', 'NO') if acc.get('votes') else 'NO'
            hallucinations = list(acc.get('votes', {}).values())[0].get('hallucinations', '') if acc.get('votes') else ''

            content = f"ANSWER: {vote}\n\nHALLUCINATIONS:\n{hallucinations}"

            if "claude" in evaluator.lower():
                entry = {
                    "custom_id": "accuracy",
                    "result": {
                        "type": "succeeded",
                        "message": {
                            "model": evaluator,
                            "content": [{"type": "text", "text": content}]
                        }
                    }
                }
            else:
                entry = {
                    "custom_id": "accuracy",
                    "response": {
                        "body": {
                            "choices": [{
                                "message": {
                                    "content": content
                                }
                            }]
                        }
                    }
                }
            batch_entries.append(entry)

        # Add completeness metric
        if 'completeness_result' in eval_results:
            comp = eval_results['completeness_result']
            vote = list(comp.get('votes', {}).values())[0].get('vote', 'NO') if comp.get('votes') else 'NO'
            missing = list(comp.get('votes', {}).values())[0].get('missing_terms', '') if comp.get('votes') else ''

            content = f"ANSWER: {vote}\n\nMISSING_TERMS:\n{missing}"

            if "claude" in evaluator.lower():
                entry = {
                    "custom_id": "completeness",
                    "result": {
                        "type": "succeeded",
                        "message": {
                            "model": evaluator,
                            "content": [{"type": "text", "text": content}]
                        }
                    }
                }
            else:
                entry = {
                    "custom_id": "completeness",
                    "response": {
                        "body": {
                            "choices": [{
                                "message": {
                                    "content": content
                                }
                            }]
                        }
                    }
                }
            batch_entries.append(entry)

        # Add consistency metric
        if 'consistency_result' in eval_results:
            cons = eval_results['consistency_result']
            content = json.dumps({
                "has_issues": cons.get('has_issues', False),
                "issues": cons.get('issues', [])
            })

            if "claude" in evaluator.lower():
                entry = {
                    "custom_id": "consistency",
                    "result": {
                        "type": "succeeded",
                        "message": {
                            "model": evaluator,
                            "content": [{"type": "text", "text": content}]
                        }
                    }
                }
            else:
                entry = {
                    "custom_id": "consistency",
                    "response": {
                        "body": {
                            "choices": [{
                                "message": {
                                    "content": content
                                }
                            }]
                        }
                    }
                }
            batch_entries.append(entry)

        # Add quality metrics (4 sub-metrics)
        if 'quality_result' in eval_results:
            qual = eval_results['quality_result']
            quality_metrics = {
                'quality_clarity': qual.get('clarity_score', 0),
                'quality_tone': qual.get('tone_score', 0),
                'quality_length': qual.get('length_score', 0),
                'quality_structure': qual.get('structure_score', 0)
            }

            for metric_name, score in quality_metrics.items():
                content = f"SCORE: {score}"

                if "claude" in evaluator.lower():
                    entry = {
                        "custom_id": metric_name,
                        "result": {
                            "type": "succeeded",
                            "message": {
                                "model": evaluator,
                                "content": [{"type": "text", "text": content}]
                            }
                        }
                    }
                else:
                    entry = {
                        "custom_id": metric_name,
                        "response": {
                            "body": {
                                "choices": [{
                                    "message": {
                                        "content": content
                                    }
                                }]
                            }
                        }
                    }
                batch_entries.append(entry)

        # Write to JSONL file
        with open(output_file, 'w') as f:
            for entry in batch_entries:
                f.write(json.dumps(entry) + '\n')

        print(f"  ✓ Saved {evaluator} results for index {idx}: {output_file.name} ({len(batch_entries)} metrics)")

    print(f"\n{'─'*70}")
    print(f"Saved {len(results_by_index_evaluator)} evaluation result files")
    print(f"{'='*70}\n")


def run_iterative_refinement(
    memos: Dict[int, Dict],
    evaluator_models: List[str],
    refinement_rounds: int,
    api_key: str
) -> Dict:
    """
    Run iterative refinement workflow for all evaluators.

    For each input and each evaluator:
    1. Evaluate memo → get feedback
    2. For each refinement round:
       - Refine memo based on feedback (using Claude)
       - Re-evaluate refined memo
    3. Return final evaluation scores

    Args:
        memos: Dict mapping index → memo info
        evaluator_models: List of evaluator model names
        refinement_rounds: Number of refinement iterations
        api_key: Anthropic API key for refinement calls

    Returns:
        Dict mapping (index, evaluator) → final evaluation results
        AND saves all intermediate artifacts (round 0 evals, round 1 memos, etc.) to disk
    """
    print(f"\n{'='*70}")
    print(f"ITERATIVE REFINEMENT WORKFLOW")
    print(f"{'='*70}")
    print(f"Refinement rounds: {refinement_rounds}")
    print(f"Evaluators: {len(evaluator_models)}")
    print(f"Inputs: {len([m for m in memos.values() if m.get('memo')])}")
    print(f"Total evaluations: {len([m for m in memos.values() if m.get('memo')]) * len(evaluator_models) * (refinement_rounds + 1)}")
    print(f"{'='*70}\n")

    # Load original prompt (for refinement context)
    if PROMPT_FILE:
        prompt_path = PROMPT_FILE
    else:
        prompt_path = Path(__file__).parent.parent.parent / "prompts" / "baseline.txt"
    with open(prompt_path, 'r') as f:
        original_prompt = f.read()

    # Track final results AND all intermediate results
    final_results = {}
    all_round_results = {}  # Track results from ALL rounds for explainability

    # Process each input
    for idx, memo_info in memos.items():
        if not memo_info.get('memo'):
            print(f"⏭️  Skipping index {idx} (no memo generated)")
            continue

        print(f"\n{'='*70}")
        print(f"Processing input {idx}")
        print(f"{'='*70}\n")

        initial_memo = memo_info['memo']
        source_document = memo_info['credit_agreement']

        # Save initial memo (round 0) to disk
        save_memo_to_disk(initial_memo, idx, round_num=0, batch_temp_dir=BATCH_TEMP_DIR)

        # Process each evaluator independently
        for evaluator_model in evaluator_models:
            print(f"\n  Evaluator: {evaluator_model}")
            print(f"  {'─'*66}")

            current_memo = initial_memo
            current_round = 0

            # Iterate through refinement rounds
            for round_num in range(refinement_rounds + 1):
                print(f"\n  Round {round_num + 1}/{refinement_rounds + 1}:")

                # Evaluate current memo
                print(f"    Evaluating...")
                eval_results = evaluate_single_memo_sync(
                    memo=current_memo,
                    source_document=source_document,
                    evaluator_model=evaluator_model,
                    input_index=idx
                )

                if not eval_results:
                    print(f"    ❌ Evaluation failed")
                    break

                # Display scores
                acc_score = eval_results.get('accuracy_result', {}).get('score', 0)
                comp_score = eval_results.get('completeness_result', {}).get('score', 0)
                cons_score = eval_results.get('consistency_result', {}).get('score', 0)
                qual_score = eval_results.get('quality_result', {}).get('quality_score', 0)
                avg_score = (acc_score + comp_score + cons_score + qual_score) / 4

                print(f"    Scores: Acc={acc_score:.2f}, Comp={comp_score:.2f}, Cons={cons_score:.2f}, Qual={qual_score:.2f}, Avg={avg_score:.2f}")

                # Save evaluation results for THIS round to disk
                round_key = (idx, evaluator_model, round_num)
                all_round_results[round_key] = eval_results
                save_single_round_eval_to_disk(
                    eval_results=eval_results,
                    idx=idx,
                    evaluator=evaluator_model,
                    round_num=round_num,
                    batch_temp_dir=BATCH_TEMP_DIR
                )
                print(f"    💾 Round {round_num} evaluation saved to disk")

                # If this is the last round, save as final results and stop
                if round_num == refinement_rounds:
                    final_results[(idx, evaluator_model)] = eval_results
                    print(f"    ✅ Final scores recorded")
                    break

                # Otherwise, refine the memo
                print(f"    Refining based on feedback...")
                refined_memo = refine_memo_based_on_feedback(
                    source_document=source_document,
                    original_prompt=original_prompt,
                    current_memo=current_memo,
                    evaluation_feedback=eval_results,
                    api_key=api_key
                )

                if refined_memo:
                    current_memo = refined_memo
                    print(f"    ✅ Memo refined ({len(refined_memo)} chars)")
                    # Save refined memo to disk
                    save_memo_to_disk(refined_memo, idx, round_num=round_num+1, batch_temp_dir=BATCH_TEMP_DIR)
                    print(f"    💾 Refined memo (round {round_num+1}) saved to disk")
                else:
                    print(f"    ❌ Refinement failed, keeping current memo")
                    # Continue with current memo

            current_round += 1

    print(f"\n{'='*70}")
    print(f"ITERATIVE REFINEMENT COMPLETE")
    print(f"{'='*70}")
    print(f"Final results: {len(final_results)} evaluations")
    print(f"All round results: {len(all_round_results)} evaluations (across all rounds)")
    print(f"{'='*70}\n")

    return final_results


def run_combined_refinement(
    memos: Dict[int, Dict],
    evaluator_models: List[str],
    refinement_rounds: int,
    api_key: str
) -> Dict:
    """
    Run combined refinement workflow where all evaluators' feedback is aggregated before refining.

    For each input:
    1. Evaluate memo with ALL evaluators → aggregate feedback
    2. For each refinement round:
       - Refine memo based on COMBINED feedback from all evaluators (using Claude)
       - Re-evaluate refined memo with ALL evaluators → aggregate feedback
    3. Return final evaluation scores for all evaluators

    Args:
        memos: Dict mapping index → memo info
        evaluator_models: List of evaluator model names
        refinement_rounds: Number of refinement iterations
        api_key: Anthropic API key for refinement calls

    Returns:
        Dict mapping (index, evaluator) → final evaluation results
        AND saves all intermediate artifacts (round 0 evals, round 1 memos, etc.) to disk
    """
    print(f"\n{'='*70}")
    print(f"COMBINED REFINEMENT WORKFLOW")
    print(f"{'='*70}")
    print(f"Refinement rounds: {refinement_rounds}")
    print(f"Evaluators: {len(evaluator_models)}")
    print(f"Inputs: {len([m for m in memos.values() if m.get('memo')])}")
    print(f"Total evaluations: {len([m for m in memos.values() if m.get('memo')]) * len(evaluator_models) * (refinement_rounds + 1)}")
    print(f"Mode: COMBINED (all evaluators' feedback aggregated before refinement)")
    print(f"{'='*70}\n")

    # Load original prompt (for refinement context)
    if PROMPT_FILE:
        prompt_path = PROMPT_FILE
    else:
        prompt_path = Path(__file__).parent.parent.parent / "prompts" / "baseline.txt"
    with open(prompt_path, 'r') as f:
        original_prompt = f.read()

    # Track final results
    final_results = {}

    # Process each input
    for idx, memo_info in memos.items():
        if not memo_info.get('memo'):
            print(f"⏭️  Skipping index {idx} (no memo generated)")
            continue

        print(f"\n{'='*70}")
        print(f"Processing input {idx}")
        print(f"{'='*70}\n")

        initial_memo = memo_info['memo']
        source_document = memo_info['credit_agreement']

        # Save initial memo (round 0) to disk
        save_memo_to_disk(initial_memo, idx, round_num=0, batch_temp_dir=BATCH_TEMP_DIR)

        current_memo = initial_memo

        # Iterate through refinement rounds
        for round_num in range(refinement_rounds + 1):
            print(f"\n  Round {round_num + 1}/{refinement_rounds + 1}:")

            # Evaluate with ALL evaluators
            print(f"    Evaluating with all {len(evaluator_models)} evaluators...")

            eval_results_list = []
            for evaluator_model in evaluator_models:
                print(f"      → {evaluator_model}...", end=' ')
                eval_result = evaluate_single_memo_sync(
                    memo=current_memo,
                    source_document=source_document,
                    evaluator_model=evaluator_model,
                    input_index=idx
                )

                if not eval_result:
                    print(f"✗ Failed")
                    continue

                eval_results_list.append(eval_result)

                # Display scores for this evaluator
                acc_score = eval_result.get('accuracy_result', {}).get('score', 0)
                comp_score = eval_result.get('completeness_result', {}).get('score', 0)
                cons_score = eval_result.get('consistency_result', {}).get('score', 0)
                qual_score = eval_result.get('quality_result', {}).get('quality_score', 0)
                avg_score = (acc_score + comp_score + cons_score + qual_score) / 4
                print(f"✓ (Avg={avg_score:.2f})")

                # Save evaluation results for THIS round and evaluator to disk
                save_single_round_eval_to_disk(
                    eval_results=eval_result,
                    idx=idx,
                    evaluator=evaluator_model,
                    round_num=round_num,
                    batch_temp_dir=BATCH_TEMP_DIR
                )

            if len(eval_results_list) != len(evaluator_models):
                print(f"    ❌ Some evaluations failed, cannot continue refinement for this input")
                break

            # Aggregate feedback from all evaluators
            print(f"    Aggregating feedback from {len(evaluator_models)} evaluators...")
            combined_feedback = aggregate_feedback_from_evaluators(eval_results_list, evaluator_models)

            # Display aggregated scores
            agg_acc = combined_feedback.get('accuracy_result', {}).get('score', 0)
            agg_comp = combined_feedback.get('completeness_result', {}).get('score', 0)
            agg_cons = combined_feedback.get('consistency_result', {}).get('score', 0)
            agg_qual = combined_feedback.get('quality_result', {}).get('quality_score', 0)
            agg_avg = (agg_acc + agg_comp + agg_cons + agg_qual) / 4
            print(f"    Aggregated scores: Acc={agg_acc:.2f}, Comp={agg_comp:.2f}, Cons={agg_cons:.2f}, Qual={agg_qual:.2f}, Avg={agg_avg:.2f}")

            # If this is the last round, save final results and stop
            if round_num == refinement_rounds:
                # Store final results for each evaluator
                for eval_result, evaluator_model in zip(eval_results_list, evaluator_models):
                    final_results[(idx, evaluator_model)] = eval_result
                print(f"    ✅ Final scores recorded for all evaluators")
                break

            # Otherwise, refine based on COMBINED feedback
            print(f"    Refining based on combined feedback from all evaluators...")
            refined_memo = refine_memo_based_on_feedback(
                source_document=source_document,
                original_prompt=original_prompt,
                current_memo=current_memo,
                evaluation_feedback=combined_feedback,  # Use combined feedback
                api_key=api_key
            )

            if refined_memo:
                current_memo = refined_memo
                print(f"    ✅ Memo refined ({len(refined_memo)} chars)")
                # Save refined memo to disk (with "combined" in the filename would be ideal but keeping it simple)
                save_memo_to_disk(refined_memo, idx, round_num=round_num+1, batch_temp_dir=BATCH_TEMP_DIR)
                print(f"    💾 Refined memo (round {round_num+1}) saved to disk")
            else:
                print(f"    ❌ Refinement failed, keeping current memo")
                # Continue with current memo

    print(f"\n{'='*70}")
    print(f"COMBINED REFINEMENT COMPLETE")
    print(f"{'='*70}")
    print(f"Final results: {len(final_results)} evaluations")
    print(f"{'='*70}\n")

    return final_results


def evaluate_memo_with_sync_api(
    memo: str,
    source_document: str,
    evaluator_model: str
) -> Optional[Dict]:
    """
    Synchronously evaluate a memo using direct API calls (not batch API).
    This is much faster for iterative refinement where we evaluate one memo at a time.

    Args:
        memo: Generated memo text
        source_document: Original credit agreement
        evaluator_model: Model to use for evaluation

    Returns:
        Dict with evaluation results, or None if evaluation failed
    """
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
    import os

    try:
        # Prepare prompts for all metrics
        accuracy_prompt = ACCURACY_PROMPT_TEMPLATE.format(
            source_document=source_document,
            memo=memo
        )
        completeness_prompt = COMPLETENESS_PROMPT_TEMPLATE.format(
            source_document=source_document,
            memo=memo
        )
        consistency_prompt = CONSISTENCY_PROMPT_TEMPLATE.format(memo=memo)
        clarity_prompt = CLARITY_PROMPT_TEMPLATE.format(memo=memo)
        tone_prompt = TONE_PROMPT_TEMPLATE.format(memo=memo)
        length_prompt = LENGTH_PROMPT_TEMPLATE.format(memo=memo)

        # Structure prompt requires a template parameter
        default_template = """1. Executive Summary/Overview
2. Transaction/Company Details
3. Financial Terms
4. Investment Strengths/Highlights
5. Risks and Concerns
6. Recommendation/Conclusion"""
        structure_prompt = STRUCTURE_PROMPT_TEMPLATE.format(template=default_template, memo=memo)

        results = {}

        if evaluator_model.startswith("gpt"):
            # Use OpenAI sync API
            print(f"      Using OpenAI sync API ({evaluator_model})...", flush=True)
            from openai import OpenAI

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment")

            client = OpenAI(api_key=api_key)

            # Evaluate each metric
            prompts = {
                'accuracy': accuracy_prompt,
                'completeness': completeness_prompt,
                'consistency': consistency_prompt,
                'clarity': clarity_prompt,
                'tone': tone_prompt,
                'length': length_prompt,
                'structure': structure_prompt
            }

            for metric, prompt in prompts.items():
                print(f"        → Evaluating {metric}...", end=' ', flush=True)

                # Retry logic for transient errors
                max_retries = 3
                retry_delay = 10  # seconds

                for attempt in range(max_retries):
                    try:
                        response = client.chat.completions.create(
                            model=evaluator_model,
                            messages=[{"role": "user", "content": prompt}]
                        )
                        results[metric] = response.choices[0].message.content
                        print("✓", flush=True)
                        break  # Success, exit retry loop
                    except Exception as e:
                        error_message = str(e)

                        # Check if it's a quota/credit error
                        is_quota, provider = is_quota_error(error_message)
                        if is_quota:
                            print(f"⚠️  Quota error detected", flush=True)
                            should_retry = prompt_user_for_credits(provider, error_message)
                            if should_retry:
                                # User added credits, retry this metric (don't count as attempt)
                                continue
                            else:
                                # User cancelled
                                print(f"✗ Cancelled by user", flush=True)
                                raise KeyboardInterrupt("User cancelled due to quota limits")

                        # Check if it's a retryable error (500, 529, rate limits)
                        is_retryable = (
                            "500" in error_message or
                            "529" in error_message or
                            "rate_limit" in error_message.lower() or
                            "overloaded" in error_message.lower()
                        )

                        if is_retryable and attempt < max_retries - 1:
                            # Retry with exponential backoff
                            wait_time = retry_delay * (2 ** attempt)
                            print(f"⚠️  (retry {attempt + 1}/{max_retries} in {wait_time}s)...", end=' ', flush=True)
                            import time
                            time.sleep(wait_time)
                        else:
                            # Non-retryable error or max retries reached
                            print(f"✗", flush=True)
                            raise  # Re-raise the exception

        elif "claude" in evaluator_model.lower():
            # Use Anthropic sync API
            print(f"      Using Anthropic sync API ({evaluator_model})...", flush=True)
            import anthropic

            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not found in environment")

            client = anthropic.Anthropic(api_key=api_key)

            # Evaluate each metric
            prompts = {
                'accuracy': accuracy_prompt,
                'completeness': completeness_prompt,
                'consistency': consistency_prompt,
                'clarity': clarity_prompt,
                'tone': tone_prompt,
                'length': length_prompt,
                'structure': structure_prompt
            }

            for metric, prompt in prompts.items():
                print(f"        → Evaluating {metric}...", end=' ', flush=True)

                # Retry logic for transient errors
                max_retries = 3
                retry_delay = 2  # seconds

                for attempt in range(max_retries):
                    try:
                        response = client.messages.create(
                            model=evaluator_model,
                            max_tokens=4096,
                            messages=[{"role": "user", "content": prompt}]
                        )
                        results[metric] = response.content[0].text
                        print("✓", flush=True)
                        break  # Success, exit retry loop
                    except Exception as e:
                        error_message = str(e)

                        # Check if it's a quota/credit error
                        is_quota, provider = is_quota_error(error_message)
                        if is_quota:
                            print(f"⚠️  Quota error detected", flush=True)
                            should_retry = prompt_user_for_credits(provider, error_message)
                            if should_retry:
                                # User added credits, retry this metric (don't count as attempt)
                                continue
                            else:
                                # User cancelled
                                print(f"✗ Cancelled by user", flush=True)
                                raise KeyboardInterrupt("User cancelled due to quota limits")

                        # Check if it's a retryable error (500, 529, rate limits)
                        is_retryable = (
                            "500" in error_message or
                            "529" in error_message or
                            "rate_limit" in error_message.lower() or
                            "overloaded" in error_message.lower()
                        )

                        if is_retryable and attempt < max_retries - 1:
                            # Retry with exponential backoff
                            wait_time = retry_delay * (2 ** attempt)
                            print(f"⚠️  (retry {attempt + 1}/{max_retries} in {wait_time}s)...", end=' ', flush=True)
                            import time
                            time.sleep(wait_time)
                        else:
                            # Non-retryable error or max retries reached
                            print(f"✗", flush=True)
                            raise  # Re-raise the exception

        elif "gemini" in evaluator_model.lower():
            # Use Gemini sync API
            print(f"      Using Gemini sync API ({evaluator_model})...", flush=True)

            # Suppress Python 3.9 deprecation warnings from google.api_core
            import warnings
            warnings.filterwarnings("ignore", category=FutureWarning, module="google.api_core")
            warnings.filterwarnings("ignore", message=".*importlib.metadata.*")

            import google.generativeai as genai

            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found in environment")

            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(evaluator_model)

            # Evaluate each metric
            prompts = {
                'accuracy': accuracy_prompt,
                'completeness': completeness_prompt,
                'consistency': consistency_prompt,
                'clarity': clarity_prompt,
                'tone': tone_prompt,
                'length': length_prompt,
                'structure': structure_prompt
            }

            for metric, prompt in prompts.items():
                print(f"        → Evaluating {metric}...", end=' ', flush=True)

                # Retry logic for transient errors
                max_retries = 3
                retry_delay = 2  # seconds

                for attempt in range(max_retries):
                    try:
                        response = model.generate_content(prompt)
                        results[metric] = response.text
                        print("✓", flush=True)
                        break  # Success, exit retry loop
                    except Exception as e:
                        error_message = str(e)

                        # Check if it's a quota/credit error
                        is_quota, provider = is_quota_error(error_message)
                        if is_quota:
                            print(f"⚠️  Quota error detected", flush=True)
                            should_retry = prompt_user_for_credits(provider, error_message)
                            if should_retry:
                                # User added credits, retry this metric (don't count as attempt)
                                continue
                            else:
                                # User cancelled
                                print(f"✗ Cancelled by user", flush=True)
                                raise KeyboardInterrupt("User cancelled due to quota limits")

                        # Check if it's a retryable error (500, 529, rate limits)
                        is_retryable = (
                            "500" in error_message or
                            "529" in error_message or
                            "rate_limit" in error_message.lower() or
                            "overloaded" in error_message.lower() or
                            "resource exhausted" in error_message.lower()
                        )

                        if is_retryable and attempt < max_retries - 1:
                            # Retry with exponential backoff
                            wait_time = retry_delay * (2 ** attempt)
                            print(f"⚠️  (retry {attempt + 1}/{max_retries} in {wait_time}s)...", end=' ', flush=True)
                            import time
                            time.sleep(wait_time)
                        else:
                            # Non-retryable error or max retries reached
                            print(f"✗", flush=True)
                            raise  # Re-raise the exception
        else:
            raise ValueError(f"Unknown evaluator model: {evaluator_model}")

        # Parse results using existing parsers and format like batch API
        # Accuracy - returns tuple (vote, hallucinations)
        accuracy_vote, accuracy_hallucinations = _parse_accuracy_response(results['accuracy'])
        accuracy_result = {
            "accurate": accuracy_vote == "NO",
            "score": 1.0 if accuracy_vote == "NO" else 0.0,
            "votes": {
                evaluator_model: {
                    "vote": accuracy_vote,
                    "hallucinations": accuracy_hallucinations
                }
            },
            "consensus_reached": True,
            "yes_votes": 1 if accuracy_vote == "YES" else 0,
            "no_votes": 1 if accuracy_vote == "NO" else 0
        }

        # Completeness - returns tuple (vote, missing_terms)
        completeness_vote, completeness_missing = _parse_completeness_response(results['completeness'])
        completeness_result = {
            "complete": completeness_vote == "NO",
            "score": 1.0 if completeness_vote == "NO" else 0.0,
            "votes": {
                evaluator_model: {
                    "vote": completeness_vote,
                    "missing_terms": completeness_missing
                }
            },
            "consensus_reached": True,
            "yes_votes": 1 if completeness_vote == "YES" else 0,
            "no_votes": 1 if completeness_vote == "NO" else 0
        }

        # Consistency - returns dict
        consistency_result = _parse_consistency_response(results['consistency'])

        # Quality - parse scores
        clarity_score = _parse_quality_score(results['clarity'])
        tone_score = _parse_quality_score(results['tone'])
        length_score = _parse_quality_score(results['length'])
        structure_score = _parse_quality_score(results['structure'])

        quality_scores = [clarity_score, tone_score, length_score, structure_score]
        quality_avg = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

        quality_result = {
            "quality_score": quality_avg,
            "clarity_score": clarity_score,
            "tone_score": tone_score,
            "length_score": length_score,
            "structure_score": structure_score,
            "votes": {
                evaluator_model: {
                    "clarity": clarity_score,
                    "tone": tone_score,
                    "length": length_score,
                    "structure": structure_score
                }
            }
        }

        return {
            'accuracy_result': accuracy_result,
            'completeness_result': completeness_result,
            'consistency_result': consistency_result,
            'quality_result': quality_result
        }

    except Exception as e:
        print(f"      Error in sync evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None


def evaluate_single_memo_sync(memo: str, source_document: str, evaluator_model: str, input_index: int) -> Optional[Dict]:
    """
    Synchronously evaluate a single memo.

    For iterative refinement, uses direct API calls for speed.
    For standard evaluation, would use batch API (but this function is only called during refinement).

    Returns:
        Dict with evaluation results, or None if evaluation failed
    """
    try:
        # Use synchronous API for fast evaluation during iterative refinement
        results = evaluate_memo_with_sync_api(
            memo=memo,
            source_document=source_document,
            evaluator_model=evaluator_model
        )

        if results:
            return results
        else:
            return None

    except Exception as e:
        print(f"      Error evaluating memo: {e}")
        return None


def save_batch_job_mappings(batch_jobs: List[Dict], temp_dir: Path):
    """
    Save batch job mappings to a JSON file for debugging and recovery.
    This allows resuming downloads even if the process is interrupted.

    If the file already exists, merges new jobs with existing ones
    (updates if same index/evaluator, otherwise appends).
    """
    mapping_file = temp_dir / "batch_job_mappings.json"

    # Load existing mappings if they exist
    if mapping_file.exists():
        with open(mapping_file, 'r') as f:
            mappings = json.load(f)
        print(f"📂 Found existing batch job mappings with {len(mappings.get('jobs', []))} jobs")
        existing_jobs = {(j['input_index'], j['evaluator_model']): j for j in mappings.get('jobs', [])}
    else:
        mappings = {
            "created_at": datetime.now().isoformat(),
            "jobs": []
        }
        existing_jobs = {}

    # Update timestamp if we're adding new jobs
    if batch_jobs:
        mappings["updated_at"] = datetime.now().isoformat()

    # Add/update new jobs
    new_jobs_count = 0
    updated_jobs_count = 0

    for job in batch_jobs:
        if job.get('batch_id'):  # Only save jobs that were successfully created
            key = (job["input_index"], job["evaluator_model"])
            job_entry = {
                "input_index": job["input_index"],
                "evaluator_model": job["evaluator_model"],
                "provider": job["provider"],
                "batch_id": job["batch_id"]
            }

            if key in existing_jobs:
                # Update existing job
                existing_jobs[key] = job_entry
                updated_jobs_count += 1
            else:
                # Add new job
                existing_jobs[key] = job_entry
                new_jobs_count += 1

    # Convert back to list
    mappings["jobs"] = list(existing_jobs.values())

    # Save to file
    with open(mapping_file, 'w') as f:
        json.dump(mappings, f, indent=2)

    if new_jobs_count > 0 or updated_jobs_count > 0:
        print(f"💾 Updated batch job mappings: {new_jobs_count} new, {updated_jobs_count} updated")
        print(f"   Total jobs in mapping file: {len(mappings['jobs'])}")
        print(f"   Location: {mapping_file.name}\n")
    else:
        print(f"💾 No new batch jobs to add to mappings file\n")


def submit_all_batch_jobs(memos: Dict[int, Dict], evaluator_models: List[str]) -> List[Dict]:
    """
    Phase 2: Submit ALL batch jobs at once WITHOUT waiting.

    Returns list of batch job info dicts.
    """
    print(f"\n{'='*70}")
    print(f"PHASE 2: SUBMITTING ALL BATCH JOBS (NO WAITING)")
    print(f"{'='*70}")
    print(f"Evaluator models: {', '.join(evaluator_models)}")
    print(f"Total memos: {len([m for m in memos.values() if m['memo'] is not None])}")
    print(f"Total batch jobs to submit: {len([m for m in memos.values() if m['memo'] is not None]) * len(evaluator_models)}")
    print(f"{'='*70}\n")

    batch_jobs = []

    # Get API keys (from environment or .env file)
    openai_key = load_api_key_from_env("OPENAI_API_KEY")
    anthropic_key = load_api_key_from_env("ANTHROPIC_API_KEY")
    gemini_key = load_api_key_from_env("GEMINI_API_KEY")

    for idx, memo_data in memos.items():
        if memo_data['memo'] is None:
            print(f"⏭️  Skipping input {idx} (no memo generated)")
            continue

        print(f"\n📝 Submitting batch jobs for input {idx}...")

        for eval_model in evaluator_models:
            try:
                print(f"  🚀 Submitting {eval_model} batch job...", end=" ")

                if eval_model.startswith("gpt"):
                    # GPT batch
                    requests = create_batch_requests_for_memo(
                        memo=memo_data['memo'],
                        source_document=memo_data['credit_agreement'],
                        template=None,
                        model=eval_model
                    )
                    file_id = upload_batch_file(requests, BATCH_TEMP_DIR, openai_key, input_index=idx)
                    batch_id = create_batch_job(file_id, openai_key, f"Eval {idx} with {eval_model}")

                    batch_jobs.append({
                        "input_index": idx,
                        "evaluator_model": eval_model,
                        "provider": "openai",
                        "batch_id": batch_id,
                        "file_id": file_id
                    })
                    print(f"✅ Job ID: {batch_id}")

                elif "claude" in eval_model.lower():
                    # Claude batch
                    requests = create_claude_batch_requests_for_memo(
                        memo=memo_data['memo'],
                        source_document=memo_data['credit_agreement'],
                        template=None,
                        model=eval_model
                    )
                    batch_id = create_claude_batch(requests, anthropic_key)

                    batch_jobs.append({
                        "input_index": idx,
                        "evaluator_model": eval_model,
                        "provider": "anthropic",
                        "batch_id": batch_id
                    })
                    print(f"✅ Job ID: {batch_id}")

                elif "gemini" in eval_model.lower():
                    # Gemini batch
                    requests = create_gemini_batch_requests_for_memo(
                        memo=memo_data['memo'],
                        source_document=memo_data['credit_agreement'],
                        template=None,
                        model=eval_model
                    )
                    batch_id = create_gemini_batch(requests, gemini_key, eval_model)

                    batch_jobs.append({
                        "input_index": idx,
                        "evaluator_model": eval_model,
                        "provider": "gemini",
                        "batch_id": batch_id
                    })
                    print(f"✅ Job ID: {batch_id}")

            except Exception as e:
                print(f"❌ Error: {e}")
                batch_jobs.append({
                    "input_index": idx,
                    "evaluator_model": eval_model,
                    "provider": None,
                    "batch_id": None,
                    "error": str(e)
                })

    print(f"\n{'='*70}")
    print(f"ALL BATCH JOBS SUBMITTED!")
    print(f"{'='*70}")
    print(f"Total jobs submitted: {len([j for j in batch_jobs if j.get('batch_id')])}")
    print(f"Failed submissions: {len([j for j in batch_jobs if not j.get('batch_id')])}")
    print(f"{'='*70}\n")

    # Save batch job mappings to file for debugging and recovery
    save_batch_job_mappings(batch_jobs, BATCH_TEMP_DIR)

    return batch_jobs


def poll_all_batch_jobs(batch_jobs: List[Dict], poll_interval: int = 60) -> Dict[str, Dict]:
    """
    Phase 3: Poll all batch jobs until complete.

    Returns dict mapping batch_id -> results.
    """
    print(f"\n{'='*70}")
    print(f"PHASE 3: POLLING ALL BATCH JOBS")
    print(f"{'='*70}")
    print(f"Total jobs to poll: {len([j for j in batch_jobs if j.get('batch_id')])}")
    print(f"Poll interval: {poll_interval} seconds")
    print(f"{'='*70}\n")

    # Get API keys (from environment or .env file)
    openai_key = load_api_key_from_env("OPENAI_API_KEY")
    anthropic_key = load_api_key_from_env("ANTHROPIC_API_KEY")
    gemini_key = load_api_key_from_env("GEMINI_API_KEY")

    results = {}
    completed = set()
    failed = set()

    jobs_to_poll = {j["batch_id"]: j for j in batch_jobs if j.get("batch_id")}

    start_time = time.time()
    check_count = 0

    while len(completed) + len(failed) < len(jobs_to_poll):
        check_count += 1
        elapsed = int(time.time() - start_time)

        print(f"\n[Check #{check_count}, {elapsed}s elapsed]")
        print(f"  Completed: {len(completed)}/{len(jobs_to_poll)}")
        print(f"  Failed: {len(failed)}/{len(jobs_to_poll)}")
        print(f"  Still running: {len(jobs_to_poll) - len(completed) - len(failed)}")

        for batch_id, job_info in jobs_to_poll.items():
            if batch_id in completed or batch_id in failed:
                continue

            try:
                provider = job_info["provider"]

                if provider == "openai":
                    status_data = check_batch_status(batch_id, openai_key)
                    status = status_data.get("status")

                    if status == "completed":
                        output_file_id = status_data.get("output_file_id")
                        output_path = download_batch_results(output_file_id, BATCH_TEMP_DIR, openai_key, input_index=job_info['input_index'])
                        batch_results = load_batch_results(output_path)
                        parsed = parse_batch_results(batch_results)
                        results[batch_id] = parsed
                        completed.add(batch_id)
                        print(f"  ✅ {job_info['evaluator_model']} for input {job_info['input_index']}: COMPLETE")

                    elif status in ["failed", "expired", "cancelled"]:
                        failed.add(batch_id)
                        print(f"  ❌ {job_info['evaluator_model']} for input {job_info['input_index']}: {status}")

                elif provider == "anthropic":
                    status_data = check_claude_batch_status(batch_id, anthropic_key)
                    processing_status = status_data.get("processing_status")

                    if processing_status == "ended":
                        results_url = status_data.get("results_url")
                        output_path = download_claude_batch_results(results_url, BATCH_TEMP_DIR, anthropic_key, input_index=job_info['input_index'])
                        with open(output_path, 'r') as f:
                            batch_results = [json.loads(line) for line in f]
                        parsed = parse_claude_batch_results(batch_results)
                        results[batch_id] = parsed
                        completed.add(batch_id)
                        print(f"  ✅ {job_info['evaluator_model']} for input {job_info['input_index']}: COMPLETE")

                    elif processing_status in ["failed", "expired", "cancelled"]:
                        failed.add(batch_id)
                        print(f"  ❌ {job_info['evaluator_model']} for input {job_info['input_index']}: {processing_status}")

                elif provider == "gemini":
                    status_data = check_gemini_batch_status(batch_id, gemini_key)
                    # State is nested in metadata for Gemini API
                    state = status_data.get("metadata", {}).get("state")

                    if state == "BATCH_STATE_SUCCEEDED":
                        output_path = extract_gemini_batch_results(status_data, BATCH_TEMP_DIR, input_index=job_info['input_index'])
                        with open(output_path, 'r') as f:
                            batch_results = [json.loads(line) for line in f]
                        parsed = parse_gemini_batch_results(batch_results)
                        results[batch_id] = parsed
                        completed.add(batch_id)
                        print(f"  ✅ {job_info['evaluator_model']} for input {job_info['input_index']}: COMPLETE")

                    elif state in ["BATCH_STATE_FAILED", "BATCH_STATE_CANCELLED"]:
                        failed.add(batch_id)
                        print(f"  ❌ {job_info['evaluator_model']} for input {job_info['input_index']}: {state}")

            except Exception as e:
                print(f"  ⚠️  Error checking {batch_id}: {e}")

        if len(completed) + len(failed) < len(jobs_to_poll):
            print(f"\n  ⏳ Waiting {poll_interval}s before next check...")
            time.sleep(poll_interval)

    print(f"\n{'='*70}")
    print(f"POLLING COMPLETE")
    print(f"{'='*70}")
    print(f"Completed: {len(completed)}/{len(jobs_to_poll)}")
    print(f"Failed: {len(failed)}/{len(jobs_to_poll)}")
    print(f"Total time: {int(time.time() - start_time)}s")
    print(f"{'='*70}\n")

    return results, jobs_to_poll


def aggregate_all_results(
    memos: Dict[int, Dict],
    batch_results: Dict[str, Dict],
    batch_jobs: List[Dict],
    indices: List[int],
    evaluator_models: List[str]
) -> Dict:
    """Phase 4: Aggregate all results."""
    print(f"\n{'='*70}")
    print(f"PHASE 4: AGGREGATING RESULTS")
    print(f"{'='*70}\n")

    # Map batch_id -> job_info
    batch_id_to_job = {j["batch_id"]: j for j in batch_jobs if j.get("batch_id")}

    # Group results by input_index
    results_by_input = {}
    for batch_id, parsed_results in batch_results.items():
        job_info = batch_id_to_job.get(batch_id)
        if not job_info:
            continue

        input_idx = job_info["input_index"]
        if input_idx not in results_by_input:
            results_by_input[input_idx] = []

        # Add evaluator model info to results
        evaluator_result = {
            "evaluator_model": job_info["evaluator_model"],
            "summary_score": None,
            "metrics": parsed_results
        }

        # Calculate summary score from metrics
        from evals.metrics import calculate_summary_score
        summary_result = calculate_summary_score(
            accuracy_result=parsed_results["accuracy_result"],
            completeness_result=parsed_results["completeness_result"],
            consistency_result=parsed_results["consistency_result"],
            quality_result=parsed_results["quality_result"]
        )
        evaluator_result["summary_score"] = summary_result["summary_score"]
        evaluator_result["metrics"] = {
            "accuracy": parsed_results["accuracy_result"],
            "completeness": parsed_results["completeness_result"],
            "consistency": parsed_results["consistency_result"],
            "quality": parsed_results["quality_result"]
        }

        results_by_input[input_idx].append(evaluator_result)

    # Aggregate per input
    detailed_results = []
    for idx in indices:
        memo_data = memos.get(idx)
        evaluator_results = results_by_input.get(idx, [])

        if not memo_data or not evaluator_results:
            detailed_results.append({
                "input_index": idx,
                "source_url": memo_data['source_url'] if memo_data else None,
                "summary_score": None,
                "metrics": None,
                "evaluator_results": None,
                "error": memo_data.get('error') if memo_data else "Unknown error"
            })
            continue

        # Aggregate evaluator results
        aggregated = aggregate_evaluator_results(evaluator_results)

        if aggregated:
            detailed_results.append({
                "input_index": idx,
                "source_url": memo_data['source_url'],
                "summary_score": aggregated["summary_score"],
                "metrics": aggregated["metrics"],
                "evaluator_results": evaluator_results,
                "error": None
            })
            print(f"✅ Input {idx}: {aggregated['summary_score']:.2f}/100")
        else:
            detailed_results.append({
                "input_index": idx,
                "source_url": memo_data['source_url'],
                "summary_score": None,
                "metrics": None,
                "evaluator_results": None,
                "error": "Failed to aggregate results"
            })
            print(f"❌ Input {idx}: Failed to aggregate")

    # Calculate summary statistics
    valid_scores = [r['summary_score'] for r in detailed_results if r['summary_score'] is not None]

    if not valid_scores:
        summary_statistics = {
            "mean_score": 0.0,
            "median_score": 0.0,
            "worst_score": 0.0,
            "best_score": 0.0,
            "std_dev": 0.0,
            "score_range": 0.0,
            "successful_evals": 0,
            "failed_evals": len(detailed_results)
        }
        metric_statistics = None
    else:
        summary_statistics = {
            "mean_score": statistics.mean(valid_scores),
            "median_score": statistics.median(valid_scores),
            "worst_score": min(valid_scores),
            "best_score": max(valid_scores),
            "std_dev": statistics.stdev(valid_scores) if len(valid_scores) > 1 else 0.0,
            "score_range": max(valid_scores) - min(valid_scores),
            "successful_evals": len(valid_scores),
            "failed_evals": len(detailed_results) - len(valid_scores)
        }

        # Per-metric aggregated statistics
        valid_results = [r for r in detailed_results if r['metrics'] is not None]

        if valid_results:
            accuracy_scores = [r['metrics']['accuracy']['score'] * 100 for r in valid_results]
            completeness_scores = [r['metrics']['completeness']['score'] * 100 for r in valid_results]
            consistency_scores = [r['metrics']['consistency']['score'] * 100 for r in valid_results]
            quality_scores = [r['metrics']['quality']['quality_score'] for r in valid_results]
            clarity_scores = [r['metrics']['quality']['clarity_score'] for r in valid_results]
            tone_scores = [r['metrics']['quality']['tone_score'] for r in valid_results]
            length_scores = [r['metrics']['quality']['length_score'] for r in valid_results]
            structure_scores = [r['metrics']['quality']['structure_score'] for r in valid_results]

            metric_statistics = {
                "accuracy": {
                    "mean": statistics.mean(accuracy_scores),
                    "median": statistics.median(accuracy_scores),
                    "min": min(accuracy_scores),
                    "max": max(accuracy_scores),
                    "std_dev": statistics.stdev(accuracy_scores) if len(accuracy_scores) > 1 else 0.0
                },
                "completeness": {
                    "mean": statistics.mean(completeness_scores),
                    "median": statistics.median(completeness_scores),
                    "min": min(completeness_scores),
                    "max": max(completeness_scores),
                    "std_dev": statistics.stdev(completeness_scores) if len(completeness_scores) > 1 else 0.0
                },
                "consistency": {
                    "mean": statistics.mean(consistency_scores),
                    "median": statistics.median(consistency_scores),
                    "min": min(consistency_scores),
                    "max": max(consistency_scores),
                    "std_dev": statistics.stdev(consistency_scores) if len(consistency_scores) > 1 else 0.0
                },
                "quality": {
                    "mean": statistics.mean(quality_scores),
                    "median": statistics.median(quality_scores),
                    "min": min(quality_scores),
                    "max": max(quality_scores),
                    "std_dev": statistics.stdev(quality_scores) if len(quality_scores) > 1 else 0.0
                },
                "clarity": {
                    "mean": statistics.mean(clarity_scores),
                    "median": statistics.median(clarity_scores),
                    "min": min(clarity_scores),
                    "max": max(clarity_scores),
                    "std_dev": statistics.stdev(clarity_scores) if len(clarity_scores) > 1 else 0.0
                },
                "tone": {
                    "mean": statistics.mean(tone_scores),
                    "median": statistics.median(tone_scores),
                    "min": min(tone_scores),
                    "max": max(tone_scores),
                    "std_dev": statistics.stdev(tone_scores) if len(tone_scores) > 1 else 0.0
                },
                "length": {
                    "mean": statistics.mean(length_scores),
                    "median": statistics.median(length_scores),
                    "min": min(length_scores),
                    "max": max(length_scores),
                    "std_dev": statistics.stdev(length_scores) if len(length_scores) > 1 else 0.0
                },
                "structure": {
                    "mean": statistics.mean(structure_scores),
                    "median": statistics.median(structure_scores),
                    "min": min(structure_scores),
                    "max": max(structure_scores),
                    "std_dev": statistics.stdev(structure_scores) if len(structure_scores) > 1 else 0.0
                }
            }
        else:
            metric_statistics = None

    results = {
        "model_evaluated": MODEL_TO_EVALUATE,
        "evaluator_models": evaluator_models,
        "dataset": str(TRAIN_FILE),
        "evaluated_indices": indices,
        "summary_statistics": summary_statistics,
        "metric_statistics": metric_statistics,
        "detailed_results": detailed_results
    }

    print(f"\n{'='*70}")
    print(f"AGGREGATION COMPLETE")
    print(f"{'='*70}")
    print(f"Successful evals: {summary_statistics['successful_evals']}")
    print(f"Failed evals: {summary_statistics['failed_evals']}")
    print(f"Mean score: {summary_statistics['mean_score']:.2f}/100")
    print(f"{'='*70}\n")

    return results


def save_results(results: Dict, sampling_info: Dict, output_dir: Path):
    """Save results."""
    comprehensive_results = {
        "model": results["model_evaluated"],
        "evaluator_models": results["evaluator_models"],
        "random_seed": sampling_info["random_seed"],
        "sample_size": sampling_info["total_sampled"],
        "sampling_breakdown": sampling_info["sampling_breakdown"],
        "sampled_indices": sampling_info["all_sampled_indices"],

        "mean_score": results["summary_statistics"]["mean_score"],
        "median_score": results["summary_statistics"]["median_score"],
        "worst_score": results["summary_statistics"]["worst_score"],
        "best_score": results["summary_statistics"]["best_score"],
        "std_dev": results["summary_statistics"]["std_dev"],
        "score_range": results["summary_statistics"]["score_range"],

        "total_inputs_in_dataset": sampling_info["total_inputs_in_dataset"],
        "successful_evals": results["summary_statistics"]["successful_evals"],
        "failed_evals": results["summary_statistics"]["failed_evals"],

        "metric_statistics": results["metric_statistics"],

        "all_results": [
            {
                "input_index": r["input_index"],
                "source_url": r["source_url"],
                "score": r["summary_score"],
                "error": r["error"]
            }
            for r in results["detailed_results"]
        ],

        "detailed_results": results["detailed_results"],
        "created_at": sampling_info["created_at"]
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"comprehensive_batch_eval_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(comprehensive_results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"RESULTS SAVED")
    print(f"{'='*70}")
    print(f"Results file: {results_file}")

    sampling_file = output_dir / f"comprehensive_sampled_indices_{timestamp}.json"
    with open(sampling_file, 'w') as f:
        json.dump(sampling_info, f, indent=2)

    print(f"Sampling info: {sampling_file}")
    print(f"{'='*70}\n")

    return results_file, sampling_file


def main():
    """Main execution function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run truly parallelized batch evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default comprehensive sample (50 indices)
  python run_truly_parallel_batch_eval.py

  # Test with specific indices
  python run_truly_parallel_batch_eval.py --indices 0 1 2 6 12

  # Test with just one index
  python run_truly_parallel_batch_eval.py --indices 128
        """
    )
    parser.add_argument(
        '--indices',
        type=int,
        nargs='+',
        help='Custom indices to evaluate (space-separated). If not provided, uses default comprehensive sample.'
    )
    parser.add_argument(
        '--parallel-memos',
        action='store_true',
        help='Generate memos in parallel using Claude Batch API (MUCH faster, recommended)'
    )
    parser.add_argument(
        '--run-name',
        type=str,
        default=DEFAULT_RUN_NAME,
        help=f'Name for this run (determines output directories: batch_temp_<name>, results_<name>). Default: {DEFAULT_RUN_NAME}'
    )
    parser.add_argument(
        '--prompt',
        type=str,
        default=None,
        help='Path to custom prompt file (e.g., prompts/my_prompt.txt). If not provided, uses prompts/baseline.txt'
    )
    parser.add_argument(
        '--evaluators',
        type=str,
        nargs='+',
        choices=['gpt-5', 'claude-sonnet-4-20250514', 'gemini-2.5-pro', 'openai', 'claude', 'gemini'],
        default=None,
        help='Evaluator(s) to run. Can use short names (openai, claude, gemini) or full model names. Default: all 3 evaluators'
    )
    parser.add_argument(
        '--skip-memo-generation',
        action='store_true',
        help='Skip memo generation and use existing batch inputs (useful for re-running specific evaluators)'
    )
    parser.add_argument(
        '--few-shot-dir',
        type=str,
        default=None,
        help='Path to directory containing few-shot examples (with input_*.txt and example_*.md files)'
    )
    parser.add_argument(
        '--use-system-parameter',
        action='store_true',
        help='Use Claude\'s native system parameter for better instruction following (only affects Claude API calls)'
    )
    parser.add_argument(
        '--use-xml-tags',
        action='store_true',
        help='Wrap inputs in XML tags for better structure. Recommended for Claude with long documents.'
    )
    parser.add_argument(
        '--refinement-rounds',
        type=int,
        default=0,
        help='Number of iterative refinement rounds per evaluator. Default: 0 (no refinement)'
    )
    parser.add_argument(
        '--refinement-mode',
        type=str,
        choices=['independent', 'combined'],
        default='independent',
        help='Refinement mode: "independent" = each evaluator creates separate trajectory, "combined" = aggregate all 3 evaluators feedback before refinement. Default: independent'
    )
    args = parser.parse_args()

    # Declare global variables that we'll modify
    global BATCH_TEMP_DIR, PROMPT_FILE, EVALUATOR_MODELS

    # Set up directories based on run name
    BATCH_TEMP_DIR = OUTPUT_DIR / args.run_name
    RESULTS_DIR = OUTPUT_DIR / f"results_{args.run_name.replace('batch_temp_', '')}"
    BATCH_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Set prompt file
    PROMPT_FILE = Path(args.prompt) if args.prompt else DEFAULT_PROMPT_FILE

    # Load few-shot examples if specified
    few_shot_examples = None
    if args.few_shot_dir:
        few_shot_dir = Path(args.few_shot_dir)
        print(f"\nLoading few-shot examples from {few_shot_dir}...")
        few_shot_examples = load_few_shot_examples(few_shot_dir)
        if few_shot_examples:
            print(f"✅ Successfully loaded {len(few_shot_examples)} few-shot examples\n")
        else:
            print(f"⚠️  No few-shot examples found in {few_shot_dir}\n")

    # Process evaluators argument
    if args.evaluators:
        # Map short names to full model names
        evaluator_map = {
            'openai': 'gpt-5',
            'claude': 'claude-sonnet-4-20250514',
            'gemini': 'gemini-2.5-pro',
            'gpt-5': 'gpt-5',
            'claude-sonnet-4-20250514': 'claude-sonnet-4-20250514',
            'gemini-2.5-pro': 'gemini-2.5-pro'
        }
        EVALUATOR_MODELS = [evaluator_map[e] for e in args.evaluators]

    print(f"✓ Run name: {args.run_name}")
    print(f"✓ Batch temp directory: {BATCH_TEMP_DIR}")
    print(f"✓ Results directory: {RESULTS_DIR}")
    print(f"✓ Prompt file: {PROMPT_FILE if PROMPT_FILE else 'prompts/baseline.txt (default)'}")
    print(f"✓ Evaluators: {', '.join(EVALUATOR_MODELS)}")
    print(f"✓ Skip memo generation: {args.skip_memo_generation}")

    print(f"\n{'='*70}")
    print(f"TRULY PARALLELIZED COMPREHENSIVE BATCH EVALUATION")
    print(f"{'='*70}\n")

    # Load API keys from environment or .env file and set in os.environ
    # This ensures subprocess calls (like model_run.py) can access them
    api_keys = {
        "OPENAI_API_KEY": load_api_key_from_env("OPENAI_API_KEY"),
        "ANTHROPIC_API_KEY": load_api_key_from_env("ANTHROPIC_API_KEY"),
        "GEMINI_API_KEY": load_api_key_from_env("GEMINI_API_KEY")
    }

    # Set in os.environ so subprocesses can access them
    for key, value in api_keys.items():
        if value:
            os.environ[key] = value
        else:
            print(f"⚠️  Warning: {key} not found in environment or .env file")

    # Determine which indices to use
    if args.indices:
        # Use custom indices from command line
        indices_to_evaluate = args.indices
        sampling_info = {
            'all_sampled_indices': indices_to_evaluate,
            'total_sampled': len(indices_to_evaluate),
            'source': 'command_line_custom',
            'baseline_count': 0,
            'first_n_count': 0,
            'random_sample_count': 0
        }
        print(f"Using custom indices from command line: {indices_to_evaluate}")
        print(f"  Total indices: {len(indices_to_evaluate)}\n")
    else:
        # Use default comprehensive sample
        # Load baseline sampled indices
        print(f"Loading baseline sampled indices from {BASELINE_SAMPLED_INDICES_FILE}...")
        baseline_indices = load_baseline_sampled_indices(BASELINE_SAMPLED_INDICES_FILE)
        print(f"  Loaded {len(baseline_indices)} baseline indices: {baseline_indices}")

        # Count total samples
        print(f"\nCounting samples in {TRAIN_FILE}...")
        total_samples = count_train_samples(TRAIN_FILE)
        print(f"  Total samples in dataset: {total_samples}")

        # Create comprehensive sample
        print(f"\nCreating comprehensive sample...")
        sampling_info = create_comprehensive_sample(
            baseline_indices=baseline_indices,
            first_n=3,
            random_sample_size=37,
            total_samples=total_samples,
            seed=RANDOM_SEED
        )

        indices_to_evaluate = sampling_info['all_sampled_indices']
        print(f"\n  Created comprehensive sample with {sampling_info['total_sampled']} total indices")

    # Phase 1: Generate all memos (or load existing)
    if args.skip_memo_generation:
        print(f"\n{'='*70}")
        print(f"PHASE 1: LOADING EXISTING MEMOS FROM BATCH INPUTS")
        print(f"{'='*70}\n")
        print(f"Loading memos from existing batch input files in {BATCH_TEMP_DIR}...")

        memos = {}
        for idx in indices_to_evaluate:
            # Find existing batch input file for this index
            input_files = list(BATCH_TEMP_DIR.glob(f"batch_input_{idx}_*.jsonl"))
            if input_files:
                # Use most recent if multiple exist
                input_file = sorted(input_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]

                # Load the memo from the batch input file
                with open(input_file, 'r') as f:
                    first_request = json.loads(f.readline())
                    content = first_request['body']['messages'][0]['content']

                    # Extract memo from content
                    if 'GENERATED MEMO:' in content:
                        memo_start = content.find('GENERATED MEMO:') + len('GENERATED MEMO:')
                        memo_end = content.find('\n\nDoes the memo contain')
                        if memo_end == -1:
                            memo_end = content.find('\n\nAre any key')
                        if memo_end == -1:
                            memo_end = content.find('\n\nGoal')

                        memo_text = content[memo_start:memo_end].strip() if memo_end != -1 else content[memo_start:].strip()

                        # Get source URL and credit agreement from train.jsonl
                        source_url, credit_agreement_text = load_training_sample(TRAIN_FILE, idx)

                        memos[idx] = {
                            'source_url': source_url,
                            'memo': memo_text,
                            'credit_agreement': credit_agreement_text
                        }
                        print(f"  ✓ Loaded memo for index {idx}")
                    else:
                        print(f"  ⚠️  Could not extract memo from batch input for index {idx}")
                        memos[idx] = {'error': 'Could not extract memo from batch input'}
            else:
                print(f"  ⚠️  No batch input file found for index {idx}")
                memos[idx] = {'error': 'No batch input file found'}

        print(f"\n✅ Loaded {len([m for m in memos.values() if 'error' not in m])}/{len(indices_to_evaluate)} memos from existing batch inputs")

    elif args.parallel_memos:
        # Use parallel batch generation (faster)
        if not api_keys["ANTHROPIC_API_KEY"]:
            print("❌ ERROR: ANTHROPIC_API_KEY not found")
            print("   Checked:")
            print("   - Environment variable ANTHROPIC_API_KEY")
            print("   - .env file at project root")
            print("   Parallel memo generation requires Claude API access")
            sys.exit(1)

        memos = generate_all_memos_parallel(
            indices=indices_to_evaluate,
            train_file=TRAIN_FILE,
            model=MODEL_TO_EVALUATE,
            api_key=api_keys["ANTHROPIC_API_KEY"],
            few_shot_examples=few_shot_examples,
            use_system_parameter=args.use_system_parameter,
            use_xml_tags=args.use_xml_tags
        )
    else:
        # Use sequential generation (slower but more reliable)
        memos = generate_all_memos(
            indices=indices_to_evaluate,
            train_file=TRAIN_FILE,
            model=MODEL_TO_EVALUATE,
            few_shot_examples=few_shot_examples,
            use_system_parameter=args.use_system_parameter,
            use_xml_tags=args.use_xml_tags
        )

    # Phase 2: Evaluation (with or without iterative refinement)
    if args.refinement_rounds > 0:
        # Use iterative refinement workflow
        print(f"\n🔄 Using iterative refinement with {args.refinement_rounds} rounds (mode: {args.refinement_mode})")

        if not api_keys.get("ANTHROPIC_API_KEY"):
            print("❌ ERROR: ANTHROPIC_API_KEY required for iterative refinement")
            print("   Iterative refinement uses Claude to refine memos based on feedback")
            sys.exit(1)

        # Route between independent and combined modes
        if args.refinement_mode == 'independent':
            # Run iterative refinement (evaluates, refines, re-evaluates for each evaluator)
            refinement_results = run_iterative_refinement(
                memos=memos,
                evaluator_models=EVALUATOR_MODELS,
                refinement_rounds=args.refinement_rounds,
                api_key=api_keys["ANTHROPIC_API_KEY"]
            )
        else:  # combined mode
            # Run combined refinement (aggregate all evaluators' feedback before refining)
            refinement_results = run_combined_refinement(
                memos=memos,
                evaluator_models=EVALUATOR_MODELS,
                refinement_rounds=args.refinement_rounds,
                api_key=api_keys["ANTHROPIC_API_KEY"]
            )

        # Convert refinement results to batch_results format for aggregation
        batch_results = {}
        for (idx, evaluator), results in refinement_results.items():
            if idx not in batch_results:
                batch_results[idx] = {}
            batch_results[idx][evaluator] = results

        # NOTE: Round-specific results are already saved by save_single_round_eval_to_disk()
        # during iterative refinement. Do NOT call save_refinement_results_to_disk() here
        # as it would create files without round numbers that overwrite the round-based data!

        # No batch jobs to track in refinement mode
        batch_jobs = []
        jobs_info = {"message": "Iterative refinement mode - no batch jobs"}

    else:
        # Use standard batch evaluation workflow
        print(f"\n📊 Using standard batch evaluation (no refinement)")

        # Phase 2: Submit all batch jobs (NO WAITING)
        batch_jobs = submit_all_batch_jobs(
            memos=memos,
            evaluator_models=EVALUATOR_MODELS
        )

        # Phase 3: Poll all batch jobs until complete
        batch_results, jobs_info = poll_all_batch_jobs(
            batch_jobs=batch_jobs,
            poll_interval=60
        )

    # Phase 4: Aggregate results
    # COMMENTED OUT: This aggregation creates comprehensive_batch_eval_results_{timestamp}.json
    # results = aggregate_all_results(
    #     memos=memos,
    #     batch_results=batch_results,
    #     batch_jobs=batch_jobs,
    #     indices=sampling_info['all_sampled_indices'],
    #     evaluator_models=EVALUATOR_MODELS
    # )

    # # Save results
    # print(f"\nSaving comprehensive results to batch_evals folder...")
    # results_file, sampling_file = save_results(results, sampling_info, OUTPUT_DIR)

    # Print final summary
    print(f"\n{'='*70}")
    print(f"COMPREHENSIVE BATCH EVALUATION COMPLETE")
    print(f"{'='*70}")
    print(f"Model evaluated:      {MODEL_TO_EVALUATE}")
    print(f"Evaluator models:     {', '.join(EVALUATOR_MODELS)}")
    print(f"Total inputs:         {sampling_info['total_sampled']}")
    # COMMENTED OUT: Summary statistics depend on aggregation results
    # print(f"Successful evals:     {results['summary_statistics']['successful_evals']}")
    # print(f"Failed evals:         {results['summary_statistics']['failed_evals']}")
    # print(f"")
    # print(f"SUMMARY STATISTICS:")
    # print(f"  Mean Score:         {results['summary_statistics']['mean_score']:.2f}/100")
    # print(f"  Median Score:       {results['summary_statistics']['median_score']:.2f}/100")
    # print(f"  Worst Score:        {results['summary_statistics']['worst_score']:.2f}/100")
    # print(f"  Best Score:         {results['summary_statistics']['best_score']:.2f}/100")
    # print(f"  Std Dev:            {results['summary_statistics']['std_dev']:.2f}")
    # print(f"  Score Range:        {results['summary_statistics']['score_range']:.2f}")
    # print(f"")
    # print(f"Results saved to: {results_file}")
    print(f"")
    print(f"✓ Batch evaluation jobs completed. Results are in {BATCH_TEMP_DIR}/ folder.")
    print(f"")
    print(f"Next step: Run generate_final_results.py to aggregate the results:")
    print(f"  python3 evals/batch_evals/generate_final_results.py \\")
    print(f"    --batch-temp-dir {BATCH_TEMP_DIR.name} \\")
    print(f"    --output-dir results_benchmark_2 \\")
    print(f"    --skip-download")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
