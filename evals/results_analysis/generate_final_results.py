#!/usr/bin/env python3
"""
Generate Final Comprehensive Evaluation Results

This script consolidates all batch evaluation result processing into one end-to-end workflow:
1. Downloads Gemini batch results from API (if not already downloaded)
2. Parses OpenAI batch outputs from batch_temp/
3. Parses Claude batch outputs from batch_temp/
4. Parses Gemini batch outputs from batch_temp/
5. Calculates comprehensive statistics across all 3 evaluators
6. Outputs final_comprehensive_eval_results.json

This replaces the previous two-step process:
- aggregate_by_index.py (OpenAI + Claude)
- compile_final_results.py (add Gemini)

Usage:
    python generate_final_results.py [--skip-download]

Options:
    --skip-download    Skip Gemini download step (use existing files in batch_temp/)
"""

import argparse
import json
import os
import re
import statistics
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Default paths (can be overridden by command line arguments)
DEFAULT_BATCH_TEMP_DIR = Path(__file__).parent / "batch_temp_2"
DEFAULT_GEMINI_JOBS_FILE = Path(__file__).parent / "gemini_batch_jobs.json"
DEFAULT_OUTPUT_FILE = Path(__file__).parent / "results_benchmark_2" / "final_comprehensive_eval_results.json"

# These will be set by command line arguments
BATCH_TEMP_DIR = DEFAULT_BATCH_TEMP_DIR
GEMINI_JOBS_FILE = DEFAULT_GEMINI_JOBS_FILE
OUTPUT_FILE = DEFAULT_OUTPUT_FILE

# Dataset indices that were evaluated (from run_truly_parallel_batch_eval.py)
EVALUATED_INDICES = [
    0, 1, 2, 6, 12, 16, 17, 19, 20, 48, 51, 52, 57, 58, 63, 71, 78,
    108, 114, 119, 120, 122, 125, 128, 134, 140, 150, 152, 224, 226,
    239, 268, 289, 297, 311, 312, 318, 327, 338, 343, 357, 370, 377,
    378, 379, 390, 392, 427, 458, 469
]


# =============================================================================
# STEP 1: DOWNLOAD GEMINI RESULTS
# =============================================================================

def get_gemini_api_key() -> Optional[str]:
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


def parse_gemini_batch_to_jsonl(batch_data: dict, input_index: int) -> List[dict]:
    """
    Parse Gemini batch API response and convert to JSONL format
    matching OpenAI/Claude output structure.
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

            # Create result object matching OpenAI/Claude format
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


def download_gemini_results() -> tuple[int, int]:
    """
    Download all Gemini batch results from API and save to batch_temp/.
    Returns (successful_count, failed_count).
    """
    print(f"\n{'='*70}")
    print(f"STEP 1: DOWNLOADING GEMINI BATCH RESULTS")
    print(f"{'='*70}\n")

    # Check if gemini_batch_jobs.json exists
    if not GEMINI_JOBS_FILE.exists():
        print(f"⚠️  No Gemini jobs file found at {GEMINI_JOBS_FILE}")
        print(f"   Skipping Gemini download.\n")
        return 0, 0

    # Load API key
    api_key = get_gemini_api_key()
    if not api_key:
        print("⚠️  GEMINI_API_KEY not found in environment or .env file")
        print("   Skipping Gemini download.\n")
        return 0, 0

    # Load batch jobs
    print(f"Loading batch jobs from {GEMINI_JOBS_FILE.name}...")
    with open(GEMINI_JOBS_FILE, 'r') as f:
        jobs_data = json.load(f)

    submitted_jobs = jobs_data.get('submitted_jobs', [])
    if not submitted_jobs:
        print(f"⚠️  No submitted jobs found in {GEMINI_JOBS_FILE.name}")
        print(f"   Skipping Gemini download.\n")
        return 0, 0

    print(f"  Found {len(submitted_jobs)} batch jobs to download\n")

    # Download and save each batch
    successful = 0
    failed = 0

    for i, job in enumerate(submitted_jobs, 1):
        batch_id = job['batch_id']
        input_index = job['input_index']

        print(f"[{i}/{len(submitted_jobs)}] Index {input_index}: ", end="")

        # Check if already downloaded
        existing_files = list(BATCH_TEMP_DIR.glob(f"gemini_batch_output_{input_index}_*.jsonl"))
        if existing_files:
            print(f"✓ Already downloaded")
            successful += 1
            continue

        try:
            # Fetch from API
            batch_data = fetch_batch_results(batch_id, api_key)

            # Check status
            state = batch_data.get('metadata', {}).get('state', 'UNKNOWN')

            if state != 'BATCH_STATE_SUCCEEDED':
                print(f"⚠️  {state} (skipping)")
                failed += 1
                continue

            # Parse to JSONL format
            results = parse_gemini_batch_to_jsonl(batch_data, input_index)

            if not results:
                print(f"⚠️  No results (skipping)")
                failed += 1
                continue

            # Save to JSONL file
            timestamp = int(time.time())
            output_file = BATCH_TEMP_DIR / f"gemini_batch_output_{input_index}_{timestamp}.jsonl"

            with open(output_file, 'w') as f:
                for result in results:
                    f.write(json.dumps(result) + '\n')

            print(f"✓ Downloaded ({len(results)} results)")
            successful += 1

        except Exception as e:
            print(f"❌ Error: {str(e)}")
            failed += 1

    print(f"\n{'─'*70}")
    print(f"Gemini Download Summary:")
    print(f"  Successfully downloaded: {successful}/{len(submitted_jobs)}")
    print(f"  Failed/Skipped: {failed}/{len(submitted_jobs)}")
    print()

    return successful, failed


# =============================================================================
# STEP 2: PARSE ALL RESULTS
# =============================================================================

def extract_score_from_content(content: str, metric: str) -> Dict[str, Any]:
    """Extract score or result from evaluation content based on metric type."""

    if metric in ["quality_clarity", "quality_tone", "quality_length", "quality_structure"]:
        # Extract numeric score - prioritize finding "SCORE: XX" pattern first
        score_match = re.search(r'SCORE:\s*(\d+)', content, re.IGNORECASE)

        # If no "SCORE:" found, look for last number in text
        if not score_match:
            score_match = re.search(r'(\d+)(?!.*\d)', content)

        if score_match:
            try:
                return {"score": int(score_match.group(1)), "metric": metric}
            except:
                pass
        return {"raw": content, "metric": metric}

    elif metric == "accuracy":
        # Parse accuracy (Gemini format)
        answer_match = re.search(r'ANSWER:\s*(YES|NO)', content, re.IGNORECASE)
        answer = answer_match.group(1).upper() if answer_match else None

        # Match everything after HALLUCINATIONS: until end of content
        hall_match = re.search(r'HALLUCINATIONS:(.*)', content, re.DOTALL)
        hallucinations = hall_match.group(1).strip() if hall_match else None

        return {
            "metric": metric,
            "answer": answer,
            "hallucinations": hallucinations,
            "has_hallucinations": answer == "YES" if answer else None
        }

    elif metric == "completeness":
        # Parse completeness (Gemini format)
        answer_match = re.search(r'ANSWER:\s*(YES|NO)', content, re.IGNORECASE)
        answer = answer_match.group(1).upper() if answer_match else None

        # Match everything after MISSING_TERMS: until end of content
        missing_match = re.search(r'MISSING_TERMS:(.*)', content, re.DOTALL)
        missing = missing_match.group(1).strip() if missing_match else None

        return {
            "metric": metric,
            "answer": answer,
            "missing_terms": missing,
            "is_incomplete": answer == "YES" if answer else None
        }

    elif metric == "consistency":
        # Try to parse JSON
        try:
            json_match = re.search(r'\{[^}]+\}', content, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
                parsed["metric"] = metric
                return parsed
        except:
            pass
        return {"raw": content, "metric": metric}

    return {"raw": content, "metric": metric}


def parse_openai_output(file_path: Path) -> Dict[str, Any]:
    """Parse OpenAI batch output file and return metrics."""
    metrics = {}
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                result = json.loads(line)
                custom_id = result.get("custom_id", "")

                # Skip memo generation entries - they're not evaluation metrics
                if custom_id.startswith("memo_generation_"):
                    continue

                response_body = result.get("response", {}).get("body", {})

                if response_body.get("choices"):
                    content = response_body["choices"][0]["message"]["content"]
                    parsed = extract_score_from_content(content, custom_id)
                    metrics[custom_id] = parsed
    return metrics


def parse_claude_output(file_path: Path) -> Dict[str, Any]:
    """Parse Claude batch output file and return metrics."""
    metrics = {}
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                result = json.loads(line)
                custom_id = result.get("custom_id", "")

                # Skip memo generation entries - they're not evaluation metrics
                if custom_id.startswith("memo_generation_"):
                    continue

                if result.get("result", {}).get("type") == "succeeded":
                    content_blocks = result["result"]["message"]["content"]
                    if content_blocks:
                        content = content_blocks[0]["text"]
                        parsed = extract_score_from_content(content, custom_id)
                        metrics[custom_id] = parsed
    return metrics


def parse_gemini_output(file_path: Path) -> Dict[str, Any]:
    """Parse Gemini batch output file and return metrics.

    Note: Gemini files may use OpenAI format when generated through compatibility layer.
    """
    metrics = {}
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                result = json.loads(line)
                custom_id = result.get("custom_id", "")

                # Skip memo generation entries - they're not evaluation metrics
                if custom_id.startswith("memo_generation_"):
                    continue

                response = result.get("response", {})

                # Try OpenAI format first (used by compatibility layer)
                response_body = response.get("body", {})
                if response_body.get("choices"):
                    # OpenAI format: response.body.choices[0].message.content
                    content = response_body["choices"][0]["message"]["content"]
                    parsed = extract_score_from_content(content, custom_id)
                    metrics[custom_id] = parsed
                else:
                    # Native Gemini format: response.candidates[0].content.parts[0].text
                    candidates = response.get("candidates", [])
                    if candidates:
                        content_obj = candidates[0].get("content", {})
                        parts = content_obj.get("parts", [])

                        if parts:
                            # Concatenate all text parts
                            content = ""
                            for part in parts:
                                if "text" in part:
                                    content += part["text"]

                            parsed = extract_score_from_content(content, custom_id)
                            metrics[custom_id] = parsed
    return metrics


def parse_all_results() -> Dict[int, Dict]:
    """
    Parse all batch results from batch_temp/ and organize by dataset index.
    Supports round-based results from iterative refinement.

    Returns dict mapping index -> evaluator -> metrics (or index -> evaluator -> rounds -> metrics).
    """
    print(f"{'='*70}")
    print(f"STEP 2: PARSING ALL BATCH RESULTS")
    print(f"{'='*70}\n")

    results_by_index = {}
    has_rounds = False  # Track if any files have round numbers

    # Get all output files
    openai_outputs = sorted(BATCH_TEMP_DIR.glob("batch_output_*.jsonl"))
    claude_outputs = sorted(BATCH_TEMP_DIR.glob("claude_batch_output_*.jsonl"))
    gemini_outputs = list(BATCH_TEMP_DIR.glob("gemini_batch_output_*.jsonl"))

    print(f"Found output files:")
    print(f"  OpenAI: {len(openai_outputs)} files")
    print(f"  Claude: {len(claude_outputs)} files")
    print(f"  Gemini: {len(gemini_outputs)} files\n")

    # Parse OpenAI outputs (extract index and optional round from filename)
    print("Parsing OpenAI results...")
    openai_parsed = 0
    for i, output_file in enumerate(openai_outputs):
        # Extract index and round from filename: batch_output_{index}_round{round}_{timestamp}.jsonl
        # OR old format: batch_output_{index}_{timestamp}.jsonl
        filename = output_file.stem
        parts = filename.split('_')

        idx = None
        round_num = None

        # Try to extract index and round from filename
        try:
            # Check for round-based format: batch_output_INDEX_roundROUND_timestamp
            if 'round' in filename:
                has_rounds = True
                # Find which part contains 'round'
                for j, part in enumerate(parts):
                    if part.startswith('round'):
                        idx = int(parts[j-1])  # Index is before 'round'
                        round_num = int(part.replace('round', ''))  # Extract round number
                        break
            # New format without rounds: batch_output_INDEX_timestamp
            elif len(parts) >= 4 and parts[2].isdigit():
                idx = int(parts[2])
                round_num = None  # No round specified
            else:
                # Old format: batch_output_timestamp - fall back to order mapping
                if i < len(EVALUATED_INDICES):
                    idx = EVALUATED_INDICES[i]
                    round_num = None
                    print(f"  ⚠️  Using order-based mapping for {output_file.name} → index {idx}")
                else:
                    print(f"  ⚠️  Could not determine index for {output_file.name}, skipping")
                    continue
        except (IndexError, ValueError) as e:
            # Fallback to order mapping
            if i < len(EVALUATED_INDICES):
                idx = EVALUATED_INDICES[i]
                round_num = None
                print(f"  ⚠️  Using order-based mapping for {output_file.name} → index {idx}")
            else:
                print(f"  ⚠️  Could not determine index for {output_file.name}, skipping")
                continue

        try:
            metrics = parse_openai_output(output_file)
            # Skip files that contain no actual evaluation metrics (e.g., only memo_generation entries)
            if not metrics:
                print(f"  ⚠️  Skipping {output_file.name} (no evaluation metrics found)")
                continue

            if idx not in results_by_index:
                results_by_index[idx] = {}

            # Store results with round structure if rounds are present
            if round_num is not None:
                if "gpt-5" not in results_by_index[idx]:
                    results_by_index[idx]["gpt-5"] = {}
                if "rounds" not in results_by_index[idx]["gpt-5"]:
                    results_by_index[idx]["gpt-5"]["rounds"] = {}
                results_by_index[idx]["gpt-5"]["rounds"][round_num] = metrics
            else:
                results_by_index[idx]["gpt-5"] = metrics

            openai_parsed += 1
        except Exception as e:
            print(f"  ⚠️  Error processing {output_file.name}: {e}")

    print(f"  ✓ Parsed {openai_parsed} files\n")

    # Parse Claude outputs (extract index and optional round from filename)
    print("Parsing Claude results...")
    claude_parsed = 0
    for i, output_file in enumerate(claude_outputs):
        # Extract index and round from filename: claude_batch_output_{index}_round{round}_{timestamp}.jsonl
        # OR old format: claude_batch_output_{index}_{timestamp}.jsonl
        filename = output_file.stem
        parts = filename.split('_')

        idx = None
        round_num = None

        # Try to extract index and round from filename
        try:
            # Check for round-based format: claude_batch_output_INDEX_roundROUND_timestamp
            if 'round' in filename:
                has_rounds = True
                # Find which part contains 'round'
                for j, part in enumerate(parts):
                    if part.startswith('round'):
                        idx = int(parts[j-1])  # Index is before 'round'
                        round_num = int(part.replace('round', ''))  # Extract round number
                        break
            # New format without rounds: claude_batch_output_INDEX_timestamp
            elif len(parts) >= 5 and parts[3].isdigit():
                idx = int(parts[3])
                round_num = None
            else:
                # Old format: claude_batch_output_timestamp - fall back to order mapping
                if i < len(EVALUATED_INDICES):
                    idx = EVALUATED_INDICES[i]
                    round_num = None
                    print(f"  ⚠️  Using order-based mapping for {output_file.name} → index {idx}")
                else:
                    print(f"  ⚠️  Could not determine index for {output_file.name}, skipping")
                    continue
        except (IndexError, ValueError):
            # Fallback to order mapping
            if i < len(EVALUATED_INDICES):
                idx = EVALUATED_INDICES[i]
                round_num = None
                print(f"  ⚠️  Using order-based mapping for {output_file.name} → index {idx}")
            else:
                print(f"  ⚠️  Could not determine index for {output_file.name}, skipping")
                continue

        try:
            metrics = parse_claude_output(output_file)
            # Skip files that contain no actual evaluation metrics (e.g., only memo_generation entries)
            if not metrics:
                print(f"  ⚠️  Skipping {output_file.name} (no evaluation metrics found)")
                continue

            if idx not in results_by_index:
                results_by_index[idx] = {}

            # Store results with round structure if rounds are present
            if round_num is not None:
                if "claude-sonnet-4-20250514" not in results_by_index[idx]:
                    results_by_index[idx]["claude-sonnet-4-20250514"] = {}
                if "rounds" not in results_by_index[idx]["claude-sonnet-4-20250514"]:
                    results_by_index[idx]["claude-sonnet-4-20250514"]["rounds"] = {}
                results_by_index[idx]["claude-sonnet-4-20250514"]["rounds"][round_num] = metrics
            else:
                results_by_index[idx]["claude-sonnet-4-20250514"] = metrics

            claude_parsed += 1
        except Exception as e:
            print(f"  ⚠️  Error processing {output_file.name}: {e}")

    print(f"  ✓ Parsed {claude_parsed} files\n")

    # Parse Gemini outputs (extract index and optional round from filename)
    print("Parsing Gemini results...")
    gemini_parsed = 0
    for file_path in gemini_outputs:
        # Extract index and round from filename: gemini_batch_output_{index}_round{round}_{timestamp}.jsonl
        # OR old format: gemini_batch_output_{index}_{timestamp}.jsonl
        filename = file_path.stem
        parts = filename.split('_')

        idx = None
        round_num = None

        try:
            # Check for round-based format
            if 'round' in filename:
                has_rounds = True
                for j, part in enumerate(parts):
                    if part.startswith('round'):
                        idx = int(parts[j-1])
                        round_num = int(part.replace('round', ''))
                        break
            else:
                idx = int(parts[3])  # gemini_batch_output_INDEX_timestamp
                round_num = None
        except (IndexError, ValueError):
            print(f"  ⚠️  Could not parse index from {file_path.name}, skipping")
            continue

        try:
            metrics = parse_gemini_output(file_path)
            # Skip files that contain no actual evaluation metrics (e.g., only memo_generation entries)
            if not metrics:
                print(f"  ⚠️  Skipping {file_path.name} (no evaluation metrics found)")
                continue

            if idx not in results_by_index:
                results_by_index[idx] = {}

            # Store results with round structure if rounds are present
            if round_num is not None:
                if "gemini-2.5-pro" not in results_by_index[idx]:
                    results_by_index[idx]["gemini-2.5-pro"] = {}
                if "rounds" not in results_by_index[idx]["gemini-2.5-pro"]:
                    results_by_index[idx]["gemini-2.5-pro"]["rounds"] = {}
                results_by_index[idx]["gemini-2.5-pro"]["rounds"][round_num] = metrics
            else:
                results_by_index[idx]["gemini-2.5-pro"] = metrics

            gemini_parsed += 1
        except Exception as e:
            print(f"  ⚠️  Error processing {file_path.name}: {e}")

    print(f"  ✓ Parsed {gemini_parsed} files\n")

    if has_rounds:
        print(f"✨ Detected round-based results from iterative refinement\n")

    print(f"{'─'*70}")
    print(f"Total unique dataset indices: {len(results_by_index)}")
    print(f"Indices: {sorted(results_by_index.keys())}\n")

    return results_by_index


# =============================================================================
# STEP 3: CALCULATE STATISTICS AND OUTPUT
# =============================================================================

def calculate_summary_scores(results_by_index: Dict[int, Dict]) -> Dict[int, Dict]:
    """
    Calculate summary scores for each memo and evaluator using all 4 metrics.
    Supports round-based results from iterative refinement.
    """
    from evals.evaluation.metrics import calculate_summary_score

    print(f"{'='*70}")
    print(f"STEP 3: CALCULATING STATISTICS")
    print(f"{'='*70}\n")

    quality_metrics = ['quality_clarity', 'quality_tone', 'quality_length', 'quality_structure']

    print("Calculating summary scores for each memo and evaluator...")

    # Detect if we have round-based results
    has_rounds = False
    num_rounds = 0
    for index_data in results_by_index.values():
        for evaluator, eval_results in index_data.items():
            if evaluator == 'summary_score':
                continue
            if 'rounds' in eval_results:
                has_rounds = True
                num_rounds = max(num_rounds, max(eval_results['rounds'].keys()) + 1)
                break
        if has_rounds:
            break

    for index_str, index_data in results_by_index.items():
        evaluator_summaries = []

        for evaluator, eval_results in index_data.items():
            if evaluator in ['summary_score', 'round_summary_scores']:  # Skip if already exists
                continue

            # Check if this evaluator has round-based results
            if 'rounds' in eval_results:
                # Process each round
                for round_num, round_metrics in eval_results['rounds'].items():
                    summary_score = calculate_round_summary_score(round_metrics, quality_metrics)
                    round_metrics['summary_score'] = summary_score

                # Use the FINAL round (highest round number) as the "official" score for averages
                final_round = max(eval_results['rounds'].keys())
                final_score = eval_results['rounds'][final_round]['summary_score']
                eval_results['summary_score'] = final_score
                evaluator_summaries.append(final_score)
            else:
                # Standard single-round evaluation
                summary_score = calculate_round_summary_score(eval_results, quality_metrics)
                eval_results['summary_score'] = summary_score
                evaluator_summaries.append(summary_score)

        # Calculate overall summary score for this memo (average across evaluators)
        if evaluator_summaries:
            index_data['summary_score'] = round(statistics.mean(evaluator_summaries), 2)

        # If round-based, also calculate memo-level average for each round
        if has_rounds:
            round_summary_scores = {}
            for round_num in range(num_rounds):
                round_scores_for_memo = []
                for evaluator, eval_results in index_data.items():
                    if evaluator in ['summary_score', 'round_summary_scores']:
                        continue
                    if 'rounds' in eval_results and round_num in eval_results['rounds']:
                        score = eval_results['rounds'][round_num].get('summary_score')
                        if score is not None:
                            round_scores_for_memo.append(score)

                if round_scores_for_memo:
                    round_summary_scores[round_num] = round(statistics.mean(round_scores_for_memo), 2)

            if round_summary_scores:
                index_data['round_summary_scores'] = round_summary_scores

    print(f"  ✓ Summary scores calculated\n")

    return results_by_index


def calculate_round_summary_score(eval_results: Dict, quality_metrics: List[str]) -> float:
    """
    Calculate summary score for a single round of evaluation.
    Helper function to avoid code duplication between round-based and standard evaluation.
    """
    from evals.evaluation.metrics import calculate_summary_score

    # 1. Convert accuracy (answer YES = has hallucinations, NO = accurate)
    accuracy_result = None
    if 'accuracy' in eval_results:
        acc_data = eval_results['accuracy']
        answer = acc_data.get('answer')
        if answer:
            accuracy_result = {
                'score': 1.0 if answer == 'NO' else 0.0,
                'accurate': answer == 'NO'
            }

    # 2. Convert completeness (answer YES = incomplete, NO = complete)
    completeness_result = None
    if 'completeness' in eval_results:
        comp_data = eval_results['completeness']
        answer = comp_data.get('answer')
        if answer:
            completeness_result = {
                'score': 1.0 if answer == 'NO' else 0.0,
                'complete': answer == 'NO'
            }

    # 3. Convert consistency (has_issues = True means inconsistent)
    consistency_result = None
    if 'consistency' in eval_results:
        cons_data = eval_results['consistency']
        has_issues = cons_data.get('has_issues', False)
        consistency_result = {
            'score': 0.0 if has_issues else 1.0,
            'consistent': not has_issues
        }

    # 4. Calculate quality score from 4 sub-metrics
    quality_result = None
    quality_scores = []
    for metric in quality_metrics:
        if metric in eval_results and eval_results[metric].get('score') is not None:
            quality_scores.append(eval_results[metric]['score'])

    if quality_scores:
        quality_avg = statistics.mean(quality_scores)
        quality_result = {
            'quality_score': quality_avg,
            'clarity_score': eval_results.get('quality_clarity', {}).get('score', 0),
            'tone_score': eval_results.get('quality_tone', {}).get('score', 0),
            'length_score': eval_results.get('quality_length', {}).get('score', 0),
            'structure_score': eval_results.get('quality_structure', {}).get('score', 0)
        }

    # Calculate summary score using proper weighted formula (accuracy, completeness, consistency, quality)
    summary_result = calculate_summary_score(
        accuracy_result=accuracy_result,
        completeness_result=completeness_result,
        consistency_result=consistency_result,
        quality_result=quality_result
    )

    return round(summary_result['summary_score'], 2)


def calculate_aggregate_statistics(results_by_index: Dict[int, Dict]) -> Dict[str, Any]:
    """Calculate aggregate statistics across all memos and evaluators.

    When iterative refinement is used, calculates statistics per round.
    """
    print("Calculating aggregate statistics...")

    quality_metrics = ['quality_clarity', 'quality_tone', 'quality_length', 'quality_structure']

    # Detect if results have rounds
    has_rounds = False
    num_rounds = 0
    for index_data in results_by_index.values():
        for evaluator, eval_results in index_data.items():
            if evaluator == 'summary_score':
                continue
            if 'rounds' in eval_results:
                has_rounds = True
                num_rounds = max(num_rounds, max(eval_results['rounds'].keys()) + 1)
                break
        if has_rounds:
            break

    if has_rounds:
        print(f"  Detected {num_rounds} rounds - calculating per-round statistics...\n")
        return calculate_aggregate_statistics_with_rounds(results_by_index, quality_metrics, num_rounds)
    else:
        return calculate_aggregate_statistics_simple(results_by_index, quality_metrics)


def calculate_aggregate_statistics_simple(results_by_index: Dict[int, Dict], quality_metrics: List[str]) -> Dict[str, Any]:
    """Calculate aggregate statistics for non-round-based results."""
    memo_averaged_scores = []  # Average score per memo (averaged across 3 evaluators)
    evaluator_memo_scores = defaultdict(list)  # Summary scores by evaluator
    metric_scores = defaultdict(list)  # Individual metric scores

    for index_str, index_data in results_by_index.items():
        # Collect the memo-level averaged score (averaged across all 3 evaluators)
        if 'summary_score' in index_data:
            memo_averaged_scores.append(index_data['summary_score'])

        for evaluator, eval_results in index_data.items():
            if evaluator == 'summary_score':  # Skip summary_score key
                continue

            # Use the calculated summary_score (which includes all 4 metrics: accuracy, completeness, consistency, quality)
            if 'summary_score' in eval_results:
                summary_score = eval_results['summary_score']
                evaluator_memo_scores[evaluator].append(summary_score)

            # Collect individual quality metric scores for additional stats
            for metric in quality_metrics:
                if metric in eval_results and eval_results[metric].get('score') is not None:
                    score = eval_results[metric]['score']
                    metric_scores[metric].append(score)

    # Calculate summary statistics using memo-averaged scores
    # (one score per memo, averaged across the 3 evaluators)
    total_evaluations = sum(len(scores) for scores in evaluator_memo_scores.values())

    summary = {
        "has_rounds": False,
        "total_memos_evaluated": len(results_by_index),
        "dataset_indices_evaluated": sorted(results_by_index.keys()),
        "total_evaluations": total_evaluations,
        "total_quality_scores": sum(len(scores) for scores in metric_scores.values()),
        "mean_score": round(statistics.mean(memo_averaged_scores), 2) if memo_averaged_scores else 0.0,
        "median_score": round(statistics.median(memo_averaged_scores), 2) if memo_averaged_scores else 0.0,
        "min_score": round(min(memo_averaged_scores), 2) if memo_averaged_scores else 0.0,
        "max_score": round(max(memo_averaged_scores), 2) if memo_averaged_scores else 0.0,
        "stdev_score": round(statistics.stdev(memo_averaged_scores), 2) if len(memo_averaged_scores) > 1 else 0.0,
        "evaluators": {},
        "metrics": {}
    }

    # Add evaluator statistics (summary scores by evaluator)
    for evaluator, scores in evaluator_memo_scores.items():
        summary['evaluators'][evaluator] = {
            'count': len(scores),
            'mean': round(statistics.mean(scores), 2),
            'median': round(statistics.median(scores), 2)
        }

    # Add metric statistics (individual quality metric scores)
    for metric, scores in metric_scores.items():
        summary['metrics'][metric] = {
            'count': len(scores),
            'mean': round(statistics.mean(scores), 2),
            'median': round(statistics.median(scores), 2)
        }

    print(f"  ✓ Aggregate statistics calculated\n")

    return summary


def calculate_aggregate_statistics_with_rounds(results_by_index: Dict[int, Dict], quality_metrics: List[str], num_rounds: int) -> Dict[str, Any]:
    """Calculate aggregate statistics with per-round breakdowns."""
    # Structure: rounds[round_num][stat_type][values]
    rounds_data = {}

    for round_num in range(num_rounds):
        memo_averaged_scores = []
        evaluator_memo_scores = defaultdict(list)
        metric_scores = defaultdict(list)

        for index_str, index_data in results_by_index.items():
            round_scores_for_memo = []

            for evaluator, eval_results in index_data.items():
                if evaluator == 'summary_score':
                    continue

                if 'rounds' in eval_results and round_num in eval_results['rounds']:
                    round_metrics = eval_results['rounds'][round_num]

                    # Collect summary score
                    if 'summary_score' in round_metrics:
                        summary_score = round_metrics['summary_score']
                        evaluator_memo_scores[evaluator].append(summary_score)
                        round_scores_for_memo.append(summary_score)

                    # Collect individual quality metric scores
                    for metric in quality_metrics:
                        if metric in round_metrics and round_metrics[metric].get('score') is not None:
                            score = round_metrics[metric]['score']
                            metric_scores[metric].append(score)

            # Calculate memo-averaged score for this round
            if round_scores_for_memo:
                memo_averaged_scores.append(round(statistics.mean(round_scores_for_memo), 2))

        # Calculate statistics for this round
        total_evaluations = sum(len(scores) for scores in evaluator_memo_scores.values())

        round_summary = {
            "total_memos_evaluated": len([s for s in memo_averaged_scores if s is not None]),
            "total_evaluations": total_evaluations,
            "total_quality_scores": sum(len(scores) for scores in metric_scores.values()),
            "mean_score": round(statistics.mean(memo_averaged_scores), 2) if memo_averaged_scores else 0.0,
            "median_score": round(statistics.median(memo_averaged_scores), 2) if memo_averaged_scores else 0.0,
            "min_score": round(min(memo_averaged_scores), 2) if memo_averaged_scores else 0.0,
            "max_score": round(max(memo_averaged_scores), 2) if memo_averaged_scores else 0.0,
            "stdev_score": round(statistics.stdev(memo_averaged_scores), 2) if len(memo_averaged_scores) > 1 else 0.0,
            "evaluators": {},
            "metrics": {}
        }

        # Add evaluator statistics for this round
        for evaluator, scores in evaluator_memo_scores.items():
            round_summary['evaluators'][evaluator] = {
                'count': len(scores),
                'mean': round(statistics.mean(scores), 2) if scores else 0.0,
                'median': round(statistics.median(scores), 2) if scores else 0.0
            }

        # Add metric statistics for this round
        for metric, scores in metric_scores.items():
            round_summary['metrics'][metric] = {
                'count': len(scores),
                'mean': round(statistics.mean(scores), 2) if scores else 0.0,
                'median': round(statistics.median(scores), 2) if scores else 0.0
            }

        rounds_data[round_num] = round_summary

    # Build final summary with rounds
    final_round = num_rounds - 1
    summary = {
        "has_rounds": True,
        "num_rounds": num_rounds,
        "rounds": rounds_data,
        # Overall stats use final round for backward compatibility
        "total_memos_evaluated": rounds_data[final_round]["total_memos_evaluated"],
        "dataset_indices_evaluated": sorted(results_by_index.keys()),
        "total_evaluations": rounds_data[final_round]["total_evaluations"],
        "total_quality_scores": rounds_data[final_round]["total_quality_scores"],
        "mean_score": rounds_data[final_round]["mean_score"],
        "median_score": rounds_data[final_round]["median_score"],
        "min_score": rounds_data[final_round]["min_score"],
        "max_score": rounds_data[final_round]["max_score"],
        "stdev_score": rounds_data[final_round]["stdev_score"],
        "evaluators": rounds_data[final_round]["evaluators"],
        "metrics": rounds_data[final_round]["metrics"]
    }

    print(f"  ✓ Per-round aggregate statistics calculated\n")

    return summary


def save_final_results(results_by_index: Dict[int, Dict], summary: Dict[str, Any]):
    """Save final comprehensive results to JSON file."""
    print(f"Saving results to {OUTPUT_FILE.name}...")

    # Prepare output data
    output_data = {
        "summary": summary,
        "results_by_index": {str(k): v for k, v in results_by_index.items()},
        "metadata": {
            "note": "Results from 50 memos evaluated by GPT-5, Claude Sonnet 4, and Gemini 2.5 Pro",
            "generated_by": "generate_final_results.py",
            "evaluators": list(summary['evaluators'].keys()),
            "total_evaluations": f"{summary['total_memos_evaluated']} memos × {len(summary['evaluators'])} evaluators = {summary['total_evaluations']}"
        }
    }

    # Save to file
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"  ✓ Saved to {OUTPUT_FILE}\n")


def print_summary(summary: Dict[str, Any]):
    """Print summary statistics to console."""
    print(f"{'='*70}")
    print(f"FINAL RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"Total memos evaluated: {summary['total_memos_evaluated']}")
    print(f"Total evaluations: {summary['total_evaluations']}")
    print(f"Total quality scores: {summary['total_quality_scores']}")
    print(f"\nOverall Quality:")
    print(f"  Mean:   {summary['mean_score']:.2f}")
    print(f"  Median: {summary['median_score']:.2f}")
    print(f"  Range:  {summary['min_score']:.2f} - {summary['max_score']:.2f}")
    print(f"  Stdev:  {summary['stdev_score']:.2f}")
    print(f"\nBy Evaluator:")
    for evaluator, stats in summary['evaluators'].items():
        print(f"  {evaluator}:")
        print(f"    Count:  {stats['count']}")
        print(f"    Mean:   {stats['mean']:.2f}")
        print(f"    Median: {stats['median']:.2f}")
    print(f"\nBy Metric:")
    for metric, stats in summary['metrics'].items():
        print(f"  {metric}:")
        print(f"    Count:  {stats['count']}")
        print(f"    Mean:   {stats['mean']:.2f}")
        print(f"    Median: {stats['median']:.2f}")
    print()


# =============================================================================
# MAIN
# =============================================================================

def main():
    global BATCH_TEMP_DIR, OUTPUT_FILE, GEMINI_JOBS_FILE

    parser = argparse.ArgumentParser(
        description="Generate final comprehensive evaluation results from batch outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default directories (batch_temp/, results_benchmark/)
  python generate_final_results.py --skip-download

  # Use custom directories (for test runs)
  python generate_final_results.py --batch-temp-dir batch_temp_2 --output-dir results_benchmark_2 --skip-download
        """
    )
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip Gemini download step (use existing files in batch_temp/)'
    )
    parser.add_argument(
        '--batch-temp-dir',
        type=str,
        default='batch_temp_2',
        help='Directory containing batch output files (default: batch_temp)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results_benchmark_2',
        help='Directory to save final results (default: results_benchmark)'
    )
    args = parser.parse_args()

    # Set global paths based on arguments
    BATCH_TEMP_DIR = Path(__file__).parent / args.batch_temp_dir
    OUTPUT_FILE = Path(__file__).parent / args.output_dir / "final_comprehensive_eval_results.json"
    GEMINI_JOBS_FILE = Path(__file__).parent / "gemini_batch_jobs.json"

    # Create output directory if it doesn't exist
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*70}")
    print(f"# GENERATE FINAL COMPREHENSIVE EVALUATION RESULTS")
    print(f"{'#'*70}")
    print(f"Batch temp directory: {BATCH_TEMP_DIR}")
    print(f"Output file:          {OUTPUT_FILE}")
    print(f"{'#'*70}\n")

    # Step 1: Download Gemini results (optional)
    if not args.skip_download:
        download_gemini_results()
    else:
        print(f"\n{'='*70}")
        print(f"STEP 1: SKIPPING GEMINI DOWNLOAD (--skip-download)")
        print(f"{'='*70}\n")

    # Step 2: Parse all results
    results_by_index = parse_all_results()

    if not results_by_index:
        print("❌ ERROR: No results found to process.")
        print("   Make sure batch output files exist in batch_temp/")
        return

    # Step 3: Calculate statistics
    results_by_index = calculate_summary_scores(results_by_index)
    summary = calculate_aggregate_statistics(results_by_index)

    # Save and display
    save_final_results(results_by_index, summary)
    print_summary(summary)

    print(f"{'#'*70}")
    print(f"# COMPLETE! Final results saved to:")
    print(f"# {OUTPUT_FILE}")
    print(f"{'#'*70}\n")


if __name__ == "__main__":
    main()
