#!/usr/bin/env python3
"""
Memo Review Generator
=====================

Generate comprehensive review documents for memo evaluations, including:
- The original credit agreement from the source
- The AI-generated investment memo
- Detailed evaluation feedback from all evaluator models

Usage:
------
    python3 generate_memo_review.py <index> [batch_temp_dir] [results_dir] [output_path]

Parameters:
-----------
    index : int (required)
        The index number from train.jsonl (0-based, e.g., 128)

    batch_temp_dir : str (optional)
        Name of batch temp directory (default: batch_temp)

    results_dir : str (optional)
        Name of results directory (default: results_benchmark)

    output_path : str (optional)
        Custom output file path (default: saves to results_dir)

Examples:
---------
    # Basic usage (uses default directories)
    python3 generate_memo_review.py 128

    # With custom directories
    python3 generate_memo_review.py 2 batch_temp_anthropic_prompt_gen results_anthropic_prompt_gen

    # With custom output path
    python3 generate_memo_review.py 128 batch_temp results_benchmark custom_output.md

Output:
-------
Generates a markdown file in the results directory: memo_review_{index}_batch_{timestamp}.md

The output includes:
    1. Metadata (batch timestamp, source URL, overall score)
    2. Source Document (original credit agreement)
    3. Generated Memo (AI-generated investment memo)
    4. Evaluation Results from each model:
       - Accuracy (hallucinations detected)
       - Completeness (missing terms)
       - Consistency (internal contradictions)
       - Quality scores: Clarity, Tone, Conciseness, Structure (0-100)
       - Model summary score

Requirements:
-------------
    - Python 3.6+
    - Standard library only (no additional dependencies)

Expected Directory Structure:
------------------------------
    AI-for-Finance/
    ├── data/
    │   └── train.jsonl
    └── evals/
        └── batch_evals/
            ├── generate_memo_review.py (this script)
            ├── batch_temp/
            │   └── batch_input_<timestamp>.jsonl
            └── results_benchmark/
                └── final_comprehensive_eval_results.json
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional


def load_source_document(index: int, train_path: Path) -> Dict[str, str]:
    """Load the source document from train.jsonl at the given index."""
    with open(train_path, 'r') as f:
        for i, line in enumerate(f):
            if i == index:
                return json.loads(line)
    raise ValueError(f"Index {index} not found in {train_path}")


def load_batch_input(batch_timestamp: str, batch_temp_dir: Path, index: int = None) -> Dict[str, Any]:
    """Load the batch input file containing source and generated memo."""
    # Try new format first: batch_input_{index}_{timestamp}.jsonl
    if index is not None:
        input_file = batch_temp_dir / f"batch_input_{index}_{batch_timestamp}.jsonl"
        if input_file.exists():
            pass  # Use this file
        else:
            # Fall back to old format
            input_file = batch_temp_dir / f"batch_input_{batch_timestamp}.jsonl"
    else:
        # Old format
        input_file = batch_temp_dir / f"batch_input_{batch_timestamp}.jsonl"

    if not input_file.exists():
        raise FileNotFoundError(f"Batch input file not found: {input_file}")

    # Read all requests from the batch input file
    requests = {}
    with open(input_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            custom_id = data.get('custom_id', '')
            # Extract the content from the user message
            content = data['body']['messages'][0]['content']
            requests[custom_id] = content

    return requests


def extract_memo_from_content(content: str) -> Optional[str]:
    """Extract the generated memo from the evaluation prompt content."""
    if "GENERATED MEMO:" in content:
        memo_start = content.find("GENERATED MEMO:")
        memo_text = content[memo_start + len("GENERATED MEMO:"):].strip()

        # Find where the evaluation question starts
        end_markers = [
            "\n\nDoes the memo contain",
            "\n\nAre any key financial terms",
            "\n\nGoal\n"
        ]

        for marker in end_markers:
            if marker in memo_text:
                memo_text = memo_text[:memo_text.find(marker)].strip()
                break

        return memo_text
    return None


def extract_source_from_content(content: str) -> Optional[str]:
    """Extract the source document from the evaluation prompt content."""
    if "SOURCE DOCUMENT:" in content:
        source_start = content.find("SOURCE DOCUMENT:")
        source_end = content.find("GENERATED MEMO:")
        if source_end == -1:
            source_end = content.find("\n\nMEMO:")

        if source_end > source_start:
            return content[source_start + len("SOURCE DOCUMENT:"):source_end].strip()
    return None


def load_evaluation_results(index: int, results_path: Path) -> Dict[str, Any]:
    """Load evaluation results for the given index."""
    with open(results_path, 'r') as f:
        data = json.load(f)

    results_by_index = data.get('results_by_index', {})

    # Try both int and string keys
    if index in results_by_index:
        return results_by_index[index]
    elif str(index) in results_by_index:
        return results_by_index[str(index)]
    else:
        raise ValueError(f"Index {index} not found in evaluation results")


def format_evaluation_results(results: Dict[str, Any]) -> str:
    """Format evaluation results into a readable markdown section."""
    output = []

    for model_name, model_results in results.items():
        if model_name == 'summary_score':
            continue

        output.append(f"\n### {model_name}\n")

        # Accuracy
        if 'accuracy' in model_results:
            acc = model_results['accuracy']
            output.append("#### Accuracy")
            output.append(f"**Has Hallucinations:** {acc.get('has_hallucinations', acc.get('answer'))}")
            if acc.get('hallucinations'):
                output.append(f"\n**Hallucinations Found:**\n{acc['hallucinations']}\n")

        # Completeness
        if 'completeness' in model_results:
            comp = model_results['completeness']
            output.append("\n#### Completeness")
            output.append(f"**Is Incomplete:** {comp.get('is_incomplete', comp.get('answer'))}")
            if comp.get('missing_terms'):
                output.append(f"\n**Missing Terms:**\n{comp['missing_terms']}\n")

        # Consistency
        if 'consistency' in model_results:
            cons = model_results['consistency']
            output.append("\n#### Consistency")
            output.append(f"**Has Issues:** {cons.get('has_issues', False)}")
            if cons.get('issues') and len(cons['issues']) > 0:
                output.append("\n**Issues Found:**")
                for issue in cons['issues']:
                    output.append(f"- {issue}")
                output.append("")

        # Quality metrics
        quality_metrics = ['quality_clarity', 'quality_tone', 'quality_length', 'quality_structure']
        quality_names = {
            'quality_clarity': 'Clarity',
            'quality_tone': 'Professional Tone',
            'quality_length': 'Conciseness',
            'quality_structure': 'Structure Match'
        }

        output.append("\n#### Quality Scores (0-100)")
        for metric in quality_metrics:
            if metric in model_results:
                score = model_results[metric].get('score', 'N/A')
                output.append(f"- **{quality_names[metric]}:** {score}")

        # Model summary score
        if 'summary_score' in model_results:
            output.append(f"\n**Model Summary Score:** {model_results['summary_score']:.2f}")

        output.append("\n---")

    # Overall summary score
    if 'summary_score' in results:
        output.append(f"\n## Overall Summary Score: {results['summary_score']:.2f}\n")

    return "\n".join(output)


def generate_review_document(batch_timestamp: str, index: int,
                            base_dir: Path, batch_temp_name: str = "batch_temp",
                            results_dir_name: str = "results_benchmark") -> str:
    """Generate a comprehensive review document."""

    # Set up paths
    train_path = base_dir / "data" / "train.jsonl"
    batch_temp_dir = base_dir / "evals" / "batch_evals" / batch_temp_name
    results_path = base_dir / "evals" / "batch_evals" / results_dir_name / "final_comprehensive_eval_results.json"

    # Load data
    print(f"Loading source document for index {index}...")
    source_doc = load_source_document(index, train_path)

    print(f"Loading batch input file for timestamp {batch_timestamp}...")
    batch_requests = load_batch_input(batch_timestamp, batch_temp_dir, index)

    print(f"Loading evaluation results...")
    eval_results = load_evaluation_results(index, results_path)

    # Extract source and memo from batch input (use any request since they all have the same source/memo)
    source_text = None
    memo_text = None
    for custom_id, content in batch_requests.items():
        if source_text is None:
            source_text = extract_source_from_content(content)
        if memo_text is None:
            memo_text = extract_memo_from_content(content)
        if source_text and memo_text:
            break

    # Build the review document
    doc = []
    doc.append(f"# Memo Review - Index {index}")
    doc.append(f"\n**Batch Timestamp:** {batch_timestamp}")
    doc.append(f"**Source URL:** {source_doc.get('source_url', 'N/A')}")
    doc.append(f"**Overall Summary Score:** {eval_results.get('summary_score', 'N/A')}")
    doc.append("\n---\n")

    # Source document section
    doc.append("## 1. SOURCE DOCUMENT\n")
    if source_text:
        doc.append(source_text)
    else:
        doc.append(source_doc.get('text', 'Source text not found'))
    doc.append("\n---\n")

    # Generated memo section
    doc.append("## 2. GENERATED MEMO\n")
    if memo_text:
        doc.append(memo_text)
    else:
        doc.append("Generated memo not found in batch input file")
    doc.append("\n---\n")

    # Evaluation results section
    doc.append("## 3. EVALUATION RESULTS\n")
    doc.append(format_evaluation_results(eval_results))

    return "\n".join(doc)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    # First argument is now index (not timestamp)
    try:
        index = int(sys.argv[1])
    except ValueError:
        print(f"Error: Index must be an integer, got '{sys.argv[1]}'")
        sys.exit(1)

    # Optional arguments for custom directories
    batch_temp_name = sys.argv[2] if len(sys.argv) > 2 else "batch_temp"
    results_dir_name = sys.argv[3] if len(sys.argv) > 3 else "results_benchmark"
    output_path = sys.argv[4] if len(sys.argv) > 4 else None

    # Assume script is in evals/batch_evals directory
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent.parent  # Go up to project root

    # Find the timestamp automatically by looking for batch input file
    batch_temp_dir = script_dir / batch_temp_name
    input_files = list(batch_temp_dir.glob(f"batch_input_{index}_*.jsonl"))

    if not input_files:
        print(f"Error: No batch input file found for index {index} in {batch_temp_dir}")
        sys.exit(1)

    if len(input_files) > 1:
        print(f"Warning: Multiple batch input files found for index {index}, using most recent")
        input_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    # Extract timestamp from filename: batch_input_{index}_{timestamp}.jsonl
    filename = input_files[0].stem
    batch_timestamp = filename.split('_')[-1]

    print(f"Found batch timestamp: {batch_timestamp}")

    try:
        print(f"Generating review for index {index}...")
        print(f"Using batch_temp: {batch_temp_name}, results: {results_dir_name}")
        review = generate_review_document(batch_timestamp, index, base_dir, batch_temp_name, results_dir_name)

        # Save to file
        if output_path:
            output_file = Path(output_path)
        else:
            # Default: save to results directory
            results_dir = script_dir / results_dir_name
            results_dir.mkdir(exist_ok=True, parents=True)
            output_file = results_dir / f"memo_review_{index}_batch_{batch_timestamp}.md"

        output_file.write_text(review)

        print(f"\n✓ Review document generated successfully!")
        print(f"  Output: {output_file}")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
