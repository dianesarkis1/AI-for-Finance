#!/usr/bin/env python3
"""
Complete Evaluation Workflow Orchestrator
==========================================

This script runs the entire evaluation workflow with customizable run names and prompts:
1. Run batch evaluations (run_truly_parallel_batch_eval.py)
2. Aggregate results (generate_final_results.py)
3. Create analysis tables (create_results_tables_v2.py)

All outputs are saved to custom-named directories based on your run name.

Usage:
------
    python run_eval_workflow.py <run_name> [--prompt PROMPT_FILE] [--indices INDEX...]

Parameters:
-----------
    run_name : str
        Name for this evaluation run. Outputs will be saved to:
        - batch_temp_{run_name}/
        - results_{run_name}/

    --prompt : str (optional)
        Path to custom prompt file (e.g., prompts/my_prompt.txt)
        If not provided, uses prompts/baseline.txt

    --indices : int... (optional)
        Custom indices to evaluate (space-separated)
        If not provided, uses default comprehensive sample (50 indices)

    --parallel-memos : flag (optional)
        Generate memos in parallel using Claude Batch API (faster)

    --evaluators : str... (optional)
        Which evaluator(s) to run. Can use short names (openai, claude, gemini)
        or full model names. Default: all 3 evaluators

    --skip-memo-generation : flag (optional)
        Skip memo generation and use existing batch inputs
        Useful for re-running specific evaluators after fixing issues

    --few-shot-dir : str (optional)
        Path to directory containing few-shot examples
        Directory should contain input_*.txt and example_*.md files
        Example: evals/few_shot_examples

    --use-system-parameter : flag (optional)
        Use Claude's native system parameter for better instruction following
        Only affects Claude API calls. Improves prompt adherence and efficiency.
        Default: False (uses old behavior with everything in user message)

    --use-xml-tags : flag (optional)
        Wrap inputs in XML tags (<credit_agreement>, <examples>, etc.)
        Helps Claude parse long, structured documents more clearly.
        Recommended for use with Claude, especially with few-shot examples.
        Default: False (no XML wrapping)

    --refinement-rounds : int (optional)
        Number of iterative refinement rounds per evaluator.
        For each round: evaluate memo → refine with Claude based on feedback → re-evaluate.
        Each evaluator gets its own refinement path.
        Default: 0 (no refinement, standard evaluation)

Examples:
---------
    # Run with default sample and baseline prompt
    python run_eval_workflow.py baseline_v1

    # Run with custom prompt
    python run_eval_workflow.py experiment_1 --prompt prompts/improved_v2.txt

    # Run with custom indices and custom prompt
    python run_eval_workflow.py test_run --indices 0 1 2 6 --prompt prompts/test.txt

    # Run with parallel memo generation (recommended for speed)
    python run_eval_workflow.py baseline_v2 --parallel-memos

    # Re-run only OpenAI evaluations (after fixing billing issue)
    python run_eval_workflow.py openai_cookbook --evaluators openai --skip-memo-generation

    # Run only Claude and Gemini evaluators
    python run_eval_workflow.py my_run --evaluators claude gemini

    # Run with few-shot examples
    python run_eval_workflow.py my_run --few-shot-dir evals/few_shot_examples --parallel-memos

    # Run with few-shot examples AND system parameter (recommended for Claude)
    python run_eval_workflow.py my_run --few-shot-dir evals/few_shot_examples --parallel-memos --use-system-parameter

    # Run with all Claude optimizations (system parameter + XML tags + few-shot)
    python run_eval_workflow.py my_run --few-shot-dir evals/few_shot_examples --parallel-memos --use-system-parameter --use-xml-tags

    # Run with iterative refinement (2 rounds per evaluator)
    python run_eval_workflow.py my_run --parallel-memos --refinement-rounds 2

Output Directories:
-------------------
    batch_temp_{run_name}/
        ├── batch_input_{index}_{timestamp}.jsonl
        ├── batch_output_{index}_{timestamp}.jsonl  (GPT-5 results)
        ├── claude_batch_output_{index}_{timestamp}.jsonl  (Claude results)
        ├── gemini_batch_output_{index}_{timestamp}.jsonl  (Gemini results)
        └── batch_job_mappings.json

    results_{run_name}/
        ├── final_comprehensive_eval_results.json
        ├── results_tables_2.md
        └── comprehensive_sampling_info.json

Notes:
------
- Each run is completely isolated in its own directories
- You can compare results across different runs by using different run names
- The workflow automatically handles all API calls and polling
- Results are resumable - batch jobs continue on provider servers
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list, description: str):
    """Run a command and handle errors."""
    print(f"\n{'='*70}")
    print(f"{description}")
    print(f"{'='*70}\n")

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"\n❌ Error: {description} failed with exit code {result.returncode}")
        sys.exit(1)

    print(f"\n✅ {description} completed successfully!")


def main():
    parser = argparse.ArgumentParser(
        description="Run complete evaluation workflow with custom run name and prompt",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default sample and baseline prompt
  python run_eval_workflow.py baseline_v1

  # Run with custom prompt
  python run_eval_workflow.py experiment_1 --prompt prompts/improved_v2.txt

  # Run with custom indices
  python run_eval_workflow.py test_run --indices 0 1 2 6

  # Run with parallel memo generation (recommended)
  python run_eval_workflow.py baseline_v2 --parallel-memos
        """
    )

    parser.add_argument(
        'run_name',
        type=str,
        help='Name for this evaluation run (determines output directories)'
    )

    parser.add_argument(
        '--prompt',
        type=str,
        default=None,
        help='Path to custom prompt file (e.g., prompts/my_prompt.txt). Default: prompts/baseline.txt'
    )

    parser.add_argument(
        '--indices',
        type=int,
        nargs='+',
        default=None,
        help='Custom indices to evaluate (space-separated). Default: 50-index comprehensive sample'
    )

    parser.add_argument(
        '--parallel-memos',
        action='store_true',
        help='Generate memos in parallel using Claude Batch API (faster, recommended)'
    )

    parser.add_argument(
        '--evaluators',
        type=str,
        nargs='+',
        choices=['gpt-5', 'claude-sonnet-4-20250514', 'gemini-2.5-pro', 'openai', 'claude', 'gemini'],
        default=None,
        help='Evaluator(s) to run. Can use short names (openai, claude, gemini). Default: all 3 evaluators'
    )

    parser.add_argument(
        '--skip-memo-generation',
        action='store_true',
        help='Skip memo generation and use existing batch inputs (useful for re-running specific evaluators after fixing issues)'
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
        help='Wrap inputs in XML tags for better structure (credit_agreement, examples). Recommended for Claude with long documents.'
    )

    parser.add_argument(
        '--refinement-rounds',
        type=int,
        default=0,
        help='Number of iterative refinement rounds. For each evaluator, refine memo based on feedback and re-evaluate. Default: 0 (no refinement)'
    )

    args = parser.parse_args()

    # Validate run_name doesn't start with batch_temp_ (we'll add it)
    run_name = args.run_name
    if run_name.startswith('batch_temp_'):
        run_name = run_name.replace('batch_temp_', '', 1)

    batch_temp_name = f"batch_temp_{run_name}"
    results_name = f"results_{run_name}"

    print(f"\n{'='*70}")
    print(f"EVALUATION WORKFLOW ORCHESTRATOR")
    print(f"{'='*70}")
    print(f"\nRun name: {run_name}")
    print(f"Batch directory: evals/batch_evals/{batch_temp_name}/")
    print(f"Results directory: evals/batch_evals/{results_name}/")

    if args.skip_memo_generation:
        print(f"Prompt file: N/A (loading existing memos from batch inputs)")
    else:
        print(f"Prompt file: {args.prompt if args.prompt else 'prompts/baseline.txt (default)'}")

    print(f"Indices: {args.indices if args.indices else 'Default comprehensive sample (50 indices)'}")
    print(f"Parallel memos: {'Yes (ignored if skipping memo generation)' if args.parallel_memos else 'No'}")
    print(f"Evaluators: {', '.join(args.evaluators) if args.evaluators else 'All (openai, claude, gemini)'}")
    print(f"Skip memo generation: {'Yes' if args.skip_memo_generation else 'No'}")
    print(f"Few-shot examples: {args.few_shot_dir if args.few_shot_dir else 'None'}")
    print(f"Use system parameter (Claude): {'Yes' if args.use_system_parameter else 'No'}")
    print(f"Use XML tags: {'Yes' if args.use_xml_tags else 'No'}")
    print(f"Refinement rounds: {args.refinement_rounds} {'(no refinement)' if args.refinement_rounds == 0 else f'({args.refinement_rounds} rounds per evaluator)'}")
    print(f"\n{'='*70}\n")

    # Get confirmation
    response = input("Proceed with this configuration? [y/N]: ")
    if response.lower() != 'y':
        print("Aborted.")
        sys.exit(0)

    script_dir = Path(__file__).parent

    # ========================================================================
    # STEP 1: Run batch evaluations
    # ========================================================================
    cmd = [
        sys.executable,
        str(script_dir / "run_truly_parallel_batch_eval.py"),
        "--run-name", batch_temp_name,
    ]

    if args.prompt:
        cmd.extend(["--prompt", args.prompt])

    if args.indices:
        cmd.append("--indices")
        cmd.extend(str(idx) for idx in args.indices)

    if args.parallel_memos:
        cmd.append("--parallel-memos")

    if args.evaluators:
        cmd.append("--evaluators")
        cmd.extend(args.evaluators)

    if args.skip_memo_generation:
        cmd.append("--skip-memo-generation")

    if args.few_shot_dir:
        cmd.extend(["--few-shot-dir", args.few_shot_dir])

    if args.use_system_parameter:
        cmd.append("--use-system-parameter")

    if args.use_xml_tags:
        cmd.append("--use-xml-tags")

    if args.refinement_rounds > 0:
        cmd.extend(["--refinement-rounds", str(args.refinement_rounds)])

    run_command(cmd, "STEP 1: Running batch evaluations")

    # ========================================================================
    # STEP 2: Aggregate results
    # ========================================================================
    cmd = [
        sys.executable,
        str(script_dir / "generate_final_results.py"),
        "--batch-temp-dir", batch_temp_name,
        "--output-dir", results_name,
        "--skip-download"
    ]

    run_command(cmd, "STEP 2: Aggregating results")

    # ========================================================================
    # STEP 3: Create analysis tables
    # ========================================================================
    cmd = [
        sys.executable,
        str(script_dir / "create_results_tables_v2.py"),
        "--results-dir", results_name
    ]

    run_command(cmd, "STEP 3: Creating analysis tables")

    # ========================================================================
    # COMPLETE
    # ========================================================================
    print(f"\n{'='*70}")
    print(f"🎉 WORKFLOW COMPLETE!")
    print(f"{'='*70}\n")
    print(f"Your results are available in:")
    print(f"  📂 {script_dir / results_name}/")
    print(f"\nKey files:")
    print(f"  📄 {script_dir / results_name / 'final_comprehensive_eval_results.json'}")
    print(f"  📊 {script_dir / results_name / 'results_tables_2.md'}")
    print(f"\nBatch data:")
    print(f"  📂 {script_dir / batch_temp_name}/")
    print(f"\nTo review a specific memo:")
    print(f"  python generate_memo_review.py <index> {batch_temp_name} {results_name}")
    print(f"  Example: python generate_memo_review.py 2 {batch_temp_name} {results_name}")
    print()


if __name__ == "__main__":
    main()
