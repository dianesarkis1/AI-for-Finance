"""
Main evaluation harness that runs all metrics on generated memos.
"""

from typing import Dict, List, Optional
from evals.metrics import (
    evaluate_accuracy,
    evaluate_completeness,
    evaluate_consistency,
    evaluate_quality,
    calculate_summary_score
)


def evaluate_memo(
    memo: str,
    source_document: str,
    template: str = None,
    eval_models: List[str] = None,
    weights: Dict[str, float] = None
) -> float:
    """
    Evaluate a single memo across all metrics and return summary score.

    This is the primary evaluation function that runs all 4 metrics
    (accuracy, completeness, consistency, quality) and aggregates them
    into a single summary score.

    Args:
        memo: Generated investment memo text
        source_document: Original credit agreement text
        template: Optional template for structure evaluation
        eval_models: Models to use for evaluation (default: gpt-5, claude, gemini)
        weights: Optional weights for summary score (default: equal 0.25 each)

    Returns:
        float: Summary score (0-100)
    """
    if eval_models is None:
        eval_models = ["gpt-5", "claude-sonnet-4-20250514", "gemini-2.5-pro"]

    # Run all 4 metrics
    accuracy_result = evaluate_accuracy(
        memo=memo,
        source_document=source_document,
        models=eval_models
    )

    completeness_result = evaluate_completeness(
        memo=memo,
        source_document=source_document,
        models=eval_models
    )

    consistency_result = evaluate_consistency(
        memo=memo,
        models=eval_models
    )

    quality_result = evaluate_quality(
        memo=memo,
        template=template,
        models=eval_models
    )

    # Calculate summary score
    summary_result = calculate_summary_score(
        accuracy_result=accuracy_result,
        completeness_result=completeness_result,
        consistency_result=consistency_result,
        quality_result=quality_result,
        weights=weights
    )

    # Return only the summary score
    return summary_result["summary_score"]


def worst_at_k(
    model: str,
    input_file: str,
    source_document: str,
    k: int = 5,
    template: str = None,
    eval_models: List[str] = None,
    weights: Dict[str, float] = None
) -> Dict:
    """
    Run model k times on same input and return the worst summary score.

    This tests the worst-case performance of a model by running it multiple times
    and returning the minimum (worst) score achieved.

    Args:
        model: Model identifier to test (e.g., 'gpt-5', 'claude-sonnet-4-20250514')
        input_file: Path to input credit agreement file (.txt, .md, or .jsonl)
        source_document: Source document text for evaluation
        k: Number of runs to perform (default: 5)
        template: Optional template for structure evaluation
        eval_models: Models to use for evaluation (default: gpt-5, claude, gemini)
        weights: Optional weights for summary score (default: equal 0.25 each)

    Returns:
        Dict with:
            - worst_score: float, minimum score across k runs (0-100)
            - all_scores: List[float], all k scores
            - mean_score: float, average score across k runs
            - best_score: float, maximum score across k runs
            - std_dev: float, standard deviation of scores
    """
    import subprocess
    import tempfile
    import statistics
    from pathlib import Path

    print(f"\n{'='*60}")
    print(f"WORST-AT-K EVALUATION: {model}")
    print(f"Running {k} iterations to find worst-case score")
    print(f"{'='*60}\n")

    all_scores = []

    for i in range(k):
        print(f"Run {i+1}/{k}...")

        # Create temporary output file for this run
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as tmp_file:
            output_path = tmp_file.name

        try:
            # Call model_run.py to generate memo
            cmd = [
                "python",
                "latest project scripts/model_run.py",
                "--model", model,
                "--input-file", input_file,
                "--output", output_path
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            if result.returncode != 0:
                print(f"  ❌ Error generating memo: {result.stderr}")
                continue

            # Read generated memo
            with open(output_path, 'r', encoding='utf-8') as f:
                memo = f.read()

            # Evaluate the memo
            print(f"  Evaluating memo...")
            score = evaluate_memo(
                memo=memo,
                source_document=source_document,
                template=template,
                eval_models=eval_models,
                weights=weights
            )

            all_scores.append(score)
            print(f"  ✅ Score: {score:.2f}/100\n")

        except subprocess.TimeoutExpired:
            print(f"  ⏰ Timeout on run {i+1}\n")
            continue
        except Exception as e:
            print(f"  💥 Error on run {i+1}: {e}\n")
            continue
        finally:
            # Clean up temporary file
            try:
                Path(output_path).unlink()
            except:
                pass

    if not all_scores:
        return {
            "worst_score": 0.0,
            "all_scores": [],
            "mean_score": 0.0,
            "best_score": 0.0,
            "std_dev": 0.0,
            "error": "No successful runs"
        }

    # Calculate statistics
    worst_score = min(all_scores)
    best_score = max(all_scores)
    mean_score = statistics.mean(all_scores)
    std_dev = statistics.stdev(all_scores) if len(all_scores) > 1 else 0.0

    print(f"{'='*60}")
    print(f"WORST-AT-K RESULTS")
    print(f"{'='*60}")
    print(f"Worst Score:  {worst_score:.2f}/100")
    print(f"Best Score:   {best_score:.2f}/100")
    print(f"Mean Score:   {mean_score:.2f}/100")
    print(f"Std Dev:      {std_dev:.2f}")
    print(f"Score Range:  {best_score - worst_score:.2f}")
    print(f"{'='*60}\n")

    return {
        "worst_score": worst_score,
        "all_scores": all_scores,
        "mean_score": mean_score,
        "best_score": best_score,
        "std_dev": std_dev,
        "score_range": best_score - worst_score
    }
