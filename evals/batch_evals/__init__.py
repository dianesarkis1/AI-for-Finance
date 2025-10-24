"""
Batch-enabled evaluation module for investment memo generation.

This module provides batch API versions of evaluation functions for faster
processing using OpenAI's Batch API.

Main functions:
- evaluate_memo_batch: Evaluate a single memo using batch API (GPT-5 only for now)
- resume_batch_evaluation: Resume monitoring an interrupted batch job
- evaluate_memo_batch_with_all_models: Evaluate with multiple models (when implemented)
"""

from evals.batch_evals.evaluator_batch import (
    evaluate_memo_batch,
    resume_batch_evaluation,
    evaluate_memo_batch_with_all_models
)

__all__ = [
    "evaluate_memo_batch",
    "resume_batch_evaluation",
    "evaluate_memo_batch_with_all_models"
]
