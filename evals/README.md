# Evaluation System

This folder contains the core evaluation engine used by the main pipeline in the project root. See the top-level README for how to run end-to-end experiments.

---

## Architecture

The evaluation system supports large-scale, parallel memo evaluation using batch APIs
from OpenAI (GPT-5), Anthropic (Claude), and Google (Gemini).

High-level flow:
1. Generate memos (optionally via Claude Batch API)
2. Submit batch evaluation jobs for each metric × evaluator
3. Poll until completion
4. Aggregate and analyze results

---

## Repository Structure

```
evals/
├── run_eval_workflow.py              # Main entry point (orchestrator)
├── evaluation/                       # Core evaluation modules
│   ├── model_run.py                 # Single memo generation
│   ├── metrics.py                   # 4 evaluation metrics
│   ├── utils.py                     # API utilities
│   ├── evaluator.py                 # Single-memo evaluation
│   ├── batch_evaluate.py            # Batch memo generation helpers
│   ├── batch_metrics.py             # Batch request creation/parsing
│   ├── batch_utils.py               # Low-level batch API interactions
│   └── run_truly_parallel_batch_eval.py  # Batch orchestrator
├── results_analysis/                 # Post-processing
│   ├── generate_final_results.py    # Aggregate batch results to JSON
│   ├── create_results_tables_v2.py  # Generate analysis tables
│   └── generate_memo_review.py      # Detailed memo review
├── few_shot_examples/                # Example memos for prompting
├── batch_outputs/                    # Batch API outputs (created at runtime)
└── results/                          # Final results (created at runtime)
```
---

### Key Benefits
- **True Parallel Evaluation**: Submit all batch jobs at once
- **Reproducible**: Deterministic dataset ensures same results every time
- **Resumable**: Batch jobs run on provider servers - can close computer while waiting
- **Cost Efficient**: Batch APIs typically offer 50% discount vs standard API
- **Custom Testing**: Test with specific indices using `--indices` flag
- **Fast Memo Generation**: Optional parallel memo generation using Claude Batch API

## Quick Start

The **easiest way** to run the full evaluation workflow is using the orchestrator script:

```bash
# Run with default settings (50 documents, baseline prompt)
python evals/run_eval_workflow.py my_experiment --parallel-memos

# Run with custom prompt
python evals/run_eval_workflow.py improved_prompt \
  --prompt prompts/improved.txt \
  --parallel-memos

# Run with specific indices for quick testing
python evals/run_eval_workflow.py test_run \
  --indices 0 1 2 6 \
  --parallel-memos
```

**Output directories:**
- `evals/batch_outputs/batch_temp_{run_name}/` - All batch input/output files
- `evals/results/results_{run_name}/` - Final results JSON and analysis tables

---

## Manual / Advanced Workflow

Use this if you want finer control than `run_eval_workflow.py` provides.

### 1. Run Batch Evaluation

```bash
python evals/evaluation/run_truly_parallel_batch_eval.py <run_name> \
  --indices 0 1 2 6 \
  --parallel-memos
```
Submits batch jobs for memo generation and evaluation.

### 2. Aggregate Results
```
python evals/results_analysis/generate_final_results.py \
  --batch-temp-dir evals/batch_outputs/batch_temp_{run_name} \
  --output-dir evals/results/results_{run_name}
```
### 3. Create Analysis Tables
```
python evals/results_analysis/create_results_tables_v2.py \
  --results-dir evals/results/results_{run_name}
```

Outputs statistical summaries to results_tables_v2.md.

### 4. Detailed Memo Review (Debugging)
```bash
python evals/results_analysis/generate_memo_review.py \
  --batch-temp-dir evals/batch_outputs/batch_temp_{run_name} \
  --index 0 \
  --output memo_review_0.md
```
Generates a markdown file with source document, generated memo, evaluator feedback

## Batch File Conventions

Each dataset index is tracked independently to ensure correctness when jobs finish out of order.

### Input
- batch_input_{INDEX}_{TIMESTAMP}.jsonl

### Outputs
- batch_output_{INDEX}_{TIMESTAMP}.jsonl (GPT-5)
- claude_batch_output_{INDEX}_{TIMESTAMP}.jsonl
- gemini_batch_output_{INDEX}_{TIMESTAMP}.jsonl

### Tracking
- batch_job_mappings.json maps batch IDs → indices