# Batch Evaluation Module

This module provides a complete workflow for evaluating investment memo generation at scale using batch APIs from OpenAI, Anthropic (Claude), and Google (Gemini).

## Overview

This system evaluates a model's ability to generate investment memos from credit agreements by:
1. Generating memos for a sampled set of credit agreements (by default, seed=42, which is our benchmark dataset)
2. Evaluating each memo using 3 different LLM evaluators (GPT-5, Claude, Gemini)
3. Aggregating results and producing statistical analysis

**Key Benefits:**
- **True Parallel Evaluation**: Submit all 150 batch jobs at once (50 inputs × 3 evaluators)
- **Reproducible**: Same seed ensures same sample every time
- **Resumable**: Batch jobs run on provider servers - can close computer while waiting
- **Cost Efficient**: Batch APIs typically offer 50% discount vs standard API

## Quick Start

### 1. Preview Sampling (Optional)
See which indices will be evaluated without running anything:

```bash
python evals/batch_evals/preview_sampling.py
```

This shows the 50 indices that will be sampled:
- 10 baseline indices (from `evals/benchmark/baseline_sampled_indices_seed42.json`)
- First 3 indices from dataset (0, 1, 2)
- 37 random indices (seed=42)

### 2. Run Batch Evaluation
Generate memos and submit evaluation jobs:

```bash
python evals/batch_evals/run_truly_parallel_batch_eval.py
```

**What it does:**
- Generates memos for 50 sampled credit agreements from `data/train.jsonl`
- Submits 150 batch evaluation jobs (50 inputs × 3 evaluators)
- Polls until all jobs complete
- Downloads results to `batch_temp/` folder

**Time:** ~30-50 minutes for memo generation, then 10-30 minutes for batch evaluations

### 3. Aggregate Results
Combine all evaluation results:

```bash
python evals/batch_evals/generate_final_results.py
```

**What it does:**
- Reads all result files from `batch_temp/`
- Parses results from each provider (OpenAI, Claude, Gemini)
- Organizes by input index
- Saves to `final_comprehensive_eval_results.json`

### 4. Create Analysis Tables
Generate markdown tables with statistics:

```bash
python evals/batch_evals/create_results_tables_v2.py
```

**What it does:**
- Reads `results_benchmark/final_comprehensive_eval_results.json`
- Creates statistical summaries
- Outputs to `results_benchmark/results_tables_2.md`

## Configuration

### Dataset
Dataset is configured in `run_truly_parallel_batch_eval.py`:

```python
TRAIN_FILE = Path("data/train.jsonl")  # Input dataset
```

The script uses a fixed sample (seed=42) of 50 indices from this dataset.

### Prompt
By default, uses `prompts/baseline.txt`. To change the prompt:

**Edit `run_truly_parallel_batch_eval.py`:**
```python
# Prompt configuration
PROMPT_FILE = None  # Uses prompts/baseline.txt (default)
# Or specify custom prompt:
PROMPT_FILE = Path("prompts/my_custom_prompt.txt")
```

### Model to Evaluate
Change which model generates the memos:

```python
MODEL_TO_EVALUATE = "claude-sonnet-4-20250514"  # Change this
```

### Evaluator Models
Change which models evaluate the memos:

```python
EVALUATOR_MODELS = ["gpt-5", "claude-sonnet-4-20250514", "gemini-2.5-pro"]
```

## Prompt Iteration Workflow

To test different prompts on the same dataset:

1. **Edit prompt configuration** in `run_truly_parallel_batch_eval.py`:
   ```python
   PROMPT_FILE = Path("prompts/version2.txt")
   ```

2. **Run evaluation** (generates new memos with new prompt):
   ```bash
   python evals/batch_evals/run_truly_parallel_batch_eval.py
   ```

3. **Aggregate results**:
   ```bash
   python evals/batch_evals/generate_final_results.py
   ```

4. **Create tables**:
   ```bash
   python evals/batch_evals/create_results_tables_v2.py
   ```

5. **Compare** results in `results_benchmark/results_tables_2.md`

**Note:** Consider archiving old `batch_temp/` files before running a new prompt iteration to avoid confusion.

## Files

### Main Scripts
- **`run_truly_parallel_batch_eval.py`** - Main execution script (generates memos + runs batch evals)
- **`generate_final_results.py`** - Aggregates results from batch_temp/ into JSON
- **`create_results_tables_v2.py`** - Creates markdown analysis tables
- **`preview_sampling.py`** - Preview which indices will be sampled (optional)

### Support Modules
- **`batch_evaluate.py`** - Memo generation and aggregation helper functions
- **`batch_metrics.py`** - Batch request creation and result parsing for all 3 providers
- **`batch_utils.py`** - Low-level API interactions (upload, submit, poll, download)

### Directories
- **`batch_temp/`** - Temporary storage for batch input/output files
- **`results_benchmark/`** - Final aggregated results and analysis tables

## How It Works

### Phase 1: Memo Generation (Sequential)
For each of the 50 sampled indices:
1. Load credit agreement from `data/train.jsonl`
2. Call `evals/model_run.py` to generate memo using specified model and prompt
3. Store memo for evaluation

**Time:** ~30-50 minutes (depends on model speed)

### Phase 2: Batch Job Submission (Parallel)
For each memo, submit batch evaluation jobs to all 3 providers:
1. Create batch requests using prompts from `evals/metrics.py`
2. Submit to OpenAI, Anthropic, and Google batch APIs
3. Get batch IDs and move to polling phase

**Time:** ~1 minute (just submission, no waiting)

### Phase 3: Polling (Parallel)
Poll all 150 batch jobs in parallel:
1. Check status every 60 seconds
2. Download results when each job completes
3. Save to `batch_temp/` with index in filename (e.g., `batch_output_0_timestamp.jsonl`)

**Time:** ~10-30 minutes (depends on provider queue times)

### Phase 4: Aggregation (Post-Processing)
Run `generate_final_results.py`:
1. Read all result files from `batch_temp/`
2. Parse results using provider-specific parsers
3. Extract index from filename
4. Organize results by index, keeping each evaluator's assessment separate
5. Save to `final_comprehensive_eval_results.json`

### Phase 5: Analysis (Post-Processing)
Run `create_results_tables_v2.py`:
1. Calculate statistics across all evaluations
2. Create markdown tables showing:
   - Summary statistics (mean, median, min, max scores)
   - Per-metric statistics
   - Per-evaluator statistics
   - Detailed results by index

## Evaluation Metrics

Each memo is evaluated on:

**Binary Metrics** (0 or 1):
- **Accuracy**: No hallucinated terms
- **Completeness**: All key terms captured
- **Consistency**: No internal contradictions

**Quality Sub-Metrics** (0-100):
- **Clarity**: Easy to understand
- **Tone**: Professional and appropriate
- **Length**: Right level of detail
- **Structure**: Well-organized

**Summary Score**: Weighted average of all metrics (0-100)

## API Keys Required

Set these environment variables:

```bash
export OPENAI_API_KEY="sk-..."          # For GPT-5 evaluation
export ANTHROPIC_API_KEY="sk-ant-..."  # For Claude evaluation
export GEMINI_API_KEY="..."            # For Gemini evaluation
```

Also needed for memo generation (depending on MODEL_TO_EVALUATE).

## Troubleshooting

### Script fails with "model_run.py not found"
Make sure `evals/model_run.py` exists. The path is now automatically resolved.

### Results seem inconsistent
Old batch result files may be interfering. Archive or delete old files in `batch_temp/` before running a new evaluation.

### Batch jobs stuck in "in_progress"
This is normal. Batch APIs can take 10-30 minutes depending on provider queue. The script will keep polling automatically.

### Want to resume after closing computer
Batch jobs continue running on provider servers. Just restart `run_truly_parallel_batch_eval.py` - it will detect completed jobs and download results.

## Output Files

### During Execution
- `batch_temp/batch_input_INDEX_timestamp.jsonl` - OpenAI batch requests
- `batch_temp/batch_output_INDEX_timestamp.jsonl` - OpenAI batch results
- `batch_temp/claude_batch_output_INDEX_timestamp.jsonl` - Claude batch results
- `batch_temp/gemini_batch_output_INDEX_timestamp.jsonl` - Gemini batch results

### Final Results
- `final_comprehensive_eval_results.json` - Complete results with all evaluator assessments
- `results_benchmark/results_tables_2.md` - Statistical analysis tables
