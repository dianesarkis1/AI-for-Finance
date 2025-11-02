# Batch Evaluation Module

This module provides a complete workflow for evaluating investment memo generation at scale using batch APIs from OpenAI, Anthropic (Claude), and Google (Gemini).

## Overview

This system evaluates a model's ability to generate investment memos from credit agreements by:
1. Generating memos for a sampled set of credit agreements (by default, 50 indices with seed=42)
2. Evaluating each memo using 3 different LLM evaluators (GPT-5, Claude, Gemini)
3. Aggregating results and producing statistical analysis

**Key Benefits:**
- **True Parallel Evaluation**: Submit all 150 batch jobs at once (50 inputs × 3 evaluators)
- **Reproducible**: Same seed ensures same sample every time
- **Resumable**: Batch jobs run on provider servers - can close computer while waiting
- **Cost Efficient**: Batch APIs typically offer 50% discount vs standard API
- **Custom Testing**: Test with specific indices using `--indices` flag
- **Fast Memo Generation**: Optional parallel memo generation using Claude Batch API

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

**Full benchmark (50 indices):**
```bash
python evals/batch_evals/run_truly_parallel_batch_eval.py
```

**Test with specific indices:**
```bash
# Sequential memo generation (recommended for <10 indices)
python evals/batch_evals/run_truly_parallel_batch_eval.py --indices 0 1 2 6

# Parallel memo generation (faster for ≥10 indices)
python evals/batch_evals/run_truly_parallel_batch_eval.py --indices 0 1 2 6 --parallel-memos
```

**What it does:**
- Generates memos for specified credit agreements from `data/train.jsonl`
- Submits batch evaluation jobs to all 3 evaluators (GPT-5, Claude, Gemini)
- Polls until all jobs complete
- Downloads results to `batch_temp_2/` folder (customizable)
- Saves batch job mappings to `batch_temp_2/batch_job_mappings.json` for tracking

**Time:**
- Sequential memos: ~1-2 min per memo
- Parallel memos: ~10-15 min fixed overhead (use for ≥10 memos)
- Batch evaluations: ~10-30 minutes

### 3. Aggregate Results
Combine all evaluation results:

```bash
python evals/batch_evals/generate_final_results.py
```

For custom directories:
```bash
python evals/batch_evals/generate_final_results.py \
  --batch-temp-dir batch_temp_2 \
  --output-dir results_benchmark_2 \
  --skip-download
```

**What it does:**
- Reads all result files from `batch_temp/` (or specified directory)
- Parses results from each provider (OpenAI, Claude, Gemini)
- Extracts index from filename for correct mapping
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

The script uses a fixed sample (seed=42) of 50 indices from this dataset by default.

### Custom Indices
Test with specific indices instead of the default 50:

```bash
# Test with 4 specific indices
python evals/batch_evals/run_truly_parallel_batch_eval.py --indices 0 1 2 6

# Test a single problematic index
python evals/batch_evals/run_truly_parallel_batch_eval.py --indices 128
```

### Parallel Memo Generation
For faster memo generation (recommended for ≥10 memos):

```bash
python evals/batch_evals/run_truly_parallel_batch_eval.py --indices 0 1 2 6 --parallel-memos
```

**How it works:**
- Uses Claude Batch API to generate all memos in parallel
- Each request tagged with `custom_id: memo_generation_{INDEX}`
- ~10-15 min fixed overhead regardless of batch size
- Much faster than sequential for large batches (≥10 memos)

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

### Output Directories
By default, outputs go to `batch_temp_2/` and `results_benchmark_2/`. These can be configured in the script:

```python
BATCH_TEMP_DIR = OUTPUT_DIR / "batch_temp_2"  # Batch files
OUTPUT_DIR = Path("evals/batch_evals/results_benchmark_2")  # Final results
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

**Note:** Use different output directories (e.g., `batch_temp_2`, `results_benchmark_2`) to avoid overwriting existing results.

## Files

### Main Scripts

- **`run_truly_parallel_batch_eval.py`** - Main execution script
  - Generates memos (sequential or parallel)
  - Submits batch evaluation jobs
  - Polls and downloads results
  - Saves batch job mappings to `batch_job_mappings.json`
  - Supports custom indices via `--indices` flag
  - Supports parallel memo generation via `--parallel-memos` flag

- **`generate_final_results.py`** - Aggregates results from batch files into JSON
  - Supports custom directories via `--batch-temp-dir` and `--output-dir`
  - Uses index from filenames for correct mapping (e.g., `batch_output_0_timestamp.jsonl`)

- **`create_results_tables_v2.py`** - Creates markdown analysis tables

- **`preview_sampling.py`** - Preview which indices will be sampled (optional)

- **`generate_memo_review.py`** - Generate detailed review of specific evaluation
  ```bash
  python evals/batch_evals/generate_memo_review.py \
    --batch-temp-dir batch_temp_2 \
    --index 128 \
    --output memo_review_128.md
  ```

### Support Modules

- **`batch_evaluate.py`** - Memo generation and aggregation helper functions
- **`batch_metrics.py`** - Batch request creation and result parsing for all 3 providers
- **`batch_utils.py`** - Low-level API interactions (upload, submit, poll, download)

### Directories

- **`batch_temp_2/`** - Temporary storage for batch input/output files (customizable)
- **`results_benchmark_2/`** - Final aggregated results and analysis tables (customizable)

### Key Files

- **`batch_temp_2/batch_job_mappings.json`** - Tracks batch_id to index mapping
  - Created automatically by `run_truly_parallel_batch_eval.py`
  - Contains mapping for all providers (OpenAI, Claude, Gemini)
  - Used for debugging and recovery

## How It Works

### Phase 1: Memo Generation

**Sequential Mode (default for custom indices):**
- For each sampled index, call `evals/model_run.py` to generate memo
- Time: ~1-2 minutes per memo

**Parallel Mode (--parallel-memos):**
- Submit all memo generation requests to Claude Batch API at once
- Each tagged with `custom_id: memo_generation_{INDEX}`
- Poll until complete, match results back to indices
- Time: ~10-15 minutes fixed overhead (recommended for ≥10 memos)

### Phase 2: Batch Job Submission (Parallel)

For each memo, submit batch evaluation jobs to all 3 providers:
1. Create batch requests using prompts from `evals/metrics.py`
2. Submit to OpenAI, Anthropic, and Google batch APIs
3. Save batch job mappings to `batch_job_mappings.json`
4. Get batch IDs and move to polling phase

**Time:** ~1 minute (just submission, no waiting)

### Phase 3: Polling (Parallel)

Poll all batch jobs in parallel:
1. Check status every 60 seconds
2. Download results when each job completes
3. Save to `batch_temp_2/` with index in filename:
   - `batch_output_{INDEX}_{TIMESTAMP}.jsonl` (GPT-5)
   - `claude_batch_output_{INDEX}_{TIMESTAMP}.jsonl` (Claude)
   - `gemini_batch_output_{INDEX}_{TIMESTAMP}.jsonl` (Gemini)

**Time:** ~10-30 minutes (depends on provider queue times)

**Note:** Index is embedded in filenames to ensure correct mapping even if batches complete out of order.

### Phase 4: Aggregation (Post-Processing)

Run `generate_final_results.py`:
1. Read all result files from `batch_temp_2/`
2. Parse results using provider-specific parsers
3. Extract index from filename (e.g., `batch_output_0_1762109716.jsonl` → index 0)
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

## File Naming Conventions

All batch files include the dataset index for correct mapping:

**Batch Input Files:**
- `batch_input_{INDEX}_{TIMESTAMP}.jsonl` - Contains 7 evaluation requests

**Batch Output Files:**
- `batch_output_{INDEX}_{TIMESTAMP}.jsonl` - GPT-5 evaluation results
- `claude_batch_output_{INDEX}_{TIMESTAMP}.jsonl` - Claude evaluation results
- `gemini_batch_output_{INDEX}_{TIMESTAMP}.jsonl` - Gemini evaluation results

**Tracking File:**
- `batch_job_mappings.json` - Maps batch IDs to indices for all providers

This ensures correct index mapping even when:
- Batches complete out of order (e.g., index 6 completes before index 1)
- Process is interrupted and resumed
- Multiple runs exist in the same directory

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

Set these environment variables or add to `.env` file:

```bash
export OPENAI_API_KEY="sk-..."          # For GPT-5 evaluation
export ANTHROPIC_API_KEY="sk-ant-..."  # For Claude evaluation and memo generation
export GEMINI_API_KEY="..."            # For Gemini evaluation
```

The script automatically loads from `.env` if environment variables are not set.

## Troubleshooting

### Gemini batches not detected as complete

**Status:** Fixed in latest version. The script now correctly checks `metadata.state` and looks for `BATCH_STATE_SUCCEEDED`.

If using an older version, update `batch_utils.py` line ~1330 to check:
```python
metadata = status_data.get("metadata", {})
state = metadata.get("state")
if state == "BATCH_STATE_SUCCEEDED":
```

### Index mismatch across evaluators

**Status:** Fixed. All batch files now include explicit index in filename (e.g., `batch_output_0_timestamp.jsonl`).

The `batch_job_mappings.json` file also tracks the batch_id → index mapping for all providers.

### Script fails with "model_run.py not found"

Make sure `evals/model_run.py` exists. The path is now automatically resolved.

### Results seem inconsistent

Old batch result files may be interfering. Use separate directories:
```bash
python evals/batch_evals/run_truly_parallel_batch_eval.py  # Uses batch_temp_2/
```

Or archive old files before running a new evaluation.

### Batch jobs stuck in "in_progress"

This is normal. Batch APIs can take 10-30 minutes depending on provider queue. The script will keep polling automatically.

### Want to resume after closing computer

Batch jobs continue running on provider servers. The script detects completed jobs automatically on restart using the `batch_job_mappings.json` file.

### Need to investigate a specific bad score

Use `generate_memo_review.py` to create a detailed review:

```bash
python evals/batch_evals/generate_memo_review.py \
  --batch-temp-dir batch_temp_2 \
  --index 128 \
  --output review_128.md
```

This creates a markdown file with:
- Original credit agreement
- Generated memo
- All evaluator feedback with explanations

## Output Files

### During Execution

**Batch Input Files (7 evaluation requests each):**
- `batch_temp_2/batch_input_{INDEX}_{TIMESTAMP}.jsonl`

**Batch Output Files:**
- `batch_temp_2/batch_output_{INDEX}_{TIMESTAMP}.jsonl` - GPT-5 results
- `batch_temp_2/claude_batch_output_{INDEX}_{TIMESTAMP}.jsonl` - Claude results
- `batch_temp_2/gemini_batch_output_{INDEX}_{TIMESTAMP}.jsonl` - Gemini results

**Tracking File:**
- `batch_temp_2/batch_job_mappings.json` - Batch ID to index mappings

### Final Results

- `results_benchmark_2/final_comprehensive_eval_results.json` - Complete results with all evaluator assessments
- `results_benchmark_2/results_tables_2.md` - Statistical analysis tables
- `results_benchmark_2/comprehensive_sampling_info.json` - Record of which indices were evaluated

## Advanced Usage

### Testing with a Subset

Test with a small subset first:

```bash
# Test with 4 indices
python evals/batch_evals/run_truly_parallel_batch_eval.py --indices 0 1 2 6

# Aggregate (note: use --skip-download since batches were already downloaded)
python evals/batch_evals/generate_final_results.py \
  --batch-temp-dir batch_temp_2 \
  --output-dir results_benchmark_2 \
  --skip-download
```

### Parallel Memo Generation

For large batches (≥10 memos), use parallel generation:

```bash
python evals/batch_evals/run_truly_parallel_batch_eval.py \
  --indices 0 1 2 6 12 52 57 71 114 125 \
  --parallel-memos
```

**When to use:**
- Sequential: <10 memos (~1-2 min/memo)
- Parallel: ≥10 memos (~10-15 min fixed)

### Separate Output Directories

Avoid overwriting existing results:

```python
# In run_truly_parallel_batch_eval.py, change:
BATCH_TEMP_DIR = OUTPUT_DIR / "batch_temp_3"  # New directory
OUTPUT_DIR = Path("evals/batch_evals/results_benchmark_3")
```

Then aggregate with matching directories:

```bash
python evals/batch_evals/generate_final_results.py \
  --batch-temp-dir batch_temp_3 \
  --output-dir results_benchmark_3 \
  --skip-download
```

## Best Practices

1. **Test with small subset first** - Use `--indices 0 1 2 6` to verify everything works
2. **Use parallel memos for large batches** - Add `--parallel-memos` for ≥10 memos
3. **Keep results organized** - Use separate directories for different runs
4. **Track batch IDs** - The `batch_job_mappings.json` file helps with debugging
5. **Review bad scores** - Use `generate_memo_review.py` to investigate issues
6. **Archive old results** - Keep old `batch_temp/` directories for comparison
