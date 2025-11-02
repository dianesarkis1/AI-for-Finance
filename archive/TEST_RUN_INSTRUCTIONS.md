# Test Run Instructions for Batch Evaluation with Index Tagging

## Summary of Changes

All necessary modifications have been made to ensure proper index tagging throughout the batch evaluation pipeline. Your existing files in `batch_temp/` and `results_benchmark/` will NOT be touched.

## Parallel vs Sequential Memo Generation

### Parallel (Recommended)
- Uses Claude Batch API to generate all memos simultaneously
- Each request tagged with `custom_id: memo_generation_{INDEX}`
- **Speed:** ~10-15 minutes for 50 memos
- Index integrity maintained via custom_id matching

### Sequential (Legacy)
- Generates memos one at a time using standard API calls
- **Speed:** ~30-50 minutes for 50 memos
- More predictable but much slower

## Flow Verification

### ✅ Step 1a: Memo Generation (Parallel Method)
- **Location:** `generate_all_memos_parallel()` in `run_truly_parallel_batch_eval.py`
- **What happens:**
  1. Loads all source documents for specified indices
  2. Creates batch requests with `custom_id: memo_generation_{INDEX}`
  3. Submits single batch job to Claude API
  4. Polls until complete (~10-15 minutes)
  5. Downloads results and matches back to indices via custom_id
- **Index tracking:** Memos stored in memory with index keys

### ✅ Step 1b: Memo Generation (Sequential Method - Legacy)
- **Location:** `generate_all_memos()` in `run_truly_parallel_batch_eval.py`
- **What happens:** Generates memos for each specified index
- **Index tracking:** Memos stored in memory with index keys

### ✅ Step 2: Batch Input Creation & Submission
- **For OpenAI (GPT-5):**
  - `upload_batch_file(..., input_index=idx)` is called
  - Creates: `batch_input_{INDEX}_{TIMESTAMP}.jsonl` in `batch_temp_2/`
  - Uploads to OpenAI API and submits batch job

- **For Claude:**
  - `create_claude_batch()` submits directly to API
  - Input file NOT saved locally (Anthropic API limitation)

- **For Gemini:**
  - `create_gemini_batch()` submits directly to API
  - Input file NOT saved locally (Google API limitation)

### ✅ Step 3: Batch Output Download
- **For OpenAI (GPT-5):**
  - `download_batch_results(..., input_index=job_info['input_index'])` is called
  - Creates: `batch_output_{INDEX}_{TIMESTAMP}.jsonl` in `batch_temp_2/`

- **For Claude:**
  - `download_claude_batch_results(..., input_index=job_info['input_index'])` is called
  - Creates: `claude_batch_output_{INDEX}_{TIMESTAMP}.jsonl` in `batch_temp_2/`

- **For Gemini:**
  - `extract_gemini_batch_results(..., input_index=job_info['input_index'])` is called
  - Creates: `gemini_batch_output_{INDEX}_{TIMESTAMP}.jsonl` in `batch_temp_2/`
  - ✨ **FIXED:** This function was updated to accept and use `input_index` parameter

### ✅ Step 4: Results Aggregation
- **Script:** `generate_final_results.py`
- **What happens:** Parses all output files from `batch_temp_2/`
- **Index extraction:** Uses the `{INDEX}` in filenames to correctly match results
- **Output:** `results_benchmark_2/final_comprehensive_eval_results.json`

## File Naming Convention

All files will follow this pattern with embedded index numbers:

```
batch_temp_2/
├── batch_input_{INDEX}_{TIMESTAMP}.jsonl        (OpenAI only)
├── batch_output_{INDEX}_{TIMESTAMP}.jsonl       (GPT-5 results)
├── claude_batch_output_{INDEX}_{TIMESTAMP}.jsonl (Claude results)
└── gemini_batch_output_{INDEX}_{TIMESTAMP}.jsonl (Gemini results)
```

**Example for index 128:**
```
batch_temp_2/
├── batch_input_128_1761699999.jsonl
├── batch_output_128_1761700500.jsonl
├── claude_batch_output_128_1761700520.jsonl
└── gemini_batch_output_128_1761700540.jsonl
```

## How to Run a Test

### Test with 5 Indices

#### Option 1: Parallel Memo Generation (RECOMMENDED - Much Faster!)

```bash
cd /Users/Diane/AI-for-Finance

# Run with parallel memo generation (faster)
python3 evals/batch_evals/run_truly_parallel_batch_eval.py \
  --indices 0 1 2 6 12 \
  --parallel-memos
```

This will:
1. **Submit 1 batch job** to generate all 5 memos in parallel (~10-15 minutes)
2. Submit 15 evaluation batch jobs (5 indices × 3 evaluators)
3. Save all results to `batch_temp_2/` with proper index tagging
4. **Total time: ~15-25 minutes** (much faster!)

#### Option 2: Sequential Memo Generation (Slower but more reliable)

```bash
cd /Users/Diane/AI-for-Finance

# Run with sequential memo generation (slower)
python3 evals/batch_evals/run_truly_parallel_batch_eval.py \
  --indices 0 1 2 6 12
```

This will:
1. Generate memos one-by-one for indices 0, 1, 2, 6, 12 (~5-10 minutes)
2. Submit 15 evaluation batch jobs (5 indices × 3 evaluators)
3. Save all results to `batch_temp_2/` with proper index tagging
4. **Total time: ~20-40 minutes**

### Aggregate Test Results

```bash
# After batch jobs complete, aggregate the results
python3 evals/batch_evals/generate_final_results.py \
  --batch-temp-dir batch_temp_2 \
  --output-dir results_benchmark_2 \
  --skip-download
```

This will:
1. Parse all files from `batch_temp_2/`
2. Extract indices from filenames
3. Correctly group evaluations by index
4. Save to `results_benchmark_2/final_comprehensive_eval_results.json`

### Verify Results

```bash
# Check that files were created with correct indices
ls -lh evals/batch_evals/batch_temp_2/batch_output_*

# View the aggregated results
cat evals/batch_evals/results_benchmark_2/final_comprehensive_eval_results.json | python3 -m json.tool | head -50
```

## Safety Guarantees

✅ **Existing files protected:**
- Original `batch_temp/` directory untouched
- Original `results_benchmark/` directory untouched

✅ **Index integrity:**
- All output files tagged with source index
- No order-based mapping (alphabetical sorting bug eliminated)
- Each evaluator's results correctly matched to source

✅ **Debugging friendly:**
- Can compare old vs new results
- Can re-run tests without data loss
- File timestamps help track submission/completion order

## Production Run (After Testing)

Once you've verified the test run works correctly:

```bash
# Run all 50 indices (default behavior)
python3 evals/batch_evals/run_truly_parallel_batch_eval.py --indices 0 1 2 6 12 16 17 19 20 48 51 52 57 58 63 71 78 108 114 119 120 122 125 128 134 140 150 152 224 226 239 268 289 297 311 312 318 327 338 343 357 370 377 378 379 390 392 427 458 469

# Or use default (loads from baseline_sampled_indices_seed42.json)
python3 evals/batch_evals/run_truly_parallel_batch_eval.py

# Aggregate results
python3 evals/batch_evals/generate_final_results.py \
  --batch-temp-dir batch_temp_2 \
  --output-dir results_benchmark_2 \
  --skip-download
```

## Troubleshooting

### Files missing index numbers
- Check that `run_truly_parallel_batch_eval.py` is using `BATCH_TEMP_DIR = OUTPUT_DIR / "batch_temp_2"`
- Verify the print statement at startup shows correct directory

### Results aggregation errors
- Ensure you used `--batch-temp-dir batch_temp_2` when running `generate_final_results.py`
- Check that files actually have indices in filenames: `ls batch_temp_2/batch_output_*`

### API rate limits
- The script polls every 60 seconds
- Batch jobs typically complete in 10-30 minutes
- If needed, increase poll interval in script

## Next Steps

1. **Run test with 5 indices** to verify everything works
2. **Check the output files** have correct index numbers in filenames
3. **Run aggregation script** and verify results are correctly grouped
4. **If successful**, run full evaluation with all 50 indices
5. **Compare** results between old `results_benchmark/` and new `results_benchmark_2/`
