# Batch Evaluation Module

This module provides batch API versions of the evaluation functions for faster processing of investment memo evaluations.

## Overview

The original `evaluator.py` and `metrics.py` make sequential API calls for each metric evaluation, which can be slow (2-3 minutes per memo). This batch version combines all metric evaluations into a single OpenAI Batch API job, significantly reducing the total time.

**Current Status:**
- ✅ GPT-5 batch evaluation implemented
- ⏳ Claude batch evaluation (coming soon)
- ⏳ Gemini batch evaluation (coming soon)

## Key Benefits

1. **Parallel Processing**: All metrics evaluated simultaneously on OpenAI's servers
2. **Resumable**: Can close your computer during processing - batch runs on OpenAI's side
3. **Cost Efficient**: Batch API often has lower costs than standard API
4. **Same Output Format**: Returns identical scores and format as original functions

## How It Works

### Standard Evaluation (Original)
```
Call GPT-5 for accuracy → Wait → Call GPT-5 for completeness → Wait →
Call GPT-5 for consistency → Wait → Call GPT-5 for clarity → Wait →
Call GPT-5 for tone → Wait → Call GPT-5 for length → Wait →
Call GPT-5 for structure → Wait → Aggregate results
Total: 7 sequential API calls (2-3 minutes)
```

### Batch Evaluation (New)
```
Create 7 requests → Submit as batch → Wait for batch to complete →
Download all results at once → Aggregate results
Total: 1 batch job (depends on queue time, but runs in parallel)
```

## Usage

### Basic Usage

```python
from evals.batch_evals import evaluate_memo_batch

# Evaluate a memo using GPT-5 via Batch API
score = evaluate_memo_batch(
    memo=generated_memo,
    source_document=credit_agreement,
    template=memo_template,  # optional
    model="gpt-5",
    poll_interval=60  # check status every 60 seconds
)

print(f"Summary Score: {score:.2f}/100")
```

### Resume Interrupted Batch

If you close your computer or stop the script, the batch continues running on OpenAI's servers. You can resume monitoring:

```python
from evals.batch_evals import resume_batch_evaluation

# When you started the batch, you saw:
# "Batch job created: batch_abc123xyz"

# Resume monitoring that batch
results = resume_batch_evaluation(
    batch_id="batch_abc123xyz",
    poll_interval=60
)
```

### Evaluate with Multiple Models (Future)

```python
from evals.batch_evals import evaluate_memo_batch_with_all_models

# Currently only GPT-5 works, but this will support all models
scores = evaluate_memo_batch_with_all_models(
    memo=generated_memo,
    source_document=credit_agreement,
    template=memo_template
)

# Returns: {"gpt-5": 85.5, "claude-sonnet-4": 87.2, "gemini-2.5-pro": 84.1}
```

## Files

### Main Files
- **`evaluator_batch.py`**: Batch version of evaluator.py with `evaluate_memo_batch()`
- **`metrics_batch.py`**: Batch versions of metric functions (accuracy, completeness, etc.)
- **`batch_utils.py`**: Low-level utilities for OpenAI Batch API operations

### Supporting Files
- **`batch_temp/`**: Temporary directory for batch input/output files (not deleted for debugging)
- **`__init__.py`**: Package initialization

## Comparison with Original

| Feature | Original (`evaluator.py`) | Batch (`evaluator_batch.py`) |
|---------|---------------------------|------------------------------|
| API Calls | 7 sequential calls per memo | 1 batch job with 7 requests |
| Time per Memo | 2-3 minutes | Depends on queue (~minutes to hours) |
| Can Close Computer | ❌ No, script must run | ✅ Yes, batch runs on OpenAI side |
| Resumable | ❌ No | ✅ Yes, with `resume_batch_evaluation()` |
| Output Format | Summary score (0-100) | Summary score (0-100) - identical |
| Multi-Model Support | GPT-5, Claude, Gemini | GPT-5 only (for now) |

## How Batch API Works

1. **Create Batch Requests**: Generate a JSONL file with all evaluation prompts
2. **Upload File**: Upload JSONL to OpenAI's file storage
3. **Create Batch Job**: Submit batch job referencing the uploaded file
4. **Poll Status**: Check job status every N seconds (free, doesn't use API credits)
5. **Download Results**: When complete, download results JSONL file
6. **Parse Results**: Extract and aggregate metric scores

## Batch Job States

- `validating`: OpenAI is validating your batch requests
- `in_progress`: Batch is being processed (this is where it spends most time)
- `finalizing`: Batch is nearly complete, finalizing results
- `completed`: Batch is done, results ready to download
- `failed`: Something went wrong (check error file)
- `cancelled`: Batch was manually cancelled
- `expired`: Batch exceeded 24-hour window

## Cost Notes

- Batch API often has 50% discount compared to standard API
- Polling (status checks) is **free** and doesn't count toward API usage
- You only pay for the actual LLM inference (the 7 evaluation prompts)

## Debugging

### Check Batch Status Manually

```bash
# Set your API key
export OPENAI_API_KEY="sk-..."

# Check batch status
curl https://api.openai.com/v1/batches/batch_abc123xyz \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

### View Temporary Files

All batch files are saved in `evals/batch_evals/batch_temp/`:
- `batch_input_*.jsonl`: Input requests sent to OpenAI
- `batch_output_*.jsonl`: Results returned by OpenAI
- `batch_*.json`: Metadata for each batch job (includes batch ID for resuming)

## Error Handling

### Batch Job Failed

If a batch job fails, the error file will be automatically downloaded to `batch_temp/`. Check it for details on which requests failed and why.

### Timeout

By default, batches have a 24-hour timeout. If your batch isn't complete after 24 hours, it will raise a `TimeoutError`. You can still check the status manually or resume later.

### API Key Missing

Make sure `OPENAI_API_KEY` is set in your environment:
```bash
export OPENAI_API_KEY="sk-..."
```

## Future Enhancements

- [ ] Add Claude batch API support
- [ ] Add Gemini batch API support
- [ ] Support for evaluating multiple memos in one batch
- [ ] Batch support for `worst_at_k()` function
- [ ] Progress bar for polling
- [ ] Email/webhook notifications when batch completes

## Questions?

See the main evaluation docs or check the original `evaluator.py` and `metrics.py` for reference on how metrics work.
