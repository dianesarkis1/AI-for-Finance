# Quick Reference: Key Functions & Files

## Most Important Files for Streamlit

| File | Purpose | Key Functions |
|------|---------|---|
| `/Users/Diane/AI-for-Finance/evals/model_run.py` | Memo generation | `call_anthropic_api()`, `build_anthropic_payload()`, `extract_credit_agreement_from_jsonl()` |
| `/Users/Diane/AI-for-Finance/evals/evaluator.py` | Evaluation harness | `evaluate_memo()`, `worst_at_k()` |
| `/Users/Diane/AI-for-Finance/evals/metrics.py` | All 4 metrics | `evaluate_accuracy()`, `evaluate_completeness()`, `evaluate_consistency()`, `evaluate_quality()` |
| `/Users/Diane/AI-for-Finance/evals/utils.py` | LLM utilities | `call_llm_for_eval()` |
| `/Users/Diane/AI-for-Finance/evals/batch_evals/batch_utils.py` | Batch API | `submit_and_wait_for_batch()`, etc. |
| `/Users/Diane/AI-for-Finance/data/data_cleaning.py` | Document processing | `fetch_and_clean()`, `clean_html_to_text()` |

---

## Minimal Working Example

```python
import os
from pathlib import Path
from evals.model_run import (
    read_text_file,
    build_anthropic_payload,
    call_anthropic_api,
    extract_output_text_anthropic
)
from evals.evaluator import evaluate_memo

# 1. Load document
doc = read_text_file(Path("credit_agreement.txt"))

# 2. Generate memo
api_key = os.getenv("ANTHROPIC_API_KEY")
payload = build_anthropic_payload(
    model="claude-sonnet-4-20250514",
    content=f"Generate an investment memo:\n{doc}",
    max_output_tokens=16000
)
response = call_anthropic_api(api_key, payload)
memo = extract_output_text_anthropic(response)

# 3. Evaluate
score = evaluate_memo(memo, doc)
print(f"Score: {score:.2f}/100")
```

---

## Input File Formats

### TXT File
```
Plain text credit agreement
```

### JSONL File
```jsonl
{"source_url": "https://...", "text": "CREDIT AGREEMENT..."}
```

### MD File
```markdown
# Credit Agreement

Contents...
```

---

## Output Formats

### Memo (Text)
```markdown
## Executive Summary
[Overview]

## Investment Highlights & Risks
- Strength 1
- Risk 1

## Key Deal Terms
| Term | Value |
|------|-------|
| Amount | $X |
```

### Evaluation Score
```
0-100 scale:
- 85-100: Excellent
- 70-84: Good
- 60-69: Acceptable
- 50-59: Needs improvement
- Below 50: Poor
```

---

## Supported Models for Memo Generation

**Anthropic (Claude):**
- `claude-sonnet-4-20250514` (RECOMMENDED)
- `claude-3-5-sonnet-20241022`
- `claude-3-sonnet-20240229`
- `claude-3-opus`

**OpenAI:**
- `gpt-5`
- `gpt-4`
- `gpt-4-turbo`

**Google:**
- `gemini-2.5-pro`
- `gemini-2.0-flash-exp`

---

## Environment Setup

```bash
# 1. Copy template and add keys
cp .env.template .env
# Edit .env with your API keys

# 2. Install dependencies
pip install -r requirements.txt

# 3. Test import
python -c "from evals.model_run import call_anthropic_api; print('OK')"
```

---

## Prompt Templates

**Location:** `/Users/Diane/AI-for-Finance/prompts/`

**Default:** `baseline.txt` - Standard 3-section structure

**Advanced:** `prompt_gen_anthropic_context.txt` - With template reference

---

## Evaluation Metrics (Detailed)

1. **Accuracy** (0-1): No hallucinated financial terms
   - Method: LLM consensus voting (3 models)
   
2. **Completeness** (0-1): All key terms captured
   - Method: LLM consensus voting (3 models)
   
3. **Consistency** (0-1): No internal contradictions
   - Method: LLM consensus voting (3 models)
   
4. **Quality** (0-100): Presentation quality
   - Sub-metrics: Clarity, Tone, Length, Structure
   - Method: LLM scoring (0-100 scale)

5. **Summary Score** (0-100): Weighted average of above
   - Default weights: 0.25 each

---

## Common Patterns for Streamlit

### File Upload
```python
uploaded_file = st.file_uploader("Upload", type=['txt', 'md', 'jsonl'])
if uploaded_file:
    content = uploaded_file.read().decode('utf-8')
```

### Progress Display
```python
progress_bar = st.progress(0)
for i in range(steps):
    # Do work
    progress_bar.progress((i+1)/steps)
```

### Error Handling
```python
try:
    memo = generate_memo(doc)
except Exception as e:
    st.error(f"Error: {str(e)}")
```

---

## Performance Expectations

| Operation | Time | Notes |
|-----------|------|-------|
| Generate memo | 2-5 sec | Claude Sonnet 4, ~4000 char output |
| Evaluate memo | 2-4 min | 3 models, 4 metrics |
| Worst-at-K (5 runs) | 15-25 min | Includes generation + evaluation |
| Batch evaluation (100 items) | 30-60 min | Parallel processing |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ANTHROPIC_API_KEY not found` | Check `.env` file, ensure key is set |
| API rate limit | Add delay between calls: `time.sleep(2)` |
| Document too large | Consider chunking or summarizing first |
| Evaluation timeout | Batch API jobs can be resumed later |
| Memory error | Process documents in batches |

---

## Key Data Structures

### Evaluation Result
```python
{
    "accuracy": {
        "score": 0.85,  # 0-1
        "accurate": True,
        "votes": {"gpt-5": "NO", ...}
    },
    "completeness": {...},
    "consistency": {...},
    "quality": {
        "quality_score": 82.5,  # 0-100
        "clarity_score": 85.0,
        "tone_score": 80.0,
        "length_score": 85.0,
        "structure_score": 78.0
    },
    "summary_score": 84.2  # Final 0-100
}
```

---

## Database of Credit Agreements

**Location:** `/Users/Diane/AI-for-Finance/data/`

- **cleaned_data.jsonl**: ~10MB, all documents
- **eval.jsonl**: Locked 15-document eval set
- **train.jsonl**: Remaining documents for training

Each entry: `{"source_url": "https://...", "text": "..."}`

---

## Useful CLI Commands

```bash
# Generate single memo
python evals/model_run.py \
  --model claude-sonnet-4-20250514 \
  --input-file data/eval.jsonl \
  --output memo.md

# Test imports
python -c "from evals import metrics; print(metrics.evaluate_accuracy.__doc__)"

# Run evaluation on test data
python -c "
from evals.evaluator import evaluate_memo
from pathlib import Path
doc = Path('data/eval.jsonl').read_text()
score = evaluate_memo('test memo', doc)
print(f'Score: {score}')
"
```

---

## Next Steps for Streamlit App

1. **UI Layout**
   - Sidebar: Model selection, prompt upload
   - Main: File upload, generate button, display memo
   - Metrics tab: Evaluation results

2. **State Management**
   - Cache generated memos
   - Store evaluation results
   - Allow export to PDF/TXT

3. **Advanced Features**
   - Batch processing
   - Prompt templates selection
   - Metric weight adjustment
   - Comparison across models

4. **Deployment**
   - Use `.streamlit/secrets.toml` for API keys
   - Set resource limits
   - Add monitoring/logging

