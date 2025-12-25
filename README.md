# Prompt Optimization and Evals for Finance

## Overview
This project explores how AI can be applied in finance to automate key workflows. The goals are to:
- Build a pipeline that ingests financial documents (publicly available credit agreements scraped from the SEC EDGAR website), cleans the data, and prompts Claude (or another model of choice) to create a structured output in the form of an investment memo
- Develop an evaluation harness to evaluate models (GPT-5, Claude Sonnet 4, Gemini 2.5 Pro) on a consistent set of metrics, and test whether prompt optimization techniques (few-shot examples, iterative refinement) can systematically improve performance
- Live demo: https://ai-for-finance-pbjgnzqdnz7ftrc3uabysv.streamlit.app/
- Project write-up and results: see [write_up.ipynb](write_up.ipynb).

---

## Repository Structure

```
AI-for-Finance/
├── data/                    # Dataset, splits, cleaning pipeline
├── evals/                   # Memo generation + evaluation system
│   ├── run_eval_workflow.py # Main entry point for full pipeline
│   ├── batch_outputs/       # Batch API outputs
│   ├── evaluation/          # Core evaluation modules
│   ├── results/             # Evaluation results
│   ├── results_analysis/    # Post-processing
│   └── few_shot_examples/   # Example memos for prompting
├── prompts/                 # Prompt templates
├── streamlit/               # For demo
└── write_up.ipynb           # Analysis and results

```
---

## Core Evaluation Metrics

The evaluation system (`evals/evaluation/metrics.py`) implements 4 metrics:

1. **Accuracy** (0-100): Detects hallucinations using LLM consensus (3 models vote)
2. **Completeness** (0-100): Detects missing terms using LLM consensus
3. **Consistency** (0-100): Detects internal contradictions using LLM consensus
4. **Quality** (0-100): Scores presentation across 4 dimensions:
   - Clarity: Clear explanations, logical flow
   - Tone: Professional, appropriate for investment committee
   - Conciseness: No unnecessary verbosity
   - Structure: Consistency with template

**Summary Score**: Weighted average of all 4 metrics (default: 0.25 each)

---

## Quick Start

### 1. Install Dependencies

```bash
# (optional) create a virtual environment
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate

# install required packages
pip install -r requirements.txt
```

**Python version:** 3.11+ recommended

---

### 2. Configure API Keys

Create a `.env` file in the project root:

```bash
cp .env.template .env
```

Edit `.env` and add your API keys:

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...
```

---

### 3. Run the Full Generation + Evaluation Pipeline

```bash
python evals/run_eval_workflow.py <run_name> [options]
```

**What this does:**
1. Generate memos using Claude (by default) Batch API 
2. Evaluate with 3 models (OpenAI, Claude, Gemini) using batch APIs
3. Aggregate results and create analysis tables

**Outputs:**
- `evals/batch_outputs/batch_temp_[run_name]/` - Batch API data
- `evals/results/results_[run_name]/` - Final results and analysis tables

### Available Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `run_name` | Name for this run (required) | Required Field |
| `--prompt PATH` | Custom prompt file | `prompts/baseline.txt` |
| `--data-file PATH` | Input data file | `data/train_final.jsonl` |
| `--indices N [N...]` | Specific indices to evaluate | All 50 documents in train_final.jsonl |
| `--parallel-memos` | Use Claude Batch API for faster memo generation | False |
| `--evaluators MODEL [MODEL...]` | Which evaluators to run (openai, claude, gemini) | All 3 |
| `--skip-memo-generation` | Skip generation, use existing memos in batch_outputs/[run_name] | False |
| `--few-shot-dir PATH` | Directory with few-shot examples | None |
| `--use-system-parameter` | Use Claude's native system parameter | False |
| `--use-xml-tags` | Wrap inputs in XML tags | False |
| `--refinement-rounds N` | Number of iterative refinement rounds | 0 |

**Note:** Batch API jobs run asynchronously on provider servers. The script polls until completion.

---

**Alternative: Try the Streamlit Demo**

For a quick interactive demo without waiting for batch APIs: https://ai-for-finance-pbjgnzqdnz7ftrc3uabysv.streamlit.app/

---

### Common Use Cases

#### 1. Test a Custom Prompt

```bash
python evals/run_eval_workflow.py my_experiment \
  --prompt prompts/my_custom_prompt.txt \
  --parallel-memos
```

#### 2. Evaluate Specific Documents

```bash
python evals/run_eval_workflow.py test_subset \
  --indices 0 1 2 6 12 16 \
  --parallel-memos
```

#### 3. Use Few-Shot Examples

```bash
python evals/run_eval_workflow.py few_shot_test \
  --few-shot-dir evals/few_shot_examples \
  --parallel-memos
```

#### 4. Evaluate on Test Set

```bash
python evals/run_eval_workflow.py test_benchmark \
  --data-file data/test.jsonl \
  --indices 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 \
  --parallel-memos
```
#### 5. Running Individual Components

**Generate a single memo:**
```bash
python evals/evaluation/model_run.py \
  --model claude-sonnet-4-20250514 \
  --input-file data/train_final.jsonl \
  --index 0 \
  --output memo.md
```

**Evaluate a memo:**
```python
from evals.evaluation.evaluator import evaluate_memo

score = evaluate_memo(
    memo=memo_text,
    source_document=document_text,
    eval_models=["gpt-5", "claude-sonnet-4-20250514", "gemini-2.5-pro"]
)
print(f"Summary Score: {score:.2f}/100")
```

#### 6. Regenerate Data Files

All data splits are deterministic and reproducible:

```bash
# Install additional dependencies
pip install requests beautifulsoup4 lxml chardet

# Run pipeline (creates 4 JSONL files in data/)
python data/data_cleaning.py

# Or test without overwriting (outputs to data_test/)
python data/data_cleaning.py data_test
```

**Process:**
1. Fetches 499 credit agreements from SEC EDGAR (~2 minutes)
2. Cleans HTML and removes artifacts
3. Creates deterministic splits using hardcoded indices

**Output:**
- `cleaned_data.jsonl` - All 499 documents
- `train_final.jsonl` - 50 documents (training set)
- `test.jsonl` - 449 documents (test set)

---

## Development

### VS Code DevContainer

Open this repo in a VS Code DevContainer (`.devcontainer/`) for a pre-configured environment.