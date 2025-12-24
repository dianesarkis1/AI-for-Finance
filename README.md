# Prompt Optimization and Evals for Finance

## Overview
This project explores how AI can be applied in finance to automate key workflows. The goals are to:
- Build an MVP pipeline that ingests financial documents (publicly available credit agreements scraped from the SEC EDGAR website), cleans the data, and prompts Claude to create a structured output in the form of an investment memo
- Develop an evaluation harness to evaluate models (GPT-5, Claude Sonnet 4, Gemini 2.5 Pro) on a consistent set of metrics, and test whether prompt optimization techniques (few-shot examples, iterative refinement) can systematically improve performance
- For a demo of the memo generation and evaluation system, see https://ai-for-finance-pbjgnzqdnz7ftrc3uabysv.streamlit.app/
- For a write up of the project and analysis of key results, see [write_up.ipynb](write_up.ipynb).

---

## Repository Structure

```
AI-for-Finance/
├── data/                          # Dataset and preprocessing
│   ├── cleaned_data.jsonl         # All 499 cleaned credit agreements
│   ├── train.jsonl                # 484 documents (training set)
│   ├── train_final.jsonl          # 50 documents (eval set)
│   ├── test.jsonl                 # 449 documents (test set)
│   ├── urls.txt                   # All source URLs
│   └── data_cleaning.py           # Data pipeline script
│
├── evals/                         # Evaluation system
│   ├── run_eval_workflow.py       # Main entry point for full eval pipeline
│   ├── evaluation/                # Core evaluation modules
│   │   ├── model_run.py          # Memo generation
│   │   ├── metrics.py            # 4 evaluation metrics
│   │   ├── utils.py              # API utilities
│   │   ├── evaluator.py          # Single-memo evaluation
│   │   ├── batch_*.py            # Batch evaluation (OpenAI/Claude/Gemini)
│   │   └── run_truly_parallel_batch_eval.py  # Batch orchestrator
│   ├── results_analysis/         # Post-processing
│   │   ├── generate_final_results.py
│   │   ├── create_results_tables_v2.py
│   │   └── generate_memo_review.py
│   ├── few_shot_examples/        # Example memos for prompting
│   ├── batch_outputs/            # Batch API outputs
│   └── results/                  # Evaluation results
│
├── prompts/                       # Prompt templates
│   ├── baseline.txt              # Standard prompt
│   └── *.txt                     # Other prompt variations
│
├── streamlit/                     # Web interface
│   ├── app.py                    # Streamlit demo app
└── └── README.md                 # Streamlit documentation
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

### 3. Run the Streamlit Demo

```bash
streamlit run streamlit/app.py
```

Opens at `http://localhost:8501`

**Features:**
- Upload credit agreements (PDF, TXT, JSON, JSONL, URL)
- Generate memos with Claude Sonnet 4, GPT-5, or Gemini 2.5 Pro
- Real-time evaluation with detailed metrics
- Download memos and evaluation results

See [streamlit/README.md](streamlit/README.md) for full documentation.

---

## Running the Full Evaluation Pipeline

The evaluation pipeline generates memos and evaluates them at scale using batch APIs.

### Basic Usage

```bash
python evals/run_eval_workflow.py <run_name> [options]
```

This runs 3 steps automatically:
1. Generate memos using Claude (by default) Batch API 
2. Evaluate with 3 models (OpenAI, Claude, Gemini) using batch APIs
3. Aggregate results and create analysis tables
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

#### 5. Claude-Optimized Settings

```bash
python evals/run_eval_workflow.py claude_optimized \
  --few-shot-dir evals/few_shot_examples \
  --use-system-parameter \
  --use-xml-tags \
  --parallel-memos
```

---

### Available Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `run_name` | Name for this run (required) | Required Field |
| `--prompt PATH` | Custom prompt file | `prompts/baseline.txt` |
| `--data-file PATH` | Input data file | `data/train.jsonl` |
| `--indices N [N...]` | Specific indices to evaluate | 50-index sample from train_final |
| `--exclude-default` | Evaluate all EXCEPT default 50 | False |
| `--parallel-memos` | Use Claude Batch API for faster memo generation | False |
| `--evaluators MODEL [MODEL...]` | Which evaluators to run (openai, claude, gemini) | All 3 |
| `--skip-memo-generation` | Skip generation, use existing memos | False |
| `--few-shot-dir PATH` | Directory with few-shot examples | None |
| `--use-system-parameter` | Use Claude's native system parameter | False |
| `--use-xml-tags` | Wrap inputs in XML tags | False |
| `--refinement-rounds N` | Number of iterative refinement rounds | 0 |

---

### Performance Expectations

| Operation | Time | Notes |
|-----------|------|-------|
| Generate 50 memos | 5-10 min | With `--parallel-memos` using Claude Batch API |
| Evaluate 50 memos | 30-60 min | Batch APIs across 3 evaluators (12 API calls per memo) |
| Full workflow | 40-70 min | End-to-end for 50 documents |

**Note:** Batch API jobs run asynchronously on provider servers. The script polls until completion.

---

## Data Pipeline

### Regenerate Data Files

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
- `train.jsonl` - 484 documents (minus 15 excluded)
- `train_final.jsonl` - 50 documents (evaluation set)
- `test.jsonl` - 449 documents (remaining)

---

## Evaluation Architecture

### Single-Memo Evaluation
For real-time evaluation (used by Streamlit):
```python
from evals.evaluation.model_run import call_anthropic_api, build_anthropic_payload
from evals.evaluation.metrics import evaluate_accuracy, evaluate_completeness

# Generate memo
payload = build_anthropic_payload(model="claude-sonnet-4-20250514",
                                   content=prompt + document)
response = call_anthropic_api(api_key, payload)
memo = extract_output_text_anthropic(response)

# Evaluate
accuracy = evaluate_accuracy(memo, document)
completeness = evaluate_completeness(memo, document)
```

### Batch Evaluation
For large-scale evaluation (50+ documents):
```bash
python evals/run_eval_workflow.py <run_name> [options]
```

Uses batch APIs for parallel processing:
- **Memo Generation**: Claude Batch API
- **Evaluation**: OpenAI, Claude, and Gemini Batch APIs

---

## Additional Resources

- **Streamlit Demo**: [streamlit/README.md](streamlit/README.md)
- **Data Pipeline**: [data/data_cleaning.py](data/data_cleaning.py) (see docstring)
- **Evaluation Metrics**: [evals/evaluation/metrics.py](evals/evaluation/metrics.py)
- **Live Demo**: https://ai-for-finance-pbjgnzqdnz7ftrc3uabysv.streamlit.app/

---

## Development

### VS Code DevContainer

Open this repo in a VS Code DevContainer (`.devcontainer/`) for a pre-configured environment.

### Running Individual Components

**Generate a single memo:**
```bash
python evals/evaluation/model_run.py \
  --model claude-sonnet-4-20250514 \
  --input-file data/train_final.jsonl \
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

---

## License

This project is for research and educational purposes.
