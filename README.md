# AI For Finance Project
## Overview 
This project explores how AI can be applied in finance to automate key workflows. The goals are twofold:
- Build an MVP pipeline that ingests financial documents (publicly available credit agreements scraped from the SEC EDGAR website), cleans the data, and prompts claude to create a structured output in the form of an investment memo
- Develop an evaluation harness to evaluate models (gpt-5, claude sonnet 4, gemini 2.5 pro) on a consistent set of metrics, and test whether prompt optimization techniques (few shot examples, iterative refinement...) can systematically improve performance

### Folder Organization
**Data folder:**
- `urls.txt` - All 499 URL links to the full dataset (SEC credit agreements)
- `eval_urls.txt` - URL links to the eval set (pre-set for consistency across runs)
- `cleaned_data.jsonl` - All 499 cleaned credit agreements after preprocessing
- `train.jsonl` - 484 documents (cleaned_data minus 15 excluded URLs)
- `train_final.jsonl` - 50 documents (evaluation set, specific indices from cleaned_data)
- `test.jsonl` - 449 documents (remaining documents for testing)

All data splits are deterministic using hardcoded indices and URL lists (no random sampling).
See [data/data_cleaning.py](data/data_cleaning.py) for detailed pipeline documentation.

If you use VS Code, you can open this repo in a devcontainer (.devcontainer/) to get a pre-configured environment.

Project Scripts folder:
- "main_exploratory" functions run "model_run" functions to generate investment memos using several models (used for initial selection of what the baseline [benchmark] model will be).
- "model_run" just runs one iteration of a chosen model to get an output memo from a given input.

Evals folder:
- **metrics.py**: Core evaluation functions implementing all 5 metrics:
  - `evaluate_accuracy()`: Detects hallucinated financial terms using LLM consensus (3 models vote YES/NO)
  - `evaluate_completeness()`: Detects missing key terms using LLM consensus
  - `evaluate_consistency()`: Detects intra-memo contradictions using LLM consensus with JSON output
  - `evaluate_quality()`: Scores presentation quality across 4 dimensions (clarity, tone, length, structure) on 0-100 scale
  - `calculate_summary_score()`: Aggregates all 4 metrics into single summary score (0-100) with configurable weights
- **evaluator.py**: Main evaluation harness with two key functions:
  - `evaluate_memo()`: Runs all 4 metrics on a single memo and returns summary score
  - `worst_at_k()`: Generates k memos from same input (via model_run.py) and returns worst-case score to test consistency across runs. Parameters include:
    - `delay_between_runs` (default: 35s) to respect API rate limits
    - `fail_fast` (default: False) to stop on first error instead of continuing all k runs
    - Returns: worst_score, best_score, mean_score, std_dev, score_range, and all_scores
- **utils.py**: Helper functions for LLM-as-judge evaluation:
  - `call_llm_for_eval()`: Unified interface to call OpenAI, Anthropic, or Google models for evaluation
  - Handles API differences, retries, and response parsing
- **helper_tests/**: Test scripts for each metric that run on exploratory outputs as sanity checks

## 🚀 Setup

### 1. Install dependencies

This project uses the versions pinned in `requirements.txt`.
Install them with:

```bash
# (optional) create a virtual environment
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate

# install required Python packages
pip install -r requirements.txt
```

> **Python version:** Use a version compatible with `requirements.txt` (e.g., Python 3.11).
> No other OS-specific assumptions—macOS, Linux, and Windows should work similarly.

---

### 2. Configure environment variables

Copy the template and fill in your API keys + model settings:

```bash
cp .env.template .env
```

Edit `.env` and set the required values (API keys + preferred model/provider).
Any defaults not explicitly provided here should be assumed to match the code’s internal configuration.

---

### 3. (Optional) Rebuild the dataset

The data pipeline is fully reproducible. To regenerate all data files:

```bash
# Install additional dependencies for data cleaning
pip install requests beautifulsoup4 lxml chardet

# Run the pipeline (creates 4 JSONL files in data/)
python data/data_cleaning.py

# Or test without overwriting existing files (outputs to data_test/)
python data/data_cleaning.py data_test
```

**What this creates:**
- `cleaned_data.jsonl` - All 499 cleaned credit agreements
- `train.jsonl` - 484 documents (subset of cleaned_data)
- `train_final.jsonl` - 50 documents (evaluation set)
- `test.jsonl` - 449 documents (remaining documents)

**Note:** The pipeline fetches from SEC servers (499 documents × 0.2s delay = ~2 minutes).
All splits are deterministic using hardcoded indices—see [data/data_cleaning.py](data/data_cleaning.py) for complete documentation.

---

### 4. Run the Streamlit app

```bash
streamlit run streamlit/app.py
```
