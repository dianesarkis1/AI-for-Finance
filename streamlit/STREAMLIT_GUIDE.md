# AI-for-Finance Repository Structure & Components
## Comprehensive Guide for Streamlit App Development

---

## 1. REPOSITORY STRUCTURE

### Top-Level Directories
```
/Users/Diane/AI-for-Finance/
├── .env                           # API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY)
├── .env.template                  # Template for configuration
├── requirements.txt               # Python dependencies
├── README.md                       # Project documentation
├── data/                           # Data and preprocessing
├── evals/                          # Evaluation metrics and harness
├── prompts/                        # Prompt templates
└── archive/                        # Legacy/experimental scripts
```

---

## 2. KEY DIRECTORIES & PURPOSE

### 2.1 DATA DIRECTORY (`/Users/Diane/AI-for-Finance/data/`)
- **cleaned_data.jsonl** - Complete dataset of processed credit agreements
- **train.jsonl** - Training split (deterministically selected)
- **eval.jsonl** - Evaluation split (locked 15-document set for reproducibility)
- **urls.txt** - All source URLs (from SEC EDGAR)
- **eval_urls.txt** - Locked list of eval URLs (ensures consistent eval set)
- **data_cleaning.py** - Preprocessing script that:
  - Fetches HTML documents from SEC EDGAR
  - Converts to plain text
  - Stores as JSONL with structure: `{"source_url": "...", "text": "..."}`
  - Uses deterministic hash-based splitting for reproducibility

### 2.2 PROMPTS DIRECTORY (`/Users/Diane/AI-for-Finance/prompts/`)
```
prompts/
├── baseline.txt                          # Standard prompt (3-section memo)
├── prompt_gen_anthropic_context.txt     # Enhanced prompt with context
├── prompt_gen_anthrop_cont_CoT.txt      # Chain-of-Thought variant
├── openai_cookbook.txt                  # OpenAI-specific prompt
└── prompt_generator_anthropic.txt       # Generation optimization prompt
```

**Key Prompt Features:**
- Structure: Executive Summary → Investment Highlights/Risks → Key Deal Terms Table
- Instruction to use ONLY facts from source (no hallucinations)
- Template reference for tone/structure consistency
- Designed for finance professionals (investment committee audience)

### 2.3 EVALS DIRECTORY (`/Users/Diane/AI-for-Finance/evals/`)
```
evals/
├── model_run.py                  # Core memo generation (multi-model)
├── utils.py                       # LLM API utilities
├── metrics.py                     # All evaluation metrics
├── evaluator.py                  # Main evaluation harness
├── batch_evals/                  # Batch API evaluation workflows
│   ├── batch_utils.py            # Batch API orchestration
│   ├── batch_metrics.py          # Batch-specific metric implementations
│   ├── batch_evaluate.py         # Batch evaluation orchestrator
│   └── run_eval_workflow.py      # Complete batch workflow
├── helper_tests/                 # Test scripts for each metric
├── benchmark/                    # Baseline benchmark runs
└── exploratory/                  # Initial model testing
```

---

## 3. CORE FUNCTIONS FOR STREAMLIT APP

### 3.1 DOCUMENT PROCESSING

**File: `/Users/Diane/AI-for-Finance/evals/model_run.py`**

#### Input Processing Functions
```python
def read_text_file(path: Path) -> str
    # Reads .txt, .md files

def extract_credit_agreement_from_jsonl(jsonl_path: Path) -> str
    # Extracts 'text' field from JSONL

def clean_and_format_credit_agreement(text: str) -> str
    # Cleans HTML artifacts, removes duplicates
    # Normalizes whitespace
    # Removes SEC formatting markers
```

**Key Features:**
- Handles multiple file formats (.txt, .md, .jsonl)
- Auto-detects and extracts credit agreements
- Cleans SEC document formatting artifacts

---

### 3.2 API CALLS TO CLAUDE/ANTHROPIC

**File: `/Users/Diane/AI-for-Finance/evals/model_run.py`**

#### Anthropic (Claude) API Functions
```python
def build_anthropic_payload(model: str, content: str, max_output_tokens: int) -> Dict
    # Builds message payload for Anthropic

def call_anthropic_api(api_key: str, payload: Dict) -> Dict
    # Makes HTTPS curl call to https://api.anthropic.com/v1/messages
    # Header: x-api-key, anthropic-version: 2023-06-01

def extract_output_text_anthropic(response: Dict) -> Optional[str]
    # Extracts text from response.content[0].text
```

#### Supported Models
```python
MODEL_CONFIGS = {
    "claude-sonnet-4-20250514": {"provider": "anthropic", "api_key_env": "ANTHROPIC_API_KEY"},
    "claude-3-5-sonnet-20241022": {"provider": "anthropic", "api_key_env": "ANTHROPIC_API_KEY"},
    "claude-3-sonnet-20240229": {"provider": "anthropic", "api_key_env": "ANTHROPIC_API_KEY"},
    "claude-3-opus": {"provider": "anthropic", "api_key_env": "ANTHROPIC_API_KEY"},
    # ... and others
}
```

**Usage Pattern:**
```python
# 1. Build payload
payload = build_anthropic_payload(
    model="claude-sonnet-4-20250514",
    content=system_preamble + "\n\n" + prompt_text + "\n\n" + document_text,
    max_output_tokens=16000
)

# 2. Call API
response = call_anthropic_api(api_key=api_key, payload=payload)

# 3. Extract memo
memo = extract_output_text_anthropic(response)
```

---

### 3.3 MEMO GENERATION

**File: `/Users/Diane/AI-for-Finance/evals/model_run.py`**

#### Main Function (CLI Version)
```python
def main()
    # Entry point for command-line usage
    # Arguments:
    #   --model: gpt-5, claude-sonnet-4-20250514, gemini-2.5-pro, etc.
    #   --input-file: Path to credit agreement (.txt, .md, or .jsonl)
    #   --prompt-file: Path to custom prompt (optional)
    #   --output: Output path for memo (optional)
    #   --max-output-tokens: Token limit (default 16000)
```

**System Preamble (Default):**
```
"You are an investment analyst. Using the provided credit agreement and any template references, 
produce a concise, structured investment memo. Keep the total output under 400 words or 15k tokens."
```

**Key Processing Flow:**
1. Read prompt (file or default baseline)
2. Read input document(s) - auto-extract from JSONL if needed
3. Combine: system preamble + prompt + documents
4. Call appropriate LLM API
5. Extract and optionally save memo

---

### 3.4 EVALUATION METRICS

**File: `/Users/Diane/AI-for-Finance/evals/metrics.py`**

#### Accuracy Evaluation
```python
def evaluate_accuracy(
    memo: str,
    source_document: str,
    models: List[str] = ["gpt-5", "claude-sonnet-4-20250514", "gemini-2.5-pro"],
    consensus_threshold: float = 0.6
) -> Dict
    # LLM consensus voting: Are there hallucinated financial terms?
    # Returns: {accurate: bool, score: 0-1, votes: Dict, consensus_reached: bool}
```

#### Completeness Evaluation
```python
def evaluate_completeness(
    memo: str,
    source_document: str,
    models: List[str] = None,
    consensus_threshold: float = 0.6
) -> Dict
    # LLM consensus: Are any key financial terms missing?
    # Returns: {complete: bool, score: 0-1, votes: Dict, consensus_reached: bool}
```

#### Consistency Evaluation
```python
def evaluate_consistency(
    memo: str,
    models: List[str] = None,
    consensus_threshold: float = 0.6
) -> Dict
    # LLM consensus: Does memo contradict itself?
    # Returns: {consistent: bool, score: 0-1, votes: Dict (JSON parsed)}
```

#### Quality Evaluation
```python
def evaluate_quality(
    memo: str,
    template: str = None,
    models: List[str] = None,
    consensus_threshold: float = 0.6
) -> Dict
    # Scores 4 dimensions (each 0-100):
    #   - Clarity: Clear explanations, logical flow
    #   - Tone: Professional, appropriate for investment committee
    #   - Length: Conciseness, no unnecessary verbosity
    #   - Structure: Consistency with template
    # Returns: {quality_score: avg, clarity_score, tone_score, length_score, structure_score}
```

#### Summary Score Calculation
```python
def calculate_summary_score(
    accuracy_result: Dict,
    completeness_result: Dict,
    consistency_result: Dict,
    quality_result: Dict,
    weights: Dict[str, float] = None  # default: {0.25, 0.25, 0.25, 0.25}
) -> Dict
    # Aggregates all metrics into single 0-100 score
    # Returns: {summary_score: float, normalized_scores: Dict, weights_used: Dict}
```

---

### 3.5 MAIN EVALUATION HARNESS

**File: `/Users/Diane/AI-for-Finance/evals/evaluator.py`**

#### Single Memo Evaluation
```python
def evaluate_memo(
    memo: str,
    source_document: str,
    template: str = None,
    eval_models: List[str] = None,
    weights: Dict[str, float] = None
) -> float
    # Runs all 4 metrics, returns single summary score (0-100)
```

#### Worst-at-K Testing
```python
def worst_at_k(
    model: str,
    input_file: str,
    source_document: str,
    k: int = 5,
    template: str = None,
    eval_models: List[str] = None,
    weights: Dict[str, float] = None,
    delay_between_runs: float = 35.0,
    fail_fast: bool = False
) -> Dict
    # Generates k memos from same input
    # Returns: {worst_score, best_score, mean_score, std_dev, all_scores}
    # Tests consistency across multiple generations
```

---

### 3.6 BATCH API UTILITIES

**File: `/Users/Diane/AI-for-Finance/evals/batch_evals/batch_utils.py`**

#### Batch Workflow Functions
```python
def upload_batch_file(requests: List[Dict], temp_dir: Path, api_key: str) -> str
    # Upload JSONL batch file, returns file_id

def create_batch_job(file_id: str, api_key: str, description: str) -> str
    # Create batch job on OpenAI, returns batch_id

def check_batch_status(batch_id: str, api_key: str) -> Dict
    # Check status of batch job

def download_batch_results(output_file_id: str, temp_dir: Path, api_key: str) -> Path
    # Download completed batch results

def poll_batch_until_complete(
    batch_id: str,
    api_key: str,
    temp_dir: Path,
    poll_interval: int = 60,
    max_wait_time: int = 86400
) -> List[Dict]
    # Poll until completion, return results

def submit_and_wait_for_batch(
    requests: List[Dict],
    api_key: str,
    temp_dir: Path,
    description: str = None,
    poll_interval: int = 60
) -> List[Dict]
    # Complete workflow: upload → create → poll → download
```

#### Claude Batch Functions (Anthropic)
```python
def create_claude_batch(requests: List[Dict], api_key: str) -> str
def check_claude_batch_status(batch_id: str, api_key: str) -> Dict
def poll_claude_batch_until_complete(batch_id: str, api_key: str, ...) -> List[Dict]
def submit_and_wait_for_claude_batch(requests: List[Dict], api_key: str, ...) -> List[Dict]
# Corresponds to: https://api.anthropic.com/v1/messages/batches
```

#### Gemini Batch Functions (Google)
```python
def create_gemini_batch(requests: List[Dict], api_key: str, model: str) -> str
def check_gemini_batch_status(batch_name: str, api_key: str) -> Dict
def poll_gemini_batch_until_complete(batch_name: str, api_key: str, ...) -> List[Dict]
def submit_and_wait_for_gemini_batch(requests: List[Dict], api_key: str, ...) -> List[Dict]
# Corresponds to: https://generativelanguage.googleapis.com/v1beta/models/{model}:batchGenerateContent
```

---

### 3.7 UNIFIED LLM EVALUATION INTERFACE

**File: `/Users/Diane/AI-for-Finance/evals/utils.py`**

#### Universal LLM Calling Function
```python
def call_llm_for_eval(model: str, prompt: str) -> str
    # Unified interface for evaluation LLM calls
    # Supports: OpenAI (gpt-5, gpt-4o), Anthropic (claude-*), Google (gemini-*)
    # Automatically routes to correct API based on model identifier
    
    # Example:
    response = call_llm_for_eval("claude-sonnet-4-20250514", evaluation_prompt)
    response = call_llm_for_eval("gpt-5", evaluation_prompt)
    response = call_llm_for_eval("gemini-2.5-pro", evaluation_prompt)
```

#### API Provider Detection
```python
EVAL_MODEL_CONFIGS = {
    # OpenAI
    "gpt-5": {"provider": "openai", "api_key_env": "OPENAI_API_KEY", "base_url": "https://api.openai.com/v1"},
    "gpt-4o": {"provider": "openai", ...},
    
    # Anthropic
    "claude-sonnet-4-20250514": {"provider": "anthropic", "api_key_env": "ANTHROPIC_API_KEY"},
    "claude-3-5-sonnet": {"provider": "anthropic", ...},
    
    # Google
    "gemini-2.5-pro": {"provider": "gemini", "api_key_env": "GEMINI_API_KEY"},
    "gemini-2.0-flash-exp": {"provider": "gemini", ...},
}
```

---

## 4. CONFIGURATION & ENVIRONMENT

### 4.1 Environment Variables (.env)
```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...
```

### 4.2 Python Dependencies (requirements.txt)
```
# Core data processing
pandas>=1.5.0
numpy>=1.21.0

# Document processing
PyPDF2>=3.0.0
python-docx>=0.8.11
PyMuPDF>=1.23.0

# LLM APIs
openai>=1.0.0
anthropic>=0.7.0

# Text processing
nltk>=3.8
spacy>=3.5.0

# Configuration
python-dotenv>=1.0.0
PyYAML>=6.0

# Utilities
requests>=2.28.0
pathlib2>=2.3.0
```

---

## 5. IMPORTANT FUNCTIONS FOR STREAMLIT APP

### 5.1 Complete Workflow: Input → Memo → Evaluation

```python
# STEP 1: Load and process input document
from evals.model_run import read_text_file, extract_credit_agreement_from_jsonl
from pathlib import Path

input_path = Path("credit_agreement.txt")
if input_path.suffix == ".jsonl":
    document_text = extract_credit_agreement_from_jsonl(input_path)
else:
    document_text = read_text_file(input_path)

# STEP 2: Generate memo using Claude Sonnet 4
from evals.model_run import call_anthropic_api, build_anthropic_payload, extract_output_text_anthropic
import os

api_key = os.getenv("ANTHROPIC_API_KEY")
prompt_text = "Write an investment memo for this credit agreement..."

payload = build_anthropic_payload(
    model="claude-sonnet-4-20250514",
    content=f"System: You are an investment analyst.\n\nUser: {prompt_text}\n\nDocument:\n{document_text}",
    max_output_tokens=16000
)

response = call_anthropic_api(api_key, payload)
memo = extract_output_text_anthropic(response)

# STEP 3: Evaluate memo
from evals.evaluator import evaluate_memo

summary_score = evaluate_memo(
    memo=memo,
    source_document=document_text,
    template=None,
    eval_models=["gpt-5", "claude-sonnet-4-20250514", "gemini-2.5-pro"]
)
print(f"Overall Score: {summary_score:.2f}/100")

# STEP 4: Get detailed metrics
from evals.metrics import (
    evaluate_accuracy, evaluate_completeness, 
    evaluate_consistency, evaluate_quality, 
    calculate_summary_score
)

accuracy_result = evaluate_accuracy(memo, document_text)
completeness_result = evaluate_completeness(memo, document_text)
consistency_result = evaluate_consistency(memo)
quality_result = evaluate_quality(memo)

print(f"Accuracy: {accuracy_result['score']*100:.1f}/100")
print(f"Completeness: {completeness_result['score']*100:.1f}/100")
print(f"Consistency: {consistency_result['score']*100:.1f}/100")
print(f"Quality: {quality_result['quality_score']:.1f}/100")
```

---

## 6. PROMPT TEMPLATES

### 6.1 Baseline Prompt (`baseline.txt`)
```
- 3-section structure: Executive Summary, Investment Highlights & Risks, Key Deal Terms
- Emphasizes factual information only
- Specifies template alignment
- Target length: ~400 words
```

### 6.2 Context-Enhanced Prompt (`prompt_gen_anthropic_context.txt`)
```
- Same 3-section structure
- Includes template memo as reference
- More explicit instructions on financial terms
- Emphasis on investment committee perspective
```

### 6.3 Using Custom Prompts in Streamlit
```python
# Option 1: Use default baseline
# → automatically loaded from prompts/baseline.txt

# Option 2: Use custom prompt file
prompt_text = read_text_file(Path("prompts/my_custom_prompt.txt"))

# Option 3: Create prompt programmatically
prompt_text = """
You are a Private Credit Analyst...
[custom instructions]
"""
```

---

## 7. DATA FORMATS

### 7.1 Input Document Formats
```
# TXT Format
Plain text credit agreement

# MD Format
Markdown-formatted credit agreement

# JSONL Format (one record per line)
{
  "source_url": "https://www.sec.gov/...",
  "text": "EXECUTION VERSION SECOND AMENDMENT TO THIRD AMENDED..."
}
```

### 7.2 Memo Output Format
Generated as markdown-formatted text, typically includes:
- Executive Summary (paragraphs)
- Investment Highlights (bullet points)
- Risks (bullet points)
- Key Deal Terms (table)

### 7.3 Evaluation Result Format
```python
{
    "summary_score": 82.5,  # 0-100
    "normalized_scores": {
        "accuracy": 85.0,
        "completeness": 78.0,
        "consistency": 89.0,
        "quality": 79.0
    },
    "weights_used": {
        "accuracy": 0.25,
        "completeness": 0.25,
        "consistency": 0.25,
        "quality": 0.25
    },
    "missing_metrics": []
}
```

---

## 8. ERROR HANDLING & EDGE CASES

### Common Issues
1. **API Key Missing**: Check `.env` file has `ANTHROPIC_API_KEY`
2. **Rate Limiting**: Use `delay_between_runs` parameter in evaluation
3. **Token Limits**: Default 16000 tokens; adjust `max_output_tokens` if needed
4. **Timeouts**: Batch API jobs can take hours; use `resume_batch_job()` if interrupted
5. **Large Documents**: JSONL files may be very large; consider streaming/chunking

### Recommended Retry Logic
```python
import time

def retry_api_call(func, max_attempts=3, delay=5):
    for attempt in range(max_attempts):
        try:
            return func()
        except Exception as e:
            if attempt == max_attempts - 1:
                raise
            time.sleep(delay)
```

---

## 9. PERFORMANCE NOTES

### API Call Latency
- **Claude Sonnet 4**: ~2-5 seconds for typical memo (4000-char output)
- **Evaluation (3 models consensus)**: ~30-60 seconds per metric × 4 metrics = ~2-4 minutes total
- **Batch API**: 100+ requests in parallel, typically 30-60 minutes for completion

### Recommended Streamlit Optimization
```python
import streamlit as st

@st.cache_resource
def get_api_client():
    return anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

@st.cache_data
def generate_memo(document, prompt, model):
    # Expensive operation - cache results
    return api_client.generate_memo(...)
```

---

## 10. ABSOLUTE FILE PATHS FOR REFERENCE

```
/Users/Diane/AI-for-Finance/evals/model_run.py                 # Main memo generation
/Users/Diane/AI-for-Finance/evals/metrics.py                   # Evaluation metrics
/Users/Diane/AI-for-Finance/evals/evaluator.py                 # Evaluation harness
/Users/Diane/AI-for-Finance/evals/utils.py                     # LLM utilities
/Users/Diane/AI-for-Finance/evals/batch_evals/batch_utils.py   # Batch API utilities
/Users/Diane/AI-for-Finance/data/data_cleaning.py              # Document preprocessing
/Users/Diane/AI-for-Finance/prompts/baseline.txt               # Default prompt
/Users/Diane/AI-for-Finance/prompts/prompt_gen_anthropic_context.txt  # Context prompt
```

---

## 11. QUICK START FOR STREAMLIT APP

```python
# Import required modules
from evals.model_run import (
    call_anthropic_api, build_anthropic_payload, 
    extract_output_text_anthropic, extract_credit_agreement_from_jsonl
)
from evals.evaluator import evaluate_memo
from evals.metrics import (
    evaluate_accuracy, evaluate_completeness, 
    evaluate_consistency, evaluate_quality
)
import streamlit as st
import os
from pathlib import Path

# Initialize Streamlit app
st.title("Investment Memo Generator & Evaluator")

# File upload
uploaded_file = st.file_uploader("Upload credit agreement", type=['txt', 'md', 'jsonl'])

if uploaded_file:
    # Read document
    if uploaded_file.name.endswith('.jsonl'):
        # Save temp file and extract
        with open("/tmp/temp.jsonl", "wb") as f:
            f.write(uploaded_file.getbuffer())
        document_text = extract_credit_agreement_from_jsonl(Path("/tmp/temp.jsonl"))
    else:
        document_text = uploaded_file.read().decode('utf-8')
    
    # Generate memo
    api_key = os.getenv("ANTHROPIC_API_KEY")
    payload = build_anthropic_payload(
        model="claude-sonnet-4-20250514",
        content=f"[system + prompt + document]",
        max_output_tokens=16000
    )
    response = call_anthropic_api(api_key, payload)
    memo = extract_output_text_anthropic(response)
    
    st.write("### Generated Memo")
    st.write(memo)
    
    # Evaluate
    if st.button("Evaluate Memo"):
        score = evaluate_memo(memo, document_text)
        st.metric("Summary Score", f"{score:.2f}/100")
```

