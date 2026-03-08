#!/usr/bin/env python3
"""
Investment Memo Generator - Streamlit Interface

A comprehensive web interface for generating investment memos from various document types
(PDF, URL, TXT, JSON/JSONL) using multiple AI models.

Features:
- Multiple input types: File upload (PDF, TXT, JSON, JSONL) or URL
- Custom prompt selection or input
- Multi-model AI integration (Claude Sonnet 4, GPT-5, Gemini 2.5 Pro)
- Few-shot examples for improved quality
- Download generated memos
- Error handling and validation

Requirements:
- Set ANTHROPIC_API_KEY environment variable
"""

import json
import os
import subprocess
import sys
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Optional

import streamlit as st

# Add parent directories to path to import from evals
sys.path.insert(0, str(Path(__file__).parent.parent))
from evals.evaluation.model_run import (
    extract_credit_agreement_from_jsonl,
    read_text_file,
)
from evals.evaluation.metrics import (
    evaluate_accuracy,
    evaluate_completeness,
    evaluate_consistency,
    evaluate_quality,
    calculate_summary_score,
)

# Try to import PDF and URL processing libraries
try:
    import PyPDF2
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


# Constants
CLAUDE_SONNET_4_MODEL = "claude-sonnet-4-20250514"
MAX_OUTPUT_TOKENS = 16000
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def load_available_prompts():
    """Load all available prompts from the prompts directory."""
    if not PROMPTS_DIR.exists():
        return {}

    prompts = {}
    for prompt_file in PROMPTS_DIR.glob("*.txt"):
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                prompts[prompt_file.stem] = f.read()
        except Exception as e:
            st.warning(f"Could not load prompt {prompt_file.name}: {e}")

    return prompts


def extract_text_from_pdf_pypdf2(pdf_bytes: bytes) -> str:
    """Extract text from PDF using PyPDF2."""
    pdf_file = BytesIO(pdf_bytes)
    reader = PyPDF2.PdfReader(pdf_file)

    text_parts = []
    for page_num, page in enumerate(reader.pages):
        try:
            text = page.extract_text()
            if text.strip():
                text_parts.append(f"--- Page {page_num + 1} ---\n{text}")
        except Exception as e:
            st.warning(f"Could not extract text from page {page_num + 1}: {e}")

    return "\n\n".join(text_parts)


def extract_text_from_pdf_pymupdf(pdf_bytes: bytes) -> str:
    """Extract text from PDF using PyMuPDF (fitz)."""
    pdf_file = BytesIO(pdf_bytes)
    doc = fitz.open(stream=pdf_file, filetype="pdf")

    text_parts = []
    for page_num in range(len(doc)):
        try:
            page = doc[page_num]
            text = page.get_text()
            if text.strip():
                text_parts.append(f"--- Page {page_num + 1} ---\n{text}")
        except Exception as e:
            st.warning(f"Could not extract text from page {page_num + 1}: {e}")

    doc.close()
    return "\n\n".join(text_parts)


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from PDF using available library."""
    if HAS_PYMUPDF:
        return extract_text_from_pdf_pymupdf(pdf_bytes)
    elif HAS_PYPDF2:
        return extract_text_from_pdf_pypdf2(pdf_bytes)
    else:
        raise RuntimeError("No PDF processing library available. Install PyPDF2 or PyMuPDF.")


def fetch_text_from_url(url: str) -> str:
    """Fetch text content from URL."""
    if not HAS_REQUESTS:
        raise RuntimeError("requests library not installed. Install it to fetch URLs.")

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to fetch URL: {e}")


def process_uploaded_file(uploaded_file) -> str:
    """Process uploaded file and extract text content."""
    file_name = uploaded_file.name
    file_extension = Path(file_name).suffix.lower()

    if file_extension == '.pdf':
        # PDF processing
        if not (HAS_PYPDF2 or HAS_PYMUPDF):
            raise RuntimeError(
                "PDF processing requires PyPDF2 or PyMuPDF. "
                "Install with: pip install PyPDF2 or pip install PyMuPDF"
            )
        pdf_bytes = uploaded_file.read()
        return extract_text_from_pdf(pdf_bytes)

    elif file_extension == '.jsonl':
        # JSONL processing - use existing function
        # Save to temp file first
        import time
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.jsonl', delete=False) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = Path(tmp.name)

        try:
            content = extract_credit_agreement_from_jsonl(tmp_path)
        finally:
            tmp_path.unlink()  # Clean up temp file

        return content

    elif file_extension in ['.json', '.txt', '.md']:
        # Text-based files
        content = uploaded_file.read().decode('utf-8', errors='ignore')

        # If JSON, try to extract 'text' field like JSONL
        if file_extension == '.json':
            try:
                data = json.loads(content)
                if 'text' in data:
                    return data['text']
            except json.JSONDecodeError:
                pass  # Fall through to return raw content

        return content

    else:
        raise ValueError(f"Unsupported file type: {file_extension}. Supported: .pdf, .txt, .md, .json, .jsonl")


COT_PREFIX = (
    "Before drafting the memo, think step by step: reason through the key financial "
    "terms, structure, and any gaps in the source document. Work through your analysis "
    "before producing the final output.\n\n"
)

OPTIMIZE_PROMPT_NAMES = ["openai_cookbook", "prompt_generator_anthropic", "meta_prompted"]

OPTIMIZE_DESCRIPTION = """
**Optimize** automatically finds the best prompt for your document by running a two-round tournament:

**Round 1 — Base prompts (no few-shot, no CoT):**
1. OAI Cookbook
2. Prompt Generator (Anthropic)
3. Meta Prompted

**Round 2 — Best prompt × 3 techniques:**
4. Best prompt + Few-Shot (FS)
5. Best prompt + Chain-of-Thought + Few-Shot (CoT + FS)
6. Best prompt + Chain-of-Thought only (CoT)

The prompt with the **highest mean eval score** across all 6 runs is selected automatically.
The final memo from the winning configuration is shown below, along with a score breakdown for every variant.
"""


def generate_memo(prompt: str, document_text: str, model: str, api_keys: dict, use_few_shot: bool = True) -> str:
    """Generate investment memo by invoking `evals.model_run` as a subprocess.

    Writes `document_text` to a temporary file and calls the model runner with
    `--model` and `--prompt`. The appropriate API key for the chosen provider
    is injected into the subprocess environment from `api_keys`.
    """

    # Load and prepend few-shot examples if requested
    if use_few_shot:
        few_shot_dir = Path(__file__).parent.parent / "evals" / "few_shot_examples"
        if few_shot_dir.exists():
            input_files = sorted(few_shot_dir.glob("input_*.txt"))
            if input_files:
                few_shot_section = "\n\n# Few-Shot Examples\n\n"
                few_shot_section += "Here are example credit agreements with their corresponding high-quality investment memos for reference:\n\n"

                for i, input_file in enumerate(input_files, 1):
                    # Read input
                    with open(input_file, 'r', encoding='utf-8') as f:
                        input_text = f.read()

                    # Read corresponding output
                    output_file = few_shot_dir / f"example_{input_file.stem.split('_')[1]}.md"
                    if output_file.exists():
                        with open(output_file, 'r', encoding='utf-8') as f:
                            output_text = f.read()

                        few_shot_section += f"## Example {i}\n\n"
                        few_shot_section += f"### Input Credit Agreement:\n```\n{input_text}\n```\n\n"
                        few_shot_section += f"### Expected Output Memo:\n{output_text}\n\n"
                        few_shot_section += "---\n\n"

                prompt = few_shot_section + prompt

    # Save document to a temporary file
    suffix = ".txt"
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False, encoding="utf-8")
    try:
        tmp.write(document_text)
        tmp.flush()
        tmp_path = tmp.name
    finally:
        tmp.close()

    # Determine which env var to set for chosen model
    MODEL_KEY_MAP = {
        "claude-sonnet-4-20250514": ("ANTHROPIC_API_KEY", api_keys.get("anthropic")),
        "gpt-5": ("OPENAI_API_KEY", api_keys.get("openai")),
        "gemini-2.5-pro": ("GEMINI_API_KEY", api_keys.get("gemini")),
    }

    key_entry = MODEL_KEY_MAP.get(model)
    if not key_entry or not key_entry[1]:
        # Fall back: try any provided key for the model provider
        raise ValueError(f"API key for model {model} not provided")

    env = os.environ.copy()
    # Inject the key for the provider used to generate
    env[key_entry[0]] = key_entry[1]

    # Also inject all provided keys so evaluation can run in the same process later
    if api_keys.get("openai"):
        env["OPENAI_API_KEY"] = api_keys.get("openai")
    if api_keys.get("anthropic"):
        env["ANTHROPIC_API_KEY"] = api_keys.get("anthropic")
    if api_keys.get("gemini"):
        env["GEMINI_API_KEY"] = api_keys.get("gemini")

    cmd = [sys.executable, "-m", "evals.evaluation.model_run", "--model", model, "--prompt", prompt, "--input-file", tmp_path]

    with st.spinner(f"Generating memo with {model}..."):
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=420)
        except subprocess.TimeoutExpired:
            raise RuntimeError("Model generation timed out")

    # Clean up temp file
    try:
        os.unlink(tmp_path)
    except Exception:
        pass

    if proc.returncode != 0:
        raise RuntimeError(f"Model run failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")

    # model_run prints the memo to stdout when --output isn't provided
    output_text = proc.stdout.strip()
    if not output_text:
        # If stdout empty, show stderr for debugging
        raise RuntimeError(f"No output from model. STDERR:\n{proc.stderr}")

    return output_text


def _run_single_variant(
    label: str,
    prompt: str,
    document_text: str,
    model: str,
    api_keys: dict,
    use_few_shot: bool,
    status_container,
    eval_models: list,
) -> dict:
    """Generate a memo and evaluate it; return a results dict."""
    status_container.info(f"Running variant: **{label}**…")
    memo = generate_memo(prompt, document_text, model, api_keys, use_few_shot)

    status_container.info(f"Evaluating variant: **{label}**…")
    acc = evaluate_accuracy(memo=memo, source_document=document_text, models=eval_models)
    comp = evaluate_completeness(memo=memo, source_document=document_text, models=eval_models)
    cons = evaluate_consistency(memo=memo, models=eval_models)
    qual = evaluate_quality(memo=memo, models=eval_models)
    summary = calculate_summary_score(
        accuracy_result=acc,
        completeness_result=comp,
        consistency_result=cons,
        quality_result=qual,
    )
    score = summary.get("summary_score", 0.0) or 0.0
    return {
        "label": label,
        "memo": memo,
        "score": score,
        "eval_results": {"accuracy": acc, "completeness": comp, "consistency": cons, "quality": qual, "summary": summary},
    }


def run_optimization(document_text: str, model: str, api_keys: dict, available_prompts: dict, status_container) -> dict:
    """
    Two-round prompt optimization:
      Round 1: openai_cookbook, prompt_generator_anthropic, meta_prompted (no FS, no CoT)
      Round 2: best_prompt + FS, best_prompt + CoT+FS, best_prompt + CoT only
    Returns dict with all variant results and the overall winner.
    """
    # Inject API keys into environment for evaluation sub-calls
    os.environ["OPENAI_API_KEY"] = api_keys.get("openai", "")
    os.environ["ANTHROPIC_API_KEY"] = api_keys.get("anthropic", "")
    os.environ["GEMINI_API_KEY"] = api_keys.get("gemini", "")

    eval_models = ["gpt-5", "claude-sonnet-4-20250514"]

    all_results = []

    # ── Round 1: base prompts ──────────────────────────────────────────────
    round1_results = []
    for name in OPTIMIZE_PROMPT_NAMES:
        prompt_text = available_prompts.get(name)
        if prompt_text is None:
            st.warning(f"Prompt '{name}' not found – skipping.")
            continue
        label_map = {
            "openai_cookbook": "OAI Cookbook",
            "prompt_generator_anthropic": "Prompt Generator (Anthropic)",
            "meta_prompted": "Meta Prompted",
        }
        result = _run_single_variant(
            label=label_map[name],
            prompt=prompt_text,
            document_text=document_text,
            model=model,
            api_keys=api_keys,
            use_few_shot=False,
            status_container=status_container,
            eval_models=eval_models,
        )
        result["round"] = 1
        result["prompt_name"] = name
        round1_results.append(result)
        all_results.append(result)

    if not round1_results:
        raise RuntimeError("No Round 1 prompts were available.")

    # Pick the best base prompt
    best_base = max(round1_results, key=lambda r: r["score"])
    best_prompt_text = available_prompts[best_base["prompt_name"]]

    # ── Round 2: best prompt × 3 technique variations ─────────────────────
    round2_variants = [
        ("Best + FS", best_prompt_text, True, False),
        ("Best + CoT + FS", COT_PREFIX + best_prompt_text, True, True),
        ("Best + CoT", COT_PREFIX + best_prompt_text, False, True),
    ]

    for label, prompt_text, use_fs, _is_cot in round2_variants:
        result = _run_single_variant(
            label=label,
            prompt=prompt_text,
            document_text=document_text,
            model=model,
            api_keys=api_keys,
            use_few_shot=use_fs,
            status_container=status_container,
            eval_models=eval_models,
        )
        result["round"] = 2
        result["prompt_name"] = best_base["prompt_name"]
        all_results.append(result)

    # Overall winner
    winner = max(all_results, key=lambda r: r["score"])

    return {
        "all_results": all_results,
        "winner": winner,
        "best_base": best_base,
    }


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Investment Memo Generator",
        page_icon="📊",
        layout="wide"
    )

    st.title("📊 Investment Memo Generator")
    st.markdown("""
    Generate and evaluate professional investment memos from various document types using **your chosen AI model**.

    **Supported inputs:**
    - 📄 PDF files
    - 📝 Text files (.txt, .md)
    - 📋 JSON/JSONL files (automatically extracts credit agreements)
    - 🌐 URLs (fetches content)
    """)

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # API Keys for all providers (required for evaluation consensus)
        openai_key = os.getenv("OPENAI_API_KEY", "")
        anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
        gemini_key = os.getenv("GEMINI_API_KEY", "")

        if not openai_key:
            openai_key = st.text_input(
                "OpenAI API Key (for GPT-5)",
                type="password",
                help="Enter your OpenAI API key or set OPENAI_API_KEY environment variable"
            )
        else:
            st.success("✓ OpenAI API Key loaded from environment")

        if not anthropic_key:
            anthropic_key = st.text_input(
                "Anthropic API Key (for Claude Sonnet 4)",
                type="password",
                help="Enter your Anthropic API key or set ANTHROPIC_API_KEY environment variable"
            )
        else:
            st.success("✓ Anthropic API Key loaded from environment")

        if not gemini_key:
            gemini_key = st.text_input(
                "Google Gemini API Key (for Gemini 2.5 Pro)",
                type="password",
                help="Enter your Google Cloud API key for Gemini or set GEMINI_API_KEY environment variable"
            )
        else:
            st.success("✓ Gemini API Key loaded from environment")

        # Model selector for generation
        MODEL_OPTIONS = [
            "claude-sonnet-4-20250514",
            "gpt-5",
            "gemini-2.5-pro",
        ]
        selected_model = st.selectbox("Select model to generate with:", MODEL_OPTIONS)

        st.info(f"**Selected model:** {selected_model}\n\n**Max Output:** {MAX_OUTPUT_TOKENS:,} tokens")

    # Main content area
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📥 Input")

        # Input method selection
        input_method = st.radio(
            "Select input method:",
            ["Upload File", "Enter URL"],
            horizontal=True
        )

        document_text = None
        document_name = None

        if input_method == "Upload File":
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=['pdf', 'txt', 'md', 'json', 'jsonl'],
                help="Upload a PDF, text file, or JSON/JSONL file"
            )

            if uploaded_file:
                document_name = uploaded_file.name
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    try:
                        document_text = process_uploaded_file(uploaded_file)
                        st.success(f"✓ File processed: {len(document_text):,} characters")
                    except Exception as e:
                        st.error(f"Error processing file: {e}")

        else:  # Enter URL
            url = st.text_input(
                "Enter URL",
                placeholder="https://example.com/document.html",
                help="Enter a URL to fetch content from"
            )

            if url:
                if not HAS_REQUESTS:
                    st.error("URL fetching requires the 'requests' library. Install with: pip install requests")
                else:
                    if st.button("Fetch URL"):
                        document_name = url
                        with st.spinner(f"Fetching {url}..."):
                            try:
                                document_text = fetch_text_from_url(url)
                                st.success(f"✓ Content fetched: {len(document_text):,} characters")
                            except Exception as e:
                                st.error(f"Error fetching URL: {e}")

        # Show document preview
        if document_text:
            with st.expander("Preview document (first 1000 characters)"):
                st.text(document_text[:1000] + ("..." if len(document_text) > 1000 else ""))

        st.divider()

        # Prompt selection
        st.subheader("📝 Prompt Configuration")

        available_prompts = load_available_prompts()

        if available_prompts:
            prompt_choice = st.selectbox(
                "Select a prompt template",
                ["Custom", "Optimize"] + list(available_prompts.keys()),
                help="Choose a pre-existing prompt, enter your own, or use Optimize to auto-select the best prompt"
            )

            if prompt_choice == "Optimize":
                st.info(OPTIMIZE_DESCRIPTION)
                prompt = None  # not used directly
            elif prompt_choice == "Custom":
                prompt = st.text_area(
                    "Enter your custom prompt",
                    height=200,
                    placeholder="Enter instructions for generating the investment memo...",
                    help="This prompt will guide the AI model in generating the memo"
                )
            else:
                prompt = available_prompts[prompt_choice]
                st.text_area(
                    "Selected prompt (read-only)",
                    value=prompt,
                    height=200,
                    disabled=True
                )
        else:
            st.warning("No prompt templates found in prompts directory")
            prompt_choice = "Custom"
            prompt = st.text_area(
                "Enter your prompt",
                height=200,
                placeholder="Enter instructions for generating the investment memo...",
                help="This prompt will guide the AI model in generating the memo"
            )

        # Few-shot examples option (hidden when Optimize is selected — it handles FS internally)
        if prompt_choice != "Optimize":
            use_few_shot = st.checkbox(
                "Include few-shot examples in prompt",
                value=True,
                help="Add example investment memos to the prompt to improve quality (recommended)"
            )
        else:
            use_few_shot = False  # Optimize manages FS per-variant

    with col2:
        st.header("📄 Output")

        # Generate button label changes for Optimize mode
        btn_label = "🔍 Run Optimization (6 prompts)" if prompt_choice == "Optimize" else "🚀 Generate & Evaluate Investment Memo"
        if st.button(btn_label, type="primary", use_container_width=True):
            if not document_text:
                st.error("⚠️ Please provide a document (upload file or enter URL)")
            elif prompt_choice not in ("Optimize",) and not prompt:
                st.error("⚠️ Please select or enter a prompt")
            elif not (openai_key and anthropic_key and gemini_key):
                st.error("⚠️ Please provide API keys for OpenAI, Anthropic, and Gemini (required for evaluation)")
            elif prompt_choice == "Optimize":
                # ── OPTIMIZE MODE ──────────────────────────────────────────
                api_keys = {"openai": openai_key, "anthropic": anthropic_key, "gemini": gemini_key}
                status_box = st.empty()
                try:
                    opt = run_optimization(document_text, selected_model, api_keys, available_prompts, status_box)
                    status_box.empty()
                    st.session_state['optimize_results'] = opt
                    st.session_state['memo'] = opt['winner']['memo']
                    st.session_state['document_name'] = document_name
                    st.session_state['eval_results'] = opt['winner']['eval_results']
                    st.session_state['is_optimize'] = True
                    st.success(f"✓ Optimization complete! Winning prompt: **{opt['winner']['label']}** ({opt['winner']['score']:.2f}/100)")
                except Exception as e:
                    status_box.empty()
                    st.error(f"Error during optimization: {e}")
            else:
                # ── STANDARD MODE ──────────────────────────────────────────
                api_keys = {"openai": openai_key, "anthropic": anthropic_key, "gemini": gemini_key}
                try:
                    memo = generate_memo(prompt, document_text, selected_model, api_keys, use_few_shot)
                    st.session_state['memo'] = memo
                    st.session_state['document_name'] = document_name
                    st.session_state['is_optimize'] = False
                    st.success("✓ Memo generated successfully!")

                    with st.spinner("Running evaluation (may take a few minutes, respecting rate limits)..."):
                        os.environ['OPENAI_API_KEY'] = openai_key
                        os.environ['ANTHROPIC_API_KEY'] = anthropic_key
                        os.environ['GEMINI_API_KEY'] = gemini_key
                        try:
                            eval_models = ["gpt-5", "claude-sonnet-4-20250514"]
                            acc = evaluate_accuracy(memo=memo, source_document=document_text, models=eval_models)
                            comp = evaluate_completeness(memo=memo, source_document=document_text, models=eval_models)
                            cons = evaluate_consistency(memo=memo, models=eval_models)
                            qual = evaluate_quality(memo=memo, models=eval_models)
                            summary = calculate_summary_score(
                                accuracy_result=acc,
                                completeness_result=comp,
                                consistency_result=cons,
                                quality_result=qual
                            )
                            st.session_state['eval_results'] = {
                                'accuracy': acc, 'completeness': comp,
                                'consistency': cons, 'quality': qual, 'summary': summary,
                            }
                        except Exception as e:
                            st.error(f"Error running evaluation: {e}")
                except Exception as e:
                    st.error(f"Error generating memo: {e}")
                    if "api_key" in str(e).lower() or "unauthorized" in str(e).lower():
                        st.error("Check that your API keys are valid and have sufficient credits")

        # ── OPTIMIZE RESULTS ───────────────────────────────────────────────
        if st.session_state.get('is_optimize') and 'optimize_results' in st.session_state:
            opt = st.session_state['optimize_results']
            winner = opt['winner']
            all_results = opt['all_results']

            st.divider()
            st.header("Optimization Results")

            # Winner banner
            st.success(f"**Selected prompt:** {winner['label']}  |  Score: **{winner['score']:.2f}/100**")

            # Score table for all 6 variants
            st.subheader("All Variant Scores")
            round1 = [r for r in all_results if r['round'] == 1]
            round2 = [r for r in all_results if r['round'] == 2]

            st.markdown("**Round 1 — Base prompts**")
            for r in round1:
                is_best_base = r['label'] == opt['best_base']['label']
                is_winner = r['label'] == winner['label']
                tag = " ← best base" if is_best_base and not is_winner else ""
                tag = " ← WINNER" if is_winner else tag
                st.markdown(f"- **{r['label']}**: {r['score']:.2f}/100{tag}")

            st.markdown("**Round 2 — Best base × techniques**")
            for r in round2:
                is_winner = r['label'] == winner['label']
                tag = " ← WINNER" if is_winner else ""
                st.markdown(f"- **{r['label']}**: {r['score']:.2f}/100{tag}")

        # ── MEMO OUTPUT (shared by both modes) ────────────────────────────
        if 'memo' in st.session_state:
            memo = st.session_state['memo']
            document_name = st.session_state.get('document_name', 'document')

            st.divider()
            st.header("Generated Investment Memo")
            st.markdown(memo)

            st.divider()

            col_download1, col_download2 = st.columns(2)
            with col_download1:
                st.download_button(
                    label="⬇️ Download as Markdown",
                    data=memo,
                    file_name=f"investment_memo_{Path(document_name).stem}.md",
                    mime="text/markdown",
                    use_container_width=True
                )
            with col_download2:
                st.download_button(
                    label="⬇️ Download as Text",
                    data=memo,
                    file_name=f"investment_memo_{Path(document_name).stem}.txt",
                    mime="text/plain",
                    use_container_width=True
                )

            # Show evaluation results if available
            if 'eval_results' in st.session_state:
                evals = st.session_state['eval_results']

                st.divider()
                st.header("Evaluation Results")

                summary = evals.get('summary', {})
                summary_score = summary.get('summary_score')
                if summary_score is not None:
                    st.metric(label="Summary Score", value=f"{summary_score:.2f}/100")

                if summary.get('normalized_scores'):
                    with st.expander("Detailed Subscores"):
                        st.json(summary.get('normalized_scores'))

                with st.expander("Per-metric Details (votes & findings)"):
                    st.subheader("Accuracy")
                    st.write(evals['accuracy'].get('votes'))
                    st.subheader("Completeness")
                    st.write(evals['completeness'].get('votes'))
                    st.subheader("Consistency")
                    st.write(evals['consistency'].get('votes'))
                    st.subheader("Quality")
                    st.write(evals['quality'].get('votes'))

                with st.expander("Download evaluation JSON"):
                    import json as _json
                    st.download_button(
                        "⬇️ Download Eval JSON",
                        data=_json.dumps(evals, indent=2),
                        file_name=f"evals_{Path(document_name).stem}.json",
                        mime="application/json"
                    )

    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: gray; padding: 20px;'>
        <small>
        Powered by Claude Sonnet 4, GPT-5 & Gemini 2.5 Pro | Built with Streamlit<br>
        Investment Memo Generator v1.0
        </small>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
