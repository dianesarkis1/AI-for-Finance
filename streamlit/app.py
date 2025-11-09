#!/usr/bin/env python3
"""
Investment Memo Generator - Streamlit Interface

A comprehensive web interface for generating investment memos from various document types
(PDF, URL, TXT, JSON/JSONL) using Claude Sonnet 4.

Features:
- Multiple input types: File upload (PDF, TXT, JSON, JSONL) or URL
- Custom prompt selection or input
- Claude Sonnet 4 integration
- Download generated memos
- Error handling and validation

Requirements:
- Set ANTHROPIC_API_KEY environment variable
"""

import json
import os
import sys
from io import BytesIO
from pathlib import Path
from typing import Optional

import streamlit as st

# Add parent directories to path to import from evals
sys.path.insert(0, str(Path(__file__).parent.parent))
from evals.model_run import (
    build_anthropic_payload,
    call_anthropic_api,
    extract_credit_agreement_from_jsonl,
    extract_output_text_anthropic,
    read_text_file,
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
        import tempfile
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


def generate_memo(prompt: str, document_text: str, api_key: str) -> str:
    """Generate investment memo using Claude Sonnet 4."""
    # Combine prompt and document
    combined_content = f"{prompt}\n\n--- DOCUMENT ---\n{document_text}"

    # Build payload
    payload = build_anthropic_payload(
        model=CLAUDE_SONNET_4_MODEL,
        content=combined_content,
        max_output_tokens=MAX_OUTPUT_TOKENS
    )

    # Call API
    with st.spinner("Generating memo with Claude Sonnet 4..."):
        response = call_anthropic_api(api_key, payload)

    # Extract output
    output = extract_output_text_anthropic(response)

    if output is None:
        raise RuntimeError(f"Failed to extract output from API response: {response}")

    return output


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Investment Memo Generator",
        page_icon="📊",
        layout="wide"
    )

    st.title("📊 Investment Memo Generator")
    st.markdown("""
    Generate professional investment memos from various document types using **Claude Sonnet 4**.

    **Supported inputs:**
    - 📄 PDF files
    - 📝 Text files (.txt, .md)
    - 📋 JSON/JSONL files (automatically extracts credit agreements)
    - 🌐 URLs (fetches content)
    """)

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # API Key
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            api_key = st.text_input(
                "Anthropic API Key",
                type="password",
                help="Enter your Anthropic API key or set ANTHROPIC_API_KEY environment variable"
            )
        else:
            st.success("✓ API Key loaded from environment")

        # Model info
        st.info(f"**Model:** {CLAUDE_SONNET_4_MODEL}\n\n**Max Output:** {MAX_OUTPUT_TOKENS:,} tokens")

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
                ["Custom"] + list(available_prompts.keys()),
                help="Choose a pre-existing prompt or enter your own"
            )

            if prompt_choice == "Custom":
                prompt = st.text_area(
                    "Enter your custom prompt",
                    height=200,
                    placeholder="Enter instructions for generating the investment memo...",
                    help="This prompt will guide Claude in generating the memo"
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
            prompt = st.text_area(
                "Enter your prompt",
                height=200,
                placeholder="Enter instructions for generating the investment memo...",
                help="This prompt will guide Claude in generating the memo"
            )

    with col2:
        st.header("📄 Output")

        # Generate button
        if st.button("🚀 Generate Investment Memo", type="primary", use_container_width=True):
            # Validation
            if not api_key:
                st.error("⚠️ Please provide an Anthropic API key")
            elif not document_text:
                st.error("⚠️ Please provide a document (upload file or enter URL)")
            elif not prompt:
                st.error("⚠️ Please select or enter a prompt")
            else:
                # Generate memo
                try:
                    memo = generate_memo(prompt, document_text, api_key)

                    # Store in session state
                    st.session_state['memo'] = memo
                    st.session_state['document_name'] = document_name

                    st.success("✓ Memo generated successfully!")

                except Exception as e:
                    st.error(f"Error generating memo: {e}")
                    if "api_key" in str(e).lower() or "unauthorized" in str(e).lower():
                        st.error("Check that your API key is valid and has sufficient credits")

        # Display memo if available
        if 'memo' in st.session_state:
            memo = st.session_state['memo']
            document_name = st.session_state.get('document_name', 'document')

            st.divider()

            # Display memo
            st.markdown(memo)

            st.divider()

            # Download button
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

    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: gray; padding: 20px;'>
        <small>
        Powered by Claude Sonnet 4 | Built with Streamlit<br>
        Investment Memo Generator v1.0
        </small>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
