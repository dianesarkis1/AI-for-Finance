#!/usr/bin/env python3
"""
Multi-Model Investment Memo Generator

This script uses curl via subprocess to:
  1) Process input files (.txt, .md, .jsonl) - automatically extracts credit agreements from JSONL files
  2) Call various LLM APIs (OpenAI, Google Gemini, Anthropic, Groq/Llama 3) with a concatenated prompt + processed texts
  3) Print the generated investment memo to stdout and optionally save it

Supported Models:
  - OpenAI: gpt-4, gpt-4-turbo, gpt-5, gpt-3.5-turbo
  - Google Gemini: gemini-2.5-pro, gemini-2.0-flash-exp, gemini-1.5-pro, gemini-1.5-flash
  - Anthropic: claude-3-opus, claude-3-sonnet, claude-3-haiku
  - Llama 3: llama-3.1-8b, llama-3.1-70b, llama-3.1-405b

Requirements:
  - For OpenAI: OPENAI_API_KEY environment variable
  - For Gemini: GEMINI_API_KEY environment variable  
  - For Anthropic: ANTHROPIC_API_KEY environment variable
  - For Llama 3: GROQ_API_KEY environment variable (using Groq API)

Examples:
  # OpenAI GPT-5 with JSONL file (auto-extracts credit agreement)
  OPENAI_API_KEY=sk-... python exploratory.py --model gpt-5 --input-file data/outputs/amg_raw.jsonl --output memo_gpt5.md
  
  # Google Gemini with text file
  GEMINI_API_KEY=... python exploratory.py --model gemini-2.0-flash-exp --input-file data/amg_credit_agreement_final.txt --output memo_gemini.md
  
  # Anthropic Claude with JSONL file
  ANTHROPIC_API_KEY=... python exploratory.py --model claude-3-sonnet --input-file data/outputs/amg_raw.jsonl --output memo_claude.md
  
  # Llama 3 70B with JSONL file
  GROQ_API_KEY=... python exploratory.py --model llama-3.1-70b --input-file data/outputs/amg_raw.jsonl --output memo_llama.md
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def read_text_file(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def clean_and_format_credit_agreement(text: str) -> str:
    """
    Clean and format extracted credit agreement text.
    
    Args:
        text: Raw text from credit agreement
        
    Returns:
        Cleaned and formatted text
    """
    # Remove extra whitespace and newlines
    text = re.sub(r'\s+', ' ', text)
    
    # Remove duplicate sections by finding unique content
    sections = []
    current_section = ""
    
    # Split by common section markers
    parts = re.split(r'(Section \d+\.|WHEREAS|NOW THEREFORE|IN WITNESS WHEREOF|ANNEX I)', text)
    
    for i, part in enumerate(parts):
        if part.strip():
            if part.startswith(('Section', 'WHEREAS', 'NOW THEREFORE', 'IN WITNESS', 'ANNEX')):
                if current_section.strip():
                    sections.append(current_section.strip())
                current_section = part
            else:
                current_section += part
    
    if current_section.strip():
        sections.append(current_section.strip())
    
    # Remove duplicates while preserving order
    seen = set()
    unique_sections = []
    for section in sections:
        # Create a key for deduplication (first 50 chars)
        key = section[:50]
        if key not in seen:
            seen.add(key)
            unique_sections.append(section)
    
    # Format the final text
    formatted_text = "\n\n".join(unique_sections)
    
    # Clean up specific artifacts
    formatted_text = re.sub(r'Exhibit 10\.1\s+', '', formatted_text)
    formatted_text = re.sub(r'EXECUTION VERSION\s+', '', formatted_text)
    formatted_text = re.sub(r'\[\d+\]', '', formatted_text)
    formatted_text = re.sub(r'\[Remainder\s+of\s+Page\s+Intentionally\s+Left\s+Blank\]', '', formatted_text)
    formatted_text = re.sub(r'£', '', formatted_text)  # Remove pound symbol
    
    return formatted_text.strip()


def extract_credit_agreement_from_jsonl(jsonl_path: Path) -> str:
    """
    Extract and clean credit agreement text from a JSONL file.
    
    Args:
        jsonl_path: Path to JSONL file containing credit agreement data
        
    Returns:
        Cleaned and formatted credit agreement text
    """
    print(f"Extracting credit agreement from {jsonl_path}...", file=sys.stderr)
    
    # Read the first line of the JSONL file
    with jsonl_path.open("r", encoding="utf-8") as f:
        line = f.readline().strip()
    
    if not line:
        raise ValueError(f"Empty JSONL file: {jsonl_path}")
    
    # Parse the JSON
    try:
        data = json.loads(line)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in file {jsonl_path}: {e}")
    
    # Extract the text field
    if 'text' not in data:
        raise ValueError(f"No 'text' field found in JSONL file: {jsonl_path}")
    
    raw_text = data['text']
    
    # Clean and format the text
    formatted_text = clean_and_format_credit_agreement(raw_text)
    
    print(f"Original text length: {len(raw_text):,} characters", file=sys.stderr)
    print(f"Formatted text length: {len(formatted_text):,} characters", file=sys.stderr)
    
    return formatted_text


def run_curl(args: List[str], stdin_bytes: Optional[bytes] = None) -> str:
    result = subprocess.run(
        args,
        input=stdin_bytes,
        capture_output=True,
        check=False,
        text=False,
    )
    if result.returncode != 0:
        stderr_text = result.stderr.decode("utf-8", errors="ignore")
        stdout_text = result.stdout.decode("utf-8", errors="ignore")
        raise RuntimeError(f"Command failed: {' '.join(args)}\nSTDERR:\n{stderr_text}\nSTDOUT:\n{stdout_text}")
    return result.stdout.decode("utf-8", errors="ignore")


# ---------------- OpenAI API ---------------- #

def build_openai_payload(model: str, system_text: str, user_text: str, max_output_tokens: int) -> Dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ],
        "max_completion_tokens": max_output_tokens,
    }


def call_openai_api(base_url: str, api_key: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{base_url}/chat/completions"
    cmd = [
        "curl",
        "-sS",
        "-X",
        "POST",
        url,
        "-H",
        f"Authorization: Bearer {api_key}",
        "-H",
        "Content-Type: application/json",
        "--data-binary",
        "@-",
    ]
    raw = run_curl(cmd, stdin_bytes=json.dumps(payload).encode("utf-8"))
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        raise RuntimeError(f"Failed to parse OpenAI output as JSON:\n{raw}")
    return data


def extract_output_text_openai(response: Dict[str, Any]) -> Optional[str]:
    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        msg = choices[0].get("message") if isinstance(choices[0], dict) else None
        if isinstance(msg, dict):
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
            # Check if there are reasoning tokens (for GPT-5)
            usage = response.get("usage", {})
            reasoning_tokens = usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0)
            if reasoning_tokens > 0:
                return f"[GPT-5 generated {reasoning_tokens} reasoning tokens but no content. This may indicate the model is processing internally.]"
    return None


# ---------------- Google Gemini API ---------------- #

def build_gemini_payload(content: str, max_output_tokens: int) -> Dict[str, Any]:
    return {
        "contents": [
            {
                "parts": [
                    {
                        "text": content
                    }
                ]
            }
        ],
        "generationConfig": {
            "maxOutputTokens": max_output_tokens,
            "temperature": 0.1,
            "topP": 0.8,
            "topK": 40
        }
    }


def call_gemini_api(api_key: str, model: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    cmd = [
        "curl",
        "-sS",
        "-X",
        "POST",
        url,
        "-H",
        "Content-Type: application/json",
        "--data-binary",
        "@-",
    ]
    raw = run_curl(cmd, stdin_bytes=json.dumps(payload).encode("utf-8"))
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        raise RuntimeError(f"Failed to parse Gemini output as JSON:\n{raw}")
    return data


def extract_output_text_gemini(response: Dict[str, Any]) -> Optional[str]:
    candidates = response.get("candidates")
    if isinstance(candidates, list) and candidates:
        candidate = candidates[0]
        content = candidate.get("content")
        if isinstance(content, dict):
            parts = content.get("parts")
            if isinstance(parts, list) and parts:
                text = parts[0].get("text")
                if isinstance(text, str) and text.strip():
                    return text.strip()
    return None


# ---------------- Anthropic API ---------------- #

def build_anthropic_payload(model: str, content: str, max_output_tokens: int) -> Dict[str, Any]:
    return {
        "model": model,
        "max_tokens": max_output_tokens,
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ]
    }


def call_anthropic_api(api_key: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    url = "https://api.anthropic.com/v1/messages"
    cmd = [
        "curl",
        "-sS",
        "-X",
        "POST",
        url,
        "-H",
        f"x-api-key: {api_key}",
        "-H",
        "Content-Type: application/json",
        "-H",
        "anthropic-version: 2023-06-01",
        "--data-binary",
        "@-",
    ]
    raw = run_curl(cmd, stdin_bytes=json.dumps(payload).encode("utf-8"))
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        raise RuntimeError(f"Failed to parse Anthropic output as JSON:\n{raw}")
    return data


def extract_output_text_anthropic(response: Dict[str, Any]) -> Optional[str]:
    content = response.get("content")
    if isinstance(content, list) and content:
        text = content[0].get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()
    return None


# ---------------- Groq API (Llama 3) ---------------- #

def build_groq_payload(model: str, content: str, max_output_tokens: int) -> Dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ],
        "max_tokens": max_output_tokens,
        "temperature": 0.1,
        "top_p": 0.8
    }


def call_groq_api(api_key: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    url = "https://api.groq.com/openai/v1/chat/completions"
    cmd = [
        "curl",
        "-sS",
        "-X",
        "POST",
        url,
        "-H",
        f"Authorization: Bearer {api_key}",
        "-H",
        "Content-Type: application/json",
        "--data-binary",
        "@-",
    ]
    raw = run_curl(cmd, stdin_bytes=json.dumps(payload).encode("utf-8"))
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        raise RuntimeError(f"Failed to parse Groq output as JSON:\n{raw}")
    return data


def extract_output_text_groq(response: Dict[str, Any]) -> Optional[str]:
    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        msg = choices[0].get("message") if isinstance(choices[0], dict) else None
        if isinstance(msg, dict):
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
    return None


# ---------------- Model Configuration ---------------- #

MODEL_CONFIGS = {
    # OpenAI models
    "gpt-5": {"provider": "openai", "api_key_env": "OPENAI_API_KEY", "base_url": "https://api.openai.com/v1"},
    "gpt-4": {"provider": "openai", "api_key_env": "OPENAI_API_KEY", "base_url": "https://api.openai.com/v1"},
    "gpt-4-turbo": {"provider": "openai", "api_key_env": "OPENAI_API_KEY", "base_url": "https://api.openai.com/v1"},
    "gpt-3.5-turbo": {"provider": "openai", "api_key_env": "OPENAI_API_KEY", "base_url": "https://api.openai.com/v1"},
    
    # Google Gemini models
    "gemini-2.5-pro": {"provider": "gemini", "api_key_env": "GEMINI_API_KEY"},
    "gemini-2.0-flash-exp": {"provider": "gemini", "api_key_env": "GEMINI_API_KEY"},
    "gemini-1.5-pro": {"provider": "gemini", "api_key_env": "GEMINI_API_KEY"},
    "gemini-1.5-flash": {"provider": "gemini", "api_key_env": "GEMINI_API_KEY"},
    
    # Anthropic models
    "claude-sonnet-4-20250514": {"provider": "anthropic", "api_key_env": "ANTHROPIC_API_KEY"},
    "claude-3-5-sonnet-20241022": {"provider": "anthropic", "api_key_env": "ANTHROPIC_API_KEY"},
    "claude-3-sonnet-20240229": {"provider": "anthropic", "api_key_env": "ANTHROPIC_API_KEY"},
    "claude-3-opus": {"provider": "anthropic", "api_key_env": "ANTHROPIC_API_KEY"},
    "claude-3-sonnet": {"provider": "anthropic", "api_key_env": "ANTHROPIC_API_KEY"},
    "claude-3-haiku": {"provider": "anthropic", "api_key_env": "ANTHROPIC_API_KEY"},
    
    # Llama 3 models (via Groq API)
    "llama-3.1-8b": {"provider": "groq", "api_key_env": "GROQ_API_KEY"},
    "llama-3.1-70b": {"provider": "groq", "api_key_env": "GROQ_API_KEY"},
    "llama-3.1-405b": {"provider": "groq", "api_key_env": "GROQ_API_KEY"},
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate investment memos using various LLM providers.")
    parser.add_argument("--model", type=str, required=True, 
                       help="Model to use (e.g., gpt-5, gemini-2.0-flash-exp, claude-3-sonnet)")
    parser.add_argument("--prompt", type=str, help="User prompt text to accompany the attached file(s).")
    parser.add_argument("--prompt-file", type=str, help="Optional path to a text file containing the prompt.")
    parser.add_argument("--input-file", action="append", required=True, help="Path to a file to process (.txt, .md, .jsonl). JSONL files will be auto-extracted. Repeat for multiple.")
    parser.add_argument("--output", type=str, help="Optional path to save the generated memo.")
    parser.add_argument("--max-output-tokens", type=int, default=2000, help="Cap on generated/new tokens.")

    args = parser.parse_args()

    # Validate model
    if args.model not in MODEL_CONFIGS:
        print(f"ERROR: Unsupported model '{args.model}'. Supported models:", file=sys.stderr)
        for model in MODEL_CONFIGS.keys():
            print(f"  - {model}", file=sys.stderr)
        sys.exit(1)

    model_config = MODEL_CONFIGS[args.model]
    provider = model_config["provider"]

    input_paths = [Path(p) for p in (args.input_file or [])]
    for p in input_paths:
        if not p.exists():
            print(f"ERROR: Input file not found: {p}", file=sys.stderr)
            sys.exit(1)

    prompt_text: Optional[str] = None
    if args.prompt_file:
        prompt_path = Path(args.prompt_file)
        if not prompt_path.exists():
            print(f"ERROR: Prompt file not found: {prompt_path}", file=sys.stderr)
            sys.exit(1)
        prompt_text = read_text_file(prompt_path).strip()
    elif args.prompt:
        prompt_text = args.prompt.strip()
    if not prompt_text:
        prompt_text = "Draft an investment memo using the credit agreement for the deal. The investment memo should have 3 main sections. The first is an executive summary with key info such as date, overview of the company, what the deal is, a brief background on the company and the purpose of the transaction. The second, \"investment highlights & risks\". includes bullets on the key highlights and risks of the transaction from the point of view of an investor. The third, key deal information, is a table that includes deal size, deal price, interest rate, key covenants, maturity date, and payment frequency. I am inputting a template memo for reference. I want you to ensure you use only facts from the attached credit agreement. If you cannot find certain data points, write \"N/A\", but do not make up numbers or terms/facts."

    # Prepare inline texts
    inline_texts: List[str] = []
    for p in input_paths:
        if p.suffix.lower() == '.jsonl':
            # Extract credit agreement from JSONL file
            content = extract_credit_agreement_from_jsonl(p)
        else:
            # Read regular text file
            content = read_text_file(p)
        inline_texts.append(f"BEGIN_FILE: {p.name}\n{content}\nEND_FILE: {p.name}")

    # Get API key
    api_key = os.getenv(model_config["api_key_env"])
    if not api_key:
        print(f"ERROR: Set {model_config['api_key_env']} in your environment.", file=sys.stderr)
        sys.exit(1)

    # Prepare content based on provider
    system_preamble = (
        "You are an investment analyst. Using the provided credit agreement and any template references, "
        "produce a concise, structured investment memo."
    )
    
    if provider == "openai":
        combined_parts: List[str] = ["USER PROMPT:\n" + prompt_text]
        for inline in inline_texts:
            combined_parts.append("\n\n" + inline)
        user_combined = "\n\n".join(combined_parts)
        
        payload = build_openai_payload(
            model=args.model,
            system_text=system_preamble,
            user_text=user_combined,
            max_output_tokens=args.max_output_tokens,
        )
        print(f"Requesting completion from {model_config.get('base_url', 'OpenAI')}/chat/completions using model {args.model}...", file=sys.stderr)
        response = call_openai_api(base_url=model_config.get("base_url", "https://api.openai.com/v1"), api_key=api_key, payload=payload)
        output_text = extract_output_text_openai(response) or json.dumps(response, indent=2)
        
    elif provider == "gemini":
        combined_parts: List[str] = [system_preamble, "USER PROMPT:\n" + prompt_text]
        for inline in inline_texts:
            combined_parts.append("\n\n" + inline)
        user_combined = "\n\n".join(combined_parts)
        
        payload = build_gemini_payload(
            content=user_combined,
            max_output_tokens=args.max_output_tokens,
        )
        print(f"Requesting completion from Google Gemini using model {args.model}...", file=sys.stderr)
        response = call_gemini_api(api_key=api_key, model=args.model, payload=payload)
        output_text = extract_output_text_gemini(response) or json.dumps(response, indent=2)
        
    elif provider == "anthropic":
        combined_parts: List[str] = [f"System: {system_preamble}\n\nUser: {prompt_text}"]
        for inline in inline_texts:
            combined_parts.append("\n\n" + inline)
        user_combined = "\n\n".join(combined_parts)
        
        payload = build_anthropic_payload(
            model=args.model,
            content=user_combined,
            max_output_tokens=args.max_output_tokens,
        )
        print(f"Requesting completion from Anthropic using model {args.model}...", file=sys.stderr)
        response = call_anthropic_api(api_key=api_key, payload=payload)
        output_text = extract_output_text_anthropic(response) or json.dumps(response, indent=2)
        
    elif provider == "groq":
        combined_parts: List[str] = [f"System: {system_preamble}\n\nUser: {prompt_text}"]
        for inline in inline_texts:
            combined_parts.append("\n\n" + inline)
        user_combined = "\n\n".join(combined_parts)
        
        payload = build_groq_payload(
            model=args.model,
            content=user_combined,
            max_output_tokens=args.max_output_tokens,
        )
        print(f"Requesting completion from Groq using model {args.model}...", file=sys.stderr)
        response = call_groq_api(api_key=api_key, payload=payload)
        output_text = extract_output_text_groq(response) or json.dumps(response, indent=2)

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(output_text, encoding="utf-8")
        print(f"Saved memo to {out_path}")
    else:
        print(output_text)


if __name__ == "__main__":
    main()
