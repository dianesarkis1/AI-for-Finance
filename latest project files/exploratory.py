#!/usr/bin/env python3
"""
Shell-driven invocation helper for OpenAI-compatible and Hugging Face providers.

This script uses curl via subprocess to:
  1) Inline non-PDF files (e.g., .txt, .md) as input_text and upload PDF files to the Files API (OpenAI only)
  2) Call either:
     - OpenAI-compatible Chat Completions API with a concatenated prompt + inlined texts, or
     - Hugging Face Inference API by sending a combined text prompt
  3) Print the generated investment memo to stdout and optionally save it

Providers:
  - openai (default): requires OPENAI_API_KEY and optionally OPENAI_BASE_URL
  - hf: Hugging Face Inference API; requires HF_API_TOKEN

Examples (OpenAI-compatible):
  OPENAI_API_KEY=sk-... \
  python latest\ project\ files/exploratory.py \
    --prompt "Draft an investment memo summarizing key risks and covenants." \
    --input-file data/cleaned_data.jsonl \
    --input-file docs/sample_memo.md \
    --model gpt-5 \
    --output memo.md

Examples (Hugging Face):
  HF_API_TOKEN=hf_... \
  python latest\ project\ files/exploratory.py \
    --provider hf \
    --prompt "Draft an investment memo summarizing key risks and covenants." \
    --input-file data/cleaned_data.jsonl \
    --input-file docs/sample_memo.md \
    --model Qwen/Qwen2.5-7B-Instruct \
    --output memo.md
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def read_text_file(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return f.read()


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


# ---------------- OpenAI-compatible: chat.completions ---------------- #

def build_chat_payload(model: str, system_text: str, user_text: str, max_output_tokens: int) -> Dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ],
        "max_tokens": max_output_tokens,
    }


def call_chat_api(base_url: str, api_key: str, payload: Dict[str, Any]) -> Dict[str, Any]:
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
        raise RuntimeError(f"Failed to parse chat completions output as JSON:\n{raw}")
    return data


def extract_output_text_chat(response: Dict[str, Any]) -> Optional[str]:
    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        msg = choices[0].get("message") if isinstance(choices[0], dict) else None
        if isinstance(msg, dict):
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
    return None


# ---------------- Hugging Face Inference ---------------- #

def call_hf_inference(model: str, hf_token: str, combined_text: str, max_new_tokens: int, temperature: float = 0.2, wait_timeout_s: int = 60) -> Dict[str, Any]:
    url = f"https://api-inference.huggingface.co/models/{model}"
    payload = {
        "inputs": combined_text,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "return_full_text": False,
        },
        "options": {
            "wait_for_model": True
        }
    }
    headers = [
        "-H", f"Authorization: Bearer {hf_token}",
        "-H", "Content-Type: application/json",
    ]

    start = time.time()
    while True:
        raw = run_curl(["curl", "-sS", "-X", "POST", url, *headers, "--data-binary", "@-"], stdin_bytes=json.dumps(payload).encode("utf-8"))
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return {"output_text": raw}
        if isinstance(data, dict) and data.get("error", "").startswith("Model") and "loading" in data.get("error", "").lower():
            if time.time() - start > wait_timeout_s:
                return data
            time.sleep(3)
            continue
        return data


def extract_output_text_hf(response: Any) -> Optional[str]:
    if isinstance(response, list) and response:
        first = response[0]
        if isinstance(first, dict):
            if isinstance(first.get("generated_text"), str):
                return first["generated_text"].strip() or None
            if "output_text" in first and isinstance(first.get("output_text"), str):
                return first["output_text"].strip() or None
    if isinstance(response, dict) and isinstance(response.get("output_text"), str):
        return response["output_text"].strip() or None
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt an LLM via shell (curl) with a prompt + one or more files.")
    parser.add_argument("--provider", choices=["openai", "hf"], default="openai", help="Backend provider: openai (default) or hf (Hugging Face)")
    parser.add_argument("--prompt", type=str, help="User prompt text to accompany the attached file(s).")
    parser.add_argument("--prompt-file", type=str, help="Optional path to a text file containing the prompt.")
    parser.add_argument("--input-file", action="append", required=True, help="Path to a file to upload/inline. Repeat for multiple.")
    parser.add_argument("--output", type=str, help="Optional path to save the generated memo.")
    parser.add_argument("--model", type=str, default="gpt-5", help="Model identifier (OpenAI model name or HF repo id).")
    parser.add_argument("--max-output-tokens", type=int, default=2000, help="Cap on generated/new tokens.")

    args = parser.parse_args()

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
        prompt_text = "Draft an investment memo summarizing key terms, covenants, and risks."

    # Prepare inline texts and PDFs list
    pdf_paths: List[Path] = []
    inline_texts: List[str] = []
    for p in input_paths:
        if p.suffix.lower() == ".pdf":
            pdf_paths.append(p)
        else:
            content = read_text_file(p)
            inline_texts.append(f"BEGIN_FILE: {p.name}\n{content}\nEND_FILE: {p.name}")

    if args.provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("ERROR: Set OPENAI_API_KEY in your environment.", file=sys.stderr)
            sys.exit(1)
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

        system_preamble = (
            "You are an investment analyst. Using the provided credit agreement and any template references, "
            "produce a concise, structured investment memo."
        )
        combined_parts: List[str] = ["USER PROMPT:\n" + prompt_text]
        for inline in inline_texts:
            combined_parts.append("\n\n" + inline)
        user_combined = "\n\n".join(combined_parts)

        payload = build_chat_payload(
            model=args.model,
            system_text=system_preamble,
            user_text=user_combined,
            max_output_tokens=args.max_output_tokens,
        )
        print(f"Requesting completion from {base_url}/chat/completions using model {args.model} ...", file=sys.stderr)
        response = call_chat_api(base_url=base_url, api_key=api_key, payload=payload)
        output_text = extract_output_text_chat(response) or json.dumps(response, indent=2)

    else:  # provider == "hf"
        hf_token = os.getenv("HF_API_TOKEN")
        if not hf_token:
            print("ERROR: Set HF_API_TOKEN in your environment.", file=sys.stderr)
            sys.exit(1)
        system_preamble = (
            "You are an investment analyst. Using the provided credit agreement and any template references, "
            "produce a concise, structured investment memo."
        )
        combined_parts: List[str] = [system_preamble, "\n\nUSER PROMPT:\n" + prompt_text]
        for inline in inline_texts:
            combined_parts.append("\n\n" + inline)
        combined_text = "\n\n".join(combined_parts)
        print(f"Requesting completion from Hugging Face model {args.model} ...", file=sys.stderr)
        response = call_hf_inference(
            model=args.model,
            hf_token=hf_token,
            combined_text=combined_text,
            max_new_tokens=args.max_output_tokens,
            temperature=0.2,
        )
        output_text = extract_output_text_hf(response) or json.dumps(response, indent=2)

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(output_text, encoding="utf-8")
        print(f"Saved memo to {out_path}")
    else:
        print(output_text)


if __name__ == "__main__":
    main()
