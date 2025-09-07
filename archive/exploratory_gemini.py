#!/usr/bin/env python3
"""
Shell-driven invocation helper for Google Gemini 2.5 Pro.

This script uses curl via subprocess to:
  1) Inline non-PDF files (e.g., .txt, .md) as input_text
  2) Call Google Gemini API with a concatenated prompt + inlined texts
  3) Print the generated investment memo to stdout and optionally save it

Requirements:
  - GEMINI_API_KEY environment variable set

Examples:
  GEMINI_API_KEY=... \
  python latest\ project\ scripts/exploratory_gemini.py \
    --prompt "Draft an investment memo summarizing key risks and covenants." \
    --input-file data/outputs/amg_credit_agreement_final.txt \
    --input-file data/sample_memo.md \
    --output amg_investment_memo_gemini.md
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


def call_gemini_api(api_key: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={api_key}"
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt Google Gemini 2.5 Pro via shell (curl) with a prompt + one or more files.")
    parser.add_argument("--prompt", type=str, help="User prompt text to accompany the attached file(s).")
    parser.add_argument("--prompt-file", type=str, help="Optional path to a text file containing the prompt.")
    parser.add_argument("--input-file", action="append", required=True, help="Path to a file to upload/inline. Repeat for multiple.")
    parser.add_argument("--output", type=str, help="Optional path to save the generated memo.")
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
        prompt_text = "Draft an investment memo using the credit agreement for the deal. The investment memo should have 3 main sections. The first is an executive summary with key info such as date, overview of the company, what the deal is, a brief background on the company and the purpose of the transaction. The second, \"investment highlights & risks\". includes bullets on the key highlights and risks of the transaction from the point of view of an investor. The third, key deal information, is a table that includes deal size, deal price, interest rate, key covenants, maturity date, and payment frequency. I am inputting a template memo for reference. I want you to ensure you use only facts from the attached credit agreement. If you cannot find certain data points, write \"N/A\", but do not make up numbers or terms/facts."

    # Prepare inline texts
    inline_texts: List[str] = []
    for p in input_paths:
        content = read_text_file(p)
        inline_texts.append(f"BEGIN_FILE: {p.name}\n{content}\nEND_FILE: {p.name}")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: Set GEMINI_API_KEY in your environment.", file=sys.stderr)
        sys.exit(1)

    system_preamble = (
        "You are an investment analyst. Using the provided credit agreement and any template references, "
        "produce a concise, structured investment memo."
    )
    combined_parts: List[str] = [system_preamble, "USER PROMPT:\n" + prompt_text]
    for inline in inline_texts:
        combined_parts.append("\n\n" + inline)
    user_combined = "\n\n".join(combined_parts)

    payload = build_gemini_payload(
        content=user_combined,
        max_output_tokens=args.max_output_tokens,
    )
    print(f"Requesting completion from Gemini 2.5 Pro...", file=sys.stderr)
    response = call_gemini_api(api_key=api_key, payload=payload)
    output_text = extract_output_text_gemini(response) or json.dumps(response, indent=2)

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(output_text, encoding="utf-8")
        print(f"Saved memo to {out_path}")
    else:
        print(output_text)


if __name__ == "__main__":
    main()
