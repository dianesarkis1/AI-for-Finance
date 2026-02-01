"""
Helper functions for evaluation.

Includes:
- LLM-as-judge API calls
- Parsers for memo structure
- Semantic matching utilities
"""

import json
import os
import subprocess
from typing import Any, Dict, List, Optional


def run_curl(args: List[str], stdin_bytes: Optional[bytes] = None) -> str:
    """Execute curl command and return stdout."""
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

def build_openai_payload(model: str, content: str) -> Dict[str, Any]:
    """Build payload for OpenAI API."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
    }
    # GPT-5 doesn't support temperature parameter, only default (1)
    if not model.startswith("gpt-5"):
        payload["temperature"] = 0.0  # Deterministic for evaluation
    return payload


def call_openai_api(base_url: str, api_key: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Call OpenAI API and return raw response."""
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
    """Extract text from OpenAI API response."""
    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        msg = choices[0].get("message") if isinstance(choices[0], dict) else None
        if isinstance(msg, dict):
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
    return None


# ---------------- Google Gemini API ---------------- #

def build_gemini_payload(content: str) -> Dict[str, Any]:
    """Build payload for Gemini API."""
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
            "temperature": 0.0,
        }
    }


def call_gemini_api(api_key: str, model: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Call Gemini API and return raw response."""
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
    """Extract text from Gemini API response."""
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

def build_anthropic_payload(model: str, content: str) -> Dict[str, Any]:
    """Build payload for Anthropic API."""
    return {
        "model": model,
        "max_tokens": 1024,
        "temperature": 0.0,
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ]
    }


def call_anthropic_api(api_key: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Call Anthropic API and return raw response."""
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
    """Extract text from Anthropic API response."""
    content = response.get("content")
    if isinstance(content, list) and content:
        text = content[0].get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()
    return None


# ---------------- Unified Eval Interface ---------------- #

# Model configurations for evaluation (matching main_exploratory.py)
EVAL_MODEL_CONFIGS = {
    # OpenAI models
    "gpt-5": {"provider": "openai", "api_key_env": "OPENAI_API_KEY", "base_url": "https://api.openai.com/v1"},
    "gpt-4o": {"provider": "openai", "api_key_env": "OPENAI_API_KEY", "base_url": "https://api.openai.com/v1"},
    "gpt-4o-mini": {"provider": "openai", "api_key_env": "OPENAI_API_KEY", "base_url": "https://api.openai.com/v1"},

    # Anthropic models
    "claude-sonnet-4-20250514": {"provider": "anthropic", "api_key_env": "ANTHROPIC_API_KEY"},
    "claude-3-5-sonnet-20241022": {"provider": "anthropic", "api_key_env": "ANTHROPIC_API_KEY"},
    "claude-3-5-sonnet": {"provider": "anthropic", "api_key_env": "ANTHROPIC_API_KEY"},

    # Google Gemini models
    "gemini-2.5-pro": {"provider": "gemini", "api_key_env": "GEMINI_API_KEY"},
    "gemini-2.0-flash-exp": {"provider": "gemini", "api_key_env": "GEMINI_API_KEY"},
}


def call_llm_for_eval(model: str, prompt: str, max_retries: int = 3) -> str:
    """
    Call an LLM for evaluation purposes using the same structure as model_run.py.

    Args:
        model: Model identifier (e.g., "gpt-4o", "claude-3-5-sonnet", "gemini-2.0-flash-exp")
        prompt: Evaluation prompt
        max_retries: Maximum number of retries for rate limit errors (default: 3)

    Returns:
        Model response as string

    Raises:
        ValueError: If model is unknown or API key not found
        RuntimeError: If API call fails after retries
    """
    import time
    import sys

    # Get model config
    if model not in EVAL_MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model}. Available models: {list(EVAL_MODEL_CONFIGS.keys())}")

    config = EVAL_MODEL_CONFIGS[model]
    provider = config["provider"]
    api_key = os.getenv(config["api_key_env"])

    if not api_key:
        raise ValueError(f"{config['api_key_env']} not found in environment")

    # Retry loop for rate limits
    for attempt in range(max_retries):
        try:
            # Call appropriate provider
            if provider == "openai":
                payload = build_openai_payload(model, prompt)
                response = call_openai_api(config["base_url"], api_key, payload)
                output = extract_output_text_openai(response)
            elif provider == "anthropic":
                payload = build_anthropic_payload(model, prompt)
                response = call_anthropic_api(api_key, payload)
                output = extract_output_text_anthropic(response)
            elif provider == "gemini":
                payload = build_gemini_payload(prompt)
                response = call_gemini_api(api_key, model, payload)
                output = extract_output_text_gemini(response)
            else:
                raise ValueError(f"Unknown provider: {provider}")

            if output is None:
                # Check if response contains rate limit or quota error
                response_str = str(response)
                if "error" in response and isinstance(response["error"], dict):
                    error_code = response["error"].get("code")
                    error_msg = response["error"].get("message", "")

                    # Rate limit or quota exceeded
                    if error_code in [429, 503] or "quota" in error_msg.lower() or "rate limit" in error_msg.lower():
                        if attempt < max_retries - 1:
                            # Extract retry delay if provided
                            retry_delay = 30  # Default 30 seconds
                            if "retry" in response_str.lower():
                                import re
                                delay_match = re.search(r'(\d+\.?\d*)\s*s', error_msg)
                                if delay_match:
                                    retry_delay = float(delay_match.group(1)) + 1  # Add 1 second buffer

                            print(f"Rate limit hit for {model}, retrying in {retry_delay}s (attempt {attempt + 1}/{max_retries})...", file=sys.stderr)
                            time.sleep(retry_delay)
                            continue
                        else:
                            raise RuntimeError(f"Rate limit exceeded for {model} after {max_retries} attempts: {error_msg}")

                raise RuntimeError(f"Failed to extract output from {model} response: {response}")

            return output

        except RuntimeError as e:
            # If it's a rate limit error and we have retries left, continue
            if "rate limit" in str(e).lower() or "quota" in str(e).lower():
                if attempt < max_retries - 1:
                    retry_delay = 30
                    print(f"Error with {model}, retrying in {retry_delay}s (attempt {attempt + 1}/{max_retries})...", file=sys.stderr)
                    time.sleep(retry_delay)
                    continue
            # Re-raise if not a rate limit error or out of retries
            raise

    # Should not reach here, but just in case
    raise RuntimeError(f"Failed to get response from {model} after {max_retries} attempts")
