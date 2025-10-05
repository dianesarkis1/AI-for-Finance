"""
Core evaluation metrics for investment memo generation.

Metrics:
- Accuracy: No hallucinated terms
- Completeness: All key terms captured
- Quality: Appropriate length, tone, structure
- Consistency (intra-memo): No internal contradictions
- Consistency (across runs): Variance in output quality
"""

from typing import Dict, List, Tuple
from evals.utils import call_llm_for_eval


ACCURACY_PROMPT_TEMPLATE = """You are evaluating an investment memo for factual accuracy against its source credit agreement.

SOURCE DOCUMENT:
{source_document}

GENERATED MEMO:
{memo}

Does the memo contain any key financial terms that are NOT supported by the source document? Key financial terms include: interest rates, dates, maturity dates, borrower/lender names, loan amounts, covenant terms, collateral details, or other material transaction terms.

DO NOT flag discrepancies related to:
- Tone or writing style differences
- Filler words or transitional phrases (e.g., "additionally", "furthermore", "however")
- Structural differences or organizational choices
- Paraphrasing or reformulation of the same information
- Summary or synthesis that accurately represents the source material
- Changes in titles, headers, or document names (focus only on substantive financial terms)

ONLY flag hallucinations where specific factual claims about key financial terms are made in the memo that are NOT present in or supported by the source document.

Answer YES if the memo contains hallucinated or unsupported key terms.
Answer NO if all key financial terms in the memo are supported by the source document.

Then list any specific hallucinations you found (or "None" if NO).

Format your response as:
ANSWER: [YES/NO]
HALLUCINATIONS: [list or "None"]"""


def evaluate_accuracy(
    memo: str,
    source_document: str,
    models: List[str] = None,
    consensus_threshold: float = 0.6
) -> Dict:
    """
    Evaluate accuracy by checking for hallucinated terms using LLM consensus.

    Args:
        memo: Generated investment memo text
        source_document: Original credit agreement text
        models: List of model identifiers to use for consensus (default: uses 3-5 models)
        consensus_threshold: Fraction of models that must agree for consensus (default: 0.6)

    Returns:
        Dict with:
            - accurate: bool, True if no hallucinations detected
            - score: float, percentage of models that voted NO (0-1)
            - votes: Dict mapping model to (vote, hallucinations)
            - consensus_reached: bool
    """
    if models is None:
        # Default to same 3 models used in main_exploratory.py
        models = ["gpt-5", "claude-sonnet-4-20250514", "gemini-2.5-pro"]

    prompt = ACCURACY_PROMPT_TEMPLATE.format(
        source_document=source_document,
        memo=memo
    )

    votes = {}
    yes_count = 0
    no_count = 0

    for model in models:
        response = call_llm_for_eval(model, prompt)
        vote, hallucinations = _parse_accuracy_response(response)
        votes[model] = {"vote": vote, "hallucinations": hallucinations}

        if vote == "YES":
            yes_count += 1
        elif vote == "NO":
            no_count += 1

    total_votes = yes_count + no_count
    no_percentage = no_count / total_votes if total_votes > 0 else 0

    # Consensus is reached if threshold of models agree
    consensus_reached = (yes_count / total_votes >= consensus_threshold or
                        no_count / total_votes >= consensus_threshold)

    # Accurate if majority votes NO (no hallucinations)
    accurate = no_count > yes_count

    return {
        "accurate": accurate,
        "score": no_percentage,
        "votes": votes,
        "consensus_reached": consensus_reached,
        "yes_votes": yes_count,
        "no_votes": no_count
    }


def _parse_accuracy_response(response: str) -> Tuple[str, str]:
    """
    Parse LLM response to extract vote and hallucinations.

    Returns:
        Tuple of (vote, hallucinations) where vote is "YES"/"NO" or "PARSE_ERROR"
    """
    import re

    # Extract ANSWER
    answer_match = re.search(r'ANSWER:\s*(YES|NO)', response, re.IGNORECASE)
    vote = answer_match.group(1).upper() if answer_match else "PARSE_ERROR"

    # Extract HALLUCINATIONS
    halluc_match = re.search(r'HALLUCINATIONS:\s*(.+?)(?:\n|$)', response, re.DOTALL)
    hallucinations = halluc_match.group(1).strip() if halluc_match else "Not provided"

    return vote, hallucinations


# ============================================================================
# COMPLETENESS METRIC
# ============================================================================

COMPLETENESS_PROMPT_TEMPLATE = """You are evaluating an investment memo for completeness against its source credit agreement.

SOURCE DOCUMENT:
{source_document}

GENERATED MEMO:
{memo}

Are any key financial terms from the source document MISSING from the memo? Key financial terms include: interest rates, dates, maturity dates, borrower/lender names, loan amounts, covenant terms, collateral details, or other material transaction terms.

DO NOT flag omissions related to:
- Minor procedural details
- Boilerplate legal language
- Redundant information already captured elsewhere in the memo
- Information appropriately summarized or synthesized

ONLY flag missing key financial terms that are material to understanding the transaction.

Answer YES if key terms are missing from the memo.
Answer NO if all key financial terms are present in the memo.

Then list any specific missing terms you found (or "None" if NO).

Format your response as:
ANSWER: [YES/NO]
MISSING_TERMS: [list or "None"]"""


def evaluate_completeness(
    memo: str,
    source_document: str,
    models: List[str] = None,
    consensus_threshold: float = 0.6
) -> Dict:
    """
    Evaluate completeness by checking for missing key terms using LLM consensus.

    Args:
        memo: Generated investment memo text
        source_document: Original credit agreement text
        models: List of model identifiers to use for consensus (default: uses 3 models)
        consensus_threshold: Fraction of models that must agree for consensus (default: 0.6)

    Returns:
        Dict with:
            - complete: bool, True if no key terms missing
            - score: float, percentage of models that voted NO (0-1)
            - votes: Dict mapping model to (vote, missing_terms)
            - consensus_reached: bool
    """
    if models is None:
        # Default to same 3 models used in main_exploratory.py
        models = ["gpt-5", "claude-sonnet-4-20250514", "gemini-2.5-pro"]

    prompt = COMPLETENESS_PROMPT_TEMPLATE.format(
        source_document=source_document,
        memo=memo
    )

    votes = {}
    yes_count = 0
    no_count = 0

    for model in models:
        response = call_llm_for_eval(model, prompt)
        vote, missing_terms = _parse_completeness_response(response)
        votes[model] = {"vote": vote, "missing_terms": missing_terms}

        if vote == "YES":
            yes_count += 1
        elif vote == "NO":
            no_count += 1

    total_votes = yes_count + no_count
    no_percentage = no_count / total_votes if total_votes > 0 else 0

    # Consensus is reached if threshold of models agree
    consensus_reached = (yes_count / total_votes >= consensus_threshold or
                        no_count / total_votes >= consensus_threshold)

    # Complete if majority votes NO (no missing terms)
    complete = no_count > yes_count

    return {
        "complete": complete,
        "score": no_percentage,
        "votes": votes,
        "consensus_reached": consensus_reached,
        "yes_votes": yes_count,
        "no_votes": no_count
    }


def _parse_completeness_response(response: str) -> Tuple[str, str]:
    """
    Parse LLM response to extract vote and missing terms.

    Returns:
        Tuple of (vote, missing_terms) where vote is "YES"/"NO" or "PARSE_ERROR"
    """
    import re

    # Extract ANSWER
    answer_match = re.search(r'ANSWER:\s*(YES|NO)', response, re.IGNORECASE)
    vote = answer_match.group(1).upper() if answer_match else "PARSE_ERROR"

    # Extract MISSING_TERMS
    missing_match = re.search(r'MISSING_TERMS:\s*(.+?)(?:\n|$)', response, re.DOTALL)
    missing_terms = missing_match.group(1).strip() if missing_match else "Not provided"

    return vote, missing_terms


# ============================================================================
# INTRA-MEMO CONSISTENCY METRIC
# ============================================================================

CONSISTENCY_PROMPT_TEMPLATE = """You are evaluating an investment memo for internal self-contradictions.

MEMO:
{memo}

Goal
Detect *genuine* self-contradictions or logical impossibilities **inside** the investment memo.

Definitions
• A contradiction = two statements or claims that cannot both be true.
• Examples of contradictions:
  - Listing the same factor as both a strength AND a weakness
  - Stating incompatible financial terms (e.g., "5-year maturity" and "3-year maturity" for the same loan)
  - Claiming both positive and negative assessments of the same specific aspect
  - Making logically impossible claims (e.g., "debt-free" but also "high leverage")

• Overlaps, redundancies, or stylistic variations are *not* contradictions.
• Different sections discussing different aspects (e.g., strengths vs. weaknesses) are NOT contradictions unless they make incompatible claims about the SAME specific item.

What you MUST do
1. Compare every claim, financial term, and assessment against all others in the memo.
2. List at most FIVE genuine contradictions (each as ONE concise bullet point).
3. If no contradiction exists, say so.

Output format (**strict JSON**)
Return **only** a JSON object that matches this schema:

```json
{{
  "has_issues": <bool>,
  "issues": [
    "<bullet 1>",
    "<bullet 2>"
  ]
}}
```

Rules:
- has_issues = true IFF the issues array is non-empty.
- Do not add extra keys, comments, or markdown formatting.
- Return ONLY the JSON object, nothing else."""


def evaluate_consistency(
    memo: str,
    models: List[str] = None,
    consensus_threshold: float = 0.6
) -> Dict:
    """
    Evaluate intra-memo consistency by checking for self-contradictions using LLM consensus.

    Args:
        memo: Generated investment memo text
        models: List of model identifiers to use for consensus (default: uses 3 models)
        consensus_threshold: Fraction of models that must agree for consensus (default: 0.6)

    Returns:
        Dict with:
            - consistent: bool, True if no contradictions detected
            - score: float, percentage of models that found no issues (0-1)
            - votes: Dict mapping model to parsed response
            - consensus_reached: bool
    """
    if models is None:
        # Default to same 3 models used in main_exploratory.py
        models = ["gpt-5", "claude-sonnet-4-20250514", "gemini-2.5-pro"]

    prompt = CONSISTENCY_PROMPT_TEMPLATE.format(memo=memo)

    votes = {}
    has_issues_count = 0
    no_issues_count = 0

    for model in models:
        response = call_llm_for_eval(model, prompt)
        parsed = _parse_consistency_response(response)
        votes[model] = parsed

        if parsed["has_issues"]:
            has_issues_count += 1
        else:
            no_issues_count += 1

    total_votes = has_issues_count + no_issues_count
    no_issues_percentage = no_issues_count / total_votes if total_votes > 0 else 0

    # Consensus is reached if threshold of models agree
    consensus_reached = (has_issues_count / total_votes >= consensus_threshold or
                        no_issues_count / total_votes >= consensus_threshold)

    # Consistent if majority finds no issues
    consistent = no_issues_count > has_issues_count

    return {
        "consistent": consistent,
        "score": no_issues_percentage,
        "votes": votes,
        "consensus_reached": consensus_reached,
        "has_issues_votes": has_issues_count,
        "no_issues_votes": no_issues_count
    }


def _parse_consistency_response(response: str) -> Dict:
    """
    Parse LLM response to extract consistency evaluation.

    Returns:
        Dict with:
            - has_issues: bool
            - issues: List[str]
            - parse_error: bool (True if parsing failed)
    """
    import json
    import re

    # Try to extract JSON from the response
    # Look for JSON object pattern
    json_match = re.search(r'\{[\s\S]*"has_issues"[\s\S]*\}', response)

    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
            return {
                "has_issues": parsed.get("has_issues", False),
                "issues": parsed.get("issues", []),
                "parse_error": False
            }
        except json.JSONDecodeError:
            pass

    # If parsing fails, return error state
    return {
        "has_issues": None,
        "issues": ["PARSE_ERROR: Could not parse response"],
        "parse_error": True
    }