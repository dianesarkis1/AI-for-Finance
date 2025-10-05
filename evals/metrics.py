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