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
from evals.evaluation.utils import call_llm_for_eval


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
        try:
            response = call_llm_for_eval(model, prompt)
            vote, hallucinations = _parse_accuracy_response(response)
            votes[model] = {"vote": vote, "hallucinations": hallucinations}

            if vote == "YES":
                yes_count += 1
            elif vote == "NO":
                no_count += 1
        except Exception as e:
            import sys
            print(f"Warning: Failed to get accuracy evaluation from {model}: {e}", file=sys.stderr)
            votes[model] = {"vote": "ERROR", "hallucinations": f"Failed: {str(e)[:100]}"}

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
        try:
            response = call_llm_for_eval(model, prompt)
            vote, missing_terms = _parse_completeness_response(response)
            votes[model] = {"vote": vote, "missing_terms": missing_terms}

            if vote == "YES":
                yes_count += 1
            elif vote == "NO":
                no_count += 1
        except Exception as e:
            import sys
            print(f"Warning: Failed to get completeness evaluation from {model}: {e}", file=sys.stderr)
            votes[model] = {"vote": "ERROR", "missing_terms": f"Failed: {str(e)[:100]}"}

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
        try:
            response = call_llm_for_eval(model, prompt)
            parsed = _parse_consistency_response(response)
            votes[model] = parsed

            if parsed["has_issues"]:
                has_issues_count += 1
            else:
                no_issues_count += 1
        except Exception as e:
            import sys
            print(f"Warning: Failed to get consistency evaluation from {model}: {e}", file=sys.stderr)
            votes[model] = {"has_issues": None, "issues": [f"Failed: {str(e)[:100]}"], "parse_error": True}

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


# ============================================================================
# QUALITY OF PRESENTATION METRIC
# ============================================================================

CLARITY_PROMPT_TEMPLATE = """You are evaluating an investment memo for clarity of presentation.

MEMO:
{memo}

Goal
Assess how clearly the memo communicates its key points and financial information.

Clarity Criteria
• Clear explanations: Financial terms and concepts are explained in an accessible yet professional manner
• Logical flow: Ideas progress naturally with smooth transitions
• Unambiguous language: Statements are precise and avoid vague or confusing phrasing
• Key points emphasized: Important financial terms and risks are highlighted appropriately
• Reader comprehension: The memo can be easily understood by an investment committee of experienced finance professionals without requiring re-reading

Evaluate the memo's clarity on a scale from 0-100:
- 90-100: Exceptionally clear, all information easy to understand
- 70-89: Clear with minor areas that could be clearer
- 50-69: Moderately clear but has some confusing sections
- 30-49: Multiple unclear or confusing sections
- 0-29: Difficult to understand, major clarity issues

Output format
Provide ONLY a number from 0-100 as your score.
SCORE: [number]"""


TONE_PROMPT_TEMPLATE = """You are evaluating an investment memo for appropriate professional tone.

MEMO:
{memo}

Goal
Assess whether the memo's tone is appropriate for presentation to an investment committee.

Tone Criteria
• Professional formality: Language is formal and businesslike, appropriate for executive decision-makers
• Objective presentation: Presents information factually without emotional language or hype
• Balanced perspective: Acknowledges both strengths and risks without being overly promotional or pessimistic
• Financial sophistication: Uses appropriate financial terminology without being overly technical or simplistic
• Confidence without arrogance: Presents analysis authoritatively but remains measured

Evaluate the memo's tone on a scale from 0-100:
- 90-100: Perfectly appropriate for investment committee, professional and balanced
- 70-89: Generally appropriate with minor tone issues
- 50-69: Somewhat appropriate but has noticeable tone problems
- 30-49: Inappropriate tone in multiple sections
- 0-29: Significantly inappropriate tone throughout

Output format
Provide ONLY a number from 0-100 as your score.
SCORE: [number]"""


LENGTH_PROMPT_TEMPLATE = """You are evaluating an investment memo for appropriate conciseness and verbosity.

MEMO:
{memo}

Goal
Assess whether the memo maintains appropriate conciseness (approx. less than 400 words) without unnecessary verbosity.

Length Criteria
• Consistent verbosity: All sections have similar levels of detail relative to their importance
• Conciseness: Information is presented efficiently without unnecessary repetition or filler
• No fluff: Avoids redundant phrases, obvious statements, or excessive elaboration
• Balanced detail: Each section provides proportional detail based on its significance
• Efficient communication: Gets to the point quickly without meandering

Note: Do NOT evaluate whether information is complete or missing - focus ONLY on whether the existing content is appropriately concise or unnecessarily verbose.

Evaluate the memo's conciseness on a scale from 0-100:
- 90-100: Perfectly concise, every section appropriately detailed with no fluff
- 70-89: Generally well-balanced with minor verbosity issues
- 50-69: Some sections too verbose or wordy
- 30-49: Multiple sections with significant verbosity problems
- 0-29: Consistently too verbose or wordy throughout

Output format
Provide ONLY a number from 0-100 as your score.
SCORE: [number]"""


STRUCTURE_PROMPT_TEMPLATE = """You are evaluating an investment memo for structural consistency with a provided template.

EXPECTED TEMPLATE:
{template}

MEMO:
{memo}

Goal
Assess whether the memo's structure matches the provided template.

Structure Criteria
• Section presence: All sections from the template are present in the memo
• Section ordering: Sections appear in the same order as the template
• Consistent formatting: Headers and formatting match the template style
• No extra sections: Memo doesn't include unnecessary sections not in the template
• Easy navigation: Structure makes it easy to find information where the template specifies

Evaluate the memo's structural consistency with the template on a scale from 0-100:
- 90-100: Perfectly matches template structure and ordering
- 70-89: Mostly matches template with minor deviations
- 50-69: Partially matches template but has notable structural differences
- 30-49: Poor match with template, significant structural issues
- 0-29: Does not follow template structure at all

Output format
Provide ONLY a number from 0-100 as your score.
SCORE: [number]"""


def evaluate_quality(
    memo: str,
    template: str = None,
    models: List[str] = None,
    consensus_threshold: float = 0.6
) -> Dict:
    """
    Evaluate presentation quality across 4 dimensions using LLM consensus scoring.

    Quality is assessed across:
    - Clarity: How clearly information is communicated
    - Tone: Appropriateness for investment committee setting
    - Length: Conciseness without unnecessary verbosity
    - Structure: Consistency with provided template

    Each dimension is scored 0-100 by multiple models, then averaged.
    Overall quality is the average of the 4 dimension scores.

    Args:
        memo: Generated investment memo text
        template: Expected memo structure/template (required for structure evaluation)
        models: List of model identifiers to use for consensus (default: uses 3 models)
        consensus_threshold: Not used for quality metric (kept for consistency)

    Returns:
        Dict with:
            - quality_score: float, overall quality score (0-100)
            - clarity_score: float, average clarity score (0-100)
            - tone_score: float, average tone score (0-100)
            - length_score: float, average length score (0-100)
            - structure_score: float, average structure score (0-100)
            - votes: Dict mapping model to scores for each dimension
    """
    if models is None:
        # Default to same 3 models used in other metrics
        models = ["gpt-5", "claude-sonnet-4-20250514", "gemini-2.5-pro"]

    # Prepare prompts for each dimension
    clarity_prompt = CLARITY_PROMPT_TEMPLATE.format(memo=memo)
    tone_prompt = TONE_PROMPT_TEMPLATE.format(memo=memo)
    length_prompt = LENGTH_PROMPT_TEMPLATE.format(memo=memo)

    # Structure prompt requires template
    if template:
        structure_prompt = STRUCTURE_PROMPT_TEMPLATE.format(template=template, memo=memo)
    else:
        # Use a default general structure if no template provided
        default_template = """1. Executive Summary/Overview
2. Transaction/Company Details
3. Financial Terms
4. Investment Strengths/Highlights
5. Risks and Concerns
6. Recommendation/Conclusion"""
        structure_prompt = STRUCTURE_PROMPT_TEMPLATE.format(template=default_template, memo=memo)

    votes = {}

    # Collect scores from each model for each dimension
    for i, model in enumerate(models):
        try:
            clarity_response = call_llm_for_eval(model, clarity_prompt)
            tone_response = call_llm_for_eval(model, tone_prompt)
            length_response = call_llm_for_eval(model, length_prompt)
            structure_response = call_llm_for_eval(model, structure_prompt)

            votes[model] = {
                "clarity": _parse_quality_score(clarity_response),
                "tone": _parse_quality_score(tone_response),
                "length": _parse_quality_score(length_response),
                "structure": _parse_quality_score(structure_response)
            }

            # Add delay between models to respect rate limits (especially Gemini free tier: 2 req/min)
            # Each model makes 4 calls, so delay ensures we don't exceed limits
            if i < len(models) - 1:  # Don't delay after the last model
                import time
                time.sleep(35)  # 35 seconds delay to stay under Gemini's 2 req/min limit
        except Exception as e:
            import sys
            print(f"Warning: Failed to get quality evaluation from {model}: {e}", file=sys.stderr)
            votes[model] = {
                "clarity": None,
                "tone": None,
                "length": None,
                "structure": None,
                "error": str(e)[:100]
            }

    # Calculate average score for each dimension
    clarity_scores = [v["clarity"] for v in votes.values() if v["clarity"] is not None]
    tone_scores = [v["tone"] for v in votes.values() if v["tone"] is not None]
    length_scores = [v["length"] for v in votes.values() if v["length"] is not None]
    structure_scores = [v["structure"] for v in votes.values() if v["structure"] is not None]

    clarity_avg = sum(clarity_scores) / len(clarity_scores) if clarity_scores else 0
    tone_avg = sum(tone_scores) / len(tone_scores) if tone_scores else 0
    length_avg = sum(length_scores) / len(length_scores) if length_scores else 0
    structure_avg = sum(structure_scores) / len(structure_scores) if structure_scores else 0

    # Overall quality is average of the 4 dimensions
    quality_score = (clarity_avg + tone_avg + length_avg + structure_avg) / 4

    return {
        "quality_score": quality_score,
        "clarity_score": clarity_avg,
        "tone_score": tone_avg,
        "length_score": length_avg,
        "structure_score": structure_avg,
        "votes": votes
    }


def _parse_quality_score(response: str) -> float:
    """
    Parse LLM response to extract numerical score (0-100).

    Returns:
        float: Score from 0-100, or None if parsing fails
    """
    import re

    # Look for "SCORE: [number]" pattern
    score_match = re.search(r'SCORE:\s*(\d+(?:\.\d+)?)', response, re.IGNORECASE)

    if score_match:
        try:
            score = float(score_match.group(1))
            # Clamp to 0-100 range
            return max(0.0, min(100.0, score))
        except ValueError:
            pass

    # Fallback: look for any number that might be a score
    numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', response)
    for num_str in numbers:
        try:
            num = float(num_str)
            if 0 <= num <= 100:
                return num
        except ValueError:
            continue

    # If no valid score found, return None
    return None


# ============================================================================
# SUMMARY SCORE AGGREGATION
# ============================================================================

def calculate_summary_score(
    accuracy_result: Dict = None,
    completeness_result: Dict = None,
    consistency_result: Dict = None,
    quality_result: Dict = None,
    weights: Dict[str, float] = None
) -> Dict:
    """
    Aggregate individual metric results into a single summary score.

    All metrics are normalized to 0-100 scale and combined using weighted average.

    Args:
        accuracy_result: Result dict from evaluate_accuracy()
        completeness_result: Result dict from evaluate_completeness()
        consistency_result: Result dict from evaluate_consistency()
        quality_result: Result dict from evaluate_quality()
        weights: Optional dict with keys 'accuracy', 'completeness', 'consistency', 'quality'
                Default weights are equal (0.25 each)

    Returns:
        Dict with:
            - summary_score: float, weighted average score (0-100)
            - normalized_scores: Dict with each metric normalized to 0-100
            - weights_used: Dict showing the weights applied
            - missing_metrics: List of metrics that were None
    """
    # Default equal weights
    if weights is None:
        weights = {
            'accuracy': 0.25,
            'completeness': 0.25,
            'consistency': 0.25,
            'quality': 0.25
        }

    normalized_scores = {}
    missing_metrics = []

    # Normalize accuracy (score is 0-1, convert to 0-100)
    if accuracy_result is not None:
        normalized_scores['accuracy'] = accuracy_result['score'] * 100
    else:
        missing_metrics.append('accuracy')

    # Normalize completeness (score is 0-1, convert to 0-100)
    if completeness_result is not None:
        normalized_scores['completeness'] = completeness_result['score'] * 100
    else:
        missing_metrics.append('completeness')

    # Normalize consistency (score is 0-1, convert to 0-100)
    if consistency_result is not None:
        normalized_scores['consistency'] = consistency_result['score'] * 100
    else:
        missing_metrics.append('consistency')

    # Quality is already 0-100
    if quality_result is not None:
        normalized_scores['quality'] = quality_result['quality_score']
    else:
        missing_metrics.append('quality')

    # Calculate weighted average only for available metrics
    if not normalized_scores:
        return {
            "summary_score": 0.0,
            "normalized_scores": {},
            "weights_used": weights,
            "missing_metrics": missing_metrics,
            "error": "No metrics provided"
        }

    # Adjust weights to account for missing metrics
    available_weight_sum = sum(weights[k] for k in normalized_scores.keys())
    adjusted_weights = {k: weights[k] / available_weight_sum for k in normalized_scores.keys()}

    # Calculate weighted summary score
    summary_score = sum(normalized_scores[k] * adjusted_weights[k] for k in normalized_scores.keys())

    return {
        "summary_score": summary_score,
        "normalized_scores": normalized_scores,
        "weights_used": adjusted_weights,
        "missing_metrics": missing_metrics
    }