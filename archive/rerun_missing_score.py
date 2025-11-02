#!/usr/bin/env python3
"""
Re-run the missing quality_structure evaluation for index 370 with GPT-5.
"""

import json
import os
import subprocess
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def extract_memo_from_batch_input(batch_input_file: Path) -> str:
    """Extract the generated memo from the batch input file."""
    with open(batch_input_file, 'r') as f:
        for line in f:
            request = json.loads(line)
            if request['custom_id'] == 'accuracy':
                content = request['body']['messages'][0]['content']

                # Find the GENERATED MEMO section
                start = content.find('GENERATED MEMO:')
                if start != -1:
                    memo_section = content[start+15:].strip()

                    # Find where the memo ends (before evaluation questions)
                    end = memo_section.find('\n\nDoes the memo contain')
                    if end == -1:
                        end = memo_section.find('\n\nAre any key')
                    if end == -1:
                        end = memo_section.find('\n\nGoal')

                    if end != -1:
                        return memo_section[:end].strip()
                    else:
                        return memo_section.strip()

    raise ValueError("Could not find memo in batch input file")

def get_quality_structure_prompt(memo: str) -> str:
    """Get the quality_structure evaluation prompt."""
    return f"""You are evaluating an investment memo for structural consistency with a provided template.

EXPECTED TEMPLATE:
1. Executive Summary/Overview
2. Transaction/Company Details
3. Financial Terms
4. Investment Strengths/Highlights
5. Risks and Concerns
6. Recommendation/Conclusion

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

def call_gpt5_for_evaluation(prompt: str) -> int:
    """Call GPT-5 to evaluate quality_structure."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment")

    # Prepare request body
    request_body = {
        "model": "gpt-5",
        "messages": [{"role": "user", "content": prompt}]
    }

    # Use curl to call OpenAI API
    cmd = [
        "curl", "-s", "-X", "POST",
        "https://api.openai.com/v1/chat/completions",
        "-H", "Content-Type: application/json",
        "-H", f"Authorization: Bearer {api_key}",
        "-d", json.dumps(request_body)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"curl failed: {result.stderr}")

    response = json.loads(result.stdout)

    if "error" in response:
        raise RuntimeError(f"API error: {response['error']}")

    content = response["choices"][0]["message"]["content"].strip()

    # Extract score from response
    if "SCORE:" in content:
        score_str = content.split("SCORE:")[-1].strip()
        # Remove any non-digit characters
        score_str = ''.join(c for c in score_str if c.isdigit())
        return int(score_str)
    else:
        # Try to extract just a number
        score_str = ''.join(c for c in content if c.isdigit())
        if score_str:
            return int(score_str)
        else:
            raise ValueError(f"Could not extract score from response: {content}")

def add_score_to_results(results_file: Path, index: int, evaluator: str, score: int):
    """Add the missing score to the results file and recompute summary."""
    with open(results_file, 'r') as f:
        data = json.load(f)

    # Add the quality_structure score
    index_str = str(index)
    data['results_by_index'][index_str][evaluator]['quality_structure'] = {
        "score": score,
        "metric": "quality_structure"
    }

    print(f"✅ Added quality_structure score for index {index}, {evaluator}: {score}")

    # Recompute summary statistics
    print("\n📊 Recomputing summary statistics...")

    results_by_index = data['results_by_index']

    # Collect scores per evaluator and per metric (for total count)
    evaluator_scores = {"gpt-5": [], "claude-sonnet-4-20250514": [], "gemini-2.5-pro": []}
    metric_scores = {
        "quality_clarity": [],
        "quality_tone": [],
        "quality_length": [],
        "quality_structure": []
    }

    # Collect averaged scores per index (average across 3 evaluators)
    averaged_scores_per_index = []

    for idx_key, idx_data in results_by_index.items():
        # Collect summary scores for this index across all evaluators
        index_summary_scores = []

        for eval_name, eval_data in idx_data.items():
            if eval_name == 'summary_score':
                continue

            # Summary score
            if 'summary_score' in eval_data and eval_data['summary_score'] is not None:
                index_summary_scores.append(eval_data['summary_score'])
                evaluator_scores[eval_name].append(eval_data['summary_score'])

            # Quality metrics (for total count)
            for metric in metric_scores.keys():
                if metric in eval_data and 'score' in eval_data[metric]:
                    metric_scores[metric].append(eval_data[metric]['score'])

        # Calculate average score for this index across evaluators
        if index_summary_scores:
            avg_score = sum(index_summary_scores) / len(index_summary_scores)
            averaged_scores_per_index.append(avg_score)

    # Update summary - total quality scores
    data['summary']['total_quality_scores'] = sum(len(scores) for scores in metric_scores.values())

    # Update metric statistics (these are still per-evaluator counts/stats)
    for metric, scores in metric_scores.items():
        if scores:
            data['summary']['metrics'][metric] = {
                "count": len(scores),
                "mean": round(sum(scores) / len(scores), 2),
                "median": sorted(scores)[len(scores) // 2] if scores else 0
            }

    # Update overall statistics - using AVERAGED scores per index
    if averaged_scores_per_index:
        data['summary']['mean_score'] = round(sum(averaged_scores_per_index) / len(averaged_scores_per_index), 2)
        sorted_avg_scores = sorted(averaged_scores_per_index)
        data['summary']['median_score'] = round(sorted_avg_scores[len(sorted_avg_scores) // 2], 2)
        data['summary']['min_score'] = round(min(averaged_scores_per_index), 2)
        data['summary']['max_score'] = round(max(averaged_scores_per_index), 2)

        # Calculate stdev
        mean = data['summary']['mean_score']
        variance = sum((x - mean) ** 2 for x in averaged_scores_per_index) / len(averaged_scores_per_index)
        data['summary']['stdev_score'] = round(variance ** 0.5, 2)

    # Update evaluator statistics
    for eval_name, scores in evaluator_scores.items():
        if scores:
            data['summary']['evaluators'][eval_name] = {
                "count": len(scores),
                "mean": round(sum(scores) / len(scores), 2),
                "median": round(sorted(scores)[len(scores) // 2], 2)
            }

    # Save updated results
    with open(results_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"✅ Updated {results_file}")
    print(f"\n📈 New statistics:")
    print(f"   Total quality scores: {data['summary']['total_quality_scores']}")
    print(f"   quality_structure count: {data['summary']['metrics']['quality_structure']['count']}")
    print(f"   quality_structure mean: {data['summary']['metrics']['quality_structure']['mean']}")
    print(f"   quality_structure median: {data['summary']['metrics']['quality_structure']['median']}")

def main():
    """Main function."""
    index = 370
    evaluator = "gpt-5"

    batch_input_file = Path("batch_temp_3/batch_input_370_1762117855.jsonl")
    results_file = Path("results_benchmark_3/final_comprehensive_eval_results.json")

    print(f"🔄 Re-running quality_structure evaluation for index {index} with {evaluator}...\n")

    # Step 1: Extract memo
    print("📄 Extracting memo from batch input file...")
    memo = extract_memo_from_batch_input(batch_input_file)
    print(f"   Memo length: {len(memo)} characters")

    # Step 2: Get prompt
    print("\n📝 Preparing evaluation prompt...")
    prompt = get_quality_structure_prompt(memo)

    # Step 3: Call GPT-5
    print(f"\n🤖 Calling {evaluator} API...")
    score = call_gpt5_for_evaluation(prompt)
    print(f"   ✅ Received score: {score}")

    # Step 4: Add to results and recompute
    print(f"\n💾 Updating results file...")
    add_score_to_results(results_file, index, evaluator, score)

    print("\n✅ Done! All 600 quality scores are now present (150 indices × 4 metrics).")

if __name__ == "__main__":
    main()
