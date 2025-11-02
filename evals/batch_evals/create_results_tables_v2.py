#!/usr/bin/env python3
"""
Extract results from final_comprehensive_eval_results.json and format into tables.
Version 2: Includes qualitative metrics (accuracy, completeness, consistency).
Outputs markdown only.
"""

import json
from pathlib import Path

# Paths
RESULTS_FILE = Path(__file__).parent / "results_benchmark_2" / "final_comprehensive_eval_results.json"
OUTPUT_FILE = Path(__file__).parent / "results_benchmark_2" / "results_tables_2.md"


def create_summary_table(data):
    """Create overall summary statistics table."""
    summary = data['summary']

    table = []
    table.append(['Metric', 'Value'])
    table.append(['Total Memos Evaluated', summary['total_memos_evaluated']])
    table.append(['Total Evaluations', summary['total_evaluations']])
    table.append(['Total Quality Scores', summary['total_quality_scores']])
    table.append(['Mean Score', f"{summary['mean_score']:.2f}"])
    table.append(['Median Score', f"{summary['median_score']:.2f}"])
    table.append(['Min Score', f"{summary['min_score']:.2f}"])
    table.append(['Max Score', f"{summary['max_score']:.2f}"])
    table.append(['Std Dev', f"{summary['stdev_score']:.2f}"])

    return table


def create_evaluator_table(data):
    """Create summary statistics by evaluator table."""
    evaluators = data['summary']['evaluators']

    table = []
    table.append(['Evaluator', 'Count', 'Mean', 'Median'])

    for evaluator, stats in evaluators.items():
        table.append([
            evaluator,
            stats['count'],
            f"{stats['mean']:.2f}",
            f"{stats['median']:.2f}"
        ])

    return table


def create_all_metrics_table(data):
    """Create summary statistics for all metrics (quantitative and qualitative)."""
    results_by_index = data['results_by_index']

    # Collect quantitative quality metrics
    quality_metrics = {}
    for metric in ['quality_clarity', 'quality_tone', 'quality_length', 'quality_structure']:
        quality_metrics[metric] = []

    # Collect qualitative metrics
    accuracy_stats = {'total': 0, 'has_hallucinations': 0}
    completeness_stats = {'total': 0, 'is_incomplete': 0}
    consistency_stats = {'total': 0, 'has_issues': 0}

    for index_str, index_data in results_by_index.items():
        for evaluator, eval_results in index_data.items():
            if evaluator == 'summary_score':
                continue

            # Collect quality scores
            for metric in ['quality_clarity', 'quality_tone', 'quality_length', 'quality_structure']:
                if metric in eval_results and eval_results[metric].get('score') is not None:
                    quality_metrics[metric].append(eval_results[metric]['score'])

            # Collect accuracy
            if 'accuracy' in eval_results:
                accuracy_stats['total'] += 1
                if eval_results['accuracy'].get('has_hallucinations'):
                    accuracy_stats['has_hallucinations'] += 1

            # Collect completeness
            if 'completeness' in eval_results:
                completeness_stats['total'] += 1
                if eval_results['completeness'].get('is_incomplete'):
                    completeness_stats['is_incomplete'] += 1

            # Collect consistency
            if 'consistency' in eval_results:
                consistency_stats['total'] += 1
                if eval_results['consistency'].get('has_issues'):
                    consistency_stats['has_issues'] += 1

    # Build table
    table = []
    table.append(['Metric', 'Type', 'Count', 'Mean / % with Issues', 'Median / Details'])

    # Add quality metrics (quantitative)
    for metric, scores in quality_metrics.items():
        if scores:
            import statistics
            mean_val = f"{statistics.mean(scores):.2f}"
            median_val = f"{statistics.median(scores):.2f}"
            table.append([
                metric,
                'Quantitative',
                len(scores),
                mean_val,
                median_val
            ])

    # Add accuracy (qualitative)
    if accuracy_stats['total'] > 0:
        pct = (accuracy_stats['has_hallucinations'] / accuracy_stats['total']) * 100
        table.append([
            'accuracy',
            'Qualitative',
            accuracy_stats['total'],
            f"{pct:.1f}% with hallucinations",
            f"{accuracy_stats['has_hallucinations']}/{accuracy_stats['total']} memos"
        ])

    # Add completeness (qualitative)
    if completeness_stats['total'] > 0:
        pct = (completeness_stats['is_incomplete'] / completeness_stats['total']) * 100
        table.append([
            'completeness',
            'Qualitative',
            completeness_stats['total'],
            f"{pct:.1f}% incomplete",
            f"{completeness_stats['is_incomplete']}/{completeness_stats['total']} memos"
        ])

    # Add consistency (qualitative)
    if consistency_stats['total'] > 0:
        pct = (consistency_stats['has_issues'] / consistency_stats['total']) * 100
        table.append([
            'consistency',
            'Qualitative',
            consistency_stats['total'],
            f"{pct:.1f}% with issues",
            f"{consistency_stats['has_issues']}/{consistency_stats['total']} memos"
        ])

    return table


def create_simple_results_table(data):
    """Create simplified results by index table (just summary scores)."""
    results_by_index = data['results_by_index']

    table = []
    table.append([
        'Index',
        'Overall Score',
        'GPT-5 Score',
        'Claude Score',
        'Gemini Score'
    ])

    for index in sorted(results_by_index.keys(), key=int):
        index_data = results_by_index[index]

        row = [index]

        # Overall score
        row.append(f"{index_data.get('summary_score', ''):.2f}" if index_data.get('summary_score') else '')

        # Evaluator summary scores
        for evaluator in ['gpt-5', 'claude-sonnet-4-20250514', 'gemini-2.5-pro']:
            if evaluator in index_data:
                summary = index_data[evaluator].get('summary_score', '')
                row.append(f"{summary:.2f}" if summary else '')
            else:
                row.append('')

        table.append(row)

    return table


def format_markdown_table(table):
    """Format table as markdown."""
    if not table:
        return ""

    # Calculate column widths
    col_widths = [max(len(str(row[i])) for row in table) for i in range(len(table[0]))]

    # Format header
    header = table[0]
    md = "| " + " | ".join(str(header[i]).ljust(col_widths[i]) for i in range(len(header))) + " |\n"
    md += "| " + " | ".join("-" * col_widths[i] for i in range(len(header))) + " |\n"

    # Format rows
    for row in table[1:]:
        md += "| " + " | ".join(str(row[i]).ljust(col_widths[i]) for i in range(len(row))) + " |\n"

    return md


def main():
    print(f"\n{'='*70}")
    print(f"CREATING RESULTS TABLES V2")
    print(f"{'='*70}\n")

    # Load results
    print(f"Loading results from {RESULTS_FILE.name}...")
    with open(RESULTS_FILE, 'r') as f:
        data = json.load(f)

    # Create tables
    print("Creating markdown tables...")

    summary_table = create_summary_table(data)
    evaluator_table = create_evaluator_table(data)
    all_metrics_table = create_all_metrics_table(data)
    results_simple_table = create_simple_results_table(data)

    # Build markdown content
    md_content = "# Comprehensive Evaluation Results (Version 2)\n\n"

    md_content += "## Summary Statistics\n\n"
    md_content += format_markdown_table(summary_table)
    md_content += "\n"

    md_content += "## Statistics by Evaluator\n\n"
    md_content += format_markdown_table(evaluator_table)
    md_content += "\n"

    md_content += "## Statistics by Metric (All Metrics)\n\n"
    md_content += "This table includes both quantitative quality metrics (0-100 scores) and qualitative metrics (presence/absence of issues).\n\n"
    md_content += format_markdown_table(all_metrics_table)
    md_content += "\n"

    md_content += "## Results by Index (Summary Scores)\n\n"
    md_content += format_markdown_table(results_simple_table)
    md_content += "\n"

    # Save markdown file
    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        f.write(md_content)

    print(f"  ✅ Saved: {OUTPUT_FILE.name}")

    print(f"\n{'='*70}")
    print(f"TABLE CREATED SUCCESSFULLY")
    print(f"{'='*70}")
    print(f"Location: {OUTPUT_FILE}")
    print()


if __name__ == "__main__":
    main()
