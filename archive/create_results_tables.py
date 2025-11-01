#!/usr/bin/env python3
"""
Extract results from final_comprehensive_eval_results.json and format into tables.
Outputs both markdown and CSV formats.
"""

import json
from pathlib import Path
import csv

# Paths
RESULTS_FILE = Path(__file__).parent / "final_comprehensive_eval_results.json"
OUTPUT_DIR = Path(__file__).parent / "results_tables"


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


def create_metric_table(data):
    """Create summary statistics by metric table."""
    metrics = data['summary']['metrics']

    table = []
    table.append(['Metric', 'Count', 'Mean', 'Median'])

    for metric, stats in metrics.items():
        table.append([
            metric,
            stats['count'],
            f"{stats['mean']:.2f}",
            f"{stats['median']:.2f}"
        ])

    return table


def create_results_by_index_table(data):
    """Create detailed results by index table."""
    results_by_index = data['results_by_index']

    table = []
    table.append([
        'Index',
        'Overall Score',
        'GPT-5 Score',
        'Claude Score',
        'Gemini Score',
        'GPT-5 Clarity',
        'GPT-5 Tone',
        'GPT-5 Length',
        'GPT-5 Structure',
        'Claude Clarity',
        'Claude Tone',
        'Claude Length',
        'Claude Structure',
        'Gemini Clarity',
        'Gemini Tone',
        'Gemini Length',
        'Gemini Structure'
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

        # Detailed metric scores for each evaluator
        for evaluator in ['gpt-5', 'claude-sonnet-4-20250514', 'gemini-2.5-pro']:
            if evaluator in index_data:
                eval_data = index_data[evaluator]
                for metric in ['quality_clarity', 'quality_tone', 'quality_length', 'quality_structure']:
                    score = eval_data.get(metric, {}).get('score')
                    row.append(str(score) if score is not None else '')
            else:
                row.extend(['', '', '', ''])

        table.append(row)

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


def save_csv(table, filename):
    """Save table as CSV."""
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(table)


def main():
    print(f"\n{'='*70}")
    print(f"CREATING RESULTS TABLES")
    print(f"{'='*70}\n")

    # Load results
    print(f"Loading results from {RESULTS_FILE.name}...")
    with open(RESULTS_FILE, 'r') as f:
        data = json.load(f)

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}\n")

    # Create tables
    tables = {
        'summary_statistics': create_summary_table(data),
        'evaluator_statistics': create_evaluator_table(data),
        'metric_statistics': create_metric_table(data),
        'results_by_index_simple': create_simple_results_table(data),
        'results_by_index_detailed': create_results_by_index_table(data)
    }

    # Save as markdown
    print("Creating markdown tables...")
    md_content = "# Comprehensive Evaluation Results\n\n"

    md_content += "## Summary Statistics\n\n"
    md_content += format_markdown_table(tables['summary_statistics'])
    md_content += "\n"

    md_content += "## Statistics by Evaluator\n\n"
    md_content += format_markdown_table(tables['evaluator_statistics'])
    md_content += "\n"

    md_content += "## Statistics by Metric\n\n"
    md_content += format_markdown_table(tables['metric_statistics'])
    md_content += "\n"

    md_content += "## Results by Index (Summary Scores)\n\n"
    md_content += format_markdown_table(tables['results_by_index_simple'])
    md_content += "\n"

    md_content += "## Results by Index (Detailed)\n\n"
    md_content += "Note: This table is very wide. See CSV for easier viewing.\n\n"
    md_content += format_markdown_table(tables['results_by_index_detailed'])
    md_content += "\n"

    md_file = OUTPUT_DIR / "results_tables.md"
    with open(md_file, 'w') as f:
        f.write(md_content)
    print(f"  ✅ Saved: {md_file.name}")

    # Save as CSV
    print("Creating CSV tables...")
    for table_name, table_data in tables.items():
        csv_file = OUTPUT_DIR / f"{table_name}.csv"
        save_csv(table_data, csv_file)
        print(f"  ✅ Saved: {csv_file.name}")

    print(f"\n{'='*70}")
    print(f"TABLES CREATED SUCCESSFULLY")
    print(f"{'='*70}")
    print(f"Location: {OUTPUT_DIR}")
    print(f"\nFiles created:")
    print(f"  - results_tables.md (all tables in markdown)")
    print(f"  - summary_statistics.csv")
    print(f"  - evaluator_statistics.csv")
    print(f"  - metric_statistics.csv")
    print(f"  - results_by_index_simple.csv")
    print(f"  - results_by_index_detailed.csv")
    print()


if __name__ == "__main__":
    main()
