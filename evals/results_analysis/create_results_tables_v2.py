#!/usr/bin/env python3
"""
Extract results from final_comprehensive_eval_results.json and format into tables.
Version 2: Includes qualitative metrics (accuracy, completeness, consistency).
Supports per-round statistics for iterative refinement.
Outputs markdown only.
"""

import argparse
import json
from pathlib import Path

# Default paths (can be overridden by command-line arguments)
DEFAULT_RESULTS_DIR = "results_benchmark_3"


def create_summary_table(data):
    """Create overall summary statistics table."""
    summary = data['summary']
    has_rounds = summary.get('has_rounds', False)

    if not has_rounds:
        # Standard table for non-round-based results
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
    else:
        # Round-based table with columns for each round
        num_rounds = summary['num_rounds']
        rounds_data = summary['rounds']

        table = []
        # Header: Metric | Round 0 | Round 1 | ...
        header = ['Metric'] + [f'Round {i}' for i in range(num_rounds)]
        table.append(header)

        # Rows for each metric
        metrics = [
            ('Total Memos Evaluated', 'total_memos_evaluated'),
            ('Total Evaluations', 'total_evaluations'),
            ('Total Quality Scores', 'total_quality_scores'),
            ('Mean Score', 'mean_score'),
            ('Median Score', 'median_score'),
            ('Min Score', 'min_score'),
            ('Max Score', 'max_score'),
            ('Std Dev', 'stdev_score')
        ]

        for metric_name, metric_key in metrics:
            row = [metric_name]
            for round_num in range(num_rounds):
                value = rounds_data[str(round_num)][metric_key]
                if isinstance(value, float):
                    row.append(f"{value:.2f}")
                else:
                    row.append(str(value))
            table.append(row)

        return table


def create_evaluator_table(data):
    """Create summary statistics by evaluator table."""
    summary = data['summary']
    has_rounds = summary.get('has_rounds', False)

    if not has_rounds:
        # Standard table for non-round-based results
        evaluators = summary['evaluators']

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
    else:
        # Round-based table
        num_rounds = summary['num_rounds']
        rounds_data = summary['rounds']

        # Get all evaluators from any round
        all_evaluators = set()
        for round_data in rounds_data.values():
            all_evaluators.update(round_data['evaluators'].keys())

        table = []
        # Header: Evaluator | Count | Mean - Round 0 | Mean - Round 1 | ... | Median - Round 0 | Median - Round 1 | ...
        header = ['Evaluator', 'Count']
        for i in range(num_rounds):
            header.append(f'Mean - Round {i}')
        for i in range(num_rounds):
            header.append(f'Median - Round {i}')
        table.append(header)

        # Rows for each evaluator
        for evaluator in sorted(all_evaluators):
            row = [evaluator]

            # Count (use final round count)
            final_round_count = rounds_data[str(num_rounds - 1)]['evaluators'].get(evaluator, {}).get('count', 0)
            row.append(str(final_round_count))

            # Mean for each round
            for round_num in range(num_rounds):
                if evaluator in rounds_data[str(round_num)]['evaluators']:
                    mean = rounds_data[str(round_num)]['evaluators'][evaluator]['mean']
                    row.append(f"{mean:.2f}")
                else:
                    row.append('-')

            # Median for each round
            for round_num in range(num_rounds):
                if evaluator in rounds_data[str(round_num)]['evaluators']:
                    median = rounds_data[str(round_num)]['evaluators'][evaluator]['median']
                    row.append(f"{median:.2f}")
                else:
                    row.append('-')

            table.append(row)

        return table


def create_all_metrics_table(data):
    """Create summary statistics for all metrics (quantitative and qualitative)."""
    summary = data['summary']
    has_rounds = summary.get('has_rounds', False)

    if not has_rounds:
        # Standard non-round-based table
        return create_all_metrics_table_simple(data)
    else:
        # Round-based table - need to collect both quantitative and qualitative metrics
        num_rounds = summary['num_rounds']
        rounds_data = summary['rounds']
        results_by_index = data['results_by_index']

        # Get quantitative metrics from rounds
        all_metrics = set()
        for round_data in rounds_data.values():
            all_metrics.update(round_data['metrics'].keys())

        # Collect qualitative metrics per round
        qualitative_by_round = {}
        for round_num in range(num_rounds):
            accuracy_stats = {'total': 0, 'has_hallucinations': 0}
            completeness_stats = {'total': 0, 'is_incomplete': 0}
            consistency_stats = {'total': 0, 'has_issues': 0}

            for index_str, index_data in results_by_index.items():
                for evaluator, eval_results in index_data.items():
                    if evaluator == 'summary_score':
                        continue

                    # Check if this evaluator has round-based results
                    if 'rounds' in eval_results and str(round_num) in eval_results['rounds']:
                        round_metrics = eval_results['rounds'][str(round_num)]

                        # Collect accuracy
                        if 'accuracy' in round_metrics:
                            accuracy_stats['total'] += 1
                            if round_metrics['accuracy'].get('has_hallucinations'):
                                accuracy_stats['has_hallucinations'] += 1

                        # Collect completeness
                        if 'completeness' in round_metrics:
                            completeness_stats['total'] += 1
                            if round_metrics['completeness'].get('is_incomplete'):
                                completeness_stats['is_incomplete'] += 1

                        # Collect consistency
                        if 'consistency' in round_metrics:
                            consistency_stats['total'] += 1
                            if round_metrics['consistency'].get('has_issues'):
                                consistency_stats['has_issues'] += 1

            qualitative_by_round[round_num] = {
                'accuracy': accuracy_stats,
                'completeness': completeness_stats,
                'consistency': consistency_stats
            }

        table = []
        # Header: Metric | Count | Mean/% - Round 0 | Mean/% - Round 1 | ... | Median/Details - Round 0 | Median/Details - Round 1 | ...
        header = ['Metric', 'Count']
        for i in range(num_rounds):
            header.append(f'Mean/% - R{i}')
        for i in range(num_rounds):
            header.append(f'Median/Details - R{i}')
        table.append(header)

        # Rows for quantitative metrics
        for metric in sorted(all_metrics):
            row = [metric]

            # Count (use final round count)
            final_round_count = rounds_data[str(num_rounds - 1)]['metrics'].get(metric, {}).get('count', 0)
            row.append(str(final_round_count))

            # Mean for each round
            for round_num in range(num_rounds):
                if metric in rounds_data[str(round_num)]['metrics']:
                    mean = rounds_data[str(round_num)]['metrics'][metric]['mean']
                    row.append(f"{mean:.2f}")
                else:
                    row.append('-')

            # Median for each round
            for round_num in range(num_rounds):
                if metric in rounds_data[str(round_num)]['metrics']:
                    median = rounds_data[str(round_num)]['metrics'][metric]['median']
                    row.append(f"{median:.2f}")
                else:
                    row.append('-')

            table.append(row)

        # Add qualitative metrics
        for qual_metric in ['accuracy', 'completeness', 'consistency']:
            row = [qual_metric]

            # Count (use final round count)
            final_count = qualitative_by_round[num_rounds - 1][qual_metric]['total']
            row.append(str(final_count))

            # Percentage for each round
            for round_num in range(num_rounds):
                stats = qualitative_by_round[round_num][qual_metric]
                if stats['total'] > 0:
                    if qual_metric == 'accuracy':
                        pct = (stats['has_hallucinations'] / stats['total']) * 100
                        row.append(f"{pct:.1f}% halluc.")
                    elif qual_metric == 'completeness':
                        pct = (stats['is_incomplete'] / stats['total']) * 100
                        row.append(f"{pct:.1f}% incomp.")
                    else:  # consistency
                        pct = (stats['has_issues'] / stats['total']) * 100
                        row.append(f"{pct:.1f}% issues")
                else:
                    row.append('-')

            # Count details for each round
            for round_num in range(num_rounds):
                stats = qualitative_by_round[round_num][qual_metric]
                if stats['total'] > 0:
                    if qual_metric == 'accuracy':
                        row.append(f"{stats['has_hallucinations']}/{stats['total']}")
                    elif qual_metric == 'completeness':
                        row.append(f"{stats['is_incomplete']}/{stats['total']}")
                    else:  # consistency
                        row.append(f"{stats['has_issues']}/{stats['total']}")
                else:
                    row.append('-')

            table.append(row)

        return table


def create_all_metrics_table_simple(data):
    """Create summary statistics for all metrics (non-round-based)."""
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
    summary = data['summary']
    results_by_index = data['results_by_index']
    has_rounds = summary.get('has_rounds', False)

    evaluator_names = ['gpt-5', 'claude-sonnet-4-20250514', 'gemini-2.5-pro']

    if not has_rounds:
        # Standard table for non-round-based results
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
            for evaluator in evaluator_names:
                if evaluator in index_data:
                    summary_score = index_data[evaluator].get('summary_score', '')
                    row.append(f"{summary_score:.2f}" if summary_score else '')
                else:
                    row.append('')

            table.append(row)

        return table
    else:
        # Round-based table
        num_rounds = summary['num_rounds']

        table = []
        # Header: Index | Overall - R0 | Overall - R1 | ... | GPT-5 - R0 | GPT-5 - R1 | ... | Claude - R0 | Claude - R1 | ... | Gemini - R0 | Gemini - R1 | ...
        header = ['Index']
        for i in range(num_rounds):
            header.append(f'Overall - R{i}')
        for evaluator in evaluator_names:
            for i in range(num_rounds):
                eval_short = evaluator.split('-')[0].title() if evaluator == 'gpt-5' else \
                             'Claude' if 'claude' in evaluator else 'Gemini'
                header.append(f'{eval_short} - R{i}')
        table.append(header)

        for index in sorted(results_by_index.keys(), key=int):
            index_data = results_by_index[index]
            row = [index]

            # Overall scores for each round (averaged across evaluators)
            for round_num in range(num_rounds):
                round_scores = []
                for evaluator in evaluator_names:
                    if evaluator in index_data and 'rounds' in index_data[evaluator]:
                        if str(round_num) in index_data[evaluator]['rounds']:
                            score = index_data[evaluator]['rounds'][str(round_num)].get('summary_score')
                            if score is not None:
                                round_scores.append(score)

                if round_scores:
                    import statistics
                    avg_score = statistics.mean(round_scores)
                    row.append(f"{avg_score:.2f}")
                else:
                    row.append('-')

            # Evaluator scores for each round
            for evaluator in evaluator_names:
                for round_num in range(num_rounds):
                    if evaluator in index_data and 'rounds' in index_data[evaluator]:
                        if str(round_num) in index_data[evaluator]['rounds']:
                            score = index_data[evaluator]['rounds'][str(round_num)].get('summary_score')
                            if score is not None:
                                row.append(f"{score:.2f}")
                            else:
                                row.append('-')
                        else:
                            row.append('-')
                    else:
                        row.append('-')

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
    parser = argparse.ArgumentParser(
        description="Create results tables from final_comprehensive_eval_results.json"
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default=DEFAULT_RESULTS_DIR,
        help=f'Results directory name (default: {DEFAULT_RESULTS_DIR})'
    )
    args = parser.parse_args()

    # Set up paths (results are in evals/results/)
    evals_dir = Path(__file__).parent.parent  # Go up to evals/ directory
    results_dir = evals_dir / "results" / args.results_dir
    RESULTS_FILE = results_dir / "final_comprehensive_eval_results.json"
    OUTPUT_FILE = results_dir / "results_tables_2.md"

    print(f"\n{'='*70}")
    print(f"CREATING RESULTS TABLES V2")
    print(f"{'='*70}\n")

    # Load results
    print(f"Loading results from {RESULTS_FILE}...")
    with open(RESULTS_FILE, 'r') as f:
        data = json.load(f)

    # Check if results have rounds
    has_rounds = data['summary'].get('has_rounds', False)
    if has_rounds:
        print(f"  ✓ Detected round-based results ({data['summary']['num_rounds']} rounds)\n")

    # Create tables
    print("Creating markdown tables...")

    summary_table = create_summary_table(data)
    evaluator_table = create_evaluator_table(data)
    all_metrics_table = create_all_metrics_table(data)
    results_simple_table = create_simple_results_table(data)

    # Build markdown content
    md_content = "# Comprehensive Evaluation Results (Version 2)\n\n"

    if has_rounds:
        md_content += f"**Iterative Refinement: {data['summary']['num_rounds']} rounds**\n\n"

    md_content += "## Summary Statistics\n\n"
    md_content += format_markdown_table(summary_table)
    md_content += "\n"

    md_content += "## Statistics by Evaluator\n\n"
    md_content += format_markdown_table(evaluator_table)
    md_content += "\n"

    md_content += "## Statistics by Metric\n\n"
    if not has_rounds:
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