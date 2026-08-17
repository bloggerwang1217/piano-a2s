#!/usr/bin/env python3
"""
Aggregate Slurm MV2H evaluation results.
Run after all slurm array tasks complete.

Usage:
    python evaluate_aggregate.py --output-folder workspace/1234/pretrain.score
"""
import os
import argparse
from collections import Counter
from utilities import load
from evaluate import summarize_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-folder', required=True)
    args = parser.parse_args()

    results_dir = f'{args.output_folder}/results'

    # Aggregate per-task summaries
    status_counts = Counter()
    all_errors = []
    for f in sorted(os.listdir(results_dir)):
        if f.startswith('slurm_task_') and f.endswith('.json'):
            summary = load(os.path.join(results_dir, f))
            for status, count in summary['status_counts'].items():
                status_counts[status] += count
            all_errors.extend(summary['errors'])

    # Total from test split
    test_dir = f'{results_dir}/test'
    total = len(os.listdir(test_dir))

    # Save aggregated errors
    with open(f'{results_dir}/errors.txt', 'w') as f:
        for error in all_errors:
            f.write(error + '\n')

    # Print summary
    print(f"\n=== Aggregated from Slurm tasks ===")
    print(f"Total chunks: {total}")
    for status, count in status_counts.most_common():
        print(f"  {status}: {count} ({count/total*100:.1f}%)")

    # MV2H summary
    summarize_results(args.output_folder, status_counts, total)


if __name__ == '__main__':
    main()
