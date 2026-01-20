#!/usr/bin/env python3
"""
Parse Zeng Pipeline Results to CSV

Converts Zeng evaluation log and MV2H JSON results into a unified CSV format
compatible with MT3 baseline analysis scripts.

Usage:
    python scripts/parse_zeng_results.py \
        --log /path/to/pipeline.log \
        --mv2h_dir /path/to/mv2h/results \
        --output data/experiments/zeng/chunk_results.csv

Output CSV format (same as MT3):
    task_id, pred_path, gt_path, status, error_message,
    Multi-pitch, Voice, Meter, Value, Harmony, MV2H, MV2H_custom, measure_idx
"""

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path


def parse_failures_from_log(log_path: str) -> dict:
    """
    Parse failure entries from Zeng pipeline log.

    Log format:
        WARNING - status [chunk_id]: message

    Args:
        log_path: Path to pipeline log file

    Returns:
        Dictionary mapping chunk_id to failure status
    """
    failures = {}
    pattern = r"WARNING - (\w+) \[([^\]]+)\]"

    with open(log_path, "r") as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                status = match.group(1)
                chunk_id = match.group(2)
                failures[chunk_id] = status

    return failures


def extract_measure_idx(chunk_id: str) -> int:
    """
    Extract measure index from chunk_id.

    chunk_id format: Piece#Subpiece#Performer.measure_idx
    Example: Bach#Prelude#bwv_875#Ahfat01M.10 -> 10

    Args:
        chunk_id: Chunk identifier string

    Returns:
        Measure index (0 if parsing fails)
    """
    if "." in chunk_id:
        try:
            return int(chunk_id.rsplit(".", 1)[1])
        except ValueError:
            pass
    return 0


def parse_mv2h_results(mv2h_dir: str, failures: dict) -> list:
    """
    Parse MV2H JSON result files.

    Args:
        mv2h_dir: Directory containing *_mv2h.json files
        failures: Dictionary of failures from log

    Returns:
        List of result dictionaries
    """
    results = []
    mv2h_path = Path(mv2h_dir)

    for fpath in mv2h_path.glob("*_mv2h.json"):
        chunk_id = fpath.stem.replace("_mv2h", "")

        try:
            with open(fpath, "r") as f:
                data = json.load(f)

            measure_idx = extract_measure_idx(chunk_id)

            # Determine status
            mv2h_score = data.get("MV2H", 0)
            if mv2h_score == 0 and chunk_id in failures:
                status = failures[chunk_id]
            elif mv2h_score == 0:
                status = "zero_mv2h"
            else:
                status = "success"

            # Calculate MV2H_custom (no Meter)
            mp = data.get("Multi-pitch", 0)
            voice = data.get("Voice", 0)
            value = data.get("Value", 0)
            harmony = data.get("Harmony", 0)
            mv2h_custom = (mp + voice + value + harmony) / 4

            results.append({
                "task_id": chunk_id,
                "pred_path": "",
                "gt_path": "",
                "status": status,
                "error_message": "",
                "Multi-pitch": mp,
                "Voice": voice,
                "Meter": data.get("Meter", 0),
                "Value": value,
                "Harmony": harmony,
                "MV2H": data.get("MV2H", 0),
                "MV2H_custom": mv2h_custom,
                "measure_idx": measure_idx,
            })

        except Exception as e:
            print(f"Error parsing {fpath.name}: {e}")

    return results


def add_missing_failures(results: list, failures: dict) -> list:
    """
    Add failure entries that don't have MV2H result files.

    Args:
        results: List of parsed results
        failures: Dictionary of all failures from log

    Returns:
        Updated results list with missing failures added
    """
    existing_ids = {r["task_id"] for r in results}

    for chunk_id, status in failures.items():
        if chunk_id not in existing_ids:
            measure_idx = extract_measure_idx(chunk_id)
            results.append({
                "task_id": chunk_id,
                "pred_path": "",
                "gt_path": "",
                "status": status,
                "error_message": status,
                "Multi-pitch": 0,
                "Voice": 0,
                "Meter": 0,
                "Value": 0,
                "Harmony": 0,
                "MV2H": 0,
                "MV2H_custom": 0,
                "measure_idx": measure_idx,
            })

    return results


def write_csv(results: list, output_path: str):
    """
    Write results to CSV file.

    Args:
        results: List of result dictionaries
        output_path: Output CSV file path
    """
    fieldnames = [
        "task_id", "pred_path", "gt_path", "status", "error_message",
        "Multi-pitch", "Voice", "Meter", "Value", "Harmony",
        "MV2H", "MV2H_custom", "measure_idx"
    ]

    # Create output directory if needed
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(results, key=lambda x: x["task_id"]):
            writer.writerow(row)


def print_summary(results: list):
    """Print summary statistics."""
    status_counts = defaultdict(int)
    for r in results:
        status_counts[r["status"]] += 1

    total = len(results)
    n_success = status_counts.get("success", 0)

    print("\n" + "=" * 60)
    print("Zeng Results Summary")
    print("=" * 60)
    print(f"Total chunks: {total}")
    print(f"Successful: {n_success} ({100*n_success/total:.1f}%)")
    print(f"Failed: {total - n_success} ({100*(total-n_success)/total:.1f}%)")
    print("\nStatus breakdown:")
    for status, count in sorted(status_counts.items(), key=lambda x: -x[1]):
        print(f"  {status}: {count} ({100*count/total:.1f}%)")

    # MV2H metrics for successful chunks
    successful = [r for r in results if r["status"] == "success"]
    if successful:
        print("\n" + "=" * 60)
        print(f"MV2H Metrics (Zeng Method, n={len(successful)})")
        print("=" * 60)
        for metric in ["Multi-pitch", "Voice", "Meter", "Value", "Harmony", "MV2H", "MV2H_custom"]:
            avg = sum(r[metric] for r in successful) / len(successful)
            print(f"  {metric}: {avg:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Parse Zeng pipeline results to CSV format"
    )
    parser.add_argument(
        "--log",
        type=str,
        required=True,
        help="Path to Zeng pipeline log file"
    )
    parser.add_argument(
        "--mv2h_dir",
        type=str,
        required=True,
        help="Directory containing MV2H JSON results"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/experiments/zeng/chunk_results.csv",
        help="Output CSV file path"
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.log):
        raise FileNotFoundError(f"Log file not found: {args.log}")
    if not os.path.isdir(args.mv2h_dir):
        raise FileNotFoundError(f"MV2H directory not found: {args.mv2h_dir}")

    print(f"Parsing log: {args.log}")
    print(f"MV2H directory: {args.mv2h_dir}")

    # Parse data
    failures = parse_failures_from_log(args.log)
    print(f"Parsed {len(failures)} failures from log")

    results = parse_mv2h_results(args.mv2h_dir, failures)
    print(f"Parsed {len(results)} MV2H result files")

    results = add_missing_failures(results, failures)
    print(f"Total results after adding missing failures: {len(results)}")

    # Write output
    write_csv(results, args.output)
    print(f"\nSaved to: {args.output}")

    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()
