"""Aggregate slurm MV2H eval output into a chunk-level CSV.

For each ASAP test chunk, emit one row: (task_id, status, Multi-pitch, Voice,
Meter, Value, Harmony, MV2H, MV2H_custom, measure_idx).

Status comes from:
- MV2H JSON present and MV2H > 0 -> 'success'
- errors.txt (written by evaluate_aggregate) + slurm .err files -> failure reason

Usage:
    python build_chunk_csv.py --output-folder workspace/1234/asap_eval.score_ft.gt
"""
import os
import re
import csv
import json
import argparse
import glob
from pathlib import Path


def parse_err_logs(err_glob: str):
    """Return dict chunk_id -> status by greping slurm .err files."""
    pattern = re.compile(r'- WARNING - (\w+) \[([^\]]+)\]')
    out = {}
    for p in glob.glob(err_glob):
        try:
            with open(p) as f:
                for line in f:
                    m = pattern.search(line)
                    if m:
                        status, chunk_id = m.group(1), m.group(2)
                        out[chunk_id] = status
        except Exception:
            continue
    return out


def extract_measure_idx(chunk_id: str) -> int:
    if '.' in chunk_id:
        try:
            return int(chunk_id.rsplit('.', 1)[1])
        except ValueError:
            pass
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-folder', required=True)
    ap.add_argument('--test-dir', default=None,
                    help='Where the per-chunk .json predictions are; default '
                         '<output-folder>/results/test')
    ap.add_argument('--output-csv', default=None,
                    help='Default: <output-folder>/results/chunk_results.csv')
    ap.add_argument('--job-id', default=None,
                    help='Slurm job id to scope err-log parsing (avoids mixing runs)')
    args = ap.parse_args()

    out_folder = Path(args.output_folder)
    mv2h_dir = out_folder / 'results' / 'mv2h'
    test_dir = Path(args.test_dir) if args.test_dir else out_folder / 'results' / 'test'
    out_csv = Path(args.output_csv) if args.output_csv else out_folder / 'results' / 'chunk_results.csv'

    # All chunk_ids come from the predictions directory (model ran on them).
    all_chunks = sorted(p.stem for p in test_dir.glob('*.json'))
    print(f'total chunks (from {test_dir}): {len(all_chunks)}')

    # MV2H results
    mv2h_by_id = {}
    for p in mv2h_dir.glob('*_mv2h.json'):
        chunk_id = p.stem.replace('_mv2h', '')
        with open(p) as f:
            data = json.load(f)
        mp = data.get('Multi-pitch', 0)
        voice = data.get('Voice', 0)
        value = data.get('Value', 0)
        harmony = data.get('Harmony', 0)
        mv2h_by_id[chunk_id] = {
            'Multi-pitch': mp,
            'Voice': voice,
            'Meter': data.get('Meter', 0),
            'Value': value,
            'Harmony': harmony,
            'MV2H': data.get('MV2H', 0),
            'MV2H_custom': (mp + voice + value + harmony) / 4,
        }
    print(f'mv2h jsons: {len(mv2h_by_id)}')

    # Failure status from slurm .err files (filter by job id when given)
    if args.job_id:
        err_pattern = f'logs/slurm/mv2h_{args.job_id}_*.err'
    else:
        err_pattern = 'logs/slurm/mv2h_*.err'
    statuses = parse_err_logs(err_pattern)
    print(f'failure statuses parsed (glob={err_pattern}): {len(statuses)}')

    # Build rows
    fieldnames = ['task_id', 'status', 'Multi-pitch', 'Voice', 'Meter', 'Value',
                  'Harmony', 'MV2H', 'MV2H_custom', 'measure_idx']
    rows = []
    for cid in all_chunks:
        m_idx = extract_measure_idx(cid)
        if cid in mv2h_by_id:
            row = dict(mv2h_by_id[cid])
            # Check if it was zero-MV2H
            status = 'success' if row['MV2H'] > 0 else statuses.get(cid, 'zero_mv2h')
            row.update({'task_id': cid, 'status': status, 'measure_idx': m_idx})
        else:
            status = statuses.get(cid, 'conversion_error')
            row = {'task_id': cid, 'status': status, 'measure_idx': m_idx,
                   'Multi-pitch': 0, 'Voice': 0, 'Meter': 0, 'Value': 0,
                   'Harmony': 0, 'MV2H': 0, 'MV2H_custom': 0}
        rows.append(row)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'wrote {out_csv} ({len(rows)} rows)')

    # Status summary
    from collections import Counter
    print('status breakdown:')
    for s, c in Counter(r['status'] for r in rows).most_common():
        print(f'  {s}: {c}')


if __name__ == '__main__':
    main()
