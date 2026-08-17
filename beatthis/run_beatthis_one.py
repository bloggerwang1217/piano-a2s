"""Run Beat This! on one ASAP performance, write a `.beats` file in
ASAP-annotation TSV format so existing `_get_anno_downbeats()` parser can read it.

Format (matches ASAP `*_annotations.txt`):
    <time>\t<time>\t<beat_type>
where beat_type is `b` (beat) or `db` (downbeat). No key/time_sig fields
(those come from the GT annotation in the eval pipeline).

Usage: python run_beatthis_one.py --task-id N --tsv beatthis/asap_test_perfs.tsv \
                                  --out-dir workspace/beatthis
"""
import os
import argparse
import numpy as np
from pathlib import Path

from beat_this.inference import File2Beats


def write_asap_format(beats: np.ndarray, downbeats: np.ndarray, out_path: Path):
    db_set = set(np.round(downbeats, 6).tolist())
    lines = []
    # Merge beats + downbeats; downbeats are already a subset of beats in BT
    # but be defensive: union and sort
    all_times = sorted(set(np.round(beats, 6).tolist()) | db_set)
    for t in all_times:
        beat_type = 'db' if t in db_set else 'b'
        lines.append(f'{t}\t{t}\t{beat_type}')
    out_path.write_text('\n'.join(lines) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task-id', type=int, required=True)
    ap.add_argument('--tsv', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--checkpoint', default='final0')
    ap.add_argument('--asap-dir', default=os.environ.get('ASAP_DIR'),
                    help='ASAP checkout the tsv paths are relative to')
    args = ap.parse_args()
    if not args.asap_dir:
        raise SystemExit('pass --asap-dir or set ASAP_DIR (see env.sh.example)')

    rows = [l.rstrip('\n').split('\t') for l in open(args.tsv) if l.strip()]
    if args.task_id >= len(rows):
        print(f'task_id {args.task_id} out of range (total {len(rows)})')
        return
    score_name, perf, rel_wav = rows[args.task_id]
    wav_path = os.path.join(args.asap_dir, rel_wav)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{score_name}#{perf}_annotations.txt'

    if out_path.exists():
        print(f'[skip] {out_path} exists')
        return

    print(f'[task {args.task_id}] {score_name} / {perf}')
    f2b = File2Beats(checkpoint_path=args.checkpoint, device='cpu', dbn=False)
    beats, downbeats = f2b(wav_path)
    print(f'  beats={len(beats)} downbeats={len(downbeats)}')
    write_asap_format(beats, downbeats, out_path)
    print(f'  wrote {out_path}')


if __name__ == '__main__':
    main()
