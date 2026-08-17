"""Process ONE ASAP performance into five-bar chunks. One slurm task = one recording.

Reads a TSV of (score_name, performance, wav path relative to the ASAP
checkout) and builds only that row, so a hold-out defined per recording can be
built without pulling in the other performances of the same piece.

Without --bt-dir the chunks are cut on GT downbeats; with it, on Beat This!
downbeats (per-bar targets still come from the GT annotation either way).

Usage:
    python build_asap_one_perf.py --task-id N \
        --tsv data_processing/metadata/test_acpas102_perfs.tsv \
        --feature-folder workspace/feature.acpas102
"""
import os
import argparse
from hyperpyyaml import load_hyperpyyaml

from datasets.asap import ProcessASAP
from utilities import mkdirs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task-id', type=int, required=True)
    ap.add_argument('--tsv', required=True)
    ap.add_argument('--hparams', default='hparams/finetune.yaml')
    ap.add_argument('--test-list',
                    default='data_processing/metadata/test_asap.txt')
    # An empty train list keeps a hold-out piece in the test split even when
    # the repo's own train list happens to name it.
    ap.add_argument('--train-list',
                    default='data_processing/metadata/train_asap.txt')
    ap.add_argument('--bt-dir', default=None,
                    help='Beat This! annotations dir; omit for GT downbeats')
    ap.add_argument('--feature-folder', required=True)
    ap.add_argument('--asap-dir', default=os.environ.get('ASAP_DIR'),
                    help='ASAP checkout the tsv paths are relative to')
    ap.add_argument('--workdir', default='temp_workers/asap_perf',
                    help='Per-task scratch root; isolates temp/<split> writes')
    args = ap.parse_args()
    if not args.asap_dir:
        raise SystemExit('pass --asap-dir or set ASAP_DIR (see env.sh.example)')

    project_dir = os.getcwd()
    feature_folder_abs = os.path.abspath(args.feature_folder)
    bt_dir_abs = os.path.abspath(args.bt_dir) if args.bt_dir else None

    rows = [l.rstrip('\n').split('\t') for l in open(args.tsv) if l.strip()]
    if args.task_id >= len(rows):
        print(f'[skip] task_id {args.task_id} >= {len(rows)}')
        return
    score_name, perf, rel_wav = rows[args.task_id]
    folder = os.path.dirname(os.path.join(args.asap_dir, rel_wav))
    mode = 'bt' if bt_dir_abs else 'gt'
    print(f'[task {args.task_id}] {mode}: {score_name} / {perf}')

    with open(args.hparams) as fh:
        hparams = load_hyperpyyaml(fh, {})
    hparams['feature_folder'] = feature_folder_abs

    # Construct before chdir: ProcessASAP reads data_processing/metadata/*
    # through relative paths.
    process = ProcessASAP(hparams, bt_dir=bt_dir_abs,
                          test_list=args.test_list,
                          train_list=args.train_list,
                          recordings=[(score_name, perf)])

    task_workdir = os.path.join(project_dir, args.workdir, str(args.task_id))
    os.makedirs(task_workdir, exist_ok=True)
    os.chdir(task_workdir)

    for split in ['train', 'test']:
        mkdirs(f'temp/{split}')
        for sub in ['wav', 'midi', 'xml', 'kern', 'target',
                    'kern_upper', 'kern_lower', 'info']:
            mkdirs(f'{feature_folder_abs}/{split}/{sub}')

    unmatched = process.process_one(folder)
    if unmatched:
        print(f'[task {args.task_id}] unmatched: {unmatched}')
    print(f'[task {args.task_id}] done')


if __name__ == '__main__':
    main()
