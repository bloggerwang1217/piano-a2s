"""Process one ASAP test piece folder with Beat This! downbeats.
One slurm task = one piece (all its performances).

Usage: python build_asap_bt_one.py --task-id N --hparams hparams/finetune.yaml \
                                   --bt-dir workspace/beatthis \
                                   --feature-folder workspace/feature.asap.beatthis
"""
import os
import argparse
import pandas as pd
from hyperpyyaml import load_hyperpyyaml

from datasets.asap import ProcessASAP
from utilities import mkdirs


def get_test_piece_folders(asap_folder: str):
    test_songs = [r['name'] for _, r in
                  pd.read_csv('data_processing/metadata/test_asap.txt').iterrows()]
    folders = [os.path.join(asap_folder, *s.split('#')) for s in test_songs]
    return sorted([f for f in folders if os.path.isdir(f)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task-id', type=int, required=True)
    ap.add_argument('--hparams', default='hparams/finetune.yaml')
    ap.add_argument('--bt-dir', required=True)
    ap.add_argument('--feature-folder', required=True)
    args = ap.parse_args()

    project_dir = os.getcwd()

    # Resolve to absolute paths BEFORE any chdir
    bt_dir_abs = os.path.abspath(args.bt_dir)
    feature_folder_abs = os.path.abspath(args.feature_folder)

    with open(args.hparams) as fh:
        hparams = load_hyperpyyaml(fh, {})
    hparams['feature_folder'] = feature_folder_abs

    folders = get_test_piece_folders(hparams['asap_folder'])
    if args.task_id >= len(folders):
        print(f'[skip] task_id {args.task_id} >= {len(folders)}')
        return
    folder = folders[args.task_id]
    print(f'[task {args.task_id}] {folder}')

    # ProcessASAP.__init__ reads data_processing/metadata/* with relative paths,
    # so build it from the project dir first, then chdir to per-task workdir.
    p = ProcessASAP(hparams, bt_dir=bt_dir_abs)

    # Per-task working dir isolates temp/<split>/lower.krn writes.
    task_workdir = os.path.join(project_dir, f'temp_workers/asap_bt/{args.task_id}')
    os.makedirs(task_workdir, exist_ok=True)
    os.chdir(task_workdir)

    for split in ['train', 'test']:
        mkdirs(f'temp/{split}')
        for sub in ['wav', 'midi', 'xml', 'kern', 'target',
                    'kern_upper', 'kern_lower', 'info']:
            mkdirs(f"{feature_folder_abs}/{split}/{sub}")

    unmatched = p.process_one(folder)
    if unmatched:
        print(f'[task {args.task_id}] unmatched: {unmatched}')
    print(f'[task {args.task_id}] done')


if __name__ == '__main__':
    main()
