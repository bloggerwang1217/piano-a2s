"""Process ONE ASAP performance with Beat This! downbeats.
One slurm task = one performance.

Reads beatthis/asap_test_perfs.tsv for the (score_name, perf, wav_path) list.
"""
import os
import argparse
from pathlib import Path
from hyperpyyaml import load_hyperpyyaml

from datasets.asap import ProcessASAP
from utilities import mkdirs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task-id', type=int, required=True)
    ap.add_argument('--tsv', default='beatthis/asap_test_perfs.tsv')
    ap.add_argument('--hparams', default='hparams/finetune.yaml')
    ap.add_argument('--bt-dir', required=True)
    ap.add_argument('--feature-folder', required=True)
    ap.add_argument('--asap-dir', default=os.environ.get('ASAP_DIR'),
                    help='ASAP checkout the tsv paths are relative to')
    args = ap.parse_args()
    if not args.asap_dir:
        raise SystemExit('pass --asap-dir or set ASAP_DIR (see env.sh.example)')

    project_dir = os.getcwd()
    bt_dir_abs = os.path.abspath(args.bt_dir)
    feature_folder_abs = os.path.abspath(args.feature_folder)

    rows = [l.rstrip('\n').split('\t') for l in open(args.tsv) if l.strip()]
    if args.task_id >= len(rows):
        print(f'[skip] task_id {args.task_id} >= {len(rows)}')
        return
    score_name, perf, rel_wav = rows[args.task_id]
    folder = os.path.dirname(os.path.join(args.asap_dir, rel_wav))
    print(f'[task {args.task_id}] {score_name} / {perf}')

    with open(args.hparams) as fh:
        hparams = load_hyperpyyaml(fh, {})
    hparams['feature_folder'] = feature_folder_abs

    # Build ProcessASAP from project_dir (its __init__ uses relative paths
    # for data_processing/metadata/*).
    p = ProcessASAP(hparams, bt_dir=bt_dir_abs)

    # Filter performances list to just the target one. Cleanest: monkey-patch
    # os.listdir for the one folder so process_one() only sees this perf's wav.
    real_listdir = os.listdir
    target_perf_files = [f'{perf}.wav', f'{perf}_annotations.txt', 'xml_score.musicxml']

    def filtered_listdir(path):
        if os.path.realpath(path) == os.path.realpath(folder):
            entries = real_listdir(path)
            return [e for e in entries if e in target_perf_files or not e.endswith('.wav')]
        return real_listdir(path)

    os.listdir = filtered_listdir

    task_workdir = os.path.join(project_dir, f'temp_workers/asap_bt_perf/{args.task_id}')
    os.makedirs(task_workdir, exist_ok=True)
    os.chdir(task_workdir)

    for split in ['train', 'test']:
        mkdirs(f'temp/{split}')
        for sub in ['wav', 'midi', 'xml', 'kern', 'target',
                    'kern_upper', 'kern_lower', 'info']:
            mkdirs(f"{feature_folder_abs}/{split}/{sub}")

    unmatched = p.process_one(folder)
    os.listdir = real_listdir
    if unmatched:
        print(f'[task {args.task_id}] unmatched: {unmatched}')
    print(f'[task {args.task_id}] done')


if __name__ == '__main__':
    main()
