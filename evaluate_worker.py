#!/usr/bin/env python3
"""
Slurm worker for MV2H evaluation.
Processes a slice of result files based on SLURM_ARRAY_TASK_ID.

Usage:
    python evaluate_worker.py --output-folder workspace/1234/pretrain.score \
                              --mv2h-bin MV2H/bin \
                              --split test \
                              --num-tasks 64
"""
import os
import signal
import logging
import argparse
import subprocess
from utilities import load, save, mkdirs
from data_processing.humdrum import get_xml_from_target

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_chunk(id, result_data, results_dir, task_dir, mv2h_bin, eval_script):
    """Process a single chunk. Returns status string."""
    pred_xml_path = f'{task_dir}/scores/pred/{id}_pred.xml'
    target_xml_path = f'{task_dir}/scores/target/{id}_target.xml'
    pred_midi_path = f'{task_dir}/midi/pred/{id}_pred.mid'
    target_midi_path = f'{task_dir}/midi/target/{id}_target.mid'
    mv2h_path = f'{results_dir}/mv2h/{id}_mv2h.json'

    if os.path.exists(mv2h_path):
        return 'cached'

    # Convert to XML/MIDI
    try:
        pred = get_xml_from_target(result_data['pred'])
        pred.write('musicxml', pred_xml_path)
        pred.write('midi', pred_midi_path)
        target = get_xml_from_target(load(result_data['target_path']))
        target.write('musicxml', target_xml_path)
        target.write('midi', target_midi_path)
    except Exception as e:
        logger.warning(f"conversion_error [{id}]: {type(e).__name__}: {e}")
        return 'conversion_error'

    # Run MV2H
    try:
        proc = subprocess.Popen(
            ['sh', eval_script, target_midi_path, pred_midi_path, mv2h_bin],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True
        )
        try:
            output, stderr = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait()
            logger.warning(f"timeout [{id}]")
            return 'timeout'
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, proc.args, output, stderr)
    except ValueError as e:
        logger.warning(f"mv2h_error [{id}]: {type(e).__name__}: {e}")
        return 'mv2h_error'
    except subprocess.CalledProcessError as e:
        logger.warning(f"mv2h_error [{id}]: Process returned {e.returncode}")
        return 'mv2h_error'

    # Parse output
    try:
        result_list = output.decode('utf-8').splitlines()[-6:]
        mv2h_result = dict([tuple(item.split(': ')) for item in result_list])
        for key, value in mv2h_result.items():
            mv2h_result[key] = float(value)
    except Exception as e:
        logger.warning(f"parse_error [{id}]: {type(e).__name__}: {e}")
        return 'parse_error'

    if 'MV2H' not in mv2h_result:
        logger.warning(f"parse_error [{id}]: MV2H key missing, got keys: {list(mv2h_result.keys())}")
        return 'parse_error'

    if mv2h_result['MV2H'] == 0:
        logger.warning(f"zero_mv2h [{id}]")
        return 'zero_mv2h'

    save(mv2h_result, mv2h_path)
    return 'success'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-folder', required=True)
    parser.add_argument('--mv2h-bin', required=True)
    parser.add_argument('--split', default='test')
    parser.add_argument('--num-tasks', type=int, default=64)
    args = parser.parse_args()

    task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))

    # Resolve all paths to absolute BEFORE chdir
    project_dir = os.getcwd()
    results_dir = os.path.abspath(f'{args.output_folder}/results')
    mv2h_bin = os.path.abspath(args.mv2h_bin)
    eval_script = os.path.abspath('evaluate_midi_mv2h.sh')

    # Create per-task output dirs
    task_dir = f'{results_dir}/tasks/{task_id}'
    for d in ['scores/pred', 'scores/target', 'midi/pred', 'midi/target']:
        mkdirs(f'{task_dir}/{d}')
    mkdirs(f'{results_dir}/mv2h')

    # CRITICAL: get_xml_from_target() uses a hardcoded 'temp/' directory for
    # intermediate krn/xml files. Each task needs its own working directory
    # so that 'temp/' is isolated per task.
    task_workdir = os.path.join(project_dir, f'temp_workers/{task_id}')
    os.makedirs(task_workdir, exist_ok=True)
    os.chdir(task_workdir)

    # Get this task's slice
    all_files = sorted(os.listdir(f'{results_dir}/{args.split}'))
    chunk_size = (len(all_files) + args.num_tasks - 1) // args.num_tasks
    start = task_id * chunk_size
    end = min(start + chunk_size, len(all_files))
    my_files = all_files[start:end]

    logger.info(f"Task {task_id}: processing {len(my_files)} files [{start}:{end}] of {len(all_files)}")

    from collections import Counter
    status_counts = Counter()
    errors = []

    for f in my_files:
        id = f[:-5]
        result_data = load(os.path.join(f'{results_dir}/{args.split}', f))
        status = process_chunk(id, result_data, results_dir, task_dir, mv2h_bin, eval_script)
        status_counts[status] += 1
        if status not in ('success', 'cached'):
            errors.append(id)

    # Save per-task summary
    summary = {'task_id': task_id, 'status_counts': dict(status_counts), 'errors': errors}
    save(summary, f'{results_dir}/slurm_task_{task_id}.json')

    logger.info(f"Task {task_id} done: {dict(status_counts)}")


if __name__ == '__main__':
    main()
