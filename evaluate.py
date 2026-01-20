import os
import signal
import logging
from collections import Counter
from tqdm import tqdm
import numpy as np
from utilities import load, save, mkdirs
from data_processing.humdrum import get_xml_from_target
from hyperpyyaml import load_hyperpyyaml
import subprocess

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_mv2h_from_test(output_folder, split, mv2h_bin):
    results_dir = f'{output_folder}/results'
    mkdirs(f'{results_dir}/mv2h')
    for dir in ['scores', 'midi']:
        for sub_dir in ['pred', 'target']:
            mkdirs(f'{results_dir}/{dir}/{sub_dir}')

    status_counts = Counter()
    errors = []
    result_files = os.listdir(f'{results_dir}/{split}')

    for result in tqdm(result_files):
        id = result[:-5]
        pred_xml_path = f'{results_dir}/scores/pred/{id}_pred.xml'
        target_xml_path = f'{results_dir}/scores/target/{id}_target.xml'
        pred_midi_path = f'{results_dir}/midi/pred/{id}_pred.mid'
        target_midi_path = f'{results_dir}/midi/target/{id}_target.mid'
        mv2h_path = f'{results_dir}/mv2h/{id}_mv2h.json'

        if os.path.exists(mv2h_path):
            status_counts['success'] += 1
            continue

        result = load(os.path.join(f'{results_dir}/{split}', result))

        # Convert xml to midi
        try:
            pred = get_xml_from_target(result['pred'])
            pred.write('musicxml', pred_xml_path)
            pred.write('midi', pred_midi_path)
            target = get_xml_from_target(load(result['target_path']))
            target.write('musicxml', target_xml_path)
            target.write('midi', target_midi_path)
        except Exception as e:
            logger.warning(f"conversion_error [{id}]: {type(e).__name__}: {e}")
            status_counts['conversion_error'] += 1
            errors.append(id)
            continue

        try:
            # Use Popen with process group to properly kill Java subprocesses on timeout
            proc = subprocess.Popen(
                ['sh', 'evaluate_midi_mv2h.sh', target_midi_path, pred_midi_path, mv2h_bin],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True  # Create new process group
            )
            try:
                output, stderr = proc.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                # Kill entire process group (sh + all java children)
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                proc.wait()  # Reap the process
                logger.warning(f"timeout [{id}]: MV2H evaluation timeout")
                status_counts['timeout'] += 1
                errors.append(id)
                continue
            if proc.returncode != 0:
                raise subprocess.CalledProcessError(proc.returncode, proc.args, output, stderr)
        except ValueError as e:
            logger.warning(f"mv2h_error [{id}]: {type(e).__name__}: {e}")
            status_counts['mv2h_error'] += 1
            errors.append(id)
            continue
        except subprocess.CalledProcessError as e:
            logger.warning(f"mv2h_error [{id}]: Process returned {e.returncode}")
            status_counts['mv2h_error'] += 1
            errors.append(id)
            continue

        try:
            result_list = output.decode('utf-8').splitlines()[-6:]
            result = dict([tuple(item.split(': ')) for item in result_list])
            for key, value in result.items():
                result[key] = float(value)
        except Exception as e:
            logger.warning(f"parse_error [{id}]: {type(e).__name__}: {e}")
            status_counts['parse_error'] += 1
            errors.append(id)
            continue

        if result['MV2H'] == 0:
            logger.warning(f"zero_mv2h [{id}]: MV2H score is 0 (likely MIDI read error)")
            status_counts['zero_mv2h'] += 1
            errors.append(id)
            continue

        status_counts['success'] += 1
        save(result, mv2h_path)

    error_path = f'{results_dir}/errors.txt'
    with open(error_path, 'w') as f:
        for error in errors:
            f.write(error + '\n')

    # Log summary
    total = len(result_files)
    logger.info(f"=== Processing Summary ===")
    logger.info(f"Total chunks: {total}")
    for status, count in status_counts.most_common():
        logger.info(f"  {status}: {count} ({count/total*100:.1f}%)")

    return status_counts, total

def summarize_syn_mv2h(results_dir, composer='all', soundfont='all', test_split='all'):
    assert composer in ['all', 'score', 'Bach', 'Mozart', 'Chopin']
    assert soundfont in ['all', 'Upright', 'Salamander', 'YDP']
    assert test_split in ['all', 'musesyn', 'humsyn']
    mv2h_folder = f'{results_dir}/results/mv2h'
    keys = ['Multi-pitch', 'Voice', 'Meter', 'Value', 'Harmony', 'MV2H']
    mv2h_metrics = {}
    for key in keys:
        mv2h_metrics[key] = 0
    n = 0
    for mv2h_file in tqdm(os.listdir(mv2h_folder)):
        id = mv2h_file[:-5]
        v, chunk_id, sf = id.split('~')

        # Only consider the specified composer and soundfont
        skip = False
        for i, c in enumerate(['score', 'Bach', 'Mozart', 'Chopin']):
            if composer == c and int(v) != i:
                skip = True
                break
        for s in ['Upright', 'Salamander', 'YDP']:
            if soundfont == s and sf[0] != s[0]:
                skip = True
                break
        if test_split == 'musesyn' and chunk_id[0].islower():
            skip = True
        if test_split == 'humsyn' and chunk_id[0].isupper():
            skip = True
        if skip: continue

        mv2h_path = os.path.join(mv2h_folder, mv2h_file)
        mv2h = load(mv2h_path)
        for key in keys:
            mv2h_metrics[key] += (mv2h[key] - mv2h_metrics[key]) / (n + 1)
        n += 1
    print(mv2h_metrics)
    print((mv2h_metrics['Multi-pitch'] + mv2h_metrics['Voice'] + mv2h_metrics['Value'] + mv2h_metrics['Harmony']) / 4)

def summarize_asap_mv2h(results_dir):
    mv2h_folder = f'{results_dir}/results/mv2h'
    keys = ['Multi-pitch', 'Voice', 'Meter', 'Value', 'Harmony', 'MV2H']
    mv2h_metrics = {}
    for key in keys:
        mv2h_metrics[key] = 0
    n = 0
    for mv2h_file in tqdm(os.listdir(mv2h_folder)):
        mv2h_path = os.path.join(mv2h_folder, mv2h_file)
        mv2h = load(mv2h_path)
        for key in keys:
            mv2h_metrics[key] += (mv2h[key] - mv2h_metrics[key]) / (n + 1)
        n += 1
    print(mv2h_metrics)
    print((mv2h_metrics['Multi-pitch'] + mv2h_metrics['Voice'] + mv2h_metrics['Value'] + mv2h_metrics['Harmony']) / 4)


def summarize_results(output_folder, status_counts, total_chunks):
    """
    Display evaluation summary and MV2H scores.

    Two calculation methods:
    1. Zeng method: Only successful samples, no Meter -> (Multi-pitch + Voice + Value + Harmony) / 4
    2. Strict method: All samples, failures count as 0, calculate full MV2H
    """
    mv2h_folder = f'{output_folder}/results/mv2h'
    keys = ['Multi-pitch', 'Voice', 'Meter', 'Value', 'Harmony', 'MV2H']
    keys_no_meter = ['Multi-pitch', 'Voice', 'Value', 'Harmony']

    # Collect scores from all successful samples
    mv2h_scores = {key: [] for key in keys}
    for mv2h_file in os.listdir(mv2h_folder):
        if not mv2h_file.endswith('.json'):
            continue
        mv2h_path = os.path.join(mv2h_folder, mv2h_file)
        mv2h = load(mv2h_path)
        for key in keys:
            mv2h_scores[key].append(mv2h[key])

    n_success = len(mv2h_scores['Multi-pitch'])
    n_failed = total_chunks - n_success

    print(f"\n{'='*50}")
    print(f"=== Zeng Pipeline Evaluation Summary ===")
    print(f"{'='*50}")
    print(f"Total chunks: {total_chunks}")
    for status, count in status_counts.most_common():
        print(f"  {status}: {count} ({count/total_chunks*100:.1f}%)")

    # === Zeng method: Only successful samples, no Meter ===
    print(f"\n{'='*50}")
    print(f"=== MV2H Scores - Zeng Method (exclude failures, no Meter, n={n_success}) ===")
    print(f"{'='*50}")

    zeng_metrics = {}
    for key in keys:
        if mv2h_scores[key]:
            zeng_metrics[key] = np.mean(mv2h_scores[key]) * 100
        else:
            zeng_metrics[key] = 0.0
        print(f"  {key}: {zeng_metrics[key]:.2f}%")

    # Zeng's MV2H (no Meter)
    zeng_mv2h = np.mean([zeng_metrics[k] for k in keys_no_meter])
    print(f"  MV2H (no Meter): {zeng_mv2h:.2f}%")

    # === Strict method: Failures count as 0 ===
    print(f"\n{'='*50}")
    print(f"=== MV2H Scores - Strict Method (failures as 0, n={total_chunks}) ===")
    print(f"{'='*50}")

    strict_metrics = {}
    for key in keys:
        # Sum of successful samples + 0 for failed samples
        total_score = sum(mv2h_scores[key])
        strict_metrics[key] = (total_score / total_chunks) * 100
        print(f"  {key}: {strict_metrics[key]:.2f}%")

    # Strict MV2H (no Meter, for fair comparison with Zeng)
    strict_mv2h_no_meter = np.mean([strict_metrics[k] for k in keys_no_meter])
    print(f"  MV2H (no Meter): {strict_mv2h_no_meter:.2f}%")

    # Strict full MV2H (with Meter)
    strict_mv2h_full = np.mean([strict_metrics[k] for k in keys])
    print(f"  MV2H (full): {strict_mv2h_full:.2f}%")

    return {
        'zeng': zeng_metrics,
        'zeng_mv2h': zeng_mv2h,
        'strict': strict_metrics,
        'strict_mv2h_no_meter': strict_mv2h_no_meter,
        'strict_mv2h_full': strict_mv2h_full,
        'n_success': n_success,
        'n_failed': n_failed,
        'total': total_chunks
    }

def summarize_WER_and_F1(results_dir):
    folder = f'{results_dir}/results/test'
    keys = ['wer_upper', 'wer_lower', 'key_f1', 'time_f1']
    metrics = {}
    for key in keys:
        metrics[key] = 0
    i = 0
    for result_file in tqdm(os.listdir(folder)):
        result_path = os.path.join(folder, result_file)
        result = load(result_path)
        for key in keys:
            metrics[key] += (result[key] - metrics[key]) / (i + 1)
        i += 1
    metrics['wer'] = (metrics['wer_upper'] + metrics['wer_lower']) / 2
    print(metrics)

def get_ER(results_dir):
    pred_scores_folder = f'{results_dir}/results/scores/pred'
    target_scores_folder = f'{results_dir}/results/scores/target'
    mv2h_folder = f'{results_dir}/results/mv2h'
    files = os.listdir(mv2h_folder)
    files = [file[:-10] for file in files if file.endswith('.json')]
    ers = np.zeros(11)
    i = 0
    for file in tqdm(files):
        try:
            pred_path = os.path.join(pred_scores_folder, file + '_pred')
            target_path = os.path.join(target_scores_folder, file + '_target')
            os.system(f'./MUSTER/evaluate_XML_voicePlus.sh {pred_path} {target_path} ER >/dev/null 2>&1')
            current_er = load('ER.txt')
            current_er = current_er[0].split(',')[12].split('\t')
            current_er = np.array([float(er) for er in current_er[1:]])
            if current_er.any() is np.nan: continue
        except Exception as e:
            continue
        for j in range(11):
            ers[j] += current_er[j]
            if ers.any() is np.nan: 
                print(ers)
                print(current_er)
        i += 1
    ers /= i
    print(ers)
    print(i)
    
    # Delete non-xml files
    for pred_file in os.listdir(pred_scores_folder):
        if not pred_file.endswith('.xml'):
            os.remove(os.path.join(pred_scores_folder, pred_file))
    for target_file in os.listdir(target_scores_folder):
        if not target_file.endswith('.xml'):
            os.remove(os.path.join(target_scores_folder, target_file))

if __name__ == '__main__':
    with open('hparams/finetune.yaml') as fin:
        hparams = load_hyperpyyaml(fin, {})
    pretrain_output_folder = hparams['pretrained_output_folder']
    finetune_output_folder = hparams['output_folder']
    mv2h_bin = hparams['mv2h_bin']

    # Get mv2h for test set
    # Only process if pretrain results exist
    if os.path.exists(os.path.join(pretrain_output_folder, 'results', 'test')):
        get_mv2h_from_test(pretrain_output_folder, 'test', mv2h_bin)
        summarize_syn_mv2h(pretrain_output_folder, composer='all', soundfont='all', test_split='all')

    # Always process finetune results
    status_counts, total = get_mv2h_from_test(finetune_output_folder, 'test', mv2h_bin)
    summarize_results(finetune_output_folder, status_counts, total)
