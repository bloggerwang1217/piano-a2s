import os
import sys
sys.path.append('.')
import argparse
import random
import numpy as np
from tqdm import tqdm
from pedalboard import *
from midi2audio import FluidSynth
import soundfile as sf
import pyloudnorm as pyln
import warnings
import music21 as m21
import pretty_midi as pm
import pandas as pd
import multiprocessing
from pathlib import Path
from functools import partial
import logging
import subprocess
from collections import Counter
import torchaudio
from utilities import set_seed, load, save, mkdirs, get_VQT, MIDIProcess
from data_processing.humdrum import Kern, LabelsMultiple, sort_chords, sort_voices, process_voices

# Tool paths for external commands
PROJECT_ROOT = Path(__file__).parent.parent
HUMEXTRA_BIN = PROJECT_ROOT / 'humextra' / 'bin'
EXTRACTX_PATH = str(HUMEXTRA_BIN / 'extractx')
HUM2XML_PATH = str(HUMEXTRA_BIN / 'hum2xml')
TIEFIX_PATH = str(HUMEXTRA_BIN / 'tiefix')
TRANSPOSE_PATH = str(HUMEXTRA_BIN / 'transpose')
VEROVIO_PATH = str(PROJECT_ROOT / 'verovio' / 'tools' / 'verovio')
VEROVIO_DATA = str(PROJECT_ROOT / 'verovio' / 'data')

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(PROJECT_ROOT / 'logs' / f'render_pipeline_{__import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")}.log'), mode='w'),
    ]
)
logger = logging.getLogger('render')

warnings.filterwarnings("ignore")
set_seed(0)

feasible_trasposes = {-6: [0, '-m2', '-m3', 'M2', 'M3'],                             # exclude '-M2', '-M3', 'm2', 'm3'
                      -5: [0, '-m2', '-m3', 'M2', 'M3'],                             # exclude '-M2', '-M3', 'm2', 'm3'
                      -4: [0, '-m2', '-M2', '-m3', 'M2', 'M3'],                      # exclude '-M3', 'm2', 'm3'
                      -3: [0, '-m2', '-M2', '-m3', 'M2', 'm3', 'M3'],                # exclude '-M3' 'm2'
                      -2: [0, '-m2', '-M2', '-m3', '-M3', 'M2', 'm3', 'M3'],         # exclude 'm2'
                      -1: [0, '-m2', '-M2', '-m3', '-M3', 'm2', 'M2', 'm3', 'M3'],   # exclude nothing
                       0: [0, '-m2', '-M2', '-m3', '-M3', 'm2', 'M2', 'm3', 'M3'],   # exclude nothing
                       1: [0, '-m2', '-M2', '-m3', '-M3', 'm2', 'M2', 'm3', 'M3'],   # exclude nothing
                       2: [0, '-m2', '-M2', '-m3', '-M3', 'm2', 'M2', 'm3', 'M3'],   # exclude nothing
                       3: [0, '-M2', '-m3', '-M3', 'm2', 'M2', 'm3', 'M3'],          # exclude '-m2'
                       4: [0, '-M2', '-m3', '-M3', 'm2', 'M2', 'm3'],                # exclude '-m2', 'M3'
                       5: [0, '-M2', '-M3', 'm2', 'M2', 'm3'],                       # exclude '-m2', '-m3', 'M3'
                       6: [0, '-M2', '-M3', 'm2', 'm3'],                             # exclude '-m2', '-m3', 'M2', 'M3'
                       7: [0, '-M2', '-M3', 'm2', 'm3'],                             # exclude '-m2', '-m3', 'M2', 'M3'
                     }

def get_staff_spines(kern_path):
    kern = Kern(Path(kern_path))
    for line in kern.header:
        if line.startswith('**'):
            spines = line.split('\t')
            break
    indices = [i for i, x in enumerate(spines) if x == '**kern']
    return indices[0] + 1, indices[1] + 1

def split_single_score(score_path,
                       feature_folder,
                       labels,
                       time_sig_list,
                       logger,
                       split='train',
                       version=0,
                       chunk_size=5):
    """Returns a Counter of status outcomes for this score."""
    counts = Counter()
    # Get score name
    score_name = score_path.split('/')[-1].split('.')[0]

    # Make directory
    output_dir = os.path.join(feature_folder, f'{split}/{version}')
    mkdirs(output_dir)
    mkdirs(f'temp/{split}/{version}')
    for dir in ['midi', 'wav', 'kern', 'xml', 'target', 'kern_upper', 'kern_lower', 'info']:
        mkdirs(f'{output_dir}/{dir}')

    # Split into staffs and clean
    try:
        spine_lower, spine_upper = get_staff_spines(score_path)
        os.system(f'{EXTRACTX_PATH} -s {spine_lower} {score_path} > temp/{split}/{version}/lower.krn')
        os.system(f'{EXTRACTX_PATH} -s {spine_upper} {score_path} > temp/{split}/{version}/upper.krn')
        lower = Kern(Path(f'temp/{split}/{version}/lower.krn'))
        upper = Kern(Path(f'temp/{split}/{version}/upper.krn'))
        full = Kern(Path(score_path))
        for kern in [lower, upper, full]:
            cleaned, _ = kern.clean()
            if not cleaned:
                logger.warning(f"clean_failed [{score_name}]: Cannot clean kern")
                counts['clean_failed'] += 1
                return counts
    except Exception as e:
        logger.warning(f"staff_extract_error [{score_name}]: {type(e).__name__}: {e}")
        counts['staff_extract_error'] += 1
        return counts

    # Split into chunks
    chunks = []
    for i, kern in enumerate([lower, upper, full]):
        try:
            stride = 2 if split == 'train' else chunk_size
            kern_chunks = kern.split(chunk_size, stride)
        except Exception as e:
            logger.warning(f"split_error [{score_name}]: {type(e).__name__}: {e}")
            counts['split_error'] += 1
            return counts

        for j, chunk in enumerate(kern_chunks):
            # Save kern
            subfolder = ['kern_lower', 'kern_upper', 'kern'][i]
            chunk_path = f'{output_dir}/{subfolder}/{score_name}.{j}.krn'
            chunk.save(Path(chunk_path))
            try:
                # Fix ties with tiefix command
                process = subprocess.run([TIEFIX_PATH, chunk_path],
                                         capture_output=True,
                                         encoding='iso-8859-1')
                if (process.returncode != 0):
                    logger.debug(f"tiefix_error [{score_name}.{j}]: returncode={process.returncode}")
                    counts['tiefix_error'] += 1
                    continue

                chunk = Kern(data=process.stdout)
                chunk.save(Path(chunk_path))
                if i == 2:
                    chunks.append(f'{score_name}.{j}.krn')
            except Exception as e:
                logger.debug(f"kern_save_error [{score_name}.{j}]: {type(e).__name__}: {e}")
                counts['kern_save_error'] += 1
                continue

    for chunk in chunks:
        if not os.path.exists(f'{output_dir}/kern_lower/{chunk}') \
            or not os.path.exists(f'{output_dir}/kern_upper/{chunk}'):
            logger.debug(f"missing_staff [{chunk}]: kern_lower or kern_upper not found")
            counts['missing_staff'] += 1
            continue

        # Information
        info_path = f'{output_dir}/info/{chunk[:-4]}.json'
        info = {'score_name': score_name, 'chunk': chunk}

        # Read as music21 score
        xml_path = f'{output_dir}/xml/{chunk[:-4]}.xml'
        try:
            os.system(f'{HUM2XML_PATH} {output_dir}/kern/{chunk} >{xml_path}')
            m21_score = m21.converter.parse(xml_path).expandRepeats()
        except Exception as e:
            logger.debug(f"hum2xml_parse_error [{chunk}]: {type(e).__name__}: {e}")
            counts['hum2xml_parse_error'] += 1
            continue

        if len(m21_score.parts[0].getElementsByClass('Measure')) != chunk_size:
            logger.debug(f"wrong_measure_count [{chunk}]: expected {chunk_size}, got {len(m21_score.parts[0].getElementsByClass('Measure'))}")
            counts['wrong_measure_count'] += 1
            continue

        # Transpose
        try:
            original_key = m21_score.parts[0].getElementsByClass('Measure')[0].keySignature.sharps
            if split == 'train':
                # Transpose to random key with feasible transpose
                transpose = random.choice(feasible_trasposes[original_key])
                info['original_key'] = original_key
                info['transpose'] = transpose
                m21_score = m21_score.transpose(transpose)
                p_transpose_lower = \
                    subprocess.run([TRANSPOSE_PATH, '-t', transpose, f'{output_dir}/kern_lower/{chunk}'],
                                    capture_output=True,
                                    encoding='iso-8859-1')
                p_transpose_upper = \
                    subprocess.run([TRANSPOSE_PATH, '-t', transpose, f'{output_dir}/kern_upper/{chunk}'],
                                    capture_output=True,
                                    encoding='iso-8859-1')
                lower = Kern(data=p_transpose_lower.stdout)
                upper = Kern(data=p_transpose_upper.stdout)
                lower.save(Path(f'{output_dir}/kern_lower/{chunk}'))
                upper.save(Path(f'{output_dir}/kern_upper/{chunk}'))
            else:
                info['original_key'] = original_key
                info['transpose'] = 0
                lower = Kern(Path(f'{output_dir}/kern_lower/{chunk}'))
                upper = Kern(Path(f'{output_dir}/kern_upper/{chunk}'))
        except Exception as e:
            logger.debug(f"transpose_error [{chunk}]: {type(e).__name__}: {e}")
            counts['transpose_error'] += 1
            continue

        # Save transposed xml
        try:
            m21_score.write('musicxml', fp=xml_path)
        except Exception as e:
            logger.debug(f"xml_write_error [{chunk}]: {type(e).__name__}: {e}")
            counts['xml_write_error'] += 1
            continue

        # Save targets
        try:
            lower = process_voices(lower)
            upper = process_voices(upper)
        except Exception as e:
            logger.debug(f"process_voices_error [{chunk}]: {type(e).__name__}: {e}")
            counts['process_voices_error'] += 1
            continue
        if lower is False or upper is False:
            logger.debug(f"process_voices_failed [{chunk}]: returned False")
            counts['process_voices_failed'] += 1
            continue
        try:
            lower = sort_voices(sort_chords(lower))
            upper = sort_voices(sort_chords(upper))
        except Exception as e:
            logger.debug(f"sort_error [{chunk}]: {type(e).__name__}: {e}")
            counts['sort_error'] += 1
            continue
        if lower is False or upper is False:
            logger.debug(f"sort_failed [{chunk}]: returned False")
            counts['sort_failed'] += 1
            continue
        lower = lower.tosequence()
        upper = upper.tosequence()
        if lower is None or upper is None:
            logger.debug(f"tosequence_failed [{chunk}]: double sharps/flats/dots")
            counts['tosequence_failed'] += 1
            continue
        current_key, current_time = None, None
        try:
            target_path = f'{output_dir}/target/{chunk[:-4]}.pkl'
            if lower.startswith('=\n'): lower = lower[2:]
            if lower.endswith('\n='): lower = lower[:-2]
            if upper.startswith('=\n'): upper = upper[2:]
            if upper.endswith('\n='): upper = upper[:-2]
            lower, upper = lower.split('\n=\n'), upper.split('\n=\n')
            target = []
            for m in range(chunk_size):
                # Get key and time signature
                key_signature = m21_score.parts[0].getElementsByClass('Measure')[m].keySignature
                time_signature = m21_score.parts[0].getElementsByClass('Measure')[m].timeSignature
                current_key = key_signature.sharps if key_signature is not None else current_key
                current_time = time_signature.ratioString if time_signature is not None else current_time
                if current_time not in time_sig_list:
                    logger.debug(f"invalid_time_sig [{chunk}]: {current_time}")
                    counts['invalid_time_sig'] += 1
                    target = []
                    break
                if current_key < -6 or current_key > 7:
                    logger.debug(f"invalid_key_sig [{chunk}]: {current_key}")
                    counts['invalid_key_sig'] += 1
                    target = []
                    break
                target.append([current_key, current_time, labels.encode(lower[m]), labels.encode(upper[m])])
            if len(target) != chunk_size: continue
            save(target, target_path)
            save(info, info_path)
            counts['success'] += 1
        except Exception as e:
            logger.debug(f"target_save_error [{chunk}]: {type(e).__name__}: {e}")
            counts['target_save_error'] += 1
            continue

    return counts

def split_datasets(versions, feature_folder, target_splits=None):
    _logger = logging.getLogger('render.split')

    # Get score paths
    score_paths = []
    for kern_file in os.listdir('data_processing/kern'):
        score_paths.append(os.path.join('data_processing/kern', kern_file))
    _logger.info(f'Number of scores: {len(score_paths)}')

    # Get test and validation split
    test_songs = set([row['name'] for i, row in
                      pd.read_csv('data_processing/metadata/test_split.txt').iterrows()])
    val_songs = set([row['name'] for i, row in
                     pd.read_csv('data_processing/metadata/valid_split.txt').iterrows()])

    labels = LabelsMultiple(extended=True)
    time_sig_list = load('data_processing/metadata/time_signature_list.json')

    total_counts = Counter()

    # Split scores
    for v in versions:
        _logger.info(f'Version {v}')
        for i, score_path in tqdm(enumerate(score_paths), total=len(score_paths)):
            score_name = score_path.split('/')[-1].split('.')[0]
            if score_name in test_songs and v == 0:
                split = 'test'
            elif score_name in val_songs and v == 0:
                split = 'valid'
            elif score_name not in test_songs and score_name not in val_songs:
                split = 'train'
            else:
                continue

            # Filter by target_splits
            if target_splits and split not in target_splits:
                continue

            result = split_single_score(score_path,
                               feature_folder,
                               labels,
                               time_sig_list,
                               logger=_logger,
                               split=split,
                               version=v)
            if result:
                total_counts += result

    # Summary
    _logger.info(f"=== split_datasets summary ===")
    for status, count in total_counts.most_common():
        _logger.info(f"  {status}: {count}")
    return total_counts

def render_all_midi(versions, feature_folder, soundfont_folder, target_splits=None):
    _logger = logging.getLogger('render.midi2wav')
    train_sondfonts = ['TimGM6mb.sf2',
                       'FluidR3_GM.sf2',
                       'UprightPianoKW-20220221.sf2',
                       'SalamanderGrandPiano-V3+20200602.sf2']
    test_soundfonts = ['UprightPianoKW-20220221.sf2',
                       'SalamanderGrandPiano-V3+20200602.sf2',
                       'YDP-GrandPiano-20160804.sf2']
    dynamic_compression = Compressor(threshold_db=-1, ratio = 18, attack_ms=50)
    counts = Counter()

    all_splits = target_splits or ['train', 'valid', 'test']

    for split in [s for s in ['train', 'valid'] if s in all_splits]:
        folder = os.path.join(feature_folder, f'{split}')
        for v in versions:
            current_folder = os.path.join(folder, str(v))
            _logger.info(f'Now processing: {split}, {v}')
            if not os.path.exists(os.path.join(current_folder, 'midi')):
                _logger.warning(f"midi folder not found: {current_folder}/midi")
                continue
            midi_files = os.listdir(os.path.join(current_folder, 'midi'))
            for midi_file in tqdm(midi_files):
                if split == 'train':
                    soundfont = random.choice(train_sondfonts)
                else:
                    soundfont = random.choice(test_soundfonts)
                midi_path = os.path.join(current_folder, 'midi', midi_file)
                wav_path = os.path.join(current_folder, 'wav', midi_file[:-4] + f'~{soundfont[:-4]}.wav')
                soundfont_path = os.path.join(soundfont_folder, soundfont)
                success = render_one_midi(FluidSynth(soundfont_path, sample_rate=44100),
                                dynamic_compression,
                                midi_path,
                                wav_path,
                                _logger)
                counts['success' if success else 'render_error'] += 1

    if 'test' in all_splits:
        for split in ['test']:
            folder = os.path.join(feature_folder, f'{split}')
            for v in versions:
                current_folder = os.path.join(folder, str(v))
                _logger.info(f'Now processing: {split}, {v}')
                if not os.path.exists(os.path.join(current_folder, 'midi')):
                    _logger.warning(f"midi folder not found: {current_folder}/midi")
                    continue
                midi_files = os.listdir(os.path.join(current_folder, 'midi'))
                for midi_file in tqdm(midi_files):
                    for soundfont in test_soundfonts:
                        midi_path = os.path.join(current_folder, 'midi', midi_file)
                        wav_path = os.path.join(current_folder, 'wav', midi_file[:-4] + f'~{soundfont[:-4]}.wav')
                        soundfont_path = os.path.join(soundfont_folder, soundfont)
                        success = render_one_midi(FluidSynth(soundfont_path, sample_rate=44100),
                                        dynamic_compression,
                                        midi_path,
                                        wav_path,
                                        _logger)
                        counts['success' if success else 'render_error'] += 1

    _logger.info(f"=== render_all_midi summary ===")
    for status, count in counts.most_common():
        _logger.info(f"  {status}: {count}")
    return counts

def render_one_midi(fs, dynamic_compression, midi_path, wav_path, _logger=None):
    """Returns True on success, False on failure."""
    # fs: FluidSynth object
    try:
        fs.midi_to_audio(midi_path, wav_path)
        data, rate = sf.read(wav_path)
        meter = pyln.Meter(rate) # Create BS.1770 meter
        if np.ndim(data) > 1:
            data = np.mean(data, axis=1) # Convert to mono

        data_copy = pyln.normalize.peak(data, -1.0)
        attempt = 0
        while meter.integrated_loudness(data_copy) < -20:
            loudness_normalized_audio = pyln.normalize.peak(data, -1.0)
            threshold = meter.integrated_loudness(loudness_normalized_audio) + 15
            if attempt % 3 == 2:
                dynamic_compression.threshold_db -= 1
                if dynamic_compression.threshold_db < threshold: break
            elif attempt % 3 == 1:
                dynamic_compression.attack_ms *= 0.7
                if dynamic_compression.attack_ms < 3: break
            else:
                dynamic_compression.ratio += 2
                if dynamic_compression.ratio > 34: break
            loudness_normalized_audio = np.array(loudness_normalized_audio)
            data_copy = dynamic_compression(loudness_normalized_audio, rate)
            data_copy = pyln.normalize.peak(data_copy, -1.0)
            attempt += 1

        dynamic_compression.threshold_db = -5
        dynamic_compression.attack_ms = 10
        dynamic_compression.ratio = 1
        attempt = 0

        data = data_copy
        data_copy = pyln.normalize.loudness(data, meter.integrated_loudness(data), -15)

        while data_copy.max() > 0.9 or data_copy.min() < -0.9:
            data_copy = pyln.normalize.loudness(data, meter.integrated_loudness(data), -15)
            if attempt % 3 == 2:
                dynamic_compression.threshold_db -= 0.5
                if dynamic_compression.threshold_db < -10: break
            elif attempt % 3 == 1:
                dynamic_compression.attack_ms *= 0.75
                if dynamic_compression.attack_ms < 1: break
            else:
                dynamic_compression.ratio += 1.5
                if dynamic_compression.ratio > 15: break
            loudness_normalized_audio = np.array(data_copy)
            data_copy = dynamic_compression(loudness_normalized_audio, rate)
            attempt += 1

        dynamic_compression.threshold_db = -1
        dynamic_compression.attack_ms = 50
        dynamic_compression.ratio = 18
        attempt = 0

        data = pyln.normalize.peak(data_copy, -1.0)

        sf.write(wav_path, data, rate)
        return True
    except Exception as e:
        if _logger:
            _logger.warning(f"render_error [{midi_path}]: {type(e).__name__}: {e}")
        return False

def xml_to_midi(versions, feature_folder, midi_syn='epr', target_splits=None):
    _logger = logging.getLogger('render.xml2midi')
    assert midi_syn in ['epr', 'score']
    train_composers = ['score', 'Bach', 'Balakirev', 'Beethoven',
                       'Brahms', 'Debussy', 'Glinka', 'Haydn',
                       'Liszt', 'Prokofiev', 'Rachmaninoff',
                       'Ravel', 'Schubert', 'Schumann', 'Scriabin']
    test_composers = ['score', 'Bach', 'Mozart', 'Chopin']
    counts = Counter()

    if midi_syn == 'epr':
        # Load virtuosoNet and work under virtuosoNet folder
        os.chdir('virtuosoNet')
        sys.path.append(os.getcwd())
        from virtuosoNet.model_run import load_file_and_generate_performance
        # We use 4 composers for test set, with each version corresponding to a composer
        for v in range(1, 4):
            os.system(f'cp -r {feature_folder}/valid/0/ {feature_folder}/valid/{v}')
            os.system(f'cp -r {feature_folder}/test/0/ {feature_folder}/test/{v}')

    all_splits = target_splits or ['train', 'test', 'valid']

    for split in [s for s in ['train', 'test', 'valid'] if s in all_splits]:
        for v in versions:
            if split != 'train':
                if midi_syn == 'epr' and v >= 4: continue
                if midi_syn == 'score' and v > 0: continue
            _logger.info(f'Now processing: {split}, {v}')
            folder = os.path.join(feature_folder, f'{split}', str(v))
            mkdirs(f'temp/{split}/{v}')
            target_dir = os.path.join(folder, 'target')
            if not os.path.exists(target_dir):
                _logger.warning(f"target folder not found: {target_dir}")
                continue
            target_files = os.listdir(target_dir)
            file_names = [file[:-4] for file in target_files]
            _logger.info(f"  {len(file_names)} target files found")
            for file_name in tqdm(file_names):
                xml_path = os.path.join(folder, 'xml', f'{file_name}.xml')
                midi_path = os.path.join(folder, 'midi', f'{file_name}.mid')
                info_path = os.path.join(folder, 'info', f'{file_name}.json')
                info = load(info_path)
                os.system(f'cp {xml_path} temp/{split}/{v}/xml.xml')
                if split == 'train':
                    composer = random.choice(train_composers) if midi_syn == 'epr' else 'score'
                else:
                    composer = test_composers[v] if midi_syn == 'epr' else 'score'
                info['composer'] = composer
                try:
                    if composer == 'score':
                        command = f'{VEROVIO_PATH} -r {VEROVIO_DATA} -f musicxml -t midi {xml_path} -o temp/{split}/{v}/temp.mid'
                        status = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                        if status.returncode != 0:
                            _logger.warning(f"verovio_error [{file_name}]: returncode={status.returncode}")
                            counts['verovio_error'] += 1
                            continue
                        elif status.stderr:
                            if 'Warning' in status.stderr or 'Error' in status.stderr:
                                _logger.warning(f"verovio_warning [{file_name}]: {status.stderr.strip()}")
                                counts['verovio_warning'] += 1
                                continue
                        midiprocess = MIDIProcess(f'temp/{split}/{v}/temp.mid', split)
                    else:
                        # EPR
                        load_file_and_generate_performance(path_name=f'temp/{split}/{v}/',
                                                        composer=composer)
                        midiprocess = MIDIProcess(f'test_result/{v}_by_isgn_z0.mid', split)
                    scaling, original_length = midiprocess.process(midi_path)
                    if scaling is not None:
                        info['scaling'] = scaling
                        info['original_length'] = original_length
                        save(info, info_path)
                        counts['success'] += 1
                    else:
                        _logger.warning(f"midi_process_null [{file_name}]: scaling is None")
                        counts['midi_process_null'] += 1
                except Exception as e:
                    _logger.warning(f"xml2midi_error [{file_name}]: {type(e).__name__}: {e}")
                    counts['xml2midi_error'] += 1
                    continue

    if midi_syn == 'epr':
        # Move back to the parent directory
        os.chdir('..')

    _logger.info(f"=== xml_to_midi summary ===")
    for status, count in counts.most_common():
        _logger.info(f"  {status}: {count}")
    return counts

def convert_xml_to_kern(xml_folder='data_processing/xml'):
    _logger = logging.getLogger('render.xml2kern')
    _logger.info('Converting MuseSyn xml files to kern files...')
    xml_files = os.listdir(xml_folder)
    counts = Counter()
    for xml_file in tqdm(xml_files):
        xml_path = os.path.join(xml_folder, xml_file)
        kern_path = os.path.join('data_processing/kern', xml_file.replace('.xml', '.krn'))
        if os.path.exists(kern_path):
            counts['skipped_exists'] += 1
            continue
        status = os.system(f'{VEROVIO_PATH} -r {VEROVIO_DATA} -f musicxml-hum -t hum {xml_path} -o {kern_path} >/dev/null 2>&1')
        if status == 0:
            counts['success'] += 1
        else:
            _logger.warning(f"xml2kern_error [{xml_file}]: exit code {status}")
            counts['error'] += 1
    _logger.info(f"=== convert_xml_to_kern summary ===")
    for status, count in counts.most_common():
        _logger.info(f"  {status}: {count}")
    return counts

def preprocess_kern():
    _logger = logging.getLogger('render.preprocess')
    _logger.info('Preprocessing kern files...')
    mkdirs('data_processing/temp')
    kern_folder = 'data_processing/kern'
    kern_files = os.listdir(kern_folder)
    counts = Counter()
    selected_chopin = set([row['name'] for i, row in pd.read_csv('data_processing/metadata/selected_chopin.txt').iterrows()])
    for kern_file in tqdm(kern_files):
        if kern_file.startswith('chopin'):
            if kern_file[:-4].split('#')[1] not in selected_chopin:
                os.remove(os.path.join(kern_folder, kern_file))
                _logger.debug(f"removed_chopin [{kern_file}]: not in selected list")
                counts['removed_chopin'] += 1
                continue
        elif kern_file.startswith('joplin'):
            kern_path = os.path.join(kern_folder, kern_file)
            if kern_file == 'joplin#school.krn':
                os.remove(kern_path)
                _logger.debug(f"removed_joplin [{kern_file}]: school.krn excluded")
                counts['removed_joplin'] += 1
                continue
            temp_xml_path = os.path.join('data_processing/temp', 'temp.xml')
            status = os.system(f'{HUM2XML_PATH} {kern_path} >{temp_xml_path}')
            if status != 0:
                os.remove(kern_path)
                _logger.warning(f"joplin_hum2xml_error [{kern_file}]: exit code {status}")
                counts['joplin_hum2xml_error'] += 1
                continue
            status = os.system(f'{VEROVIO_PATH} -r {VEROVIO_DATA} -f musicxml-hum -t hum {temp_xml_path} -o {kern_path} >/dev/null 2>&1')
            if status != 0:
                os.remove(kern_path)
                _logger.warning(f"joplin_verovio_error [{kern_file}]: exit code {status}")
                counts['joplin_verovio_error'] += 1
                continue
            counts['joplin_reformatted'] += 1
        else:
            counts['kept'] += 1
    _logger.info(f"=== preprocess_kern summary ===")
    for status, count in counts.most_common():
        _logger.info(f"  {status}: {count}")
    return counts

def prepare_spectrograms(versions, feature_folder, hparams, target_splits=None):
    _logger = logging.getLogger('render.spectrogram')
    counts = Counter()
    all_splits = target_splits or ['train', 'valid', 'test']
    for split in [s for s in ['train', 'valid', 'test'] if s in all_splits]:
        for v in versions:
            folder = os.path.join(feature_folder, f'{split}', str(v))
            if not os.path.exists(folder): continue
            _logger.info(f'Now processing: {split}, {v}')
            spectrogram_folder = os.path.join(folder, 'spectrogram')
            mkdirs(spectrogram_folder)
            wav_dir = os.path.join(folder, 'wav')
            if not os.path.exists(wav_dir):
                _logger.warning(f"wav folder not found: {wav_dir}")
                continue
            wav_files = os.listdir(wav_dir)
            _logger.info(f"  {len(wav_files)} wav files found")
            for wav_file in tqdm(wav_files):
                wav_path = os.path.join(folder, 'wav', wav_file)
                spectrogram_path = os.path.join(spectrogram_folder, wav_file[:-4] + '.npy')
                if os.path.exists(spectrogram_path):
                    counts['skipped_exists'] += 1
                    continue
                try:
                    # Get wav duration
                    waveform, sample_rate = torchaudio.load(wav_path)
                    duration = waveform.shape[1] / sample_rate
                    if duration > hparams['max_duration']:
                        _logger.debug(f"too_long [{wav_file}]: {duration:.1f}s > {hparams['max_duration']}s")
                        counts['too_long'] += 1
                        continue
                    # Get spectrogram
                    spectrogram = get_VQT(wav_path, hparams["VQT_params"])
                    save(spectrogram, spectrogram_path)
                    counts['success'] += 1
                except Exception as e:
                    _logger.warning(f"spectrogram_error [{wav_file}]: {type(e).__name__}: {e}")
                    counts['spectrogram_error'] += 1
                    continue
    _logger.info(f"=== prepare_spectrograms summary ===")
    for status, count in counts.most_common():
        _logger.info(f"  {status}: {count}")
    return counts

def clean_files(versions, feature_folder, target_splits=None):
    _logger = logging.getLogger('render.clean')
    time_sig_list = load('data_processing/metadata/time_signature_list.json')
    counts = Counter()
    all_splits = target_splits or ['train', 'valid', 'test']
    for split in [s for s in ['train', 'valid', 'test'] if s in all_splits]:
        for v in versions:
            folder = os.path.join(feature_folder, f'{split}', str(v))
            if not os.path.exists(folder): continue
            midi_dir = os.path.join(folder, 'midi')
            if not os.path.exists(midi_dir): continue
            midi_files = os.listdir(midi_dir)
            deleted = 0
            for midi in tqdm(midi_files):
                name = midi[:-4]
                midi_path = f'{folder}/midi/{midi}'
                target_path = f'{folder}/target/{name}.pkl'
                if not os.path.exists(target_path):
                    os.remove(midi_path)
                    _logger.debug(f"no_target [{name}]: target pkl not found")
                    counts['no_target'] += 1
                    deleted += 1
                    continue

                # Check if midi is out of range
                try:
                    midi_data = pm.PrettyMIDI(midi_path)
                except Exception as e:
                    _logger.warning(f"midi_parse_error [{name}]: {type(e).__name__}: {e}")
                    counts['midi_parse_error'] += 1
                    deleted += 1
                    continue
                duration = midi_data.get_end_time()
                if duration > 12:
                    os.remove(target_path)
                    os.remove(midi_path)
                    _logger.debug(f"too_long [{name}]: {duration:.1f}s")
                    counts['too_long'] += 1
                    deleted += 1
                    continue

                flag_to_delete = False
                for instrument in midi_data.instruments:
                    for note in instrument.notes:
                        if note.pitch < 21 or note.pitch > 108:
                            os.remove(target_path)
                            os.remove(midi_path)
                            _logger.debug(f"out_of_range [{name}]: pitch={note.pitch}")
                            counts['out_of_range'] += 1
                            deleted += 1
                            flag_to_delete = True
                            break
                if flag_to_delete:
                    continue

                # Check if key or time signature is out of range
                target = load(target_path)
                bad = False
                for measure in target:
                    key = measure[0]
                    time = measure[1]
                    if key < -6 or key > 7 or time not in time_sig_list:
                        os.remove(target_path)
                        os.remove(midi_path)
                        _logger.debug(f"invalid_sig [{name}]: key={key}, time={time}")
                        counts['invalid_sig'] += 1
                        deleted += 1
                        bad = True
                        break
                if not bad:
                    counts['kept'] += 1
            _logger.info(f'{split}, {v}: {deleted} files deleted, {len(midi_files) - deleted} kept')
    _logger.info(f"=== clean_files summary ===")
    for status, count in counts.most_common():
        _logger.info(f"  {status}: {count}")
    return counts

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-only', action='store_true',
                        help='Only generate test split (version 0)')
    args = parser.parse_args()

    # Make sure workspace is set correctly in hparams/pretrain.yaml before running
    hparams = load('hparams/pretrain.yaml')
    midi_syn = hparams['midi_syn'] # ['epr', 'score'] indicating how the midi files are synthesized
    feature_folder = hparams["feature_folder"]
    soundfont_folder = hparams["soundfont_folder"]

    if args.test_only:
        logger.info("=== TEST-ONLY MODE ===")
        target_splits = ['test']
        versions = [0]

        # Step 1: Convert MuseSyn XML files to Kern files
        logger.info("[Step 1/6] Converting XML to Kern...")
        convert_xml_to_kern()

        # Step 2: Preprocess kern files
        logger.info("[Step 2/6] Preprocessing kern files...")
        preprocess_kern()

        # Step 3: Split scores and cut into chunks (test only)
        logger.info("[Step 3/6] Splitting test scores into chunks...")
        split_datasets(versions, feature_folder=feature_folder, target_splits=target_splits)

        # Step 4: Convert xml to midi
        logger.info("[Step 4/6] Converting XML to MIDI...")
        xml_to_midi(versions, feature_folder=feature_folder, midi_syn=midi_syn, target_splits=target_splits)

        # Step 5: Clean files
        logger.info("[Step 5/6] Cleaning files...")
        clean_files(versions, feature_folder=feature_folder, target_splits=target_splits)

        # Step 6: Synthesize midi files
        logger.info("[Step 6/6] Synthesizing MIDI to WAV and preparing spectrograms...")
        render_all_midi(versions, feature_folder=feature_folder, soundfont_folder=soundfont_folder,
                        target_splits=target_splits)

        # Step 7: Prepare spectrograms
        logger.info("[Step 7/6] Preparing spectrograms...")
        prepare_spectrograms(versions, feature_folder=feature_folder, hparams=hparams,
                             target_splits=target_splits)

        logger.info("=== TEST-ONLY PIPELINE COMPLETE ===")
        # Print final file counts
        test_folder = os.path.join(feature_folder, 'test/0')
        for sub in ['target', 'midi', 'wav', 'spectrogram']:
            path = os.path.join(test_folder, sub)
            if os.path.exists(path):
                count = len(os.listdir(path))
                logger.info(f"  {sub}: {count} files")
            else:
                logger.info(f"  {sub}: NOT FOUND")
    else:
        # Original full pipeline
        # Convert MuseSyn XML files to Kern files
        convert_xml_to_kern()

        # Select certain Chopin scores, and reformat Joplin scores
        preprocess_kern()

        # Split scores into train, valid, test and cut into chunks
        logger.info('Splitting scores into train, valid, test and cutting into chunks...')
        partial_work = partial(split_datasets, feature_folder=feature_folder)
        with multiprocessing.Pool(processes=5) as pool:
            versions_list = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
            pool.map(partial_work, versions_list)

        # Convert xml to midi
        logger.info('Converting xml to midi...')
        xml_to_midi(range(10), feature_folder=feature_folder, midi_syn=midi_syn)

        # Remove files with invalid key or time signature or length > 12s
        logger.info('Cleaning files...')
        clean_files(range(10), feature_folder=feature_folder)

        # Synthesize midi files
        logger.info('Synthesizing midi files...')
        partial_work = partial(render_all_midi, feature_folder=feature_folder, soundfont_folder=soundfont_folder)
        with multiprocessing.Pool(processes=5) as pool:
            versions_list = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
            pool.map(partial_work, versions_list)

        # Prepare spectrograms
        logger.info('Preparing spectrograms...')
        partial_work = partial(prepare_spectrograms, feature_folder=feature_folder, hparams=load('hparams/pretrain.yaml'))
        with multiprocessing.Pool(processes=5) as pool:
            versions_list = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
            pool.map(partial_work, versions_list)
