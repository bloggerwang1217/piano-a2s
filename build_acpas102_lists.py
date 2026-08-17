"""Derive the ACPAS-102 hold-out lists from the ACPAS subset_R metadata.

ACPAS rows are per recording, while this repo's split files are per piece
folder, so both are written:

    data_processing/metadata/test_acpas102.txt       46 piece folders
    data_processing/metadata/test_acpas102_perfs.tsv 102 recordings

The recording list is the authoritative one — a piece folder holds more
performances than the hold-out selects, so driving the pipeline off the
folder list alone would evaluate recordings that are not in the benchmark.

Usage:
    python build_acpas102_lists.py --acpas-metadata <ACPAS>/metadata_R.csv \
                                   [--asap-dir <asap checkout>]
"""
import argparse
import csv
import os
import sys
from pathlib import Path

META_DIR = Path(__file__).parent / 'data_processing' / 'metadata'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--acpas-metadata', required=True,
                    help='ACPAS subset_R metadata_R.csv')
    ap.add_argument('--asap-dir', default=os.environ.get('ASAP_DIR'),
                    help='ASAP checkout, for its metadata.csv and existence checks')
    args = ap.parse_args()
    if not args.asap_dir:
        sys.exit('pass --asap-dir or set ASAP_DIR (see env.sh.example)')
    asap = Path(args.asap_dir)

    with open(args.acpas_metadata) as fh:
        hold_out = [r for r in csv.DictReader(fh)
                    if r['source'] == 'ASAP' and r['split'] == 'test']

    # ACPAS names a recording by its path in the ASAP checkout; ASAP's own
    # metadata maps that path to the piece folder.
    with open(asap / 'metadata.csv') as fh:
        by_midi = {r['midi_performance']: r for r in csv.DictReader(fh)}

    rows, missing = [], []
    for entry in hold_out:
        key = entry['performance_MIDI_external'].replace('{ASAP}/', '')
        asap_row = by_midi.get(key)
        if asap_row is None:
            missing.append(key)
            continue
        folder = asap_row['folder']
        perf = Path(key).stem
        wav = f'{folder}/{perf}.wav'
        if not (asap / wav).is_file():
            missing.append(wav)
            continue
        rows.append((folder.replace('/', '#'), perf, wav))

    if missing:
        sys.exit('unresolved ACPAS entries:\n  ' + '\n  '.join(missing))

    rows.sort()
    pieces = sorted({score for score, _, _ in rows})

    piece_path = META_DIR / 'test_acpas102.txt'
    with open(piece_path, 'w') as fh:
        fh.write('name\n')
        for piece in pieces:
            fh.write(f'{piece}\n')

    perf_path = META_DIR / 'test_acpas102_perfs.tsv'
    with open(perf_path, 'w') as fh:
        for score, perf, wav in rows:
            fh.write(f'{score}\t{perf}\t{wav}\n')

    print(f'wrote {len(pieces)} pieces to {piece_path}')
    print(f'wrote {len(rows)} recordings to {perf_path}')


if __name__ == '__main__':
    main()
