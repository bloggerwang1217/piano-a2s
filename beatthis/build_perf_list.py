"""Build the (piece, performance) list for ASAP test split → newline-delimited TSV.

Output: beatthis/asap_test_perfs.tsv  (col1=score_name, col2=performance,
col3=wav path relative to ASAP_DIR, so the list stays machine-independent)
"""
import os
import sys
import pandas as pd
from pathlib import Path

ASAP = os.environ.get('ASAP_DIR')
if not ASAP:
    sys.exit('ASAP_DIR is unset — source env.sh (see env.sh.example)')
META = Path(__file__).parent.parent / 'data_processing/metadata/test_asap.txt'
OUT = Path(__file__).parent / 'asap_test_perfs.tsv'

test_songs = [r['name'] for _, r in pd.read_csv(META).iterrows()]
rows = []
for s in test_songs:
    folder = os.path.join(ASAP, *s.split('#'))
    if not os.path.isdir(folder):
        print(f'MISSING folder: {folder}')
        continue
    for f in sorted(os.listdir(folder)):
        if f.endswith('.wav'):
            perf = f[:-4]
            rows.append((s, perf, os.path.join(*s.split('#'), f)))

with open(OUT, 'w') as fh:
    for s, p, w in rows:
        fh.write(f'{s}\t{p}\t{w}\n')

print(f'wrote {len(rows)} performances to {OUT}')
