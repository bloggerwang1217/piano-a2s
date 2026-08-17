#!/bin/bash
#SBATCH --job-name=asap-perf
#SBATCH --partition=compute
#SBATCH --array=0-101
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm_asap_perf/%a.out
#SBATCH --error=logs/slurm_asap_perf/%a.err

# One array task per recording. Defaults build the ACPAS-102 hold-out on GT
# downbeats; set BT_DIR to cut on Beat This! downbeats instead. Keep --array
# in step with the row count of TSV.

set -eo pipefail

# sbatch executes a spooled copy of this file, so BASH_SOURCE points into
# /var/spool and cannot locate env.sh; SLURM_SUBMIT_DIR is where sbatch ran.
# set -e above matters: without it a missing env.sh used to fall through to the
# system python.
ROOT="${PIANO_A2S_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}"
source "$ROOT/env.sh"

source "$CONDA_SH"
conda activate "$CONDA_ENV"

cd "$PROJECT_ROOT"
export PATH="$PROJECT_ROOT/humextra/bin:$PROJECT_ROOT/verovio/tools:$PATH"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

TSV=${TSV:-data_processing/metadata/test_acpas102_perfs.tsv}
TEST_LIST=${TEST_LIST:-data_processing/metadata/test_acpas102.txt}
FEATURE_FOLDER=${FEATURE_FOLDER:-workspace/feature.acpas102}
BT_DIR=${BT_DIR:-}

python build_asap_one_perf.py \
    --task-id $SLURM_ARRAY_TASK_ID \
    --tsv "$TSV" \
    --test-list "$TEST_LIST" \
    --feature-folder "$FEATURE_FOLDER" \
    ${BT_DIR:+--bt-dir "$BT_DIR"}
