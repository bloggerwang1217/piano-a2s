#!/bin/bash
#SBATCH --job-name=beatthis
#SBATCH --partition=compute
#SBATCH --array=0-79
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=logs/slurm_beatthis/bt_%a.out
#SBATCH --error=logs/slurm_beatthis/bt_%a.err

set -eo pipefail

# sbatch runs a spooled copy, so BASH_SOURCE cannot locate env.sh.
ROOT="${PIANO_A2S_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
source "$ROOT/env.sh"

source "$CONDA_SH"
conda activate "$CONDA_ENV_BEATTHIS"

cd "$PROJECT_ROOT"

python beatthis/run_beatthis_one.py \
    --task-id $SLURM_ARRAY_TASK_ID \
    --tsv beatthis/asap_test_perfs.tsv \
    --out-dir workspace/beatthis
