#!/bin/bash
#SBATCH --job-name=mv2h-eval
#SBATCH --partition=compute
#SBATCH --array=0-63
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/mv2h_%A_%a.out
#SBATCH --error=logs/slurm/mv2h_%A_%a.err

set -eo pipefail

# sbatch runs a spooled copy, so BASH_SOURCE cannot locate env.sh.
ROOT="${PIANO_A2S_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}"
source "$ROOT/env.sh"

source "$CONDA_SH"
conda activate "$CONDA_ENV"

cd "$PROJECT_ROOT"

OUTPUT_FOLDER=${OUTPUT_FOLDER:-workspace/1234/finetune.epr}

python evaluate_worker.py \
    --output-folder $OUTPUT_FOLDER \
    --mv2h-bin MV2H/bin \
    --split test \
    --num-tasks 64
