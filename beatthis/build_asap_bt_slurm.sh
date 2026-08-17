#!/bin/bash
#SBATCH --job-name=asap-bt-build
#SBATCH --partition=compute
#SBATCH --array=0-24
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm_beatthis/build_%a.out
#SBATCH --error=logs/slurm_beatthis/build_%a.err

set -eo pipefail

# sbatch runs a spooled copy, so BASH_SOURCE cannot locate env.sh.
ROOT="${PIANO_A2S_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
source "$ROOT/env.sh"

source "$CONDA_SH"
conda activate "$CONDA_ENV"

cd "$PROJECT_ROOT"
export PATH="$PROJECT_ROOT/humextra/bin:$PROJECT_ROOT/verovio/tools:$PATH"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

python beatthis/build_asap_bt_one.py \
    --task-id $SLURM_ARRAY_TASK_ID \
    --hparams hparams/finetune.yaml \
    --bt-dir workspace/beatthis \
    --feature-folder workspace/feature.asap.beatthis
