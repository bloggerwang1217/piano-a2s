#!/bin/bash
#SBATCH --job-name=asap-bt-perf
#SBATCH --partition=compute
#SBATCH --array=0-79
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm_beatthis/perf_%a.out
#SBATCH --error=logs/slurm_beatthis/perf_%a.err

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"

source "$CONDA_SH"
conda activate "$CONDA_ENV"

cd "$PROJECT_ROOT"
export PATH="$PROJECT_ROOT/humextra/bin:$PROJECT_ROOT/verovio/tools:$PATH"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

python beatthis/build_asap_bt_one_perf.py \
    --task-id $SLURM_ARRAY_TASK_ID \
    --tsv beatthis/asap_test_perfs.tsv \
    --hparams hparams/finetune.yaml \
    --bt-dir workspace/beatthis \
    --feature-folder workspace/feature.asap.beatthis
