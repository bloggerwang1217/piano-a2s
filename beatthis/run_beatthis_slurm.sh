#!/bin/bash
#SBATCH --job-name=beatthis
#SBATCH --partition=compute
#SBATCH --array=0-79
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=logs/slurm_beatthis/bt_%a.out
#SBATCH --error=logs/slurm_beatthis/bt_%a.err

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"

source "$CONDA_SH"
conda activate "$CONDA_ENV_BEATTHIS"

cd "$PROJECT_ROOT"

python beatthis/run_beatthis_one.py \
    --task-id $SLURM_ARRAY_TASK_ID \
    --tsv beatthis/asap_test_perfs.tsv \
    --out-dir workspace/beatthis
