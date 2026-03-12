#!/bin/bash
#SBATCH --job-name=cc-selfplay
#SBATCH --array=0-15
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_selfplay_%A_%a.out

set -euo pipefail

DATA_DIR=${DATA_DIR:-/shared/chinesechess}
GPU_ID=${GPU_ID:-0}

python ./cchess_alphazero/run.py self \
  --data-dir "${DATA_DIR}" \
  --type local_torch \
  --gpu "${GPU_ID}" \
  --cluster-mode \
  --worker-id "${SLURM_ARRAY_TASK_ID}" \
  --auto-reload-best \
  --reload-best-interval 300 \
  --safe-write-play-data
