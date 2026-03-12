#!/bin/bash
#SBATCH --job-name=cc-eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_eval_%j.out

set -euo pipefail

DATA_DIR=${DATA_DIR:-/shared/chinesechess}
GPU_ID=${GPU_ID:-0}

python ./cchess_alphazero/run.py eval \
  --data-dir "${DATA_DIR}" \
  --type local_torch \
  --gpu "${GPU_ID}" \
  --cluster-mode \
  --evaluator-poll-interval 120
