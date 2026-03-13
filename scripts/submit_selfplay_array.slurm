#!/bin/bash
#SBATCH --job-name=cc-selfplay
#SBATCH --array=0-15
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_selfplay_%A_%a.out

set -euo pipefail

module load StdEnv/2023
module load python/3.12

cd /scratch/liuhon33/chinese_chess/chinese_chess_ai
source /scratch/liuhon33/chinese_chess/venvs/torch210_cu128/bin/activate

DATA_DIR=${DATA_DIR:-/scratch/liuhon33/chinese_chess/chinese_chess_ai/mydata}
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
  



