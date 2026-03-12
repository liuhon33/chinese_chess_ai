# Cluster Self-Play Pipeline

This repository now supports an opt-in shared-filesystem pipeline for Torch training without changing the default local workflow.

## Default local workflow

If you do not pass any new cluster flags, the existing Windows-friendly commands still work as before:

```powershell
python .\cchess_alphazero\run.py self --data-dir mydata --type local_torch
python .\cchess_alphazero\run.py opt --data-dir mydata --type local_torch
python .\cchess_alphazero\run.py eval --data-dir mydata --type local_torch
```

## Cluster mode design

All workers share one `--data-dir` on a shared filesystem.

- Many independent self-play jobs write `play_*.json` into `play_data/`.
- One optimizer polls `play_data/`, atomically claims stable files into `play_data/inflight/`, trains, then deletes or archives them.
- One optional evaluator polls the shared `next_generation` model, evaluates it against the current best model, and promotes it when it passes.
- Self-play workers can periodically reload the current best model without restart.

## New flags

These flags are opt-in. If you do not pass them, the code stays on the current behavior.

- `--cluster-mode`
- `--worker-id <id>`
- `--auto-reload-best`
- `--reload-best-interval <seconds>`
- `--safe-write-play-data`
- `--archive-consumed-data`
- `--optimizer-poll-interval <seconds>`
- `--evaluator-poll-interval <seconds>`

## Recommended cluster commands

Self-play worker:

```bash
python ./cchess_alphazero/run.py self \
  --data-dir /shared/chinesechess \
  --type local_torch \
  --cluster-mode \
  --worker-id ${SLURM_ARRAY_TASK_ID} \
  --auto-reload-best \
  --reload-best-interval 300 \
  --safe-write-play-data
```

Optimizer worker:

```bash
python ./cchess_alphazero/run.py opt \
  --data-dir /shared/chinesechess \
  --type local_torch \
  --cluster-mode \
  --archive-consumed-data \
  --optimizer-poll-interval 60
```

Evaluator worker:

```bash
python ./cchess_alphazero/run.py eval \
  --data-dir /shared/chinesechess \
  --type local_torch \
  --cluster-mode \
  --evaluator-poll-interval 120
```

## Shared-filesystem behavior

### Self-play

- `--cluster-mode` switches self-play filenames to globally unique names containing timestamp, host, worker id, pid, and a random token.
- `--safe-write-play-data` writes to a temp file first, then publishes the final JSON with `os.replace`.
- In cluster mode, self-play no longer prunes old files from the shared `play_data/` directory.

### Optimizer

- In cluster mode, the optimizer only considers stable `play_*.json` files.
- Claimed files are moved atomically into `play_data/inflight/` before loading.
- `--archive-consumed-data` moves consumed files into `trained/` instead of deleting them.

### Evaluator

- The optimizer publishes candidate weights through a ready-marker flow in `model/next_generation/`.
- The evaluator promotes the candidate to best using atomic replace of the destination files.
- Elo history is still written locally under `logs/elo_history.csv`.

## Operational notes

- Do not run multiple optimizer workers against the same `--data-dir` unless you accept duplicated training-control decisions. File claiming prevents duplicate file consumption, but the pipeline is still designed around one optimizer.
- `--safe-write-play-data` is strongly recommended for cluster self-play.
- The cluster scripts in `scripts/` are examples only. They are separate from the default local flow and pass all cluster flags explicitly.
