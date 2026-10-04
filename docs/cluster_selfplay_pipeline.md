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

### Plotting recorded evaluations

Use the same Torch environment as the evaluator. The PNG writer requires Pillow;
if it is missing from an existing environment, install it with
`python -m pip install Pillow`. Then run from the repository root:

```bash
python tools/plot_elo_vs_games.py --data-dir testdata --type local_torch
```

This rebuilds `logs/elo_vs_games.png` from `logs/elo_history.csv`. The evaluator
also updates both files after each completed candidate evaluation. These log paths
are under the project log directory; `--data-dir` selects checkpoints and play data.
With no completed evaluations, the plot says "No Elo history recorded yet".
Self-play games and optimizer losses alone cannot supply Elo measurements.

Each point is the candidate's match Elo relative to the best model it faced,
computed from wins, losses, and draws. The reference changes after a promotion,
so this is not an absolute or cumulative rating of model strength.

### Following model progress

`logs/main.log` includes `MODEL_EVENT` entries from self-play, optimization,
and evaluation. Filter that timeline with:

```bash
grep -n 'MODEL_EVENT' logs/main.log
```

- `CANDIDATE_PUBLISHED`: the optimizer saved trained weights for evaluation.
- `OPTIMIZER_WAITING_FOR_EVALUATION`: training pauses until the evaluator accepts
  or rejects the candidate; this does not reload the old weights.
- `EVALUATION_STARTED`: identifies the candidate and current best, along with
  the number of evaluation games and simulations per move.
- `BEST_MODEL_PROMOTED`: `previous_best` and `current_best` identify the actual
  change in best weights. Includes the match score and promotion threshold.
- `CANDIDATE_REJECTED`: the current best stays the same; the optimizer can continue.
- `SELFPLAY_MODEL_RELOADED` / `OPTIMIZER_MODEL_RELOADED`: a process has loaded
  the promoted weights. Self-play polls at `--reload-best-interval`.

Model identifiers are checkpoint SHA-256 digests, not filenames (the filenames
stay the same when weights change). `BestModel unchanged (digest check only,
no reload)` means exactly that: a check, not a model copy or a training reset.

On POSIX systems, cluster file handlers lock each append and reopen inherited
descriptors after fork so multiple processes and NFS clients cannot overwrite
each other's log records. All writers must run the updated code. Host and PID
fields identify the process that wrote each record. Existing damaged log lines
cannot be repaired by updating the code.

### Deploying source updates

Push source commits from the development checkout, then use `git pull --ff-only`
on the cluster. Generated Slurm logs and diagnostic directories are ignored.
If the cluster has its own commits, inspect and preserve them before aligning
the branch; do not force-push over either history. Updating files does not update
already-running Python workers: restart the affected jobs to load new code.

The 2026-10-03 stalled candidate was published at 13:20 (98 optimizer steps).
Evaluator job 1020849 had started before the sender-lock fix and still used the
old code in memory. Its first comparison was still running hours later, so no
promotion decision existed and the optimizer correctly waited. Git pull also
failed because cluster commit `ab4e643` contained only generated logs while the
actual sender fix was uncommitted. That history was preserved on
`codex/cluster-before-sync-20261003` before synchronizing with GitHub.

- Do not run multiple optimizer workers against the same `--data-dir` unless you accept duplicated training-control decisions. File claiming prevents duplicate file consumption, but the pipeline is still designed around one optimizer.
- `--safe-write-play-data` is strongly recommended for cluster self-play.
- The cluster scripts in `scripts/` are examples only. They are separate from the default local flow and pass all cluster flags explicitly.
