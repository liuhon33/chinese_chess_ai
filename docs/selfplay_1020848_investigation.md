# Self-play slowdown in Slurm job 1020848

## Confirmed cause

`CChessPlayer.sender()` slept for 1 ms while holding `q_lock` when its
prediction queue was empty. MCTS producers need that same lock in
`expand_and_evaluate()` to enqueue work. On Linux, the polling sender repeatedly
reacquired the lock and starved those producers. Short sleeps accumulated into
hundreds of seconds per move. Windows also suffered, but much less severely.

On `trig0021`, live inspection of array task `1020848_1` (Slurm job `1020967`)
showed:

- Parent process `2240947`: about 0.7% CPU; H100 utilization sampled at 0%.
- Worker `2240982`: 16 search threads blocked on the same native lock
  (`0xc295dc0`), while its sender thread `2241176` was in `clock_nanosleep`.
- GDB identified the sleeping Python frame as `sender` in `agent/player.py`.
- No cgroup CPU throttling (`nr_throttled=0`).
- Normalized source hashes matched the Windows checkout before the fix.

The initial 96-Torch-threads/8-allocated-CPUs suspicion was not the cause of
this stall: the successful H100 trial retained 96 Torch threads, Linux fork,
the same 256-filter/7-block model, 200 simulations, and two self-play workers.

## Change

Release the prediction queue lock before releasing `run_lock` and sleeping on
an empty queue. Policy/value evaluation, search settings, batching, and the
one-batch-in-flight synchronization are unchanged.

`test_self_play_sender.py` verifies that a producer can enqueue while the sender
sleeps, then verifies that the queued request is sent. It also checks the existing
256-position batch limit and ordering. The first test failed before the fix and
passed after it.

## Measurements (2026-10-03)

| Run | Logged moves | Median seconds/move | Maximum |
| --- | ---: | ---: | ---: |
| Copied job 1020848, all 16 tasks | 182 | 370.05 | 1257.6 |
| Copied task 1020848_1 | 12 | 222.3 | 951.2 |
| Windows RTX 4090, before | 30 | 1.05 | 2.1 |
| Windows RTX 4090, fixed | 36 | 0.3 | 0.5 |
| H100 on trig0021, isolated fixed trial | 854 | 0.1 | 0.5 |

Timings are rounded to 0.1 seconds by existing logs. The trials are stochastic
self-play, not identical move sequences. Windows used an isolated fresh model of
the same architecture; the H100 trial copied the exact current cluster checkpoint
(digest `ef48a6d16a35c60dc61038f316f72582c2b6a86eb454353c153e255c2d4bd191`).
The 70-second H100 trial ran inside an existing allocation via SSH to the compute
node, with its own source/data directory and without stopping the original jobs.
Only the sender change differed from the original source.

## Validation

Windows commands used Conda `pytorch_learn`:

```powershell
conda activate pytorch_learn
python validation_local_torch_perf_1020848/run_cli.py baseline
python validation_local_torch_perf_1020848/run_cli.py fixed
python -m unittest cchess_alphazero.tests.test_self_play_sender cchess_alphazero.tests.test_torch_backend cchess_alphazero.tests.test_self_play_history_mode -v
```

The CLI harness launches the real `run.py self` command with the Slurm flags,
using isolated data, and stops only its subprocess tree after at least 30 moves.
All eight focused tests passed. Inference profiling on the 4090 measured about
2 ms/batch at both 1 and 96 explicitly configured Torch threads.

The cluster has no Conda installation. Its trial used the deployed
`torch210_cu128` virtual environment from the submission script. The exact
cluster launch is saved in the evidence's `run_trial.sh`.

Local evidence: `validation_local_torch_perf_1020848/`, including Windows logs,
inference profiles, summaries, and `cluster_evidence/` with the H100 log and
native stack captures. Cluster evidence:
`/scratch/liuhon33/chinese_chess/chinese_chess_ai/validation_perf_1020848/`.

The fix is applied to both local and cluster source. A backup of the original
cluster source is in the cluster evidence directory as `player_before.py`.

## Restart and production verification

The user approved restarting all 16 tasks. Trillium rejected
`scontrol requeue 1020848` with `Requested operation is presently disabled`.
Submitted the unchanged `scripts/submit_selfplay_array.slurm` as replacement
job **1021273**, confirmed acceptance, then cancelled **1020848**.

At 13:19:41 America/New_York, all 16 replacement tasks were running. Their logs
contained **20,784 moves**, a **0.1-second median**, **112 completed games**,
and **5 data-file flushes**, with no tracebacks. Three published play-data files
were opened and parsed successfully (833, 796, and 736 records). The original
array was no longer running. Details are saved in
`cluster_evidence/restarted_array_summary.json`.
