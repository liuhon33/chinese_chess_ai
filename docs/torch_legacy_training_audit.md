# Torch / legacy training audit

Audited on 2026-10-03 against the last pre-migration repository revision,
`7f45b0c` (`96dde65^`), and its pinned Keras 2.0.8 / TensorFlow 1.3.0 sources.
The scope is the `self -> opt -> eval` workflow. Torch remains the runtime.

## Main findings

1. **Candidate publication reset the optimizer's progress.** `CChessModel.save`
   changes `model.digest` to the checkpoint it saves. Publishing a candidate
   therefore made the next best-model check compare the candidate digest against
   the unchanged best digest. The optimizer reloaded best and rebuilt SGD,
   discarding both the trained weights and momentum after every cycle. The legacy
   loop kept its training model across batches. Publication now preserves the
   last loaded best digest. A real best-model promotion still triggers reload.
   Reproduced with actual checkpoints in both local and cluster publication modes.

2. **`self --new` disabled all subsequent best-model reloads.** The initial random
   model could continue generating data even after another process promoted a
   trained candidate. Fresh initialization now finishes once a new best has been
   saved; subsequent local promotions can load. Startup still skips old models,
   stale candidates still fail the freshness check, and fresh runs do not fetch
   internet checkpoints. Repeated loads by a fresh evaluator no longer rebuild
   random best models.

3. **The supplied cluster logs show data starvation before training.** They are
   from March 12-13, 2026, use `testdata`, and contain no completed training or
   evaluation. This is separate evidence from the reproducible reset bug.

## Evidence in the supplied logs

| File | Observed evidence | What it establishes |
| --- | --- | --- |
| `E:/cc/opt.log` | Line 13: zero files; requires eight files. Zero epoch, training-metric, candidate-publication, or completed-cycle records through the following day. | This recorded optimizer never started training. |
| `E:/cc/eval.log` | Eight lines total; waits for `testdata/model/next_generation/next_generation_weight.h5`. No matches or promotion decisions. | These logs do not contain the reported repeated candidate defeats. |
| `E:/cc/main.log` | Four fresh initializations; 14 occurrences of `play game`, including incomplete/interleaved records; no `save play data` records. Examples: 36,039.5 seconds for 18.5 full moves, 66,881.6 seconds for 111.5 moves. | Self-play was extremely slow; the five-game per-process buffer could retain completed games for an entire job. |
| `E:/cc/main.log` | 389,878 NUL bytes out of 677,126 bytes, plus interleaved/truncated lines. | This is an incomplete/corrupted record; absence of a self-play message alone is not proof that an event never happened. |

The active best digest remains `ef48a6d16a35c60dc61038f316f72582c2b6a86eb454353c153e255c2d4bd191`
in these records. The exact cause of the multi-hour games is not established by
the logs: GPU assignment, CPU fallback, MCTS execution, resource contention, and
time spent in inference require runtime measurements. No speculative throughput
tuning was applied. New device/CPU-thread logging and an explicit CUDA fallback
warning make the next run easier to diagnose.

`local_torch.play_data.max_buffer_seconds = 300` now permits an early flush at a
completed-game boundary after that much time since the previous flush/start.
This is **not a timer interrupting an active game**. It prevents a worker whose
first completed game takes hours from withholding that game until game five.
Set it to zero to retain only the game-count trigger. The optimizer still waits
for the configured minimum number of files.

## Step-by-step comparison

| Stage | Comparison and result |
| --- | --- |
| Board rules and action labels | `environment/` and `agent/player.py` have no differences between `7f45b0c` and pre-patch HEAD. The side-to-move board rotation and action indexing are unchanged. |
| Input planes | NCHW, 14 x 10 x 9 for the default model. The same `state_to_planes` function supplies search and training. Optional historical inputs are outside the default 14-plane path. |
| Outcome labels | Self-play converts the final result to red's perspective, then alternates signs for each recorded side-to-move state. Training does not flip this value again. No migration sign mismatch found. |
| Policy targets | Both legacy and current workers store the selected move; training reconstructs a one-hot target. Neither path stores the full MCTS visit distribution. This inherited design was retained. |
| Multiple games per file | The migration correctly added splitting at each initial-state string. Keep that fix; applying the legacy single-game parser to five-game files would be wrong. |
| Search | PUCT, virtual loss, legal-move prior normalization, temperature, terminal reward scaling, and alternating-sign backup match the final legacy revision. No search semantics changed. |
| Network topology | Same convolution/residual/BN/ReLU trunk, four policy channels, two value channels, dense policy softmax and dense value tanh. Kernel and dense transposes verified against an independent NumPy interpretation of the saved graph. |
| Fresh initialization | **Fixed:** Torch's default Conv/Linear initialization differed from Keras Glorot uniform; linear biases were random. Use Xavier/Glorot uniform and zero biases. Existing loaded weights are preserved. |
| Batch normalization | Epsilon 0.001 and update fraction 0.01 were already correct. **Fixed:** Keras 2.0.8 stores population variance, whereas Torch's standard running variance uses the unbiased estimate. Training now updates the saved statistics using the population moments. Inference continues using saved statistics. |
| Policy loss | **Fixed:** match legacy normalization and clipping to `[1e-7, 1-1e-7]` before categorical cross-entropy. Previously Torch only clamped the lower bound to `1e-8`. |
| Value loss | Mean squared error on the scalar tanh prediction already matched. Configured policy/value loss weights remain unchanged. |
| L2 regularization | The workspace already contained an uncommitted fix adding `l2 * sum(kernel^2)` for Conv/Linear weights only. Preserved and verified its value and gradient; BN affine parameters and biases are excluded. **Also fixed:** loading an explicit zero coefficient no longer substitutes the runtime config's nonzero coefficient. |
| SGD | At constant learning rate the old implementation matched. **Fixed:** legacy momentum stores velocity in parameter units; standard Torch SGD scales the accumulated gradient by the current LR. A small Torch optimizer now preserves the legacy equation across LR changes. The `opt` workflow uses SGD; supervised Adam workflows were not included in this audit. |
| Validation/shuffle | **Fixed:** hold out the last `N - int(N * .98)` samples before shuffling the training data, matching Keras. Previously the split was randomized and rounded differently. |
| Step schedule | **Fixed:** include partial training batches and exclude validation rows when counting steps. Resume loads `training_state.json` unless `--total-step` or `--new` overrides it. |
| H5/JSON serialization | HWIO convolution and input-by-output dense storage are preserved. **Fixed:** the exported Flatten config no longer contains `data_format`, which the pinned Keras 2.0.8 Flatten constructor does not accept. Its flatten operation preserves NCHW order. Trained save/reload predictions and BN buffers were verified. |
| Candidate lifecycle | **Fixed:** saving a candidate no longer changes the optimizer's best-checkpoint identity. Rejection preserves the in-memory training model and SGD state. Real promotions remain detectable. H5 files still contain model weights, not a persisted optimizer state. |
| Evaluation | Alternating candidate colors, win/loss assignment, draws worth 0.5, and the 0.55 promotion threshold are correct. **Fixed inherited behavior:** the evaluator honors the configured simulation count instead of replacing it with a random 800-1200 simulations per move. Evaluation noise and temperature settings otherwise remain unchanged. |
| Configuration | `local_torch` intentionally uses different data thresholds, batch/epoch counts, search budgets and polling from `normal`. These are configuration differences, not mathematical backend mismatches. Added only the configurable data-flush interval; retained the other training settings. |

## Validation

All commands used the `pytorch_learn` Conda environment. TensorFlow/Keras are not
installed there, so this is a source-based comparison plus independent numerical
checks, **not a live TensorFlow-vs-Torch execution comparison**.

The new regressions failed before their fixes: candidate resets in local/cluster
publication, initialization, BN variance, SGD at an LR boundary, validation split,
Flatten export, fresh reloads, resume steps, step counting, and slow-game flushing.

The focused suite passes **40 tests**:

```powershell
conda run --no-capture-output -n pytorch_learn python -m unittest cchess_alphazero.tests.test_training_parity cchess_alphazero.tests.test_local_training_workflow cchess_alphazero.tests.test_torch_backend cchess_alphazero.tests.test_torch_regularization_parity cchess_alphazero.tests.test_torch_pipeline_smoke cchess_alphazero.tests.test_fresh_start cchess_alphazero.tests.test_self_play_history_mode cchess_alphazero.tests.test_cluster_pipeline cchess_alphazero.tests.test_terminal_logging cchess_alphazero.tests.test_training_monitor -q
```

Coverage includes independently calculated loss/head-gradient updates, independent
NumPy execution of the exported Keras graph, trained checkpoint round trips,
both candidate colors, promotion detection, and data publication. An existing
config assertion expected a 90-second polling interval although the source sets
300; updated the stale assertion without changing that setting.

A deterministic two-position fitting check (8 filters, one residual block,
250 SGD updates, seed 17) reduced inference policy cross-entropy from **7.6785 to
0.00105** and value MSE from **0.2703 to 0.0522**. Training-mode policy loss fell
from 7.8307 to 0.0000177. These measurements establish that the corrected model
can fit targets, not that it has gained playing strength.

Two isolated real CLI workflows completed on the local RTX 4090. The final run
used the following commands, each stopped after its first completed stage by a
supervisor that terminated only its own process tree:

```powershell
conda run --no-capture-output -n pytorch_learn python .\cchess_alphazero\run.py self --type mini --gpu 0 --data-dir .\validation_local_torch_parity_20261003\workflow_final\data --new
conda run --no-capture-output -n pytorch_learn python .\cchess_alphazero\run.py opt --type mini --gpu 0 --data-dir .\validation_local_torch_parity_20261003\workflow_final\data
conda run --no-capture-output -n pytorch_learn python .\cchess_alphazero\run.py eval --type mini --gpu 0 --data-dir .\validation_local_torch_parity_20261003\workflow_final\data
```

`PROJECT_DIR` was set to the isolated validation directory so its logs did not
overwrite workflow logs. First run: 200 positions, 98 steps, 1 win / 1 loss /
2 draws. Final run: 65 positions, 32 steps, 1 win / 2 losses / 1 draw. Both correctly
kept best and removed the rejected candidate. Both optimizer logs show that best
remained unchanged after candidate publication. These four-game checks are
workflow smoke tests, not statistically meaningful strength benchmarks.

Artifacts are under `validation_local_torch_parity_20261003/`; the final worker
logs are in `workflow_final/`, and the match CSV is in `workflow_final/logs/`.
`git diff --check` passes. No cluster jobs, cluster checkpoints, or GUI files were
changed. Cluster execution was not rerun: unattended SSH requires unavailable
keyboard-interactive authentication; the supplied local copies were analyzed.

## Continuing the existing run

Apply the code changes to the cluster and restart the workers using the same
actual data directory; the supplied logs use `testdata`, while the checked-in
Slurm scripts default to `mydata`. Existing checkpoints remain loadable; another
`--new` is not required to use the fixes. Use `--new` only for an intentional new
run and initialize a shared best once before launching the other workers.

For the next run, first confirm the new device log reports CUDA, completed games
produce play-data files, and optimizer epochs actually occur. Then verify rejected
candidates no longer trigger a best reload, and gather evaluation results over
many candidates/games. The historical cause of repeated losses and sustained
strength improvement remain unverified by the supplied logs and short local runs.

## Legacy sources

- Repository baseline: `git show 7f45b0c:cchess_alphazero/agent/model.py` and the
  corresponding optimizer, self-play, evaluator, environment, and player files.
- [Keras 2.0.8 SGD](https://github.com/keras-team/keras/blob/2.0.8/keras/optimizers.py)
- [Keras 2.0.8 BN layer](https://github.com/keras-team/keras/blob/2.0.8/keras/layers/normalization.py)
- [Keras 2.0.8 TensorFlow moments and losses](https://github.com/keras-team/keras/blob/2.0.8/keras/backend/tensorflow_backend.py)
- [Keras 2.0.8 Flatten and Dense](https://github.com/keras-team/keras/blob/2.0.8/keras/layers/core.py)
- [Keras 2.0.8 training split](https://github.com/keras-team/keras/blob/2.0.8/keras/engine/training.py)
