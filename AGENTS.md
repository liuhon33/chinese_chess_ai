# AGENTS.md

Project scope:
- This repository is a Torch-first local migration of ChineseChess-AlphaZero.

Core rules:
- Torch is the primary backend for inference and training.
- Do not switch back to legacy Keras/TensorFlow unless explicitly asked.
- Fresh-start self-play must not depend on old checkpoints.
- Do not use distributed mode unless explicitly requested.
- Keep CLI entrypoints stable unless there is a strong reason to change them.
- Do not touch GUI or pygame code unless the issue is directly in GUI behavior.
- Prefer minimal, scoped patches over broad refactors.

Current workflow:

Environment:
- Use the Conda environment `pytorch_learn` for all repo commands and validation unless explicitly told otherwise.

- Main local workflow:
  - `python .\cchess_alphazero\run.py self --data-dir <dir>`
  - `python .\cchess_alphazero\run.py opt --data-dir <dir>`
  - `python .\cchess_alphazero\run.py eval --data-dir <dir>`
- First fresh run may use:
  - `python .\cchess_alphazero\run.py self --new --data-dir <dir>`

Debugging expectations:
- Always reproduce the exact failing command first.
- After each code change, rerun the exact command that motivated the fix when feasible.
- If a new blocker appears, report:
  - root cause
  - files changed
  - exact validation command
  - exact new error

Performance expectations:
- Profile before optimizing.
- Do not optimize blindly for GPU utilization.
- For self-play, consider CPU/MCTS/process bottlenecks before assuming GPU is the main limiter.
- Keep model policy/value semantics unchanged while optimizing.
- Prefer config flags for performance-related changes.

Validation expectations:
- Use the smallest relevant validation command.
- For workflow changes, validate real commands such as `self`, `opt`, and `eval`, not only unit tests.

Style:
- Keep edits small and easy to review.
- Preserve existing structure and naming where possible.
- Summaries should say what changed, why, and how it was validated.