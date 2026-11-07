# CLAUDE.md

C++26 Pommerman environment + libtorch self-play trainers (PPO, DQN, MuZero). See README.md.

## Build & test
- `scripts/rl.py build` (uses `build/`, g++-14, libtorch at `~/.local/libtorch`, nvcc at `/usr/local/cuda/bin`).
- `build/test_pommerman` must stay green; engine behaviour changes need a regression test in `test/pommerman/`.

## Driving training runs
Use `scripts/rl.py` for everything; never launch trainer binaries by hand for real runs.

1. Start: `scripts/rl.py train <ppo|dqn|muzero> --name <tag> -- <trainer args>`.
2. Monitor: run `scripts/rl.py watch <run>` with `run_in_background: true`. It exits on an event and
   the harness notifies you; read the output (first line `EVENT: <name>`), act, then start a new
   background `watch` on the same run. Don't poll with sleep loops.
3. React to events:
   - `heartbeat`: check the summary trend; if healthy, just re-arm `watch`.
   - `anomaly` / `crashed`: read the log tail, diagnose; fix code or `stop` → `resume -- <overrides>`
     (e.g. lower `--lr`, raise `--entropy-coef`, lower `--grad-clip`).
   - `plateau`: consider hyperparameter changes via `stop` + `resume`, or note it and continue.
   - `stall`: process alive but silent; check GPU (`nvidia-smi`) and the log.
4. Record every hyperparameter change and its reason when reporting to the user.

## Reading metrics
- Main signal: eval `score` vs 3 SimpleAgents (random ≈ 0.01, standing still ≈ 0.08,
  SimpleAgent ≈ 0.3; 20-game evals are ±0.08 noisy). Win rate vs random agents saturates
  almost immediately and is only a sanity check.
- `mean_bombs == 0` with decent survival means a passive policy (it hides instead of fighting).
- PPO entropy starts at ln 6 ≈ 1.79; a fast collapse below ~0.3 early on is a warning sign.
- Self-play shaped reward is not a progress measure (opponents improve too).
