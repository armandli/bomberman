# bomberman

Pommerman (4-player Bomberman) environment rewritten in C++, with self-play RL
trainers built on libtorch:

| Algo     | Binary         | Model                                                     |
|----------|----------------|-----------------------------------------------------------|
| `ppo`    | `train_ppo`    | Transformer actor-critic, clipped PPO + GAE               |
| `dqn`    | `train_dqn`    | Transformer dueling double DQN, uniform replay            |
| `muzero` | `train_muzero` | MuZero / EfficientZero (learned dynamics, PUCT MCTS, SimSiam consistency) |

All four seats are played by the same network (self-play, FFA).

## Build

Requirements: CMake ≥ 3.25, g++-14 (C++26), a CUDA build of libtorch, CUDA toolkit (nvcc).
CLI11, FTXUI and Catch2 are fetched automatically.

```bash
scripts/rl.py build                     # configures build/ if needed, builds trainers + tests
# or manually:
cmake -S . -B build -DCMAKE_CXX_COMPILER=g++-14 \
      [-DTORCH_PATH=~/.local/libtorch] [-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc]
cmake --build build -j
build/test_pommerman                    # unit tests
build/pommerman_ffa                     # watch 4 random agents in the terminal
```

Defaults: `TORCH_PATH=$HOME/.local/libtorch`, nvcc from `/usr/local/cuda/bin`, `TORCH_CUDA_ARCH_LIST=12.0`.

## Training and monitoring

`scripts/rl.py` (Python 3 stdlib only) manages runs. Each run gets its own
directory `runs/<algo>-<timestamp>[-name]/` holding `train.log`,
`metrics.jsonl`, `checkpoint.pt` (+ `.optim`), and `run.json` (args + launch history).

```bash
scripts/rl.py train ppo --name baseline -- --eval-interval 50    # args after -- go to the trainer
scripts/rl.py status                                             # all runs, one line each
scripts/rl.py summary ppo-2026                                   # metrics, trends, eval table (prefix ok)
scripts/rl.py watch ppo-2026                                     # block until something notable happens
scripts/rl.py stop ppo-2026                                      # SIGTERM → trainer saves and exits
scripts/rl.py resume ppo-2026 -- --lr 1e-4 --entropy-coef 0.02   # continue with changed hyperparameters
```

`resume` merges the overrides into the run's previous args, restarts from the
last checkpoint (model **and** optimizer state; `--lr` is re-applied), and
continues the iteration count (`--start-iter`), so DQN's epsilon schedule and
the metrics timeline carry on. To extend a finished run, pass a larger budget
(`--num-updates` / `--total-steps` / `--total-games`).

### Evaluation

Every `--eval-interval` iterations (and once at iteration 0) the trainer plays
`--eval-games` games of its greedy policy against **3 rule-based SimpleAgents**
(main benchmark) and against 3 random agents (sanity check), always on the
same seed so evaluations are comparable. Random agents mostly blow themselves
up, so beating them says little; SimpleAgents dodge bombs and attack.

`score = win_rate + 0.5·tie_rate + 0.25·survival/800` (vs SimpleAgents).
Reference values: random play ≈ 0.01, standing still ≈ 0.08, SimpleAgent ≈ 0.3.
With 20 games an eval is noisy (about ±0.08); raise `--eval-games` for finer comparisons.

### `watch` events

`watch` polls the run and exits as soon as one of these happens, printing
`EVENT: <name>`, the reason, a summary and (for problems) the log tail. Running
it in the background gives a notification when anything needs attention.

| exit | event       | meaning                                                         |
|-----:|-------------|-----------------------------------------------------------------|
| 0    | `heartbeat` | `--timeout` (default 30 min) elapsed with nothing notable        |
| 10   | `eval`      | new eval result (only with `--on-eval`)                          |
| 20   | `completed` | trainer finished its budget                                     |
| 21   | `stopped`   | trainer stopped by signal                                       |
| 30   | `crashed`   | non-zero exit, killed (137), segfault (139), or vanished process |
| 40   | `anomaly`   | NaN/Inf metric, loss spike (>10× rolling median), entropy < 0.15 |
| 50   | `stall`     | process alive but no metrics for `--stall` s (default 900)       |
| 60   | `plateau`   | no new best eval score (+0.01) in `--plateau` evals (default 8)  |

Each run keeps a cursor (`.watch_state`), so repeated `watch` calls only report
new events (`--reset` re-scans everything).

### Metrics records (`metrics.jsonl`)

One JSON object per line with a `kind`: `start`, `warmup`, `train`, `eval`,
`checkpoint`, `stopped`, `complete`. `iter` is the update (PPO), env step (DQN)
or game (MuZero) index. Train records include losses, entropy (PPO), epsilon
(DQN), mean shaped reward, mean game length and throughput.
