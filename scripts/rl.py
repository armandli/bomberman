#!/usr/bin/env python3
"""Launch, resume, stop and monitor Pommerman RL training runs.

Every run lives in runs/<id>/:
    run.json            algo, current args, launch history, pid
    train.log           trainer stdout/stderr
    metrics.jsonl       one JSON record per line, written by the trainer
    checkpoint.pt(.optim)
    exit_code           written when the trainer process exits
    .watch_state        cursor used by `watch` so each call reports only new events

Typical loop (also how Claude drives it):
    scripts/rl.py train ppo --name baseline -- --eval-interval 50
    scripts/rl.py watch runs/<id>             # blocks until something happens, then exits
    scripts/rl.py summary runs/<id>
    scripts/rl.py stop runs/<id>
    scripts/rl.py resume runs/<id> -- --lr 1e-4 --entropy-coef 0.02

`watch` exit codes (also printed as the first line, "EVENT: <name>"):
    0 heartbeat   timeout reached, nothing notable (summary printed)
    10 eval       new evaluation result (only with --on-eval)
    20 completed  trainer finished normally
    21 stopped    trainer stopped by signal (e.g. `rl.py stop`)
    30 crashed    trainer exited with an error
    40 anomaly    non-finite metric, loss spike or entropy collapse
    50 stall      process alive but no new metrics for --stall seconds
    60 plateau    eval win rate has not improved for --plateau evaluations
    2  error      bad arguments / run not found
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import signal
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BUILD_DIR = REPO / "build"
RUNS_DIR = REPO / "runs"
ALGOS = {
    "ppo": "train_ppo",
    "dqn": "train_dqn",
    "muzero": "train_muzero",
}
# Option holding each trainer's absolute iteration budget (for progress display)
TOTAL_OPT = {"ppo": "--num-updates", "dqn": "--total-steps", "muzero": "--total-games"}
TOTAL_DEFAULT = {"ppo": 10_000, "dqn": 5_000_000, "muzero": 10_000}
MANAGED_OPTS = {"--checkpoint", "--metrics", "--start-iter"}

EXIT = {"heartbeat": 0, "eval": 10, "completed": 20, "stopped": 21, "crashed": 30,
        "anomaly": 40, "stall": 50, "plateau": 60, "error": 2}

# Keys checked for spikes (value > factor × rolling median) per algo
SPIKE_KEYS = ("value_loss", "loss", "reward_loss", "policy_loss")
SPARK = "▁▂▃▄▅▆▇█"


# ── helpers ──────────────────────────────────────────────────────────────────

def die(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(EXIT["error"])


def resolve_run(arg: str) -> Path:
    p = Path(arg)
    for cand in (p, RUNS_DIR / arg):
        if (cand / "run.json").exists():
            return cand.resolve()
    # unique prefix match inside runs/
    if RUNS_DIR.exists():
        matches = [d for d in RUNS_DIR.iterdir() if d.name.startswith(arg) and (d / "run.json").exists()]
        if len(matches) == 1:
            return matches[0].resolve()
        if len(matches) > 1:
            die(f"ambiguous run '{arg}': {', '.join(m.name for m in matches)}")
    die(f"run not found: {arg}")
    raise AssertionError


def load_json(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def save_json(path: Path, data) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def read_metrics(run: Path, offset: int = 0) -> tuple[list[dict], int]:
    """Read complete JSONL records starting at byte offset; returns (records, new_offset)."""
    path = run / "metrics.jsonl"
    if not path.exists():
        return [], offset
    with path.open("rb") as f:
        f.seek(offset)
        data = f.read()
    end = data.rfind(b"\n")
    if end < 0:
        return [], offset
    records = []
    for line in data[: end + 1].splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records, offset + end + 1


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        state = Path(f"/proc/{pid}/stat").read_text().split(")")[-1].split()[0]
        return state != "Z"
    except (FileNotFoundError, IndexError):
        return False


def run_state(run: Path) -> tuple[str, int | None]:
    """Returns (state, exit_code): running | completed | stopped | crashed | unknown."""
    meta = load_json(run / "run.json", {})
    if pid_alive(meta.get("pid")):
        return "running", None
    code_txt = (run / "exit_code").read_text().strip() if (run / "exit_code").exists() else ""
    code = int(code_txt) if code_txt.lstrip("-").isdigit() else None
    recs, _ = read_metrics(run)
    last_kind = next((r["kind"] for r in reversed(recs) if r.get("kind") in ("complete", "stopped")), None)
    if code == 0 and last_kind == "complete":
        return "completed", code
    if code == 0 and last_kind == "stopped":
        return "stopped", code
    if code is None:
        return "unknown", None
    return "crashed", code


def parse_opts(tokens: list[str]) -> list[tuple[str, str | None]]:
    """Parse ['--a', '1', '--flag', '--b', 'x'] into ordered (key, value|None) pairs."""
    out: list[tuple[str, str | None]] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if not tok.startswith("--"):
            die(f"unexpected trainer argument '{tok}' (options must be --key [value])")
        if "=" in tok:
            k, v = tok.split("=", 1)
            out.append((k, v))
            i += 1
        elif i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
            out.append((tok, tokens[i + 1]))
            i += 2
        else:
            out.append((tok, None))
            i += 1
    return out


def merge_opts(base: list[str], overrides: list[str]) -> list[str]:
    merged: dict[str, str | None] = {}
    for k, v in parse_opts(base) + parse_opts(overrides):
        if k in MANAGED_OPTS:
            continue
        merged.pop(k, None)  # keep the override's position at the end
        merged[k] = v
    flat: list[str] = []
    for k, v in merged.items():
        flat.append(k)
        if v is not None:
            flat.append(v)
    return flat


def opt_value(args: list[str], key: str) -> str | None:
    for k, v in parse_opts(args):
        if k == key:
            return v
    return None


def sparkline(values: list[float]) -> str:
    vals = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    if not vals:
        return ""
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return SPARK[3] * len(vals)
    return "".join(SPARK[min(7, int((v - lo) / (hi - lo) * 7.999))] for v in vals)


def fmt(v) -> str:
    if isinstance(v, float):
        if not math.isfinite(v):
            return str(v)
        if v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e5):
            return f"{v:.3e}"
        return f"{v:.4f}".rstrip("0").rstrip(".") if abs(v) < 1000 else f"{v:.0f}"
    return str(v)


def human_secs(s: float) -> str:
    s = int(s)
    if s < 120:
        return f"{s}s"
    if s < 7200:
        return f"{s // 60}m"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


# ── build ────────────────────────────────────────────────────────────────────

def ensure_build(targets: list[str]) -> None:
    if not (BUILD_DIR / "CMakeCache.txt").exists():
        print(f"Configuring {BUILD_DIR} ...", flush=True)
        subprocess.run(["cmake", "-S", str(REPO), "-B", str(BUILD_DIR),
                        "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CXX_COMPILER=g++-14"], check=True)
    print(f"Building {' '.join(targets)} ...", flush=True)
    res = subprocess.run(["cmake", "--build", str(BUILD_DIR), "-j", str(os.cpu_count() or 4),
                          "--target", *targets], capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stdout[-4000:], res.stderr[-4000:], sep="\n", file=sys.stderr)
        die("build failed")


def cmd_build(a) -> None:
    ensure_build(list(ALGOS.values()) + ["test_pommerman", "test_rl"])
    print("Build OK")


# ── launch / resume / stop ───────────────────────────────────────────────────

def launch(run: Path, algo: str, user_args: list[str], start_iter: int) -> int:
    binary = BUILD_DIR / ALGOS[algo]
    full = [str(binary), *user_args,
            "--checkpoint", str(run / "checkpoint.pt"),
            "--metrics", str(run / "metrics.jsonl"),
            "--start-iter", str(start_iter)]
    (run / "exit_code").unlink(missing_ok=True)
    # The sh wrapper records the exit code; its no-op TERM/INT trap lets a
    # group-wide SIGTERM reach the trainer (which saves and exits) while sh
    # survives long enough to write exit_code.
    wrapper = 'trap ":" TERM INT; "$@"; echo $? > "$RL_RUN_DIR/exit_code"'
    log = open(run / "train.log", "a")
    log.write(f"\n===== launch {datetime.now().isoformat(timespec='seconds')}: {shlex.join(full)}\n")
    log.flush()
    proc = subprocess.Popen(["sh", "-c", wrapper, "sh", *full], cwd=run, stdout=log,
                            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            start_new_session=True, env={**os.environ, "RL_RUN_DIR": str(run)})
    meta = load_json(run / "run.json", {})
    meta.update({"id": run.name, "algo": algo, "binary": str(binary), "args": user_args,
                 "pid": proc.pid})
    meta.setdefault("launches", []).append({
        "time": datetime.now().isoformat(timespec="seconds"),
        "args": user_args, "start_iter": start_iter, "pid": proc.pid})
    save_json(run / "run.json", meta)
    return proc.pid


def resume_iter(run: Path) -> int:
    recs, _ = read_metrics(run)
    for r in reversed(recs):
        if r.get("kind") in ("checkpoint", "stopped", "complete") and "iter" in r:
            return int(r["iter"])
    return 0


def cmd_train(a, trainer_args: list[str]) -> None:
    if a.algo not in ALGOS:
        die(f"algo must be one of {', '.join(ALGOS)}")
    if not a.no_build:
        ensure_build([ALGOS[a.algo]])
    RUNS_DIR.mkdir(exist_ok=True)
    rid = f"{a.algo}-{datetime.now():%Y%m%d-%H%M%S}" + (f"-{a.name}" if a.name else "")
    run = RUNS_DIR / rid
    run.mkdir()
    args = merge_opts([], trainer_args)
    pid = launch(run, a.algo, args, 0)
    print(f"Started {rid}  pid={pid}\n  dir: {run}\n  args: {shlex.join(args) or '(defaults)'}")


def cmd_resume(a, overrides: list[str]) -> None:
    run = resolve_run(a.run)
    meta = load_json(run / "run.json", {})
    if pid_alive(meta.get("pid")):
        die(f"{run.name} is still running (pid {meta['pid']}); stop it first")
    algo = meta["algo"]
    if not a.no_build:
        ensure_build([ALGOS[algo]])
    args = merge_opts(meta.get("args", []), overrides)
    it = resume_iter(run)
    if not (run / "checkpoint.pt").exists():
        it = 0
    total = int(float(opt_value(args, TOTAL_OPT[algo]) or TOTAL_DEFAULT[algo]))
    if it >= total:
        die(f"run already at iteration {it} (budget {TOTAL_OPT[algo]} {total}); "
            f"pass a larger {TOTAL_OPT[algo]} to extend it")
    pid = launch(run, algo, args, it)
    print(f"Resumed {run.name} from iter {it}  pid={pid}\n  args: {shlex.join(args) or '(defaults)'}")


def cmd_stop(a) -> None:
    run = resolve_run(a.run)
    meta = load_json(run / "run.json", {})
    pid = meta.get("pid")
    if not pid_alive(pid):
        print(f"{run.name} is not running ({run_state(run)[0]})")
        return
    os.killpg(pid, signal.SIGTERM)
    deadline = time.time() + a.timeout
    while pid_alive(pid) and time.time() < deadline:
        time.sleep(1)
    if pid_alive(pid):
        print(f"Trainer did not exit within {a.timeout}s; sending SIGKILL (last checkpoint is kept)")
        os.killpg(pid, signal.SIGKILL)
        time.sleep(1)
    print(f"{run.name}: {run_state(run)[0]}  (resume from iter {resume_iter(run)})")


# ── status / summary ─────────────────────────────────────────────────────────

def last_of(recs: list[dict], kind: str) -> dict | None:
    return next((r for r in reversed(recs) if r.get("kind") == kind), None)


def progress(run: Path, recs: list[dict]) -> tuple[int, int]:
    meta = load_json(run / "run.json", {})
    algo = meta.get("algo", "ppo")
    total = int(float(opt_value(meta.get("args", []), TOTAL_OPT[algo]) or TOTAL_DEFAULT[algo]))
    it = max((int(r["iter"]) for r in recs if "iter" in r), default=0)
    return it, total


def cmd_status(a) -> None:
    runs = [resolve_run(r) for r in a.runs] if a.runs else sorted(
        (d for d in RUNS_DIR.glob("*") if (d / "run.json").exists()), key=lambda d: d.name)
    if not runs:
        print("No runs yet. Start one with: scripts/rl.py train <ppo|dqn|muzero>")
        return
    print(f"{'run':40} {'state':10} {'progress':>18} {'score':>6} {'best':>6} {'win':>5} {'updated':>8}")
    for run in runs:
        recs, _ = read_metrics(run)
        state, code = run_state(run)
        it, total = progress(run, recs)
        evals = [r for r in recs if r.get("kind") == "eval"]
        score = fmt(evals[-1]["score"]) if evals else "-"
        best = fmt(max(e["score"] for e in evals)) if evals else "-"
        win = fmt(evals[-1]["win_rate"]) if evals else "-"
        mpath = run / "metrics.jsonl"
        age = human_secs(time.time() - mpath.stat().st_mtime) if mpath.exists() else "-"
        st = state + (f"({code})" if state == "crashed" else "")
        print(f"{run.name:40} {st:10} {f'{it}/{total}':>18} {score:>6} {best:>6} {win:>5} {age:>8}")


def summarize(run: Path, n_train: int = 5, n_eval: int = 8) -> str:
    meta = load_json(run / "run.json", {})
    recs, _ = read_metrics(run)
    state, code = run_state(run)
    it, total = progress(run, recs)
    lines = [f"run: {run.name}   algo: {meta.get('algo')}   state: {state}"
             + (f" (exit {code})" if code not in (None, 0) else "")
             + f"   progress: {it}/{total} ({100 * it / max(total, 1):.1f}%)",
             f"args: {shlex.join(meta.get('args', [])) or '(defaults)'}"]
    if len(meta.get("launches", [])) > 1:
        lines.append(f"launches: {len(meta['launches'])} (last at iter {meta['launches'][-1]['start_iter']})")

    train = [r for r in recs if r.get("kind") == "train"]
    if train:
        last = train[-1]
        keys = [k for k in last if k not in ("kind", "wall")]
        lines.append(f"\nlatest train @ iter {last.get('iter')}:")
        lines.append("  " + "  ".join(f"{k}={fmt(last[k])}" for k in keys if k != "iter"))
        # trend: mean of last n vs the n before that
        recent, prev = train[-n_train:], train[-2 * n_train:-n_train]
        if prev:
            trend = []
            for k in ("mean_reward", "mean_ep_len", "entropy", "explained_var", "value_loss",
                      "policy_loss", "loss", "epsilon"):
                if k in last:
                    r_mean = statistics.fmean(x[k] for x in recent if k in x)
                    p_mean = statistics.fmean(x[k] for x in prev if k in x)
                    trend.append(f"{k} {fmt(p_mean)}→{fmt(r_mean)}")
            lines.append(f"  trend (last {len(recent)} vs prior {len(prev)} logs): " + ", ".join(trend))
        for k in ("mean_reward", "value_loss", "loss", "entropy"):
            if k in last:
                lines.append(f"  {k:12} {sparkline([x.get(k) for x in train[-40:]])}")
    else:
        warm = last_of(recs, "warmup")
        lines.append("\nno training updates logged yet"
                     + (f" (warmup: buffer {fmt(warm.get('buffer_size'))} @ iter {warm.get('iter')})" if warm else ""))

    evals = [r for r in recs if r.get("kind") == "eval"]
    if evals:
        best = max(evals, key=lambda e: e["score"])
        lines.append(f"\neval ({len(evals)} evals, {evals[-1].get('games')} games each; "
                     f"best score {fmt(best['score'])} @ iter {best['iter']}):")
        lines.append("  score = win + 0.5*tie + 0.25*survival/800 vs 3 SimpleAgents; reference: "
                     "random ≈ 0.01, stand-still ≈ 0.08, SimpleAgent ≈ 0.3 (20-game evals are ±0.08 noisy)")
        lines.append(f"  score {sparkline([e['score'] for e in evals[-40:]])}")
        lines.append(f"  {'iter':>10} {'score':>6} | {'win':>5} {'tie':>5} {'loss':>5} {'surv':>6} {'bombs':>5}"
                     f" | {'rnd_win':>7} {'rnd_surv':>8}")
        for e in evals[-n_eval:]:
            lines.append(f"  {e['iter']:>10} {fmt(e['score']):>6} | {fmt(e['win_rate']):>5} {fmt(e['tie_rate']):>5} "
                         f"{fmt(e['loss_rate']):>5} {fmt(e['mean_survival']):>6} {fmt(e.get('mean_bombs', 0)):>5}"
                         f" | {fmt(e.get('rand_win_rate', 0)):>7} {fmt(e.get('rand_survival', 0)):>8}")
    return "\n".join(lines)


def log_tail(run: Path, n: int = 25) -> str:
    path = run / "train.log"
    if not path.exists():
        return "(no train.log)"
    return "\n".join(path.read_text(errors="replace").splitlines()[-n:])


def cmd_summary(a) -> None:
    run = resolve_run(a.run)
    print(summarize(run, n_eval=a.evals))
    if a.log:
        print("\n--- train.log tail ---\n" + log_tail(run, a.log))


# ── watch ────────────────────────────────────────────────────────────────────

def check_record(r: dict, st: dict, a) -> tuple[str, str] | None:
    """Update watch state with one metrics record; return (event, reason) if notable."""
    kind = r.get("kind")
    for k, v in r.items():
        if isinstance(v, float) and not math.isfinite(v):
            return "anomaly", f"non-finite {k}={v} in {kind} record @ iter {r.get('iter')}"

    if kind == "train":
        hist = st.setdefault("hist", {})
        for k in SPIKE_KEYS:
            if k not in r:
                continue
            h = hist.setdefault(k, [])
            if len(h) >= 10:
                med = statistics.median(h)
                if r[k] > a.spike_factor * max(med, 1e-3) and r[k] > a.spike_min:
                    h.append(r[k]); del h[:-20]
                    return "anomaly", (f"{k} spiked to {fmt(r[k])} (rolling median {fmt(med)}) "
                                       f"@ iter {r.get('iter')}")
            h.append(r[k]); del h[:-20]
        if "entropy" in r and r["entropy"] < a.min_entropy:
            if not st.get("entropy_reported"):
                st["entropy_reported"] = True
                return "anomaly", (f"policy entropy collapsed to {fmt(r['entropy'])} "
                                   f"(< {a.min_entropy}; uniform = {math.log(6):.3f}) @ iter {r.get('iter')}")
        else:
            st["entropy_reported"] = False

    if kind == "eval":
        sc = r["score"]
        if sc > st.get("best_score", -1.0) + a.plateau_eps:
            st["best_score"], st["best_iter"], st["since_best"] = sc, r["iter"], 0
        else:
            st["since_best"] = st.get("since_best", 0) + 1
        if a.plateau and st["since_best"] >= a.plateau:
            st["since_best"] = 0  # re-arm: report again after another `plateau` evals
            return "plateau", (f"eval score has not beaten {fmt(st['best_score'])} "
                               f"(iter {st['best_iter']}) for {a.plateau} evaluations")
        if a.on_eval:
            return "eval", (f"eval @ iter {r['iter']}: score {fmt(sc)} | vs simple win {fmt(r['win_rate'])} "
                            f"tie {fmt(r['tie_rate'])} loss {fmt(r['loss_rate'])} "
                            f"survive {fmt(r['mean_survival'])} | vs random win {fmt(r.get('rand_win_rate', 0))}")
    return None


def cmd_watch(a) -> None:
    run = resolve_run(a.run)
    state_path = run / ".watch_state"
    st = load_json(state_path, {}) if not a.reset else {}
    st.setdefault("offset", 0)
    start = time.time()
    event: tuple[str, str] | None = None

    while event is None:
        recs, st["offset"] = read_metrics(run, st["offset"])
        for r in recs:
            ev = check_record(r, st, a)
            if ev and (event is None or EXIT[ev[0]] > EXIT[event[0]]):
                event = ev
        if event:
            break

        state, code = run_state(run)
        if state != "running":
            launches = load_json(run / "run.json", {}).get("launches", [])
            key = f"{len(launches)}:{state}"
            if st.get("exit_reported") != key:
                st["exit_reported"] = key
                if state == "completed":
                    event = ("completed", "trainer finished all iterations")
                elif state == "stopped":
                    event = ("stopped", "trainer stopped by signal; resume with `rl.py resume`")
                elif code is None:
                    event = ("crashed", "trainer process is gone and left no exit code "
                                        "(wrapper killed, OOM-killer, or machine restart)")
                else:
                    event = ("crashed", f"trainer exited with code {code}"
                                        + (" (killed: SIGKILL/OOM?)" if code == 137 else "")
                                        + (" (segfault)" if code == 139 else ""))
                break
            save_json(state_path, st)
            print(f"EVENT: error\n{run.name} is not running ({state}); nothing to watch.")
            sys.exit(EXIT["error"])

        mpath = run / "metrics.jsonl"
        last_write = mpath.stat().st_mtime if mpath.exists() else start
        if time.time() - max(last_write, start) > a.stall:
            event = ("stall", f"no new metrics for {human_secs(time.time() - last_write)} "
                              f"while the trainer is alive")
            break
        if time.time() - start >= a.timeout:
            event = ("heartbeat", f"no notable events in {human_secs(a.timeout)}")
            break
        time.sleep(a.poll)

    save_json(state_path, st)
    name, reason = event
    print(f"EVENT: {name}\nreason: {reason}\n")
    print(summarize(run))
    if name in ("crashed", "stall", "anomaly") or a.log:
        print("\n--- train.log tail ---\n" + log_tail(run, a.log or 25))
    sys.exit(EXIT[name])


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    argv = sys.argv[1:]
    trainer_args: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, trainer_args = argv[:i], argv[i + 1:]

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("build", help="configure (if needed) and build trainers + tests")

    t = sub.add_parser("train", help="start a new run: train <algo> [--name N] -- <trainer args>")
    t.add_argument("algo", choices=list(ALGOS))
    t.add_argument("--name", default="")
    t.add_argument("--no-build", action="store_true")

    r = sub.add_parser("resume", help="resume a stopped/crashed run: resume <run> -- <overrides>")
    r.add_argument("run")
    r.add_argument("--no-build", action="store_true")

    s = sub.add_parser("stop", help="gracefully stop a run (trainer saves a checkpoint)")
    s.add_argument("run")
    s.add_argument("--timeout", type=int, default=180)

    st = sub.add_parser("status", help="one-line status of all (or given) runs")
    st.add_argument("runs", nargs="*")

    sm = sub.add_parser("summary", help="detailed metrics summary of one run")
    sm.add_argument("run")
    sm.add_argument("--evals", type=int, default=8, help="eval rows to show")
    sm.add_argument("--log", type=int, default=0, help="also show N lines of train.log")

    w = sub.add_parser("watch", help="block until something notable happens, then report and exit")
    w.add_argument("run")
    w.add_argument("--timeout", type=float, default=1800, help="heartbeat after this many seconds")
    w.add_argument("--stall", type=float, default=900, help="alert if no metrics for this many seconds")
    w.add_argument("--poll", type=float, default=10)
    w.add_argument("--on-eval", action="store_true", help="return on every new eval result")
    w.add_argument("--plateau", type=int, default=8, help="evals without a new best score (0 = off)")
    w.add_argument("--plateau-eps", type=float, default=0.01, help="min score gain that counts as improvement")
    w.add_argument("--spike-factor", type=float, default=10.0)
    w.add_argument("--spike-min", type=float, default=1.0)
    w.add_argument("--min-entropy", type=float, default=0.15)
    w.add_argument("--log", type=int, default=0, help="always include N lines of train.log")
    w.add_argument("--reset", action="store_true", help="forget the cursor and re-scan all metrics")

    a = p.parse_args(argv)
    if trainer_args and a.cmd not in ("train", "resume"):
        die("'--' trainer arguments are only valid for train/resume")

    {"build": lambda: cmd_build(a),
     "train": lambda: cmd_train(a, trainer_args),
     "resume": lambda: cmd_resume(a, trainer_args),
     "stop": lambda: cmd_stop(a),
     "status": lambda: cmd_status(a),
     "summary": lambda: cmd_summary(a),
     "watch": lambda: cmd_watch(a)}[a.cmd]()


if __name__ == "__main__":
    main()
