"""Run DiffMPC's CPU benchmark for mpc.pytorch, DiffMPC and trajax in fresh processes.

    examples/case_studies/diffmpc/baseline/setup.sh
    uv run examples/case_studies/diffmpc/baseline/run_baselines.py --out examples/case_studies/diffmpc/results/baselines.json

Each run is one of the benchmark's own scripts (`benchmarking/reinforcement-learning/benchmark_*.py`),
run from a copy of that directory with these changes, and nothing else:

- `utils.py` gets the problem's state and control dimensions (the scripts read them from there);
- the seed loop runs `--seeds` seeds instead of 10 (the scripts time one forward and one gradient per
  seed, after an untimed warm-up and, for JAX, a compile that is repeated for each seed);
- each script saves the timed gradient, so the gradients can be compared across implementations;
- `benchmark_trajax.py` reads a JAX array's device with the current API (`.device()` was removed in
  JAX 0.4.27).

The environments: Python 3.13; PyTorch 2.14 and NumPy < 2 for mpc.pytorch (its `setup.py` asks for
NumPy < 2); JAX 0.5.3 for DiffMPC and trajax. `--threads 1` pins every library to one thread
(`OMP_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, and XLA's `--xla_cpu_multi_thread_eigen=false
intra_op_parallelism_threads=1`); `default` leaves the libraries' own thread pools.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TP = HERE / "third_party"
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE.parent))
from _common import wait_for_quiet  # noqa: E402
from scaly_impl import PROBLEMS  # noqa: E402

PYTHON = "3.13"
TORCH = ["--with", "torch==2.14.0", "--with", "numpy<2"]
JAX = ["--with", "jax==0.5.3", "--with", "jaxlib==0.5.3", "--with", "pyyaml", "--with", "scipy", "--with", "absl-py", "--with", "numpy<2.3"]
IMPLEMENTATIONS = {
  "mpcpytorch": {"script": "benchmark_mpcpytorch.py", "deps": TORCH, "path": [TP / "mpc.pytorch"], "args": ["--device", "cpu"]},
  "diffmpc": {"script": "benchmark_diffmpc.py", "deps": JAX, "path": [TP / "diffmpc"], "args": []},
  "trajax": {"script": "benchmark_trajax.py", "deps": JAX, "path": [TP / "trajax"], "args": []},
}
SINGLE_THREAD = {
  "OMP_NUM_THREADS": "1",
  "MKL_NUM_THREADS": "1",
  "VECLIB_MAXIMUM_THREADS": "1",
  "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
}
# After the timed gradient, save it: (the line it follows, the line to add), per script.
SAVE_GRADIENT = {
  "mpcpytorch": ("grad_time = time.monotonic() - grad_start_time", 'np.save(f"grad_{seed}.npy", Q_weights.grad.detach().cpu().numpy())'),
  "diffmpc": (
    "grad_time = time.monotonic() - start_time",
    'np.save(f"grad_{seed}.npy", np.asarray(gradients["weights_penalization_reference_state_trajectory"]))',
  ),
  "trajax": ("grad_time = time.monotonic() - start_time", 'np.save(f"grad_{seed}.npy", np.asarray(gradients))'),
}


def prepare(work: Path, impl: str, nx: int, nu: int, seeds: int) -> Path:
  """A patched copy of the benchmark directory for one run."""
  rl = work / "rl"
  if rl.exists():
    shutil.rmtree(rl)
  shutil.copytree(TP / "diffmpc" / "benchmarking" / "reinforcement-learning", rl)
  utils = rl / "utils.py"
  text = utils.read_text()
  text = re.sub(r"^N_STATE = \d+", f"N_STATE = {nx}", text, flags=re.M)
  text = re.sub(r"^N_CTRL = \d+", f"N_CTRL = {nu}", text, flags=re.M)
  utils.write_text(text)
  script = rl / IMPLEMENTATIONS[impl]["script"]
  text = script.read_text()
  text = text.replace("for seed in range(10):", f"for seed in range({seeds}):")
  after, line = SAVE_GRADIENT[impl]
  text = re.sub(rf"^(\s*){re.escape(after)}$", lambda m: f"{m.group(0)}\n{m.group(1)}{line}", text, count=1, flags=re.M)
  assert line in text, f"{impl}: gradient save not inserted"
  if impl == "trajax":
    text = text.replace("A_matrix.device()", "A_matrix.devices()").replace("jnp.zeros(1).device()", "jnp.zeros(1).device")
  script.write_text(text)
  return rl


def run(impl: str, problem: int, threads: str, seeds: int, work: Path) -> dict:
  p = PROBLEMS[problem]
  rl = prepare(work, impl, p.nx, p.nu, seeds)
  spec = IMPLEMENTATIONS[impl]
  cmd = [
    "uv",
    "run",
    "--no-project",
    "--python",
    PYTHON,
    *spec["deps"],
    "python",
    spec["script"],
    "--batch_size",
    str(p.batch),
    "--horizon",
    str(p.horizon),
    *spec["args"],
  ]
  if impl == "diffmpc":
    cmd += ["--num_repeats", str(seeds)]
  env = {**os.environ, "JAX_PLATFORMS": "cpu", "PYTHONPATH": os.pathsep.join(map(str, spec["path"]))}
  if threads == "1":
    env.update(SINGLE_THREAD)
  load = wait_for_quiet()
  t0 = time.perf_counter()
  done = subprocess.run(cmd, cwd=rl, env=env, capture_output=True, text=True)
  wall = time.perf_counter() - t0
  if done.returncode:
    raise RuntimeError(f"{impl} problem {problem} failed:\n{done.stderr[-3000:]}")
  fwd = [float(x) * 1e-3 for x in re.findall(r"^Time: ([\d.]+) ms", done.stdout, flags=re.M)]
  bwd = [float(x) * 1e-3 for x in re.findall(r"^grad time: ([\d.]+) ms", done.stdout, flags=re.M)]
  costs = [float(x) for x in re.findall(r"^Aggregate cost: ([-\d.]+)", done.stdout, flags=re.M)]
  if not (len(fwd) == len(bwd) == len(costs) == seeds):
    raise RuntimeError(f"{impl} problem {problem}: parsed {len(fwd)} forward, {len(bwd)} gradient times for {seeds} seeds\n{done.stdout[-2000:]}")
  grads = [np.load(rl / f"grad_{s}.npy").reshape(-1).tolist() for s in range(seeds)]
  return {
    "implementation": impl,
    "problem": problem,
    "threads": threads,
    "forward_s": fwd,
    "backward_s": bwd,
    "aggregate_cost": costs,
    "gradient": grads,
    "process_wall_s": wall,
    "load": load,
  }


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--problems", type=int, nargs="+", default=sorted(PROBLEMS))
  ap.add_argument("--implementations", nargs="+", default=list(IMPLEMENTATIONS))
  ap.add_argument("--threads", nargs="+", default=["1", "default"])
  ap.add_argument("--seeds", type=int, default=5)
  ap.add_argument("--out", type=Path, required=True)
  args = ap.parse_args()
  work = Path(tempfile.mkdtemp(prefix="diffmpc-baselines-"))
  rows = json.loads(args.out.read_text())["rows"] if args.out.exists() else []
  done = {(r["implementation"], r["problem"], r["threads"]) for r in rows}
  for threads in args.threads:
    for problem in args.problems:
      for impl in args.implementations:
        if (impl, problem, threads) in done:
          continue
        row = run(impl, problem, threads, args.seeds, work)
        rows.append(row)
        print(
          f"problem {problem} {impl:10s} threads={threads:7s} forward {1e3 * np.median(row['forward_s']):8.1f} ms  gradient {1e3 * np.median(row['backward_s']):8.1f} ms",
          flush=True,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"seeds": args.seeds, "rows": rows}, indent=1))


if __name__ == "__main__":
  main()
