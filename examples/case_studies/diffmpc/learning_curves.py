"""The learning curves of the DiffMPC study: descent on Problem 1's state weights through each gradient.

    uv run examples/case_studies/diffmpc/learning_curves.py --out examples/case_studies/diffmpc/results/learning_curves.json

Scaly's full and truncated gradients run here; mpc.pytorch's and DiffMPC's run `baseline/learning_curve.py`
in their environments (the ones `baseline/run_baselines.py` uses), from a copy of the benchmark
directory. Every curve starts at `qd = 3` and takes the same normalized steps.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE / "baseline")]
import scaly_impl as si  # noqa: E402
from run_baselines import IMPLEMENTATIONS, PYTHON, SINGLE_THREAD, TP  # noqa: E402


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--steps", type=int, default=20)
  ap.add_argument("--out", type=Path, required=True)
  args = ap.parse_args()
  curves = {
    "scaly_full": si.learning_curve(si.PROBLEMS[1], "implicit", args.steps),
    "scaly_truncated": si.learning_curve(si.PROBLEMS[1], "implicit_no_x0", args.steps),
  }
  work = Path(tempfile.mkdtemp(prefix="diffmpc-curves-"))
  rl = work / "rl"
  shutil.copytree(TP / "diffmpc" / "benchmarking" / "reinforcement-learning", rl)
  shutil.copy(HERE / "baseline" / "learning_curve.py", rl)
  for impl in ("mpcpytorch", "diffmpc"):
    spec = IMPLEMENTATIONS[impl]
    env = {**os.environ, **SINGLE_THREAD, "JAX_PLATFORMS": "cpu", "PYTHONPATH": os.pathsep.join(map(str, spec["path"]))}
    out = work / f"{impl}.json"
    cmd = [
      "uv",
      "run",
      "--no-project",
      "--python",
      PYTHON,
      *spec["deps"],
      "python",
      "learning_curve.py",
      "--impl",
      impl,
      "--steps",
      str(args.steps),
      "--out",
      str(out),
    ]
    subprocess.run(cmd, cwd=rl, env=env, check=True)
    curves[impl] = json.loads(out.read_text())
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(curves))
  for k, c in curves.items():
    print(f"{k:16s}", " ".join(f"{v:.1f}" for v in c["loss"][:: max(1, args.steps // 10)]))


if __name__ == "__main__":
  main()
