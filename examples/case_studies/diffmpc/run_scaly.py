"""The Scaly side of the DiffMPC study: one problem, every variant, the benchmark's protocol.

    uv run examples/case_studies/diffmpc/run_scaly.py --problem 1 --out results/scaly_1.json

Per variant: build the episode and its gradient, call each once on seed 0's data to compile (the
benchmark's warm-up), then per seed time one forward call and one gradient call through the Function's
Python call, as the benchmark times one call of its jitted or eager function per seed. The data of
every seed go through the same compiled Functions (the JAX baselines recompile per seed, untimed).

Variants: `per_solve` (every step of every batch element solves its own MPC) with the implicit rule
and with AD through the Riccati recursion, and `hoisted` (the benchmark's code as written, whose
Riccati recursion Scaly hoists out of the episode) with the implicit rule. The gradients returned
are the full one; the truncated one that DiffMPC and trajax compute (no path through the MPC's state
input) is recorded for seed 0, also with one more knot (`T + 1`), the problem trajax's script poses
(its `U` has `T` rows and `X` has `T + 1`).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

VARIANTS = {"per_solve_implicit": (False, "implicit"), "per_solve_ad": (False, None), "hoisted_implicit": (True, "implicit")}


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--problem", type=int, required=True)
  ap.add_argument("--seeds", type=int, default=5)
  ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
  ap.add_argument("--out", type=Path, required=True)
  args = ap.parse_args()
  os.environ.setdefault("SCALY_CACHE_DIR", tempfile.mkdtemp(prefix="diffmpc-scaly-"))
  import scaly_impl as si

  p = si.PROBLEMS[args.problem]
  data = [si.problem_data(p.nx, p.nu, p.batch, s) for s in range(args.seeds)]
  rows = []
  for variant in args.variants:
    hoist, rule = VARIANTS[variant]
    t0 = time.perf_counter()
    episode = si.episode_function(p, hoist, rule)
    grad = si.gradient_function(p, episode)
    warm = si.arguments(data[0])
    episode(*warm)
    grad(*warm)
    t_first = time.perf_counter() - t0
    fwd, bwd, costs, grads = [], [], [], []
    for d in data:
      a = si.arguments(d)
      t = time.perf_counter()
      cost = episode(*a)
      fwd.append(time.perf_counter() - t)
      t = time.perf_counter()
      g = grad(*a)
      bwd.append(time.perf_counter() - t)
      costs.append(float(cost))
      grads.append(np.asarray(g).tolist())
    row = {
      "variant": variant,
      "problem": args.problem,
      "forward_s": fwd,
      "backward_s": bwd,
      "aggregate_cost": costs,
      "gradient": grads,
      "t_first_s": t_first,
    }
    if variant == "per_solve_implicit":
      truncated = si.gradient_function(p, si.episode_function(p, False, "implicit_no_x0"))
      row["gradient_truncated_seed0"] = np.asarray(truncated(*si.arguments(data[0]))).tolist()
      longer = dataclasses.replace(p, horizon=p.horizon + 1, name=f"{p.name}_long")
      truncated = si.gradient_function(longer, si.episode_function(longer, False, "implicit_no_x0"))
      row["gradient_truncated_long_seed0"] = np.asarray(truncated(*si.arguments(data[0]))).tolist()
    rows.append(row)
    print(
      f"problem {args.problem} {variant:20s} forward {1e3 * np.median(fwd):8.2f} ms  gradient {1e3 * np.median(bwd):8.1f} ms  first {t_first:.1f} s",
      flush=True,
    )
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps({"problem": args.problem, "seeds": args.seeds, "rows": rows}, indent=1))


if __name__ == "__main__":
  main()
