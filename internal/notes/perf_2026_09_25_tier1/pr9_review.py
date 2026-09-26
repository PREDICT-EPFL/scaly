"""PR 9: the review round's optimizations, measured on the same workloads as PR 8 plus the gradient,
the Jacobian of the rollout and a trivial call.

Usage: ``python pr9_review.py N...``. Run once with the tree under test on ``PYTHONPATH`` and once
with the PR 8 tree (``git archive 1287945 src``) to compare; each run prints one line per case.
"""

from __future__ import annotations

import re
import sys
import time

import numpy as np

import scaly as sc
from scaly.codegen import render_c_module

from bench_common import median_us
from pr8_multiseed import mpc_cost, rk4_cost


def cases(n: int) -> dict[str, tuple[sc.Function, tuple[np.ndarray, np.ndarray]]]:
  mpc, rk4 = mpc_cost(n), rk4_cost(n)
  mpc_point = (np.array([1.0, -0.5]), np.sin(np.arange(n)) * 0.3)
  rk4_point = (np.array([0.1, 0.4, -0.2, 0.05]), np.sin(np.arange(n)) * 0.3)
  z0, us = rk4.inputs
  return {
    "mpc hessian": (sc.hessian(mpc, "f", "U"), mpc_point),
    "mpc gradient": (sc.gradient(mpc, "f", "U"), mpc_point),
    "rk4 hessian": (sc.hessian(rk4, "f", "us"), rk4_point),
    "rk4 gradient": (sc.gradient(rk4, "f", "us"), rk4_point),
    "rk4 fwd-mode": (sc.Function._from_exprs(f"r9_jac{n}", [z0, us], [sc.jacobian(rk4.outputs[0], us)], ["z0", "us"], ["j"]), rk4_point),
  }


def main() -> None:
  x = sc.sym("r9_x", 3)
  trivial = sc.Function._from_exprs("r9_trivial", [x], [x * 2.0], ["x"], ["y"])
  v = np.ones(3)
  print(f"{'trivial call':<14} {'':>5} us={median_us(lambda: trivial(v)):>9.2f}", flush=True)
  for n in (int(a) for a in sys.argv[1:]):
    for tag, (fn, point) in cases(n).items():
      t0 = time.perf_counter()
      body = render_c_module(fn).body
      fn(point)
      first = (time.perf_counter() - t0) * 1e3
      us = median_us(lambda fn=fn, point=point: fn(point), repeat=21)
      print(
        f"{tag:<14} {n:>5} lines={len(body.splitlines()):>6} loops={body.count('for ('):>5} procs={len(re.findall(r'void \w+_raw[(]', body)):>3} first_ms={first:>7.0f} us={us:>9.2f}",
        flush=True,
      )


if __name__ == "__main__":
  main()
