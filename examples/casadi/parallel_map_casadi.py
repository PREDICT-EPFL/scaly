# /// script
# requires-python = ">=3.12"
# dependencies = ["casadi"]
#
# [tool.ty.environment]
# extra-paths = ["."]  # the modules beside this script, which it imports
# ///
"""Evaluating one expensive scalar function at 300 points: a Python loop of calls against ``map`` (CasADi).

The function is sin applied 100 000 times. The original times three ways of evaluating it on a
grid: 300 call nodes in one MX graph, ``map`` (one node, a serial loop), and ``map`` with OpenMP; this
keeps the two maps. The pip wheel of CasADi is built without OpenMP and falls back to serial.

After casadi/docs/examples/python/parallel_map.py.
"""

import time

import casadi as ca
import numpy as np

from _common import casadi_jit, show

N = 300
DEPTH = 100_000
POINTS = np.linspace(0.0, 2.0 * np.pi, N)


def build(verbose: bool = False, parallelization: str = "serial"):
  x = ca.SX.sym("x")
  y = x
  for _ in range(DEPTH):
    y = ca.sin(y)
  f0 = ca.Function("f", [x], [y], casadi_jit(expand=False))
  f_map = f0.map(N, parallelization)

  def run():
    return {"y": f_map(POINTS).full().reshape(-1)}

  return run


if __name__ == "__main__":
  for parallelization in ["serial", "openmp"]:
    t0 = time.perf_counter()
    run = build(parallelization=parallelization)
    print(f"built in {time.perf_counter() - t0:.2f} s")
    t0 = time.perf_counter()
    out = run()
    print(f"map ({parallelization}): {time.perf_counter() - t0:.3f} s")
  show(out)
