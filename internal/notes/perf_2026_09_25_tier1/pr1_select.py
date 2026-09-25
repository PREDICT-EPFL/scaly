"""PR 1: cost of ``select``-based piecewise code against ``fmax`` and NumPy, and its build time."""

from __future__ import annotations

import sys

import numpy as np

import scaly as sc

from bench_common import median_us, timed


def relu_functions(n: int) -> dict[str, sc.Function]:
  x = sc.sym("x", n)
  return {
    "where(x>0,x,0)": sc.Function._from_exprs(f"b_relu_where_{n}", [x], [sc.where(x > 0.0, x, 0.0)], ["x"], ["y"]),
    "maximum(x,0)": sc.Function._from_exprs(f"b_relu_fmax_{n}", [x], [sc.maximum(x, 0.0)], ["x"], ["y"]),
    "saturate(where)": sc.Function._from_exprs(f"b_sat_{n}", [x], [sc.where(x > 1.0, 1.0, sc.where(x < -1.0, -1.0, x))], ["x"], ["y"]),
    "saturate(fmin/fmax)": sc.Function._from_exprs(f"b_satm_{n}", [x], [sc.minimum(sc.maximum(x, -1.0), 1.0)], ["x"], ["y"]),
  }


def main() -> None:
  rows = []
  for n in (1_000, 100_000, 1_000_000):
    xv = np.random.default_rng(0).normal(size=n)
    for label, fun in relu_functions(n).items():
      _, build_ms = timed(lambda: fun(xv))
      rows.append((n, label, build_ms, median_us(lambda: fun(xv))))
    rows.append((n, "numpy where", 0.0, median_us(lambda: np.where(xv > 0.0, xv, 0.0))))
    rows.append((n, "numpy clip", 0.0, median_us(lambda: np.clip(xv, -1.0, 1.0))))
  print(f"{'n':>9}  {'variant':<22}{'first call (build) ms':>22}{'median us/call':>16}")
  for n, label, build_ms, us in rows:
    print(f"{n:>9}  {label:<22}{build_ms:>22.1f}{us:>16.1f}")
  sys.stdout.flush()


if __name__ == "__main__":
  main()
