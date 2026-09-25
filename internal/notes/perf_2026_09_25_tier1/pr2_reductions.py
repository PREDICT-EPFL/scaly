"""PR 2: ``reduce_max`` / ``norm_inf`` against ``sum`` and NumPy, and the cost of the tie-aware gradient."""

from __future__ import annotations

import numpy as np

import scaly as sc

from bench_common import median_us, timed


def main() -> None:
  print(f"{'n':>9}  {'variant':<28}{'build ms':>10}{'median us/call':>16}")
  for n in (1_000, 100_000, 1_000_000):
    x = sc.sym("x", n)
    xv = np.random.default_rng(0).normal(size=n)
    variants = {
      "sum(x)": x.sum(),
      "reduce_max(x)": x.max(),
      "norm_inf(x)": sc.norm_inf(x),
      "grad reduce_max (split)": sc.vjp((x.max(),), (x,), (sc.const(1.0),))[0],
    }
    with sc.options(nonsmooth="first"):
      variants["grad reduce_max (first)"] = sc.vjp((x.max(),), (x,), (sc.const(1.0),))[0]
    for label, out in variants.items():
      fun = sc.Function._from_exprs(f"b2_{label.replace(' ', '_').replace('(', '').replace(')', '')}_{n}", [x], [out], ["x"], ["y"])
      _, build_ms = timed(lambda: fun(xv))
      print(f"{n:>9}  {label:<28}{build_ms:>10.1f}{median_us(lambda: fun(xv)):>16.1f}")
    for label, fn in {"numpy max": lambda: np.max(xv), "numpy abs().max()": lambda: np.abs(xv).max(), "numpy sum": lambda: np.sum(xv)}.items():
      print(f"{n:>9}  {label:<28}{0.0:>10.1f}{median_us(fn):>16.1f}")


if __name__ == "__main__":
  main()
