"""Generated C size and evaluation time of the dense solves' Jacobians in ``K``, for #491.

Usage: ``uv run internal/notes/tangent_rules_2026_11_23/measure.py <label>``, which writes
``measurements-<label>.json`` beside this script. Times are the median of repeated calls of the
compiled function on one machine; they compare before and after, not with other tools.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

import scaly as sc
from scaly import linalg
from scaly.codegen import render_c_source


def _matrix(n: int, assume: str) -> np.ndarray:
  rng = np.random.default_rng(n)
  m = rng.normal(size=(n, n))
  if assume == "gen":
    return m + n * np.eye(n)
  return m @ m.T + n * np.eye(n)


def _solve_jacobian(n: int, assume: str):
  @sc.function(sc.arg("K", (n, n)), sc.arg("b", n), outputs=sc.arg("x", n), name=f"solve_{assume}_{n}")
  def solve(K, b):
    return linalg.solve(K, b, assume=assume)[0]

  return sc.jacobian(solve, "x", "K")


def _cholesky_jacobian(n: int):
  @sc.function(sc.arg("A", (n, n)), outputs=sc.arg("L", (n, n)), name=f"factor_{n}")
  def factor(A):
    return linalg.cholesky(A)[0]

  return sc.jacobian(factor, "L", "A")


def _time(fn, *args, budget: float = 2.0) -> float:
  fn(*args)
  samples, spent = [], 0.0
  while spent < budget or len(samples) < 5:
    start = time.perf_counter()
    fn(*args)
    samples.append(time.perf_counter() - start)
    spent += samples[-1]
  return statistics.median(samples) * 1e3


def main() -> None:
  label = sys.argv[1]
  rows = []
  for assume in ("pos", "sym", "gen"):
    for n in (4, 16):
      start = time.perf_counter()
      source = render_c_source(_solve_jacobian(n, assume))
      rows.append({"what": f"jacobian of solve(assume={assume!r}) in K", "n": n, "c_bytes": len(source.encode()), "generate_s": round(time.perf_counter() - start, 2)})
      print(rows[-1], flush=True)
  for n in (8, 16, 24, 32):
    a = _matrix(n, "pos")
    for what, jac, args in (
      ("jacobian of cholesky in A", _cholesky_jacobian(n), (a,)),
      ("jacobian of solve(assume='pos') in K", _solve_jacobian(n, "pos"), (a, np.ones(n))),
    ):
      start = time.perf_counter()
      jac.compile()
      build = time.perf_counter() - start
      rows.append({"what": what, "n": n, "build_s": round(build, 2), "run_ms": round(_time(jac, *args), 3)})
      print(rows[-1], flush=True)
  out = Path(__file__).with_name(f"measurements-{label}.json")
  out.write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
  main()
