"""C code generation: the gradient of the determinant of a 7x7 matrix, compiled at three levels (Scaly).

A ``Function`` compiles on its first call, and ``SCALY_CC_OPT`` sets the optimization level (the
flag set is part of the cache key, so each level is its own library). ``write_module`` writes the
same C to a directory for use outside Python.

Scaly has no ``det``; this one expands by minors along the rows and computes each minor once (there
are 2^7 of them), which is what keeps the expression small.

After casadi/docs/examples/python/c_code_generation.py (Joel Andersson, 2016).
"""

import functools
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import scaly as sc
from _common import show
from scaly.codegen import write_module

N = 7
X0 = np.random.default_rng(0).random((N, N))


def det(a: sc.Expr) -> sc.Expr:
  n = a.shape[0]

  @functools.cache
  def minor(cols: tuple[int, ...]) -> sc.Expr:  # the determinant of the last len(cols) rows, restricted to cols
    row = n - len(cols)
    if len(cols) == 1:
      return a[row, cols[0]]
    total = a[row, cols[0]] * minor(cols[1:])
    for i in range(1, len(cols)):
      term = a[row, cols[i]] * minor(cols[:i] + cols[i + 1 :])
      total = total - term if i % 2 else total + term
    return total

  return minor(tuple(range(n)))


def build(verbose: bool = False, opt: str = "-O3"):
  os.environ["SCALY_CC_OPT"] = opt

  @sc.function((N, N), output="gd")
  def grad_det(x):
    return sc.gradient(det(x), x)

  grad_det(X0)  # the first call generates, compiles and loads
  if verbose:
    out_dir = Path(tempfile.mkdtemp(prefix="grad_det_"))
    write_module(grad_det, out_dir)
    size = sum(p.stat().st_size for p in out_dir.glob("*.c"))
    print(f"{size / 1e3:.0f} kB of C in {out_dir}")

  def run():
    return {"gd": grad_det(X0).reshape(-1)}

  return run


if __name__ == "__main__":
  for opt in ["-O0", "-O3", "-Os"] if len(sys.argv) < 2 else sys.argv[1:]:
    t0 = time.perf_counter()
    run = build(verbose=True, opt=opt)
    print(f"{opt}: generated and compiled in {time.perf_counter() - t0:.2f} s")
    t0 = time.perf_counter()
    for _ in range(1000):
      out = run()
    print(f"{opt}: {(time.perf_counter() - t0) * 1e3:.3f} us per evaluation")
  show(out)
