"""PR 3: SpMV from ``gather`` + ``segment_sum`` against dense ``matmul``, SciPy and NumPy, and the
build cost of a ``gather`` adjoint before (one sum per entry) and after (one accumulating scatter)."""

from __future__ import annotations

import time

import numpy as np
from scipy import sparse

import scaly as sc
from scaly.codegen import render_c_source

from bench_common import median_us, timed


def spmv(m: int, n: int, density: float) -> None:
  pat = sparse.random_array((m, n), density=density, format="coo", random_state=0)
  rows, cols = pat.row.astype(np.int64), pat.col.astype(np.int64)
  a, x = sc.sym("a", pat.nnz), sc.sym("x", n)
  sp = sc.Function._from_exprs(f"b3_sp_{m}_{pat.nnz}", [a, x], [sc.segment_sum(a * sc.gather(x, cols), rows, m)], ["a", "x"], ["y"])
  av, xv = np.random.default_rng(1).normal(size=pat.nnz), np.random.default_rng(2).normal(size=n)
  csr = sparse.coo_array((av, (rows, cols)), shape=(m, n)).tocsr()
  dense = csr.toarray()
  _, build = timed(lambda: sp((av, xv)))
  row = [f"{m}x{n}", f"{pat.nnz}", f"{build:.0f}", f"{median_us(lambda: sp((av, xv))):.1f}", f"{median_us(lambda: csr @ xv):.1f}", f"{median_us(lambda: dense @ xv):.1f}"]
  if m * n <= 250_000:
    A = sc.sym("A", (m, n))
    dn = sc.Function._from_exprs(f"b3_dn_{m}", [A, x], [A @ x], ["A", "x"], ["y"])
    _, bd = timed(lambda: dn((dense, xv)))
    row += [f"{median_us(lambda: dn((dense, xv))):.1f}"]
  else:
    row += ["-"]
  print("  ".join(f"{c:>10}" for c in row))


def old_gather_vjp(cot: sc.Expr, indices: np.ndarray, size: int) -> sc.Expr:
  vals = []
  for i in range(size):
    positions = np.nonzero(indices == i)[0]
    vals.append(sc.gather(cot, positions).sum() if positions.size else sc.const(0.0))
  return sc.stack(vals)


def gather_adjoint(size: int) -> None:
  picks = np.random.default_rng(0).integers(0, size, size=2 * size)
  x = sc.sym("x", size)
  cot = sc.sym("cot", 2 * size)
  t0 = time.perf_counter()
  new = sc.vjp((sc.gather(x, picks),), (x,), (cot,))[0]
  t_new = time.perf_counter() - t0
  t0 = time.perf_counter()
  old = old_gather_vjp(cot, picks, size)
  t_old = time.perf_counter() - t0
  r_new = r_old = float("nan")
  t0 = time.perf_counter()
  render_c_source(sc.Function._from_exprs(f"b3_gn_{size}", [x, cot], [new], ["x", "cot"], ["g"]))
  r_new = time.perf_counter() - t0
  if size <= 2000:
    t0 = time.perf_counter()
    render_c_source(sc.Function._from_exprs(f"b3_go_{size}", [x, cot], [old], ["x", "cot"], ["g"]))
    r_old = time.perf_counter() - t0
  print(f"{size:>10}{t_old * 1e3:>14.1f}{r_old * 1e3:>14.1f}{t_new * 1e3:>14.2f}{r_new * 1e3:>14.1f}")


def main() -> None:
  print("SpMV, median us per call (build ms is the first call: render + compile)")
  print("  ".join(f"{c:>10}" for c in ("shape", "nnz", "build ms", "scaly sp", "scipy csr", "numpy dn", "scaly dn")))
  for m, n, d in ((100, 100, 0.05), (500, 500, 0.01), (2000, 2000, 0.002), (20000, 20000, 0.0002)):
    spmv(m, n, d)
  print("\ngather adjoint build, ms: graph construction and C render, old (sum per entry) vs new (one scatter)")
  print(f"{'size':>10}{'old build':>14}{'old render':>14}{'new build':>14}{'new render':>14}")
  for size in (200, 2000, 20000):
    gather_adjoint(size)


if __name__ == "__main__":
  main()
