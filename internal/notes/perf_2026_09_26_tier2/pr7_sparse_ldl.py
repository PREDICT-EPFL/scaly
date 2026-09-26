"""PR 7 (T2-7): the generated sparse LDL^T and solve against an up-looking C factorization of the
QDLDL kind (``ldl_baseline.c``, compiled at ``-O3`` with the native CPU flag), on the same permuted
matrix with the same symbolic analysis.

Matrices: stage-ordered MPC KKT systems, random QP KKT systems ``[[P + rho I, A^T], [A, -delta I]]``
and a grid Laplacian with a negative block. Scaly's times are the best JIT call minus an empty
call; the baseline times the factorization (or solve) alone in C. The residual is
``||K x - b||_inf / (||K||_inf ||x||_inf + ||b||_inf)``.

Usage: ``python pr7_sparse_ldl.py`` (``SCALY_CC_OPT`` sets Scaly's level; the baseline is ``-O3``).
"""

from __future__ import annotations

import ctypes
import os
import platform
import subprocess
import tempfile
import time

import numpy as np
from scipy import sparse

import scaly as sc
from scaly.linalg import SparseLDL, SparseMatrix

from pr5_symbolic import grid, mpc_kkt

HERE = os.path.dirname(os.path.abspath(__file__))


def baseline() -> ctypes.CDLL:
  out = os.path.join(tempfile.mkdtemp(), "ldl_baseline.so")
  cpu = "-mcpu=native" if platform.machine().lower() in {"arm64", "aarch64"} else "-march=native"
  subprocess.run(["cc", "-O3", cpu, "-shared", "-fPIC", os.path.join(HERE, "ldl_baseline.c"), "-o", out], check=True)
  lib = ctypes.CDLL(out)
  lib.bench_factor.restype = ctypes.c_double
  lib.bench_solve.restype = ctypes.c_double
  return lib


def random_qp(n: int, m: int, seed: int, delta: float) -> sparse.csc_array:
  rng = np.random.default_rng(seed)
  p = sparse.random_array((n, n), density=3.0 / n, random_state=rng)
  p = p @ p.T + sparse.eye_array(n) * 1e-2
  a = sparse.random_array((m, n), density=4.0 / n, random_state=rng)
  return sparse.csc_array(sparse.bmat([[p + 1e-6 * sparse.eye_array(n), a.T], [a, -delta * sparse.eye_array(m)]]))


def quasi_grid(k: int) -> sparse.csc_array:
  g = grid(k)
  n = g.shape[0]
  signs = np.where(np.arange(n) % 5 == 4, -1.0, 1.0)
  return sparse.csc_array(g.multiply(np.outer(signs, signs) > 0) + sparse.diags_array(signs * 0.5))


def best_us(f, reps: int = 30) -> float:
  f()
  best = np.inf
  for _ in range(reps):
    t0 = time.perf_counter()
    f()
    best = min(best, time.perf_counter() - t0)
  return best * 1e6


def main() -> None:
  lib = baseline()
  x = sc.sym("p7_x", ())
  trivial = sc.Function._from_exprs("p7_trivial", [x], [x * 2.0], ["x"], ["y"])
  empty = best_us(lambda: trivial._flat_numerical_call(np.array(1.0)), 200)
  cases = {
    "mpc N=100": sparse.csc_array(mpc_kkt(100)),
    "mpc N=1000": sparse.csc_array(mpc_kkt(1000)),
    "qp 200+100": random_qp(200, 100, 1, 1e-4),
    "qp 500+250": random_qp(500, 250, 2, 1e-4),
    "qp 1000+500": random_qp(1000, 500, 4, 1e-4),
    "grid 30x30": quasi_grid(30),
    "grid 60x60": quasi_grid(60),
  }
  for name, k in cases.items():
    lower = sparse.csc_array(sparse.tril(k))
    lower.sort_indices()
    K = SparseMatrix.symbol("K", lower)
    kv = np.asarray(lower.tocsr()[K.coordinates()]).reshape(-1)
    t0 = time.perf_counter()
    fact = SparseLDL(K)
    b = sc.sym("b", k.shape[0])
    factor_fn = sc.Function._from_exprs(f"p7f_{name.replace(' ', '_').replace('=', '').replace('+', '_').replace('x', 'by')}", [K.values], [fact.values], ["K"], ["F"])
    fv = factor_fn._flat_numerical_call(kv)[0]
    gen_s = time.perf_counter() - t0
    solver = fact._solver_function()
    solve_fn = sc.Function._from_exprs(f"{factor_fn.name}_s", [solver.inputs[0], solver.inputs[1], solver.inputs[2]], [solver._flat_symbolic_call(list(solver.inputs))[0]], ["f", "kv", "b"], ["x"])
    bv = np.cos(np.arange(k.shape[0]))
    xv = solve_fn._flat_numerical_call(fv, kv, bv)[0]
    full = (lower + sparse.tril(lower, -1).T).tocsr()
    resid = np.abs(full @ xv - bv).max() / (abs(full).sum(axis=1).max() * np.abs(xv).max() + np.abs(bv).max())
    t_factor = best_us(lambda: factor_fn._flat_numerical_call(kv)) - empty
    t_solve = best_us(lambda: solve_fn._flat_numerical_call(fv, kv, bv)) - empty
    # The baseline on the same permuted matrix and analysis.
    s = fact.symbolic
    n = s.n
    pl = sparse.csc_array((kv[s.a_source], s.a_rows, s.a_ptr), shape=(n, n))
    up = sparse.csc_array(pl.T)
    up.sort_indices()
    i32 = lambda a: np.ascontiguousarray(a, dtype=np.int32)  # noqa: E731
    ap, ai, ax = i32(up.indptr), i32(up.indices), np.ascontiguousarray(up.data)
    lp, par = i32(s.l_ptr), i32(s.parent)
    li, lx, d = np.zeros(s.nnz_l, np.int32), np.zeros(s.nnz_l), np.zeros(n)
    y, pat, flag, lnz = np.zeros(n), np.zeros(n, np.int32), np.zeros(n, np.int32), np.zeros(n, np.int32)
    P = lambda a: a.ctypes.data_as(ctypes.c_void_p)  # noqa: E731
    c_factor = lib.bench_factor(n, P(ap), P(ai), P(ax), P(lp), P(par), P(li), P(lx), P(d), P(y), P(pat), P(flag), P(lnz), 50) * 1e6
    bp = np.ascontiguousarray(bv[s.perm])
    xo = np.zeros(n)
    c_solve = lib.bench_solve(n, P(lp), P(li), P(lx), P(d), P(bp), P(xo), 200) * 1e6
    np.testing.assert_allclose(fv[s.nnz_l : s.nnz_l + n], d, rtol=1e-9, atol=1e-12)
    st = s.stats()
    print(
      f"{name:<12} n={n:>6} nnzL={s.nnz_l:>7} lanes={st['update_lanes']:>8} order={s.method:<12} segs={len(fact.segments):>3} gen={gen_s:>5.1f}s "
      f"factor scaly={t_factor:>9.1f}us c={c_factor:>8.1f}us ratio={t_factor / c_factor:>5.2f}  solve scaly={t_solve:>7.1f}us c={c_solve:>7.1f}us ratio={t_solve / c_solve:>5.2f}  resid={resid:.1e}",
      flush=True,
    )


if __name__ == "__main__":
  main()
