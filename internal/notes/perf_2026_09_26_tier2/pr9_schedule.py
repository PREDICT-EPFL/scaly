"""PR 9 (T2-9): the sparse ``L D L^T`` schedules, refinement and the solve's sparsity override.

1. ``schedule="unroll"`` (straight-line code) against ``schedule="scan"`` (column loops) on small
   KKT systems, around the ``sparse_unroll`` default of 2000 multiply-adds: generation time, factor
   and solve time through the JIT (minus an empty call), and the C baseline of PR 7.
2. Iterative refinement: a solve with 0, 2 fixed and up to 5 adaptive steps at ``delta = 1e-10``,
   its time and backward error.
3. ``jacobian_sparsity`` of a solve with the component override against the body's pattern walked
   through the loops.

Usage: ``python pr9_schedule.py``.
"""

from __future__ import annotations

import ctypes
import time

import numpy as np
from scipy import sparse

import scaly as sc
from scaly.ad.sparsity import jacobian_sparsity
from scaly.linalg import SparseLDL, SparseMatrix

from pr5_symbolic import mpc_kkt
from pr7_sparse_ldl import baseline, best_us, random_qp

_ID = iter(range(10_000))


def lower_of(k: sparse.csc_array) -> tuple[SparseMatrix, sparse.csc_array, np.ndarray]:
  lower = sparse.csc_array(sparse.tril(k))
  lower.sort_indices()
  mat = SparseMatrix.symbol("K", lower)
  return mat, lower, np.asarray(lower.tocsr()[mat.coordinates()]).reshape(-1)


def c_times(lib: ctypes.CDLL, fact: SparseLDL, kv: np.ndarray, bv: np.ndarray) -> tuple[float, float]:
  s, n = fact.symbolic, fact.n
  pl = sparse.csc_array((kv[s.a_source], s.a_rows, s.a_ptr), shape=(n, n))
  up = sparse.csc_array(pl.T)
  up.sort_indices()
  i32 = lambda a: np.ascontiguousarray(a, dtype=np.int32)  # noqa: E731
  P = lambda a: a.ctypes.data_as(ctypes.c_void_p)  # noqa: E731
  ap, ai, ax = i32(up.indptr), i32(up.indices), np.ascontiguousarray(up.data)
  lp, par = i32(s.l_ptr), i32(s.parent)
  li, lx, d = np.zeros(s.nnz_l, np.int32), np.zeros(s.nnz_l), np.zeros(n)
  y, pat, flag, lnz = np.zeros(n), np.zeros(n, np.int32), np.zeros(n, np.int32), np.zeros(n, np.int32)
  cf = lib.bench_factor(n, P(ap), P(ai), P(ax), P(lp), P(par), P(li), P(lx), P(d), P(y), P(pat), P(flag), P(lnz), 2000) * 1e6
  bp, xo = np.ascontiguousarray(bv[s.perm]), np.zeros(n)
  cs = lib.bench_solve(n, P(lp), P(li), P(lx), P(d), P(bp), P(xo), 5000) * 1e6
  return cf, cs


BATCH = 256


def batched(tag: str, fn: sc.Function, sizes: list[int]) -> sc.Function:
  """``fn`` over ``BATCH`` instances in one call (a ``vmap``), so the per-instance time excludes the
  Python call."""
  ins = [sc.sym(f"{tag}_{i}", BATCH * size) for i, size in enumerate(sizes)]
  out = sc.vmap(fn, BATCH, [(x, 0, size) for x, size in zip(ins, sizes, strict=True)])
  return sc.Function._from_exprs(tag, ins, [out], [f"i{i}" for i in range(len(ins))], ["o"])


def schedules(lib: ctypes.CDLL) -> None:
  print(f"# schedule: unroll vs scan (per instance, {BATCH} per call)")
  cases = {
    "mpc N=2": mpc_kkt(2),
    "mpc N=5": mpc_kkt(5),
    "mpc N=10": mpc_kkt(10),
    "qp 10+5": random_qp(10, 5, 1, 1e-4),
    "qp 20+10": random_qp(20, 10, 2, 1e-4),
    "qp 40+20": random_qp(40, 20, 3, 1e-4),
  }
  for name, k in cases.items():
    mat, lower, kv = lower_of(sparse.csc_array(k))
    n = k.shape[0]
    bv = np.cos(np.arange(n))
    row = []
    for schedule in ("scan", "unroll"):
      t0 = time.perf_counter()
      fact = SparseLDL(mat, schedule=schedule)
      tag = f"p9s{next(_ID)}"
      factor_fn = sc.Function._from_exprs(tag, [mat.values], [fact.values], ["K"], ["F"])
      fv = factor_fn._flat_numerical_call(kv)[0]
      solver = fact._solver_function()
      solve_fn = sc.Function._from_exprs(f"{tag}_s", list(solver.inputs), [solver._flat_symbolic_call(list(solver.inputs))[0]], ["f", "kv", "b"], ["x"])
      solve_fn._flat_numerical_call(fv, kv, bv)
      gen = time.perf_counter() - t0
      bf = batched(f"{tag}_bf", factor_fn, [mat.nnz])
      bs = batched(f"{tag}_bs", solve_fn, [fv.size, mat.nnz, n])
      kvs, fvs, bvs = np.tile(kv, BATCH), np.tile(fv, BATCH), np.tile(bv, BATCH)
      tf = best_us(lambda: bf._flat_numerical_call(kvs), 100) / BATCH
      ts = best_us(lambda: bs._flat_numerical_call(fvs, kvs, bvs), 100) / BATCH
      row.append((schedule, gen, tf, ts))
    cf, cs = c_times(lib, fact, kv, bv)
    cells = "  ".join(f"{s}: gen={g:4.2f}s factor={tf:6.3f}us solve={ts:6.3f}us" for s, g, tf, ts in row)
    print(f"{name:<9} n={fact.n:>3} work={fact.work:>5}  {cells}  C: factor={cf:6.3f}us solve={cs:6.3f}us", flush=True)


def refinement(empty: float) -> None:
  print("# refinement at delta=1e-10 (qp 200+100)")
  k = random_qp(200, 100, 1, 1e-10)
  mat, lower, kv = lower_of(k)
  full = (lower + sparse.tril(lower, -1).T).tocsr()
  bv = np.cos(np.arange(k.shape[0]))
  fact = SparseLDL(mat)
  b = sc.sym("b", k.shape[0])
  for label, opts in (("none", {}), ("fixed 1", {"refine": 1}), ("fixed 2", {"refine": 2}), ("adaptive <=5, tol 1e-14", {"refine": 5, "tol": 1e-14})):
    fn = sc.Function._from_exprs(f"p9r{next(_ID)}", [mat.values, b], [fact.solve(b, **opts)], ["K", "b"], ["x"])
    x = fn._flat_numerical_call(kv, bv)[0]
    back = np.abs(full @ x - bv).max() / (abs(full).sum(axis=1).max() * np.abs(x).max() + np.abs(bv).max())
    t = best_us(lambda: fn._flat_numerical_call(kv, bv), 100) - empty
    print(f"{label:<24} factor+solve={t:8.1f}us  backward={back:.1e}  residual={np.abs(full @ x - bv).max():.1e}", flush=True)


def sparsity() -> None:
  print("# jacobian_sparsity of a solve in b: the override against the body's pattern")
  for name, k in (("mpc N=20", mpc_kkt(20)), ("qp 100+50", random_qp(100, 50, 2, 1e-4))):
    mat, _, _ = lower_of(sparse.csc_array(k))
    fact = SparseLDL(mat, schedule="scan")
    b = sc.sym("b", k.shape[0])
    x = fact.solve(b)
    callee = x.attrs["callee"]
    t0 = time.perf_counter()
    with_override = jacobian_sparsity(x, b)
    t_override = time.perf_counter() - t0
    saved, callee.custom_sparsity = callee.custom_sparsity, None
    try:
      t0 = time.perf_counter()
      body = jacobian_sparsity(x, b)
      t_body = time.perf_counter() - t0
    finally:
      callee.custom_sparsity = saved
    print(f"{name:<10} n={fact.n:>4} override={t_override * 1e3:8.1f}ms nnz={with_override.nnz}  body={t_body * 1e3:8.1f}ms nnz={body.nnz}", flush=True)


def main() -> None:
  lib = baseline()
  x = sc.sym("p9_x", ())
  trivial = sc.Function._from_exprs("p9_trivial", [x], [x * 2.0], ["x"], ["y"])
  empty = best_us(lambda: trivial._flat_numerical_call(np.array(1.0)), 300)
  schedules(lib)
  refinement(empty)
  sparsity()


if __name__ == "__main__":
  main()
