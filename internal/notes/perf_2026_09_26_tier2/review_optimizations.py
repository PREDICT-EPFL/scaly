"""T2-R: the optimizations of the Tier 2 review round.

1. The cost of a JIT call on small arrays (``CompiledFunction.run``), now that array addresses are
   read through the buffer protocol instead of ``arr.ctypes.data``.
2. Adaptive refinement (``solve(b, refine=5, tol=1e-14)``) on a random QP at ``delta = 1e-10``,
   against fixed refinement: its loop now updates the solution and residual in place.
3. Python generation time (lowering, program passes, rendering) of an unrolled sparse factorization,
   with ``fuse_elementwise`` and ``pack_workspace`` no longer quadratic.

Usage: ``python review_optimizations.py``.
"""

from __future__ import annotations

import time

import numpy as np
from scipy import sparse

import scaly as sc
from scaly.codegen import render_c_module
from scaly.linalg import SparseLDL, SparseMatrix

from bench_common import median_us
from pr5_symbolic import mpc_kkt
from pr7_sparse_ldl import random_qp


def call_cost() -> None:
  x, y = sc.sym("rv_x", 3), sc.sym("rv_y", 3)
  fn = sc.Function._from_exprs("rv_trivial", [x, y], [x * 2.0 + y], ["x", "y"], ["z"])
  xv, yv = np.ones(3), np.ones(3)
  compiled = fn._compile()
  print(f"# call cost: run {median_us(lambda: compiled.run([xv, yv])):.2f} us, _flat_numerical_call {median_us(lambda: fn._flat_numerical_call(xv, yv)):.2f} us")


def refinement() -> None:
  k = random_qp(200, 100, 1, 1e-10)
  lower = sparse.csc_array(sparse.tril(k))
  lower.sort_indices()
  mat = SparseMatrix.symbol("K", lower)
  kv = np.asarray(lower.tocsr()[mat.coordinates()]).reshape(-1)
  bv = np.cos(np.arange(k.shape[0]))
  fact = SparseLDL(mat)
  b = sc.sym("b", k.shape[0])
  print("# refinement, qp 200+100, delta 1e-10 (factor + solve)")
  for label, opts in (("none", {}), ("fixed 2", {"refine": 2}), ("adaptive <=5, tol 1e-14", {"refine": 5, "tol": 1e-14})):
    fn = sc.Function._from_exprs(f"rv_r{len(label)}", [mat.values, b], [fact.solve(b, **opts)], ["K", "b"], ["x"])
    print(f"{label:<24} {median_us(lambda: fn._flat_numerical_call(kv, bv)):8.1f} us", flush=True)


def generation() -> None:
  print("# Python generation of an unrolled factorization")
  for tag, k in (("qp 40+20", random_qp(40, 20, 3, 1e-4)), ("mpc N=20", mpc_kkt(20))):
    lower = sparse.csc_array(sparse.tril(sparse.csc_array(k)))
    lower.sort_indices()
    mat = SparseMatrix.symbol("K", lower)
    fact = SparseLDL(mat, schedule="unroll", name=f"rv{len(tag)}{k.shape[0]}")
    fn = sc.Function._from_exprs(f"rv_gen{k.shape[0]}", [mat.values], [fact.values], ["K"], ["F"])
    t0 = time.perf_counter()
    render_c_module(fn)
    print(f"{tag:<10} work={fact.work:>5} render={time.perf_counter() - t0:5.2f} s", flush=True)


if __name__ == "__main__":
  call_cost()
  refinement()
  generation()
