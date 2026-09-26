"""PR 4 (T2-4): what ``SparseMatrix`` code costs. A KKT matrix ``[[P + rho I, A^T], [A, -delta I]]``
is assembled from the values of ``P`` and ``A`` (inputs) and multiplied by a vector, and ``A^T A``
is formed by the build-time product pattern. Timed through the JIT against SciPy doing the same in
Python (assembly with ``bmat``, the product with ``@``).

Usage: ``python pr4_sparse_matrix.py n...`` (``A`` is ``n/2 x n``, about 4 entries per row of ``P``).
"""

from __future__ import annotations

import sys
import time

import numpy as np
from scipy import sparse

import scaly as sc
from scaly.codegen import render_c_module
from scaly.linalg import SparseMatrix

from bench_common import median_us


def problem(n: int) -> tuple[sparse.csc_array, sparse.csc_array]:
  rng = np.random.default_rng(n)
  p = sparse.random_array((n, n), density=2.0 / n, random_state=rng, format="csc")
  p = sparse.csc_array(p + p.T + sparse.eye_array(n))
  a = sparse.csc_array(sparse.random_array((n // 2, n), density=3.0 / n, random_state=rng, format="csc"))
  p.sort_indices()
  a.sort_indices()
  return p, a


def main() -> None:
  for n in (int(v) for v in sys.argv[1:]):
    p_np, a_np = problem(n)
    P, A = SparseMatrix.symbol("P", p_np), SparseMatrix.symbol("A", a_np)
    rho, delta, z = sc.sym("rho", ()), sc.sym("delta", ()), sc.sym("z", n + n // 2)
    kkt = SparseMatrix.block([[P.add_diagonal(rho), A.T], [A, SparseMatrix.identity(n // 2) * (-delta)]])
    ata = A.T @ A
    cases = {
      "kkt values": sc.Function._from_exprs(f"p4k{n}", [P.values, A.values, rho, delta], [kkt.values], ["P", "A", "rho", "delta"], ["K"], output_sparsities=[kkt.sparsity]),
      "kkt @ z": sc.Function._from_exprs(f"p4kz{n}", [P.values, A.values, rho, delta, z], [kkt @ z], ["P", "A", "rho", "delta", "z"], ["Kz"]),
      "A^T A values": sc.Function._from_exprs(f"p4ata{n}", [A.values], [ata.values], ["A"], ["AtA"], output_sparsities=[ata.sparsity]),
    }
    pv = np.asarray(p_np.tocsr()[P.coordinates()]).reshape(-1)
    av = np.asarray(a_np.tocsr()[A.coordinates()]).reshape(-1)
    zv = np.sin(np.arange(n + n // 2))
    points = {"kkt values": (pv, av, 1e-6, 1e-4), "kkt @ z": (pv, av, 1e-6, 1e-4, zv), "A^T A values": (av,)}
    ref_k = sparse.bmat([[p_np + 1e-6 * sparse.eye_array(n), a_np.T], [a_np, -1e-4 * sparse.eye_array(n // 2)]], format="csc")
    for tag, fn in cases.items():
      t0 = time.perf_counter()
      body = render_c_module(fn).body
      out = np.asarray(fn._flat_numerical_call(*points[tag])[0]).reshape(-1)
      first = (time.perf_counter() - t0) * 1e3
      if tag == "kkt @ z":
        err = np.abs(out - ref_k @ zv).max()
      elif tag == "kkt values":
        err = np.abs(sparse.csc_array((out, kkt.indices, kkt.indptr), shape=kkt.shape).toarray() - ref_k.toarray()).max()
      else:
        err = np.abs(sparse.csc_array((out, ata.indices, ata.indptr), shape=ata.shape).toarray() - (a_np.T @ a_np).toarray()).max()
      us = median_us(lambda fn=fn, pt=points[tag]: fn._flat_numerical_call(*pt), repeat=21)
      print(f"{tag:<13} n={n:>5} nnz={len(out) if tag != 'kkt @ z' else kkt.nnz:>6} lines={len(body.splitlines()):>5} first_ms={first:>6.0f} us={us:>8.2f} err={err:.1e}", flush=True)
    eye_n, eye_m = sparse.eye_array(n), sparse.eye_array(n // 2)
    t_asm = median_us(lambda: sparse.bmat([[p_np + 1e-6 * eye_n, a_np.T], [a_np, -1e-4 * eye_m]], format="csc"), repeat=21)
    t_mv = median_us(lambda: ref_k @ zv, repeat=21)
    t_ata = median_us(lambda: a_np.T @ a_np, repeat=21)
    print(f"{'scipy':<13} n={n:>5} bmat us={t_asm:.1f}  K@z us={t_mv:.2f}  A^T A us={t_ata:.1f}", flush=True)


if __name__ == "__main__":
  main()
