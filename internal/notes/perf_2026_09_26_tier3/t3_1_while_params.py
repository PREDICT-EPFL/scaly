"""T3-1 (C-112): loop-invariant ``while_loop`` inputs, measured on SparseLDL's adaptive refinement.

Before C-112 the refinement loop carried ``[factor | K values | b | x | r | threshold]`` and so
copied the factor, the matrix and the right-hand side into its carry on every solve; now they are
params and the carry is ``[x | r]``. This times ``solve(b, refine=5, tol=1e-14)`` (the factorization
included) against the plain solve on random quasi-definite KKT systems of several sizes at
``delta = 1e-10``, where refinement is needed. Run it on two checkouts to compare.

Usage: ``uv run internal/notes/perf_2026_09_26_tier3/t3_1_while_params.py`` from the repository root.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import sparse

import scaly as sc
from scaly.linalg import SparseLDL, SparseMatrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "perf_2026_09_26_tier2"))
from bench_common import median_us  # noqa: E402
from pr7_sparse_ldl import random_qp  # noqa: E402


def main() -> None:
  print("| n + m | nnz(K lower) | plain solve (us) | adaptive refinement <= 5 (us) | overhead (us) |")
  print("| ---: | ---: | ---: | ---: | ---: |")
  for n, m in ((50, 25), (200, 100), (800, 400)):
    k = random_qp(n, m, 1, 1e-10)
    lower = sparse.csc_array(sparse.tril(k))
    lower.sort_indices()
    mat = SparseMatrix.symbol("K", lower)
    kv = np.asarray(lower.tocsr()[mat.coordinates()]).reshape(-1)
    bv = np.cos(np.arange(k.shape[0]))
    fact = SparseLDL(mat, name=f"wpb{n}")
    b = sc.sym("b", k.shape[0])
    plain = sc.Function._from_exprs(f"wpb_plain{n}", [mat.values, b], [fact.solve(b)], ["K", "b"], ["x"])
    refined = sc.Function._from_exprs(f"wpb_ref{n}", [mat.values, b], [fact.solve(b, refine=5, tol=1e-14)], ["K", "b"], ["x"])
    t_plain = median_us(lambda: plain._flat_numerical_call(kv, bv))
    t_ref = median_us(lambda: refined._flat_numerical_call(kv, bv))
    print(f"| {n + m} | {mat.nnz} | {t_plain:.1f} | {t_ref:.1f} | {t_ref - t_plain:.1f} |", flush=True)


if __name__ == "__main__":
  main()
