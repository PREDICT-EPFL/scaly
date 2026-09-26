"""C-119: the cost of ``sc.S`` at a numerical call.

A ``Function`` taking ``sc.S("K", pattern)`` checks each SciPy matrix against the declared pattern
(and converts it to canonical CSC first when it is not) before the values reach the C call. This
compares ``fn((K, x))`` with the raw seam ``fn._flat_numerical_call(K.data, x)`` for a sparse
``K @ x`` at several sizes, for a CSC argument, a CSR argument (a conversion per call) and a
sparse result (``csc_array`` built on the way out).

Usage: ``python followup_sparse_args.py``.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

import scaly as sc

from bench_common import median_us


def main() -> None:
  rng = np.random.default_rng(0)
  print("| n | nnz | flat seam (us) | S, CSC in (us) | S, CSR in (us) | S out (us) |")
  print("| ---: | ---: | ---: | ---: | ---: | ---: |")
  for n in (10, 100, 1000, 10000):
    k = sparse.random_array((n, n), density=min(1.0, 5.0 / n), rng=rng, format="csc") + sparse.eye_array(n, format="csc")
    k = sparse.csc_array(k)
    k.sort_indices()

    @sc.function(sc.G(sc.S("K", k), sc.L("x", n)), sc.L("y", ...), name=f"fs_matvec_{n}")
    def matvec(inputs):
      m, x = inputs
      return m @ x

    @sc.function(sc.S("K", k), sc.S("J", ...), name=f"fs_scale_{n}")
    def scale(m):
      return m * 2.0

    x, csr = rng.standard_normal(n), sparse.csr_array(k)
    np.testing.assert_allclose(matvec((k, x)), k @ x, rtol=1e-12)
    flat = median_us(lambda: matvec._flat_numerical_call(k.data, x))
    csc_in = median_us(lambda: matvec((k, x)))
    csr_in = median_us(lambda: matvec((csr, x)))
    out = median_us(lambda: scale(k))
    print(f"| {n} | {k.nnz} | {flat:.1f} | {csc_in:.1f} | {csr_in:.1f} | {out:.1f} |")


if __name__ == "__main__":
  main()
