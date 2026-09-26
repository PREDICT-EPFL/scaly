"""PR 2 (T2-2): products with a sparse matrix whose column indices are read at run time, against the
Tier 1 form whose indices are baked into the code, and against SciPy.

``A`` is ``n x n`` with about ``k`` entries per row (random positions, fixed seed); its values and
``x`` are inputs.

``static``      ``segment_sum(vals * gather(x, cols), rows, n)``: indices fixed when generated.
``take``        one ``take`` of ``x`` by an ``int64`` index input padded to ``w`` lanes per row.
``scan-take``   a scan over rows that slices one padded row of the CSR tables per step (the factorization's shape).
``scan-put``    ``A^T y`` as a scan accumulating each row into the carry with ``put_add``.
``scipy``       ``csr_array @ x`` (``A.T @ y`` for the transpose), timed in Python.

Usage: ``python pr2_take.py n...``.
"""

from __future__ import annotations

import re
import sys
import time

import numpy as np
from scipy import sparse

import scaly as sc
from scaly.codegen import render_c_module

from bench_common import median_us

K = 5


def matrix(n: int) -> sparse.csr_array:
  rng = np.random.default_rng(n)
  a = sparse.random_array((n, n), density=K / n, random_state=rng, format="csr") + sparse.eye_array(n, format="csr")
  a.sort_indices()
  return sparse.csr_array(a)


def padded(a: sparse.csr_array) -> tuple[np.ndarray, np.ndarray, int]:
  n = a.shape[0]
  width = int(np.diff(a.indptr).max())
  cols = np.full((n, width), n, dtype=np.int64)
  pos = np.full((n, width), a.nnz, dtype=np.int64)
  for r in range(n):
    lo, hi = a.indptr[r], a.indptr[r + 1]
    cols[r, : hi - lo] = a.indices[lo:hi]
    pos[r, : hi - lo] = np.arange(lo, hi)
  return cols.reshape(-1), pos.reshape(-1), width


def variants(a: sparse.csr_array) -> dict[str, tuple[sc.Function, tuple[np.ndarray, ...]]]:
  n, nnz = a.shape[0], a.nnz
  cols_t, pos_t, width = padded(a)
  rows = np.repeat(np.arange(n), np.diff(a.indptr))
  vals, x = sc.sym("vals", nnz), sc.sym("x", n)
  xv = np.sin(np.arange(n))
  out: dict[str, tuple[sc.Function, tuple[np.ndarray, ...]]] = {}
  static = sc.segment_sum(vals * sc.gather(x, a.indices), rows, n)
  out["static"] = (sc.Function._from_exprs(f"p2s{n}", [vals, x], [static], ["vals", "x"], ["y"]), (a.data, xv))
  idx = sc.sym("idx", cols_t.size, dtype="int64")
  vpad = sc.sym("vpad", cols_t.size)
  taken = (sc.take(x, idx) * vpad).reshape((n, width)) @ sc.const(np.ones(width))
  vpad_v = np.where(cols_t < n, np.append(a.data, 0.0)[pos_t], 0.0)
  out["take"] = (sc.Function._from_exprs(f"p2t{n}", [vpad, idx, x], [taken], ["vpad", "idx", "x"], ["y"]), (vpad_v, cols_t.astype(float), xv))
  # The scan slices one padded row of each table per step: a pointer offset, no index arithmetic.
  c, xb, vb = sc.sym("c", 1), sc.sym("xb", n), sc.sym("vb", nnz)
  cols, pos = sc.sym("cols", width, dtype="int64"), sc.sym("pos", width, dtype="int64")
  row_vals = sc.take(vb, pos)
  body = sc.Function._from_exprs(
    f"p2row{n}", [c, cols, pos, xb, vb], [c, sc.stack([(row_vals * sc.take(xb, cols)).sum()])], ["c", "cols", "pos", "xb", "vb"], ["n", "y"]
  )
  tables = [(sc.const(cols_t, dtype="int64"), 0, width), (sc.const(pos_t, dtype="int64"), 0, width)]
  _, ys = sc.scan(body, sc.const(np.zeros(1)), [*tables, (x, 0, 0), (vals, 0, 0)], length=n)
  out["scan-take"] = (sc.Function._from_exprs(f"p2st{n}", [vals, x], [ys], ["vals", "x"], ["y"]), (a.data, xv))
  y, acc, yk = sc.sym("y", n), sc.sym("acc", n), sc.sym("yk", ())
  tbody = sc.Function._from_exprs(
    f"p2tr{n}", [acc, cols, pos, vb, yk], [sc.put_add(acc, cols, row_vals * yk)], ["acc", "cols", "pos", "vb", "yk"], ["n"]
  )
  (aty,) = sc.scan(tbody, sc.const(np.zeros(n)), [*tables, (vals, 0, 0), (y, 0, 1)], length=n)
  out["scan-put"] = (sc.Function._from_exprs(f"p2sp{n}", [vals, y], [aty], ["vals", "y"], ["aty"]), (a.data, np.cos(np.arange(n))))
  return out


def main() -> None:
  for n in (int(v) for v in sys.argv[1:]):
    a = matrix(n)
    cases = variants(a)
    xv = np.sin(np.arange(n))
    for tag, (fn, point) in cases.items():
      t0 = time.perf_counter()
      body = render_c_module(fn).body
      got = np.asarray(fn(point)).reshape(-1)
      first = (time.perf_counter() - t0) * 1e3
      want = a.T @ np.cos(np.arange(n)) if tag == "scan-put" else a @ xv
      err = float(np.max(np.abs(got - want)))
      us = median_us(lambda fn=fn, point=point: fn(point), repeat=21)
      print(f"{tag:<10} n={n:>5} nnz={a.nnz:>6} lines={len(body.splitlines()):>5} first_ms={first:>6.0f} us={us:>8.2f} err={err:.1e}", flush=True)
    print(f"{'scipy':<10} n={n:>5} nnz={a.nnz:>6} us={median_us(lambda: a @ xv, repeat=21):>8.2f} (A^T y {median_us(lambda: a.T @ xv, repeat=21):.2f})", flush=True)


if __name__ == "__main__":
  main()
