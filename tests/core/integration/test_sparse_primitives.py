"""Sparse kernels written from ``gather`` and ``segment_sum`` over a fixed pattern match SciPy.

Each pattern is compiled once and evaluated on many value draws, the way a generated solver is used:
the structure is fixed when the code is generated and only the numbers change.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

import scaly as sc
from scaly.codegen import render_c_source


def _random(m: int, n: int, density: float, seed: int) -> sparse.coo_array:
  a = sparse.random_array((m, n), density=density, format="coo", random_state=seed)
  return sparse.coo_array((np.ones(a.nnz), (a.row, a.col)), shape=(m, n))


def _banded_kkt(stages: int, nx: int, nu: int) -> sparse.coo_array:
  """The equality-constraint pattern of a multiple-shooting OCP: [A B -I] blocks along a diagonal."""
  blocks = []
  for k in range(stages):
    row = np.zeros((nx, stages * (nx + nu) + nx))
    row[:, k * (nx + nu) : (k + 1) * (nx + nu) + nx] = 1.0
    blocks.append(row)
  return sparse.coo_array(np.vstack(blocks))


PATTERNS = {
  "random_1pct": _random(40, 50, 0.01, 0),
  "random_5pct": _random(40, 50, 0.05, 1),
  "random_20pct": _random(40, 50, 0.20, 2),
  "banded_kkt": _banded_kkt(6, 4, 2),
  "permutation": sparse.coo_array((np.ones(30), (np.random.default_rng(3).permutation(30), np.arange(30))), shape=(30, 30)),
  "one_full_row": sparse.coo_array((np.ones(25), (np.zeros(25, dtype=int), np.arange(25))), shape=(3, 25)),
}


def _spmv(name: str, pattern: sparse.coo_array) -> sc.Function:
  rows, cols = pattern.row.astype(np.int64), pattern.col.astype(np.int64)
  (m, n), nnz = pattern.shape, pattern.nnz
  a, x, y = sc.sym("a", nnz), sc.sym("x", n), sc.sym("y", m)
  ax = sc.segment_sum(a * sc.gather(x, cols), rows, m)
  aty = sc.segment_sum(a * sc.gather(y, rows), cols, n)
  return sc.Function.from_exprs(f"spmv_{name}", [a, x, y], [ax, aty, sc.dot(y, ax)], ["a", "x", "y"], ["ax", "aty", "yax"])


@pytest.mark.parametrize("name", sorted(PATTERNS))
def test_spmv_and_transpose_match_scipy_over_many_value_draws(name: str) -> None:
  pattern = PATTERNS[name]
  fun = _spmv(name, pattern)
  rng = np.random.default_rng(10)
  for _ in range(20):
    a = rng.normal(size=pattern.nnz)
    x, y = rng.normal(size=pattern.shape[1]), rng.normal(size=pattern.shape[0])
    mat = sparse.coo_array((a, (pattern.row, pattern.col)), shape=pattern.shape).tocsr()
    ax, aty, yax = fun((a, x, y))
    np.testing.assert_allclose(ax, mat @ x, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(aty, mat.T @ y, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(yax, y @ (mat @ x), rtol=1e-12, atol=1e-12)
  # The same structure renders the same source, so a rebuilt Function reuses the cached library.
  assert render_c_source(_spmv(name, pattern)) == render_c_source(fun)


@pytest.mark.parametrize("name", ["random_5pct", "banded_kkt", "one_full_row"])
def test_spmv_derivatives_with_respect_to_data_and_vector(name: str) -> None:
  pattern = PATTERNS[name]
  fun = _spmv(name + "_d", pattern)
  rng = np.random.default_rng(4)
  a, x, y = rng.normal(size=pattern.nnz), rng.normal(size=pattern.shape[1]), rng.normal(size=pattern.shape[0])
  mat = sparse.coo_array((a, (pattern.row, pattern.col)), shape=pattern.shape).toarray()
  np.testing.assert_allclose(sc.jacobian(fun, "ax", "x")((a, x, y)), mat, rtol=1e-12, atol=1e-12)
  expected_da = np.zeros((pattern.shape[0], pattern.nnz))
  expected_da[pattern.row, np.arange(pattern.nnz)] = x[pattern.col]
  np.testing.assert_allclose(sc.jacobian(fun, "ax", "a")((a, x, y)), expected_da, rtol=1e-12, atol=1e-12)
  sp = sc.sparse_jacobian(fun, "ax", "a").output_sparsities[0]
  assert sp is not None and sp.nnz == pattern.nnz
  # Reverse mode: the gradient of y'Ax is A'y in x and y[r] x[c] entry by entry in the data.
  grad_x = sc.gradient(fun, "yax", "x")((a, x, y))
  grad_a = sc.gradient(fun, "yax", "a")((a, x, y))
  np.testing.assert_allclose(grad_x, mat.T @ y, rtol=1e-12, atol=1e-12)
  np.testing.assert_allclose(grad_a, y[pattern.row] * x[pattern.col], rtol=1e-12, atol=1e-12)


def test_spmv_flops_scale_with_nonzeros_not_the_dense_size() -> None:
  pattern = _random(400, 500, 0.002, 5)
  source = render_c_source(_spmv("flops", pattern))
  loop_bounds = [int(part.split(";")[0]) for part in source.split(" < ")[1:] if part.split(";")[0].isdigit()]
  assert max(loop_bounds, default=0) <= max(pattern.nnz, 500)  # no loop runs over the 200 000 dense entries
  assert "200000" not in source
