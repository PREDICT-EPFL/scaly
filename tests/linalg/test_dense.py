"""``scaly.linalg`` triangular solves and Cholesky against NumPy and SciPy, and their derivatives
against an unrolled reference that Scaly differentiates as ordinary expressions.

Every matrix carries garbage in the entries a routine must not read, so a read of the wrong
triangle, or of a unit diagonal, changes the result.
"""

from __future__ import annotations

import re

import numpy as np
import pytest
import scipy.linalg

import scaly as sc
from scaly import linalg
from scaly.codegen import render_c_source

N = 4
STATUS = sc.TensorType((), sc.dtypes.int64)


def _spd(rng: np.random.Generator, n: int = N) -> np.ndarray:
  m = rng.normal(size=(n, n))
  return m @ m.T + n * np.eye(n) + np.triu(rng.normal(size=(n, n)), 1)


def _triangular(rng: np.random.Generator) -> np.ndarray:
  return rng.normal(size=(N, N)) + 3.0 * np.eye(N)


@pytest.mark.parametrize(
  ("lower", "trans", "unit", "rhs"),
  [(True, False, False, (N,)), (False, True, True, (N, 2)), (True, True, False, (N, 3))],
)
def test_triangular_solve_matches_scipy(lower, trans, unit, rhs) -> None:
  rng = np.random.default_rng(0)
  t, b = _triangular(rng), rng.normal(size=rhs)

  @sc.function(sc.arg("T", (N, N)), sc.arg("b", rhs))
  def solve(T, b):
    return linalg.solve_triangular(T, b, lower=lower, trans=trans, unit_diagonal=unit)

  expected = scipy.linalg.solve_triangular(t, b, lower=lower, trans=int(trans), unit_diagonal=unit)
  np.testing.assert_allclose(solve(t, b), expected, rtol=1e-13, atol=1e-14)


def test_cholesky_matches_numpy_and_reads_the_lower_triangle() -> None:
  a = _spd(np.random.default_rng(1))

  @sc.function(sc.arg("A", (N, N)), outputs=sc.group(sc.arg("L", (N, N)), sc.arg("status", STATUS)))
  def factor(A):
    return linalg.cholesky(A)

  factor_l, status = factor(a)
  np.testing.assert_allclose(factor_l, np.linalg.cholesky(np.tril(a) + np.tril(a, -1).T), rtol=1e-13, atol=1e-14)
  assert status == 0


EPS = np.finfo(np.float64).eps


@pytest.mark.parametrize(
  ("a", "status"),
  [
    ([[4.0, 2.0], [2.0, 1.0]], 2),  # a zero pivot
    ([[1.0, 1.0 - EPS / 2], [1.0 - EPS / 2, 1.0]], 2),  # the pivot 1 - fl((1 - eps/2)^2) is exactly eps
    ([[4.0, 2.0], [2.0, 1.0 + 2 * EPS]], 0),  # a pivot of twice that is usable
    ([[1.0, 0.0, 0.0], [2.0, -1.0, 0.0], [0.0, 0.0, -1.0]], 2),  # indefinite: the first unusable pivot
  ],
)
def test_cholesky_reports_the_first_unusable_pivot(a, status) -> None:
  a = np.asarray(a)
  n = a.shape[0]

  @sc.function(sc.arg("A", (n, n)), outputs=sc.group(sc.arg("L", (n, n)), sc.arg("status", STATUS)))
  def factor(A):
    return linalg.cholesky(A)

  factor_l, got = factor(a)
  assert got == status
  np.testing.assert_array_equal(np.triu(factor_l, 1), 0.0)


def test_factorizations_map_over_leading_stage_axes() -> None:
  rng = np.random.default_rng(2)
  a = np.stack([np.stack([_spd(rng) for _ in range(3)]) for _ in range(2)])
  a[1, 2, 0, 0] = -1.0

  @sc.function(sc.arg("A", (2, 3, N, N)), outputs=sc.group(sc.arg("L", (2, 3, N, N)), sc.arg("status", sc.TensorType((2, 3), sc.dtypes.int64))))
  def factor(A):
    return linalg.cholesky(A)

  factor_l, status = factor(a)
  np.testing.assert_array_equal(status, [[0, 0, 0], [0, 0, 1]])
  np.testing.assert_allclose(factor_l[0], np.linalg.cholesky(np.tril(a[0]) + np.swapaxes(np.tril(a[0], -1), -1, -2)), rtol=1e-13)


# --- derivatives against an unrolled reference -------------------------------------------------


def _unrolled_cholesky(a: sc.Expr) -> list[list[sc.Expr | float]]:
  n = a.shape[0]
  low: list[list[sc.Expr | float]] = [[0.0] * n for _ in range(n)]
  for j in range(n):
    for i in range(j, n):
      s = a[i, j] - sum((low[i][k] * low[j][k] for k in range(j)), sc.const(0.0))
      low[i][j] = s.sqrt() if i == j else s / low[j][j]
  return low


def _unrolled_substitution(t: sc.Expr, b: sc.Expr, lower: bool, trans: bool, unit: bool) -> sc.Expr:
  n = t.shape[0]
  entry = (lambda i, k: t[k, i]) if trans else (lambda i, k: t[i, k])
  order = range(n) if lower != trans else range(n - 1, -1, -1)
  x: list[sc.Expr | None] = [None] * n
  for i in order:
    s = b[i] - sum((entry(i, k) * x[k] for k in range(n) if x[k] is not None), sc.const(0.0))
    x[i] = s if unit else s / entry(i, i)
  return sc.stack(x)


def _check_derivatives(fn, ref, p: np.ndarray, of: str) -> None:
  """Forward, reverse and second derivatives of ``fn`` against those of ``ref``, all in one packed input."""
  np.testing.assert_allclose(sc.jacobian(fn, of, "p")(p), sc.jacobian(ref, of, "p")(p), rtol=1e-10, atol=1e-12)
  np.testing.assert_allclose(sc.gradient(fn, "c", "p")(p), sc.gradient(ref, "c", "p")(p), rtol=1e-10, atol=1e-12)
  np.testing.assert_allclose(sc.hessian(fn, "c", "p")(p), sc.hessian(ref, "c", "p")(p), rtol=1e-9, atol=1e-11)


def _cost(x: sc.Expr) -> sc.Expr:
  return (x * x * x).sum() / 3.0 + x[0] * x[-1]


def test_cholesky_derivatives_match_the_unrolled_reference() -> None:
  outputs = sc.group(sc.arg("L", (N, N)), sc.arg("c", ()))

  @sc.function(sc.arg("p", N * N), outputs=outputs)
  def factor(p):
    low = linalg.cholesky(p.reshape((N, N)))[0]
    return low, _cost(low.reshape((N * N,)))

  @sc.function(sc.arg("p", N * N), outputs=outputs)
  def reference(p):
    low = sc.stack([sc.stack([sc.const(v) if isinstance(v, float) else v for v in row]) for row in _unrolled_cholesky(p.reshape((N, N)))])
    return low, _cost(low.reshape((N * N,)))

  _check_derivatives(factor, reference, _spd(np.random.default_rng(3)).reshape(-1), "L")


@pytest.mark.parametrize(("lower", "trans", "unit"), [(True, False, False), (False, True, True)])
def test_triangular_solve_derivatives_match_the_unrolled_reference(lower, trans, unit) -> None:
  outputs = sc.group(sc.arg("x", N), sc.arg("c", ()))
  size = N * N + N

  @sc.function(sc.arg("p", size), outputs=outputs)
  def solve(p):
    x = linalg.solve_triangular(p[: N * N].reshape((N, N)), p[N * N :], lower=lower, trans=trans, unit_diagonal=unit)
    return x, _cost(x)

  @sc.function(sc.arg("p", size), outputs=outputs)
  def reference(p):
    x = _unrolled_substitution(p[: N * N].reshape((N, N)), p[N * N :], lower, trans, unit)
    return x, _cost(x)

  rng = np.random.default_rng(4)
  _check_derivatives(solve, reference, np.concatenate([_triangular(rng).reshape(-1), rng.normal(size=N)]), "x")


@pytest.mark.parametrize("routine", ["triangular", "cholesky"])
def test_reverse_over_reverse_matches_forward_over_reverse(routine) -> None:
  """Differentiating a reverse rule in reverse reaches the loops, which then keep their trajectories."""

  @sc.function(sc.arg("p", N * N + N), outputs=sc.arg("c", ()))
  def cost(p):
    a = p[: N * N].reshape((N, N))
    x = linalg.solve_triangular(a, p[N * N :]) if routine == "triangular" else linalg.cholesky(a)[0].reshape((N * N,))
    return _cost(x)

  gradient = sc.gradient(cost, "c", "p")

  @sc.function(sc.arg("p", N * N + N), outputs=sc.arg("g", N * N + N))
  def twice(p):
    return sc.gradient(gradient(p).sum(), p)

  rng = np.random.default_rng(8)
  p = np.concatenate([_spd(rng).reshape(-1), rng.normal(size=N)])
  np.testing.assert_allclose(twice(p), sc.hessian(cost, "c", "p")(p) @ np.ones(N * N + N), rtol=1e-10, atol=1e-12)


def test_the_primal_loops_update_their_carries_in_place() -> None:
  @sc.function(sc.arg("K", (N, N)), sc.arg("b", N), outputs=sc.arg("x", N))
  def solve(K, b):
    return linalg.solve(K, b)[0]

  procedures = set(re.findall(r"(\w+?)_\d+x\d+\w*?_inplace\w*_raw\(", render_c_source(solve)))
  assert procedures == {"cholesky_column", "solve_triangular_lnn_row", "solve_triangular_ltn_row"}
