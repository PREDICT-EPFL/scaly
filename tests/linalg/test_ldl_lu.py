"""``scaly.linalg`` LDL^T and LU, and ``solve(assume="sym"|"gen")``: factors against SciPy, pivot
status, and solve derivatives against an unrolled elimination that Scaly differentiates as ordinary
expressions."""

from __future__ import annotations

import re

import numpy as np
import pytest
import scipy.linalg

import scaly as sc
from scaly import linalg
from scaly.codegen import render_c_source
from scaly.function.model import as_concrete
from scaly.ir.expr import ExprOp, topo

from .test_dense import EPS, N, STATUS, _check_derivatives, _cost


def _quasi_definite(rng: np.random.Generator) -> np.ndarray:
  """``[[P, B^T], [B, -R]]`` with garbage above the diagonal, which ``ldl`` must not read."""
  m = rng.normal(size=(N, N))
  m[:2, :2] += 6.0 * np.eye(2)
  m[2:, 2:] -= 6.0 * np.eye(2)
  return m


def _needs_swaps(rng: np.random.Generator) -> np.ndarray:
  """Pivots that partial pivoting takes from rows 1, 2, 0, 3, a permutation that is not its own inverse."""
  m = rng.normal(size=(N, N))
  m[[1, 2, 0, 3], range(N)] += 10.0
  return m


def _symmetric(a: np.ndarray) -> np.ndarray:
  return np.tril(a) + np.swapaxes(np.tril(a, -1), -1, -2)


def test_ldl_matches_scipy_over_a_stack() -> None:
  rng = np.random.default_rng(10)
  a = np.stack([_quasi_definite(rng) for _ in range(2)])

  @sc.function(
    sc.arg("A", (2, N, N)), outputs=sc.group(sc.arg("L", (2, N, N)), sc.arg("d", (2, N)), sc.arg("status", sc.TensorType((2,), sc.dtypes.int32)))
  )
  def factor(A):
    return linalg.ldl(A)

  factor_l, d, status = factor(a)
  np.testing.assert_array_equal(status, [0, 0])
  for k in range(2):
    expected_l, expected_d, perm = scipy.linalg.ldl(_symmetric(a[k]), lower=True)
    np.testing.assert_array_equal(perm, np.arange(N))  # diagonally dominant enough that SciPy does not pivot either
    np.testing.assert_allclose(factor_l[k], expected_l, rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(np.diag(d[k]), expected_d, rtol=1e-13, atol=1e-14)
  assert (d < 0).any()


def test_lu_factor_matches_scipy_over_a_stack() -> None:
  rng = np.random.default_rng(11)
  a = np.stack([_needs_swaps(rng) for _ in range(2)])

  @sc.function(
    sc.arg("A", (2, N, N)),
    outputs=sc.group(
      sc.arg("lu", (2, N, N)), sc.arg("piv", sc.TensorType((2, N), sc.dtypes.int64)), sc.arg("status", sc.TensorType((2,), sc.dtypes.int32))
    ),
  )
  def factor(A):
    return linalg.lu_factor(A)

  lu, piv, status = factor(a)
  np.testing.assert_array_equal(status, [0, 0])
  for k in range(2):
    expected_lu, expected_piv = scipy.linalg.lu_factor(a[k])
    np.testing.assert_array_equal(expected_piv, [1, 2, 2, 3])
    np.testing.assert_array_equal(np.asarray(piv)[k], expected_piv)
    np.testing.assert_allclose(lu[k], expected_lu, rtol=1e-13, atol=1e-14)


@pytest.mark.parametrize(
  ("routine", "a", "status"),
  [
    ("ldl", [[4.0, 2.0], [2.0, 1.0]], 2),  # a zero pivot
    ("ldl", [[1.0, 1.0 - EPS / 2], [1.0 - EPS / 2, 1.0]], 2),  # the pivot 1 - fl((1 - eps/2)^2) is exactly eps
    ("ldl", [[2.0, 1.0], [1.0, -3.0]], 0),  # quasi-definite: a negative pivot is usable
    ("ldl", [[-1.0, 0.0], [1.0 - EPS / 2, -1.0]], 2),  # the same eps pivot, negative, against a negative diagonal
    ("ldl", [[1e-160, 0.0], [1e160, -1.0]], 2),  # a pivot that overflows to -inf
    ("lu", [[0.0, 1.0], [1.0, 0.0]], 0),  # a zero diagonal that a row swap moves away
    ("lu", [[1.0, 1.0], [1.0, 1.0 + EPS]], 2),  # the pivot fl(1 + eps) - 1 is eps times its entry, rounded down
    ("lu", [[1.0, 1.0], [1.0, 1.0 + 2 * EPS]], 0),
    ("lu", [[1.0, 1.0 + 2 * EPS], [8.0, 8.0]], 0),  # the pivot 2 eps is compared with the entry of the chosen row, 1 + 2 eps
    ("lu", [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [0.0, 1.0, 1.0]], 3),  # singular: rank 2
  ],
)
def test_factorizations_report_the_first_unusable_pivot(routine, a, status) -> None:
  a = np.asarray(a)
  n = a.shape[0]

  @sc.function(sc.arg("A", (n, n)), outputs=sc.arg("status", STATUS))
  def factor(A):
    return linalg.ldl(A)[2] if routine == "ldl" else linalg.lu_factor(A)[2]

  assert factor(a) == status


def test_ldl_keeps_a_unit_diagonal_after_an_infinite_pivot() -> None:
  @sc.function(sc.arg("A", (2, 2)), outputs=sc.arg("L", (2, 2)))
  def factor(A):
    return linalg.ldl(A)[0]

  np.testing.assert_array_equal(np.diag(factor(np.array([[np.inf, 0.0], [1.0, 1.0]]))), [1.0, 1.0])


def _unrolled_elimination(k: sc.Expr) -> sc.Expr:
  """Gaussian elimination without pivoting, entry by entry: the multipliers below the diagonal and ``U`` on and above it."""
  n = k.shape[0]
  a = [[k[i, j] for j in range(n)] for i in range(n)]
  for p in range(n):
    for i in range(p + 1, n):
      a[i][p] = a[i][p] / a[p][p]
      a[i] = [a[i][j] - a[i][p] * a[p][j] if j > p else a[i][j] for j in range(n)]
  return sc.stack([sc.stack(row) for row in a])


def _symmetric_from_lower(m: sc.Expr) -> sc.Expr:
  low = m * sc.const(np.tril(np.ones((N, N)), -1))
  return low + low.T + m * sc.const(np.eye(N))


def _row_permutation(k: np.ndarray) -> np.ndarray:
  """The rows partial pivoting picks for ``k``, as a matrix, constant near ``k``."""
  return np.eye(N)[np.argsort(scipy.linalg.lu(k, p_indices=True)[0])]


@pytest.mark.parametrize("routine", ["ldl", "lu"])
def test_factor_derivatives_match_the_unrolled_reference(routine) -> None:
  """The forward rules, and the factorization loops in reverse. ``ldl``'s ``L`` and ``d`` are compared packed, ``d`` on the diagonal."""
  rng = np.random.default_rng(15)
  a = _quasi_definite(rng) if routine == "ldl" else _needs_swaps(rng)
  permute = _row_permutation(a) if routine == "lu" else np.eye(N)
  outputs = sc.group(sc.arg("F", (N, N)), sc.arg("c", ()))

  @sc.function(sc.arg("p", N * N), outputs=outputs)
  def factor(p):
    if routine == "ldl":
      low, d, _ = linalg.ldl(p.reshape((N, N)))
      f = low * sc.const(np.tril(np.ones((N, N)), -1)) + sc.const(np.eye(N)) * d.reshape((1, N))
    else:
      f = linalg.lu_factor(p.reshape((N, N)))[0]
    return f, _cost(f.reshape((N * N,)))

  @sc.function(sc.arg("p", N * N), outputs=outputs)
  def reference(p):
    m = p.reshape((N, N))
    if routine == "ldl":
      f = _unrolled_elimination(_symmetric_from_lower(m)) * sc.const(np.tril(np.ones((N, N))))
    else:
      f = _unrolled_elimination(sc.const(permute) @ m)
    return f, _cost(f.reshape((N * N,)))

  _check_derivatives(factor, reference, a.reshape(-1), "F")


def test_the_lu_forward_rule_survives_a_tiny_pivot() -> None:
  """``dU`` is formed without dividing by the pivot and multiplying back, which overflows here."""

  @sc.function(sc.arg("A", (1, 1)), outputs=sc.arg("lu", (1, 1)))
  def factor(A):
    return linalg.lu_factor(A)[0]

  np.testing.assert_array_equal(sc.jacobian(factor, "lu", "A")(np.array([[1e-310]])), [[1.0]])


# --- solves ----------------------------------------------------------------------------------


@pytest.mark.parametrize("assume", ["sym", "gen"])
def test_solve_and_its_gradient_over_a_stage_axis(assume) -> None:
  rng = np.random.default_rng(12)
  k = np.stack([(_quasi_definite if assume == "sym" else _needs_swaps)(rng) for _ in range(3)])
  b = rng.normal(size=(3, N, 2))
  if assume == "sym":
    k[2, 1, :2] = 0.0  # the second pivot is zero
  else:
    k[2, :, 1] = k[2, :, 0]  # singular: the second column eliminates to zero
  dense = _symmetric(k) if assume == "sym" else k

  @sc.function(
    sc.arg("K", (3, N, N)),
    sc.arg("b", (3, N, 2)),
    outputs=sc.group(sc.arg("x", (3, N, 2)), sc.arg("status", sc.TensorType((3,), sc.dtypes.int32)), sc.arg("c", ())),
  )
  def solve(K, b):
    x, status = linalg.solve(K, b, assume=assume)
    return x, status, 0.5 * sc.sumsqr(x)

  x, status, _ = solve(k, b)
  np.testing.assert_array_equal(status, [0, 0, 2])
  np.testing.assert_allclose(x[:2], np.linalg.solve(dense[:2], b[:2]), rtol=1e-12)
  b_bar = np.linalg.solve(np.swapaxes(dense[:2], -1, -2), x[:2])
  k_bar = -b_bar @ np.swapaxes(x[:2], -1, -2)
  if assume == "sym":
    k_bar = np.tril(k_bar + np.swapaxes(k_bar, -1, -2), -1) + k_bar * np.eye(N)
  np.testing.assert_allclose(sc.gradient(solve, "c", "K")(k, b)[:2], k_bar, rtol=1e-11, atol=1e-13)
  np.testing.assert_allclose(sc.gradient(solve, "c", "b")(k, b)[:2], b_bar, rtol=1e-11)


def _unrolled_solve(k: sc.Expr, b: sc.Expr) -> sc.Expr:
  """Gaussian elimination without pivoting, entry by entry."""
  n = k.shape[0]
  a = [[k[i, j] for j in range(n)] for i in range(n)]
  y = [b[i] for i in range(n)]
  for p in range(n):
    for i in range(p + 1, n):
      f = a[i][p] / a[p][p]
      a[i] = [a[i][j] - f * a[p][j] if j > p else a[i][j] for j in range(n)]
      y[i] = y[i] - f * y[p]
  x: list[sc.Expr] = [sc.const(0.0)] * n
  for i in reversed(range(n)):
    x[i] = (y[i] - sum((a[i][j] * x[j] for j in range(i + 1, n)), sc.const(0.0))) / a[i][i]
  return sc.stack(x)


@pytest.mark.parametrize("assume", ["sym", "gen"])
def test_solve_derivatives_match_the_unrolled_reference(assume) -> None:
  """The matrix and the right-hand side are both active, packed in one input. The reference
  eliminates without pivoting on the rows LU chose, which are constant near the point."""
  rng = np.random.default_rng(13)
  k = _quasi_definite(rng) if assume == "sym" else _needs_swaps(rng)
  permute = _row_permutation(k) if assume == "gen" else np.eye(N)
  outputs = sc.group(sc.arg("x", N), sc.arg("c", ()))
  size = N * N + N

  @sc.function(sc.arg("p", size), outputs=outputs)
  def solve(p):
    x = linalg.solve(p[: N * N].reshape((N, N)), p[N * N :], assume=assume)[0]
    return x, _cost(x)

  @sc.function(sc.arg("p", size), outputs=outputs)
  def reference(p):
    m = p[: N * N].reshape((N, N))
    if assume == "sym":
      m = _symmetric_from_lower(m)
    x = _unrolled_solve(sc.const(permute) @ m, sc.const(permute) @ p[N * N :])
    return x, _cost(x)

  _check_derivatives(solve, reference, np.concatenate([k.reshape(-1), rng.normal(size=N)]), "x")


@pytest.mark.parametrize("assume", ["sym", "gen"])
def test_reverse_over_reverse_matches_forward_over_reverse(assume) -> None:
  @sc.function(sc.arg("p", N * N + N), outputs=sc.arg("c", ()))
  def cost(p):
    return _cost(linalg.solve(p[: N * N].reshape((N, N)), p[N * N :], assume=assume)[0])

  gradient = sc.gradient(cost, "c", "p")

  rng = np.random.default_rng(14)
  weights = rng.normal(size=N * N + N)

  @sc.function(sc.arg("p", N * N + N), outputs=sc.arg("g", N * N + N))
  def twice(p):
    return sc.gradient((gradient(p) * sc.const(weights)).sum(), p)

  p = np.concatenate([(_quasi_definite if assume == "sym" else _needs_swaps)(rng).reshape(-1), rng.normal(size=N)])
  np.testing.assert_allclose(twice(p), sc.hessian(cost, "c", "p")(p) @ weights, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize(("assume", "factor"), [("sym", "ldl"), ("gen", "lu")])
def test_a_value_and_its_gradient_share_one_factorization(assume, factor) -> None:
  @sc.function(sc.arg("K", (N, N)), sc.arg("b", N))
  def value_and_gradient(K, b):
    value = _cost(linalg.solve(K, b, assume=assume)[0])
    return value, *sc.vjp((value,), (K, b), (sc.const(1.0),))

  found, todo, factorizations = [], [as_concrete(value_and_gradient)], set()
  while todo:
    graph = todo.pop()
    if graph in found:
      continue
    found.append(graph)
    for node in topo(graph.results):
      if node.op in (ExprOp.CALL, ExprOp.LOOP):
        todo.append(node.attrs["callee"])
        if node.attrs["callee"].name.startswith(f"{factor}_{N}x{N}"):
          factorizations.add((graph.name, node.args))
  assert len(factorizations) == 1


@pytest.mark.parametrize(
  ("assume", "loops"),
  [
    ("sym", {"ldl_column", "solve_triangular_lnu_row", "solve_triangular_ltu_row"}),
    ("gen", {"lu_column", "lu_row", "solve_triangular_lnu_row", "solve_triangular_unn_row"}),
  ],
)
def test_the_primal_loops_update_their_carries_in_place(assume, loops) -> None:
  @sc.function(sc.arg("K", (N, N)), sc.arg("b", N), outputs=sc.arg("x", N))
  def solve(K, b):
    return linalg.solve(K, b, assume=assume)[0]

  assert set(re.findall(r"(\w+?)_\d+x?\d*\w*?_inplace\w*_raw\(", render_c_source(solve))) == loops
