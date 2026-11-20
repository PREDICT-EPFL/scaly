"""``scaly.linalg.solve``: values against NumPy, derivatives against an unrolled reference, one
factorization shared by the value and its gradient, and the entry point's checks."""

from __future__ import annotations

import numpy as np
import pytest

import scaly as sc
from scaly import linalg
from scaly.function.model import as_concrete
from scaly.ir.expr import ExprOp, topo

from .test_dense import N, _check_derivatives, _cost, _spd, _unrolled_cholesky, _unrolled_substitution


def _symmetric(a: np.ndarray) -> np.ndarray:
  return np.tril(a) + np.swapaxes(np.tril(a, -1), -1, -2)


def test_solve_and_its_gradient_over_a_stage_axis() -> None:
  rng = np.random.default_rng(5)
  k = np.stack([_spd(rng) for _ in range(3)])
  b = rng.normal(size=(3, N, 2))
  k[2, 1, 1] = -50.0
  status_type = sc.TensorType((3,), sc.dtypes.int32)

  @sc.function(
    sc.arg("K", (3, N, N)), sc.arg("b", (3, N, 2)), outputs=sc.group(sc.arg("x", (3, N, 2)), sc.arg("status", status_type), sc.arg("c", ()))
  )
  def solve(K, b):
    x, status = linalg.solve(K, b, assume="pos")
    return x, status, 0.5 * sc.sumsqr(x)

  x, status, _ = solve(k, b)
  np.testing.assert_array_equal(status, [0, 0, 2])
  np.testing.assert_allclose(x[:2], np.linalg.solve(_symmetric(k[:2]), b[:2]), rtol=1e-12)
  b_bar = np.linalg.solve(_symmetric(k[:2]), x[:2])
  outer = -b_bar @ np.swapaxes(x[:2], -1, -2)
  k_bar = np.tril(outer + np.swapaxes(outer, -1, -2), -1) + outer * np.eye(N)
  np.testing.assert_allclose(sc.gradient(solve, "c", "K")(k, b)[:2], k_bar, rtol=1e-11, atol=1e-13)
  np.testing.assert_allclose(sc.gradient(solve, "c", "b")(k, b)[:2], b_bar, rtol=1e-11)


def test_solve_derivatives_match_the_unrolled_reference() -> None:
  """The matrix and the right-hand side are both active, packed in one input."""
  outputs = sc.group(sc.arg("x", N), sc.arg("c", ()))
  size = N * N + N

  @sc.function(sc.arg("p", size), outputs=outputs)
  def solve(p):
    x = linalg.solve(p[: N * N].reshape((N, N)), p[N * N :], assume="pos")[0]
    return x, _cost(x)

  @sc.function(sc.arg("p", size), outputs=outputs)
  def reference(p):
    low = sc.stack([sc.stack([sc.const(v) if isinstance(v, float) else v for v in row]) for row in _unrolled_cholesky(p[: N * N].reshape((N, N)))])
    x = _unrolled_substitution(low, _unrolled_substitution(low, p[N * N :], True, False, False), True, True, False)
    return x, _cost(x)

  rng = np.random.default_rng(6)
  _check_derivatives(solve, reference, np.concatenate([_spd(rng).reshape(-1), rng.normal(size=N)]), "x")


def test_a_value_and_its_gradient_share_one_factorization() -> None:
  """The reverse rule reads the factor as a residual of the one solve invocation."""

  @sc.function(sc.arg("K", (N, N)), sc.arg("b", N))
  def value_and_gradient(K, b):
    x = linalg.solve(K, b, assume="pos")[0]
    value = _cost(x)
    return value, *sc.vjp((value,), (K, b), (sc.const(1.0),))

  rng = np.random.default_rng(7)
  k, b = _spd(rng), rng.normal(size=N)
  value, k_bar, b_bar = value_and_gradient(k, b)
  x = np.linalg.solve(_symmetric(k), b)
  np.testing.assert_allclose(b_bar, np.linalg.solve(_symmetric(k), x * x + np.eye(N)[0] * x[-1] + np.eye(N)[-1] * x[0]), rtol=1e-12)
  outer = -np.outer(b_bar, x)
  np.testing.assert_allclose(k_bar, np.tril(outer + outer.T, -1) + np.diag(np.diag(outer)), rtol=1e-12, atol=1e-14)

  found, todo, factorizations = [], [as_concrete(value_and_gradient)], set()
  while todo:
    graph = todo.pop()
    if graph in found:
      continue
    found.append(graph)
    for node in topo(graph.results):
      if node.op in (ExprOp.CALL, ExprOp.LOOP):
        todo.append(node.attrs["callee"])
        if node.attrs["callee"].name.startswith(f"cholesky_{N}x{N}"):
          factorizations.add((graph.name, node.args))
  assert len(factorizations) == 1


def test_solve_checks_its_arguments() -> None:
  k, b = sc.sym("K", (N, N)), sc.sym("b", N)
  with pytest.raises(ValueError, match="assume must be one of"):
    linalg.solve(k, b, assume="her")
  with pytest.raises(ValueError, match="right-hand side has shape"):
    linalg.solve(k, sc.sym("c", N + 1), assume="pos")
  with pytest.raises(TypeError, match="dense matrix"):
    linalg.solve(sc.sparsify(sc.sym("S", (N, N)) * sc.const(np.eye(N))), b, assume="pos")
  with pytest.raises(ValueError, match="right-hand side has shape"):
    linalg.solve(k, sc.sym("s", ()), assume="pos")
  with pytest.raises(TypeError, match="float32 or float64 matrix"):
    linalg.solve(sc.sym("F", (N, N), dtype=sc.dtypes.int32), b, assume="pos")
