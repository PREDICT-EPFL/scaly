"""``scaly.linalg.solve``: values against NumPy, derivatives against an unrolled reference, one
factorization shared by the value and its derivatives, and the entry point's checks."""

from __future__ import annotations

import numpy as np
import pytest

import scaly as sc
from scaly import linalg
from scaly.function.model import as_concrete
from scaly.ir.expr import ExprOp, is_map, topo

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


def _runs(fn, loop: str) -> int:
  """How many times evaluating ``fn`` runs a loop whose body is named ``loop``: one invocation of
  a function, its outputs and residuals together, counts once, and a map multiplies by its length."""

  def count(graph, times: int) -> int:
    total, seen = 0, set()
    for node in topo(graph.results):
      if node.op not in (ExprOp.CALL, ExprOp.LOOP):
        continue
      callee, attrs = node.attrs["callee"], node.attrs
      key = (callee, node.args, attrs.get("starts"), attrs.get("strides"), attrs.get("length"))
      if key in seen:
        continue
      seen.add(key)
      if node.op == ExprOp.LOOP and callee.name.startswith(loop):
        total += times
      else:
        total += count(callee, times * (attrs["length"] if is_map(node) else 1))
    return total

  return count(as_concrete(fn), 1)


@pytest.mark.parametrize(("assume", "loop"), [("pos", "cholesky_column"), ("sym", "ldl_column"), ("gen", "lu_column")])
def test_a_jacobian_with_several_seeds_factors_once(assume, loop) -> None:
  """The forward rule reads the factorization and the solution of the one solve invocation, for
  every one of the ``N * N`` seeds in ``K``, so a solution and its Jacobian run one factorization."""

  @sc.function(sc.arg("K", (N, N)), sc.arg("b", N), outputs=sc.group(sc.arg("x", N), sc.arg("dx", (N, N * N))))
  def value_and_jacobian(K, b):
    x = linalg.solve(K, b, assume=assume)[0]
    return x, sc.jacobian(x, K)

  assert _runs(value_and_jacobian, loop) == 1

  @sc.function(sc.arg("K", (N, N)), sc.arg("b", N), outputs=sc.arg("c", ()))
  def cost(K, b):
    return _cost(linalg.solve(K, b, assume=assume)[0])

  # The Hessian differentiates the residuals through the factorization's own forward rule, which
  # reads the factor of its call. The forward helpers of the two residual results recompute the
  # call they differentiate (#90), which makes three, whatever the number of seeds.
  assert _runs(sc.hessian(cost, "c", "K"), loop) == 3
  rng = np.random.default_rng(21)
  k, b = _spd(rng) + (rng.normal(size=(N, N)) if assume == "gen" else 0.0), rng.normal(size=N)
  dense = _symmetric(k) if assume != "gen" else k
  x = np.linalg.solve(dense, b)
  # dx/dK_ij = -K^-1 (e_i x_j), and with one triangle read the mirrored entry adds -K^-1 (e_j x_i).
  want = -np.einsum("ai,j->aij", np.linalg.inv(dense), x)
  if assume != "gen":
    want = np.tril(want + np.swapaxes(want, 1, 2), -1) + want * np.eye(N)
  x_got, jacobian = value_and_jacobian(k, b)
  np.testing.assert_allclose(x_got, x, rtol=1e-12)
  np.testing.assert_allclose(jacobian.reshape((N, N, N)), want, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize(("assume", "loop"), [("pos", "cholesky_column"), ("gen", "lu_column")])
def test_a_jacobian_over_a_stage_axis_factors_once_per_stage(assume, loop) -> None:
  """The solve is mapped over the stages, and the mapped forward rule reads the residuals of the
  mapped solve, one window per stage."""
  stages = 3

  @sc.function(
    sc.arg("K", (stages, N, N)), sc.arg("b", (stages, N)), outputs=sc.group(sc.arg("x", (stages, N)), sc.arg("dx", (stages * N, stages * N * N)))
  )
  def value_and_jacobian(K, b):
    x = linalg.solve(K, b, assume=assume)[0]
    return x, sc.jacobian(x, K)

  assert _runs(value_and_jacobian, loop) == stages
  rng = np.random.default_rng(22)
  k = np.stack([_spd(rng) + (rng.normal(size=(N, N)) if assume == "gen" else 0.0) for _ in range(stages)])
  b = rng.normal(size=(stages, N))
  got = value_and_jacobian(k, b)[1].reshape((stages, N, stages, N, N))
  for s in range(stages):
    dense = k[s] if assume == "gen" else _symmetric(k[s])
    want = -np.einsum("ai,j->aij", np.linalg.inv(dense), np.linalg.solve(dense, b[s]))
    if assume != "gen":
      want = np.tril(want + np.swapaxes(want, 1, 2), -1) + want * np.eye(N)
    np.testing.assert_allclose(got[s, :, s], want, rtol=1e-10, atol=1e-12)
    np.testing.assert_array_equal(np.delete(got[s], s, axis=1), 0.0)


def test_solve_checks_its_arguments() -> None:
  k, b = sc.sym("K", (N, N)), sc.sym("b", N)
  with pytest.raises(ValueError, match="assume must be one of"):
    linalg.solve(k, b, assume="her")
  with pytest.raises(ValueError, match="right-hand side has shape"):
    linalg.solve(k, sc.sym("c", N + 1), assume="pos")
  with pytest.raises(ValueError, match="assume 'pos' or 'sym'"):
    linalg.solve(sc.sparsify(sc.sym("S", (N, N)) * sc.const(np.eye(N))), b, assume="gen")
  with pytest.raises(ValueError, match="right-hand side has shape"):
    linalg.solve(k, sc.sym("s", ()), assume="pos")
  with pytest.raises(TypeError, match="float32 or float64 matrix"):
    linalg.solve(sc.sym("F", (N, N), dtype=sc.dtypes.int32), b, assume="pos")
