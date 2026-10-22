"""Independent differentiation inputs and NumPy references for composed expressions."""

from __future__ import annotations

import numpy as np
import pytest

import scaly as sc


POINT = np.array([0.3, 1.0, 2.0, -1.5, 0.7, 5.0])

DERIVATIVES = ["grad", "hess", "jac", "sparse", "colored", "reference", "sparse-hess", "pattern", "single", "many", "reverse"]


def _request(kind: str, expr: sc.Expr, wrt: sc.Expr) -> object:
  if kind == "single":
    return sc.jvp(expr, wrt, sc.const(np.ones(wrt.shape)))
  if kind == "many":
    return sc.jvp_many(expr, wrt, sc.const(np.ones((0, *wrt.shape))))
  if kind == "reverse":
    return sc.vjp((expr,), (wrt,), (sc.const(np.ones(expr.shape)),))
  builders = {
    "grad": sc.gradient,
    "hess": sc.hessian,
    "jac": sc.jacobian,
    "sparse": sc.sparse_jacobian,
    "colored": sc.sparse_jacobian_colored,
    "reference": sc.sparse_jacobian_reference,
    "sparse-hess": sc.sparse_hessian,
    "pattern": sc.jacobian_sparsity,
  }
  return builders[kind](expr, wrt)


@pytest.mark.parametrize("kind", DERIVATIVES)
@pytest.mark.parametrize("selection", ["scaled", "slice", "reshape", "call", "map", "stopped", "constant", "empty-slice", "empty-constant"])
def test_rejects_non_input_wrt(kind: str, selection: str) -> None:
  x = sc.sym("x", 4)
  wrt = {
    "scaled": 2 * x,
    "slice": x[1:3],
    "reshape": x.reshape((2, 2)),
    "call": _pair(x[:2]),
    "map": sc.vmap(_pair, 2)(x.reshape((2, 2))),
    "stopped": sc.stop_gradient(x),
    "constant": sc.const(np.ones(2)),
    "empty-slice": x[:0],
    "empty-constant": sc.const(np.empty(0)),
  }[selection]
  with pytest.raises(ValueError, match="wrt must be an independent input"):
    _request(kind, sc.sumsqr(wrt) + sc.sumsqr(x), wrt)


@pytest.mark.parametrize("reverse_order", [False, True])
def test_reverse_rejects_a_non_input_among_inputs(reverse_order: bool) -> None:
  x = sc.sym("x", 3)
  selected = x[1:]
  wrts = (selected, x) if reverse_order else (x, selected)
  with pytest.raises(ValueError, match="wrt must be an independent input"):
    sc.vjp((sc.sumsqr(x) + sc.sumsqr(selected),), wrts, (sc.const(1.0),))


@pytest.mark.parametrize("kind", DERIVATIVES)
def test_rejects_non_input_wrt_for_a_constant_output(kind: str) -> None:
  x = sc.sym("x", 3)
  with pytest.raises(ValueError, match="wrt must be an independent input"):
    _request(kind, sc.const(1.0), 2 * x)


def _energy(p: sc.Expr, c: sc.Expr) -> sc.Expr:
  return (p * p * p).sum() + p[0] * p[1] * c[5] + c[0] * p[2].sin() + sc.sumsqr(c[2:5])


def _vector(p: sc.Expr, c: sc.Expr) -> sc.Expr:
  return sc.concat([p * p * p, (p[0] * p[2] * c[5]).reshape((1,)), c[4:6]])


def _references(c: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  p = c[1:4]
  grad = 3 * p**2 + np.array([p[1] * c[5], p[0] * c[5], c[0] * np.cos(p[2])])
  hess = np.diag(6 * p)
  hess[0, 1] = hess[1, 0] = c[5]
  hess[2, 2] -= c[0] * np.sin(p[2])
  jac = np.vstack([np.diag(3 * p**2), [p[2] * c[5], 0, p[0] * c[5]], np.zeros((2, 3))])
  return grad, hess, jac


@pytest.mark.parametrize("kind", ["grad", "hess", "jac", "sparse", "colored", "reference", "hess-full", "hess-lower", "hess-upper", "later-lower"])
def test_dense_and_sparse_forms(kind: str) -> None:
  @sc.function(sc.arg("p", 3), sc.arg("c", 6), outputs=sc.arg("derivative"), name=f"input_{kind}")
  def fn(p: sc.Expr, c: sc.Expr) -> sc.Expr:
    energy, vector = _energy(p, c), _vector(p, c)
    if kind == "grad":
      return sc.gradient(energy, p)
    if kind == "hess":
      return sc.hessian(energy, p)
    if kind == "jac":
      return sc.jacobian(vector, p)
    if kind in {"sparse", "colored", "reference"}:
      builder = {"sparse": sc.sparse_jacobian, "colored": sc.sparse_jacobian_colored, "reference": sc.sparse_jacobian_reference}[kind]
      return builder(vector, p).to_dense()
    if kind == "later-lower":
      return sc.sparse_hessian(energy, p).triangle("lower").to_dense()
    return sc.sparse_hessian(energy, p, triangle="lower" if kind == "hess-lower" else "upper" if kind == "hess-upper" else "full").to_dense()

  grad, hess, jac = _references(POINT)
  expected = grad if kind == "grad" else hess if "hess" in kind or kind == "later-lower" else jac
  if "lower" in kind:
    expected = np.tril(expected)
  if "upper" in kind:
    expected = np.triu(expected)
  np.testing.assert_allclose(fn(POINT[1:4], POINT), expected, rtol=1e-13, atol=1e-13)


def test_forward_reverse_and_sparsity() -> None:
  seeds = np.arange(9.0).reshape(3, 3) - 2
  weights = np.linspace(1.0, 2.0, 6)

  @sc.function(sc.arg("p", 3), sc.arg("c", 6), outputs=sc.group(sc.arg("single"), sc.arg("many"), sc.arg("adj")))
  def fn(p: sc.Expr, c: sc.Expr) -> tuple[sc.Expr, sc.Expr, sc.Expr]:
    vector = _vector(p, c)
    pattern = sc.jacobian_sparsity(vector, p)
    reference = _references(POINT)[2] != 0
    np.testing.assert_array_equal(pattern.to_mask(), reference)
    return sc.jvp(vector, p, sc.const(seeds[0])), sc.jvp_many(vector, p, sc.const(seeds)), sc.vjp((vector,), (p,), (sc.const(weights),))[0]

  single, many, adj = fn(POINT[1:4], POINT)
  jac = _references(POINT)[2]
  np.testing.assert_allclose(single, jac @ seeds[0])
  np.testing.assert_allclose(many, seeds @ jac.T)
  np.testing.assert_allclose(adj, jac.T @ weights)


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("reverse_order", [False, True])
def test_overlapping_values_at_separate_inputs(nested: bool, reverse_order: bool) -> None:
  @sc.function(sc.arg("a", 3), sc.arg("b", 2 if nested else 3), sc.arg("c", 6), outputs=sc.group(sc.arg("a"), sc.arg("b"), sc.arg("c")))
  def fn(a: sc.Expr, b: sc.Expr, c: sc.Expr) -> tuple[sc.Expr, sc.Expr, sc.Expr]:
    out = sc.sumsqr(a) + (b**3).sum() + c[0] * a[0]
    wrts = (c, b, a) if reverse_order else (a, b, c)
    da, db, dc = sc.vjp((out,), wrts, (sc.const(1.0),))
    return da, db, dc

  a, b = POINT[1:4], POINT[2:4] if nested else POINT[2:5]
  direct = np.zeros(6)
  direct[0] = a[0]
  expected = (2 * a + np.array([POINT[0], 0, 0]), 3 * b**2, direct)
  for got, want in zip(fn(a, b, POINT), expected[::-1] if reverse_order else expected, strict=True):
    np.testing.assert_allclose(got, want)


@sc.function(sc.arg("a", 2), outputs=sc.arg("out"))
def _pair(a: sc.Expr) -> sc.Expr:
  return sc.stack([a[0] * a[1], a[0].sin() * a[1] * a[1]])


@pytest.mark.parametrize("mapped", [False, True])
@pytest.mark.parametrize("slice_output", [False, True])
def test_call_and_map(mapped: bool, slice_output: bool) -> None:
  @sc.function(sc.arg("p", 3), sc.arg("c", 6), outputs=sc.group(sc.arg("jac"), sc.arg("sparse"), sc.arg("adj")))
  def fn(p: sc.Expr, c: sc.Expr) -> tuple[sc.Expr, sc.Expr, sc.Expr]:
    args = sc.concat([p, c[4:5]])
    value = sc.vmap(_pair, 2)(args.reshape((2, 2))).vec() if mapped else sc.concat([_pair(args[:2]), _pair(args[2:])])
    out = value[1:3] ** 3 + value[0] if slice_output else value
    pattern = sc.jacobian_sparsity(out, p)
    assert pattern.nnz == (5 if slice_output else 6)
    return sc.jacobian(out, p), sc.sparse_jacobian(out, p).to_dense(), sc.vjp((out,), (p,), (sc.const(np.ones(out.shape)),))[0]

  a, b, d, e = POINT[1:5]
  values = np.array([a * b, np.sin(a) * b**2, d * e, np.sin(d) * e**2])
  jac = np.array([[b, a, 0], [np.cos(a) * b**2, 2 * np.sin(a) * b, 0], [0, 0, e], [0, 0, np.cos(d) * e**2]])
  if slice_output:
    jac = 3 * values[1:3, None] ** 2 * jac[1:3] + jac[0]
  dense, sparse, adj = fn(POINT[1:4], POINT)
  np.testing.assert_allclose(dense, jac)
  np.testing.assert_allclose(sparse, jac)
  np.testing.assert_allclose(adj, jac.sum(axis=0))


def test_newton_calls_derivatives_on_carry_slices() -> None:
  @sc.function(sc.arg("x", 3), sc.arg("a", ()), outputs=sc.arg("energy"))
  def energy(x: sc.Expr, a: sc.Expr) -> sc.Expr:
    return sc.sumsqr(x * x - a) + 0.1 * sc.sumsqr(x)

  grad_energy = sc.gradient(energy, "energy", "x")
  hess_energy = sc.hessian(energy, "energy", "x")

  @sc.function(sc.arg("carry", 4), outputs=sc.group(sc.arg("next"), sc.arg("grad")))
  def body(carry: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
    x, a = carry[:3], carry[3]
    grad = grad_energy(x, a)
    diagonal = sc.gather(hess_energy(x, a).reshape((9,)), [0, 4, 8])
    return sc.concat([x - grad / diagonal, a.reshape((1,))]), grad

  carry = np.array([1.3, 1.5, 1.4, 2.0])
  for _ in range(10):
    x, a = carry[:3], carry[3]
    grad_ref = 4 * x * (x**2 - a) + 0.2 * x
    next_ref = np.r_[x - grad_ref / (12 * x**2 - 4 * a + 0.2), a]
    next_carry, grad = body(carry)
    np.testing.assert_allclose(grad, grad_ref, atol=1e-13)
    np.testing.assert_allclose(next_carry, next_ref, atol=1e-13)
    carry = next_carry
    if np.max(np.abs(grad)) < 1e-12:
      break
  else:
    pytest.fail("Newton iteration did not converge")
  np.testing.assert_allclose(carry[:3], np.full(3, np.sqrt(2.0 - 0.05)), rtol=1e-12)


@pytest.mark.parametrize("window", [False, True])
def test_structured_map_of_an_input_slice(window: bool) -> None:
  @sc.function(sc.arg("c", 6), outputs=sc.group(sc.arg("jac"), sc.arg("sparse")))
  def fn(c: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
    p = c[1:5]
    value = sc.vmap(_pair, 2)(sc.window(p, 0, 2) if window else p.reshape((2, 2))).vec()
    return sc.jacobian(value, c), sc.sparse_jacobian(value, c).to_dense()

  expected = np.zeros((4, 6))
  for i, (a, b) in enumerate(POINT[1:5].reshape(2, 2)):
    expected[2 * i : 2 * i + 2, 1 + 2 * i : 3 + 2 * i] = [[b, a], [np.cos(a) * b**2, 2 * np.sin(a) * b]]
  for got in fn(POINT):
    np.testing.assert_allclose(got, expected)


@pytest.mark.parametrize("selection", ["output", "absent"])
def test_input_output_and_unused_input(selection: str) -> None:
  @sc.function(sc.arg("p", 3), sc.arg("unused", 2), outputs=sc.group(sc.arg("jac"), sc.arg("adj")))
  def fn(p: sc.Expr, unused: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
    out = p
    wrt = p if selection == "output" else unused
    pattern = sc.jacobian_sparsity(out, wrt)
    np.testing.assert_array_equal(pattern.to_mask(), np.eye(3, dtype=bool) if selection == "output" else np.zeros((3, 2), dtype=bool))
    return sc.jacobian(out, wrt), sc.vjp((out,), (wrt,), (sc.const(np.ones(3)),))[0]

  jac, adj = fn(POINT[1:4], POINT[4:])
  np.testing.assert_array_equal(jac, np.eye(3) if selection == "output" else np.zeros((3, 2)))
  np.testing.assert_array_equal(adj, np.ones(3) if selection == "output" else np.zeros(2))
