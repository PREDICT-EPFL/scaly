"""Typed mapped calls, template slices, and affine view reads."""

from __future__ import annotations

import numpy as np
import pytest

import scaly as sc
from scaly.function.sugar import _mapped_call
from scaly.ir.expr import topo
from scaly.codegen import render_c_source


def test_mapped_template_keeps_parameter_and_nested_output_trees() -> None:
  traces = []

  @sc.function(sc.group(sc.arg("x"), sc.arg("p", ())), outputs=sc.group(sc.arg("y"), sc.group(sc.arg("cost", ()))))
  def stage(inputs: tuple[sc.Expr, sc.Expr]) -> tuple[sc.Expr, tuple[sc.Expr]]:
    x, p = inputs
    traces.append(x.shape)
    return x * p, ((x * x).sum(),)

  mapped = sc.vmap(stage, 3)
  x = np.arange(6.0).reshape(3, 2)
  values, (cost,) = mapped((x, sc.broadcast(np.array(2.0))))
  np.testing.assert_array_equal(values, 2.0 * x)
  np.testing.assert_array_equal(cost, np.sum(x * x, axis=1))
  symbolic, (symbolic_cost,) = mapped((sc.sym("x", (3, 2)), sc.broadcast(sc.const(2.0))))
  assert symbolic.shape == (3, 2) and symbolic_cost.shape == (3,)
  assert symbolic.args[0].op == sc.ExprOp.VMAP
  assert not any(node.op == sc.ExprOp.CALL for node in topo([symbolic, symbolic_cost]))
  assert mapped.instantiate(((3, 2), (3,))).output_shapes == ((3, 2), (3,))
  assert traces == [(2,)]


def test_affine_views_equal_explicit_ir_windows_and_emit_identical_c() -> None:
  @sc.function(sc.arg("x", 2), sc.arg("u", ()), outputs=sc.arg("y", 2))
  def stage(x: sc.Expr, u: sc.Expr) -> sc.Expr:
    return x * x + u

  z = sc.sym("z", 12)
  Z = z.reshape((4, 3))
  viewed = sc.vmap(stage, 4)(Z[:, :2], Z[:, 2]).vec()
  explicit = _mapped_call(stage, 4, [(z, 0, 3), (z, 2, 3)])
  assert viewed.args[0].args[0] is explicit  # output reshapes retain the original VMAP node
  left = sc.function(sc.arg("z", 12), outputs=sc.arg("y", 8), name="views")(
    lambda x: sc.vmap(stage, 4)(x.reshape((4, 3))[:, :2], x.reshape((4, 3))[:, 2]).vec()
  )
  right = sc.function(sc.arg("z", 12), outputs=sc.arg("y", 8), name="views")(lambda x: _mapped_call(stage, 4, [(x, 0, 3), (x, 2, 3)]))
  assert render_c_source(left) == render_c_source(right)


@pytest.mark.parametrize("kind", ["transpose", "negative", "gapped", "nested_slice"])
def test_views_and_their_derivatives_match_numpy(kind: str) -> None:
  @sc.function(sc.arg("x"), outputs=sc.arg("y"))
  def square(x: sc.Expr) -> sc.Expr:
    return x * x

  def view(x):
    if kind == "transpose":
      return x.reshape((2, 3)).T
    if kind == "negative":
      return x[::-1].reshape((3, 2))
    if kind == "gapped":
      return x.reshape((3, 2))[:, ::2]
    return x[1:][1:].reshape((2, 2))

  n = 2 if kind == "nested_slice" else 3

  @sc.function(sc.arg("x", 6), outputs=sc.arg("y"))
  def mapped(x: sc.Expr) -> sc.Expr:
    return sc.vmap(square, n)(view(x)).vec()

  x = np.arange(1.0, 7.0)
  expected = view(x).reshape(-1)
  np.testing.assert_array_equal(mapped(x), expected * expected)
  indices = view(np.arange(6)).reshape(-1)
  jac = np.zeros((indices.size, 6))
  jac[np.arange(indices.size), indices] = 2.0 * expected
  np.testing.assert_array_equal(sc.jacobian(mapped)(x), jac)
  np.testing.assert_array_equal(
    sc.hessian(sc.function(sc.arg("x", 6), outputs=sc.arg("f", ()))(lambda z: mapped(z).sum()))(x), np.diag(np.bincount(indices, minlength=6) * 2.0)
  )


def test_overlapping_windows_broadcast_and_sparse_derivatives_match_unrolled() -> None:
  @sc.function(sc.arg("x", 2), sc.arg("p", 2), outputs=sc.arg("y", 2))
  def stage(x: sc.Expr, p: sc.Expr) -> sc.Expr:
    return x * x + p

  @sc.function(sc.arg("z", 5), sc.arg("p", 2), outputs=sc.arg("y", (4, 2)))
  def mapped(z: sc.Expr, p: sc.Expr) -> sc.Expr:
    return sc.vmap(stage, 4)(sc.window(z, 0, 1), sc.broadcast(p))

  @sc.function(sc.arg("z", 5), sc.arg("p", 2), outputs=sc.arg("y", (4, 2)))
  def unrolled(z: sc.Expr, p: sc.Expr) -> sc.Expr:
    return sc.stack([stage(z[i : i + 2], p) for i in range(4)])

  z, p = np.arange(5.0), np.array([2.0, 3.0])
  np.testing.assert_array_equal(mapped(z, p), unrolled(z, p))
  for operation in (sc.jacobian, sc.sparse_jacobian):
    np.testing.assert_array_equal(operation(mapped, "y", "z")(z, p), operation(unrolled, "y", "z")(z, p))


def test_zero_length_and_zero_parameter_maps() -> None:
  @sc.function(sc.arg("x", 2), outputs=sc.arg("y", 2))
  def identity(x: sc.Expr) -> sc.Expr:
    return x

  assert sc.vmap(identity, 0)(np.empty((0, 2))).shape == (0, 2)

  @sc.function(outputs=sc.arg("value", ()))
  def constant() -> sc.Expr:
    return sc.const(2.0)

  np.testing.assert_array_equal(sc.vmap(constant, 3)(), np.full(3, 2.0))
  assert sc.vmap(constant, 3).symbolic_call().shape == (3,)


def test_bare_map_and_resolution_errors() -> None:
  @sc.function()
  def square(x: sc.Expr) -> sc.Expr:
    return x * x

  np.testing.assert_array_equal(sc.vmap(square, 2)(np.arange(4.0).reshape(2, 2)), [[0.0, 1.0], [4.0, 9.0]])
  with pytest.raises(TypeError, match="declared slice shape"):
    sc.vmap(square, 2)(sc.window(np.ones(5), 0, 1))
  with pytest.raises(ValueError, match="cannot map shape"):
    sc.vmap(square, 2)(np.ones(3))
  with pytest.raises(ValueError, match="non-negative"):
    sc.window(np.ones(3), 0, -1)
  with pytest.raises(ValueError, match="non-negative"):
    sc.vmap(square, -1)
  with pytest.raises(ValueError, match="Expr leaves"):
    sc.vmap(square, 2).symbolic_call(np.ones((2, 3)))  # ty: ignore[invalid-argument-type]


def test_repeated_maps_share_instances_and_derivatives() -> None:
  @sc.function(sc.arg("x", 2), outputs=sc.arg("y", 2))
  def square(x: sc.Expr) -> sc.Expr:
    return x * x

  left, right = sc.vmap(square, 3), sc.vmap(square, 3)
  assert left.instantiate() is right.instantiate()
  assert sc.jacobian(left).instantiate() is sc.jacobian(right).instantiate()

  @sc.function(sc.arg("x", (3, 2)), outputs=sc.arg("y", (6, 6)))
  def host(x: sc.Expr) -> sc.Expr:
    return sc.jacobian(left)(x) + sc.jacobian(right)(x)

  np.testing.assert_array_equal(host(np.ones((3, 2))), 4.0 * np.eye(6))


def test_contiguous_maps_do_not_allocate_view_indices(monkeypatch) -> None:
  @sc.function(sc.arg("x", 2), outputs=sc.arg("y", 2))
  def identity(x: sc.Expr) -> sc.Expr:
    return x

  def refuse(*args, **kwargs):
    raise AssertionError("unexpected view index allocation")

  monkeypatch.setattr(np, "arange", refuse)
  assert sc.vmap(identity, 3)(sc.sym("x", (3, 2))).shape == (3, 2)


def test_sparse_map_coloring_survives_output_shape_wrappers() -> None:
  @sc.function(sc.arg("x", 2), outputs=sc.arg("y", 2))
  def square(x: sc.Expr) -> sc.Expr:
    return x * x

  widths = []
  for n in (2, 8, 32):

    @sc.function(sc.arg("x", 2 * n), outputs=sc.arg("y", (n, 2)))
    def host(x: sc.Expr) -> sc.Expr:
      return sc.vmap(square, n)(x)

    derivative = sc.sparse_jacobian(host)
    widths.append(derivative.instantiate().output_coloring_widths[0])
    direct = sc.sparse_jacobian(sc.vmap(square, n))
    assert direct.instantiate().output_coloring_widths[0] == 1
    np.testing.assert_array_equal(direct(np.arange(2.0 * n).reshape(n, 2)), 2.0 * np.arange(2.0 * n))
    np.testing.assert_array_equal(derivative(np.arange(2.0 * n)), 2.0 * np.arange(2.0 * n))
  assert widths == [1, 1, 1]
