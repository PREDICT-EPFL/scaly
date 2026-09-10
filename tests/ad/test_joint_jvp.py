"""Joint call tangents and packed mapped derivatives share work without changing seed layout."""

from __future__ import annotations

import numpy as np
import pytest

import alloy as al
from alloy.ad.forward import _call_jvp_function
from alloy.ir.expr import ExprOp, topo
from alloy.ir.program import ProgramOp
from alloy.passes.lowering import lower_function
from alloy.passes.program._common import _walk


@pytest.mark.parametrize("mapped", [False, True])
@pytest.mark.parametrize("constant", [False, True])
@pytest.mark.parametrize("nseed", [1, 2, 3])
def test_joint_active_formals(mapped: bool, constant: bool, nseed: int) -> None:
  x, y, unused = al.sym("x", 2), al.sym("y", 2), al.sym("unused", 2)
  wx, wy = np.array([[1.0, 0.2], [-0.3, 0.7]]), np.array([[0.8, -0.1], [0.4, 0.9]])
  a, b = x @ al.const(wx), y @ al.const(wy)
  body = (a * b).sin() + a / b
  stage = al.Function._from_exprs("joint_stage", [x, y, unused], [body], ["x", "y", "unused"], ["out"])
  length = 4 if mapped else 1
  z = al.sym("z", 2 * length + 1)
  value = al.vmap(stage, length, [(z, 0, 2), (z, 1, 2), (z, 0, 2)]) if mapped else stage((z[:2], z[1:], z[:2]))
  sv = np.random.default_rng(3).normal(size=(nseed, z.size))
  if nseed > 1:
    sv[1] = 0
  seed = al.const(sv) if constant else al.sym("seed", sv.shape)
  derivative = al.jvp(value, z, seed[0]) if nseed == 1 else al.jvp_many(value, z, seed)
  inputs = [z] if constant else [z, seed]
  fn = al.Function._from_exprs("joint_actual", inputs, [derivative], ["z"] if constant else ["z", "seed"], ["dy"])
  zv = np.linspace(0.5, 1.5, z.size)
  expected = np.empty((nseed, 2 * length))
  for it in range(length):
    a, b = zv[2 * it : 2 * it + 2] @ wx, zv[2 * it + 1 : 2 * it + 3] @ wy
    da, db = sv[:, 2 * it : 2 * it + 2] @ wx, sv[:, 2 * it + 1 : 2 * it + 3] @ wy
    expected[:, 2 * it : 2 * it + 2] = np.cos(a * b) * (da * b + a * db) + da / b - a * db / b**2
  actual = fn(zv) if constant else fn((zv, sv))
  np.testing.assert_allclose(actual, expected[0] if nseed == 1 else expected, atol=1e-12, rtol=1e-12)
  calls = [node for node in topo([derivative]) if node.op in {ExprOp.CALL, ExprOp.VMAP}]
  assert len(calls) == 1


def test_packed_mapped_hessian_shares_primal_and_preserves_zero_seed_rows() -> None:
  x, u = al.sym("value", 2), al.sym("value", 1)
  weights = np.array([0.3, -0.7, 0.4])
  body = (x @ al.const(weights[:2]) + weights[2] * u[0]).exp().scalar()
  stage = al.Function._from_exprs("packed_stage", [x, u], [body], ["x", "u"], ["cost"])
  length = 4
  z = al.sym("z", 3 * length)
  mapped = al.vmap(stage, length, [(z, 0, 3), (z, 2, 3)])
  grad = al.vjp((mapped,), (z,), (al.const(np.ones(length)),))[0]
  seeds = np.tile(np.vstack([np.eye(3), np.zeros(3)]), (1, length))
  hess = al.jvp_many(grad, z, al.const(seeds))
  fn = al.Function._from_exprs("packed_hess", [z], [hess], ["z"], ["h"])
  zv = np.linspace(-0.8, 1.0, z.size)
  expected = np.zeros((4, z.size))
  for it in range(length):
    expected[:3, 3 * it : 3 * it + 3] = np.exp(zv[3 * it : 3 * it + 3] @ weights) * np.outer(weights, weights)
  np.testing.assert_allclose(fn(zv), expected, atol=1e-12, rtol=1e-12)
  calls = [node for node in topo([hess]) if node.op == ExprOp.VMAP]
  assert len(calls) == 1 and "_fwd_pack_" in calls[0].attrs["callee"].name
  prog = lower_function(fn)
  exp_nodes = {node for node in _walk(prog) if node.op == ProgramOp.EXP}
  assert len(exp_nodes) == 1


def test_joint_helper_cache_distinguishes_formal_sets() -> None:
  a, b, c = (al.sym(name, 2) for name in ("a_b", "a", "b"))
  stage = al.Function._from_exprs("joint_names", [a, b, c], [a * b + c.sin()], ["a_b", "a", "b"], ["y"])
  first = _call_jvp_function(stage, 0, (0,))
  second = _call_jvp_function(stage, 0, (1, 2))
  assert first[0] is _call_jvp_function(stage, 0, (0,))[0]
  assert first[0].name != second[0].name
  assert first[0] is not second[0]


@pytest.mark.parametrize("matrix", [False, True])
def test_self_products_keep_matrix_product_rule(matrix: bool) -> None:
  shape = (2, 2) if matrix else (4,)
  x, seed = al.sym("x", shape), al.sym("seed", (2, *shape))
  dot = x @ x
  outputs = [al.jvp(dot, x, seed[0]), al.jvp_many(dot, x, seed), al.jvp_many(x * x, x, seed)]
  fn = al.Function._from_exprs("self_products", [x, seed], outputs, ["x", "seed"], ["one", "many", "square"])
  xv = np.array([0.2, -0.7, 1.1, 0.9]).reshape(shape)
  sv = np.arange(8.0).reshape((2, *shape)) / 5 - 0.6
  one, many, square = fn((xv, sv))
  expected = np.stack([tangent @ xv + xv @ tangent for tangent in sv])
  np.testing.assert_allclose(one, expected[0], atol=1e-12, rtol=1e-12)
  np.testing.assert_allclose(many, expected, atol=1e-12, rtol=1e-12)
  np.testing.assert_allclose(square, 2 * xv * sv, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("scale", [1e-200, 1.0, 1e200])
def test_division_and_sqrt_derivatives_across_finite_scales(scale: float) -> None:
  x, y = al.sym("x", ()), al.sym("y", ())
  quotient = x / y
  dx = al.jvp(quotient, x, al.const(1.0))
  dy = al.jvp_many(quotient, y, al.const([1.0, -0.5]))
  gx, gy = al.vjp((quotient,), (x, y), (al.const(1.0),))
  root = al.jvp_many(x.sqrt(), x, al.const([1.0, -0.5]))
  fn = al.Function._from_exprs("finite_derivatives", [x, y], [dx, dy, gx, gy, root], ["x", "y"], ["dx", "dy", "gx", "gy", "root"])
  actual = fn((scale, scale))
  expected = [1.0 / scale, np.array([-1.0, 0.5]) / scale, 1.0 / scale, -1.0 / scale, np.array([0.5, -0.25]) / np.sqrt(scale)]
  for value, reference in zip(actual, expected, strict=True):
    assert np.isfinite(value).all()
    np.testing.assert_allclose(value, reference, rtol=1e-12, atol=0)


def test_mapped_local_and_generic_seeds_share_one_callee() -> None:
  x, u = al.sym("x", 2), al.sym("u", 1)
  weights = np.array([[1.0, 0.2], [-0.3, 0.7]])
  stage = al.Function._from_exprs("mixed_seed_stage", [x, u], [((x @ al.const(weights)) * u).sin()], ["x", "u"], ["out"])
  z, seeds = al.sym("z", 12), al.sym("seeds", (2, 12))
  mapped = al.vmap(stage, 4, [(z, 0, 3), (z, 2, 3)])
  derivative = al.jvp_many(mapped, z, seeds)
  fn = al.Function._from_exprs("mixed_seed_actual", [z, seeds], [derivative], ["z", "seeds"], ["dy"])
  zv = np.linspace(0.1, 1.2, 12)
  sv = np.random.default_rng(2).normal(size=(2, 12))
  expected = np.empty((2, 8))
  for it in range(4):
    xval, uval = zv[3 * it : 3 * it + 2] @ weights, zv[3 * it + 2]
    dx, du = sv[:, 3 * it : 3 * it + 2] @ weights, sv[:, 3 * it + 2 : 3 * it + 3]
    expected[:, 2 * it : 2 * it + 2] = np.cos(xval * uval) * (dx * uval + xval * du)
  np.testing.assert_allclose(fn((zv, sv)), expected, atol=1e-12, rtol=1e-12)
  calls = [node for node in topo([derivative]) if node.op == ExprOp.VMAP]
  assert len(calls) == 1 and "_fwd_pack_" in calls[0].attrs["callee"].name


def test_call_combines_constant_and_runtime_tangents() -> None:
  x, y = al.sym("x", 2), al.sym("y", 2)
  stage = al.Function._from_exprs("mixed_call", [x, y], [(x * y).sin()], ["x", "y"], ["out"])
  z = al.sym("z", 2)
  value = stage((z, z.sin() + 2))
  seeds = np.vstack([np.eye(2), np.zeros(2)])
  derivative = al.jvp_many(value, z, al.const(seeds))
  fn = al.Function._from_exprs("mixed_call_actual", [z], [derivative], ["z"], ["dy"])
  zv = np.array([0.2, -0.7])
  expected = seeds * (np.cos(zv * (np.sin(zv) + 2)) * (np.sin(zv) + 2 + zv * np.cos(zv)))
  np.testing.assert_allclose(fn(zv), expected, atol=1e-12, rtol=1e-12)
  assert sum(node.op == ExprOp.CALL for node in topo([derivative])) == 1
