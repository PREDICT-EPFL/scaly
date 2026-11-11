"""Derivatives of ``where``, the predicates and ``cast``.

Forward, reverse, Jacobians and Hessians against closed forms and finite differences: a NaN or
infinite derivative in the branch ``where`` does not choose stays out of the result in both modes,
predicates and casts to integers carry no derivative, float casts do, and the structural pattern of
a select is the union of its branches. Values are in ``tests/codegen/test_select_cast.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

import scaly as sc
from scaly.ad import finite_difference

WHERE_POINT = np.array([0.0, -1.0, -0.0, 4.0])
WHERE_GRAD = np.array([1.0, 0.0, 1.0, 0.25 + 0.25 - 1.0 / 16.0])


def _guarded(x: sc.Expr) -> sc.Expr:
  """Each unchosen branch has an infinite or NaN derivative at some entry of ``WHERE_POINT``."""
  return sc.where(x > 0.0, x.sqrt(), 0.0) + sc.where(x > 0.0, x.log(), x) + sc.where(sc.not_equal(x, 0.0), 1.0 / x, 0.0)


def test_forward_mode_ignores_a_nan_or_infinite_derivative_in_the_unchosen_branch() -> None:
  @sc.function(sc.arg("x", 4), outputs=sc.group(sc.arg("y"), sc.arg("t"), sc.arg("j")), name="sd_where_fwd")
  def forward(x: sc.Expr) -> tuple[sc.Expr, sc.Expr, sc.Expr]:
    y = _guarded(x)
    return y, sc.jvp(y, x, sc.const(np.ones(4))), sc.jacobian(y, x)

  value, tangent, jac = forward(WHERE_POINT)
  np.testing.assert_array_equal(value, [0.0, -2.0, 0.0, 2.0 + np.log(4.0) + 0.25])
  np.testing.assert_array_equal(tangent, WHERE_GRAD)
  np.testing.assert_array_equal(jac, np.diag(WHERE_GRAD))


def test_reverse_mode_ignores_a_nan_or_infinite_derivative_in_the_unchosen_branch() -> None:
  @sc.function(sc.arg("x", 4), outputs=sc.group(sc.arg("g"), sc.arg("h")), name="sd_where_rev")
  def reverse(x: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
    cost = _guarded(x).sum()
    return sc.gradient(cost, x), sc.hessian(cost, x)

  grad, hess = (np.asarray(v) for v in reverse(WHERE_POINT))
  np.testing.assert_array_equal(grad, WHERE_GRAD)
  np.testing.assert_array_equal(np.diag(hess), [0.0, -2.0, 0.0, -1.0 / 32.0 - 1.0 / 16.0 + 2.0 / 64.0])


def test_reverse_mode_keeps_a_nested_mask_through_a_chain_of_elementwise_ops() -> None:
  """The unchosen entries stay exactly zero through ``sin(2 sqrt(x))``, through a ``where`` inside a
  ``where`` where only the outer one leaves ``sqrt`` unchosen, and in the one ``sqrt`` node both
  read, whose cotangent sums two masked ones: as forward mode gives them."""

  @sc.function(sc.arg("x", 4), outputs=sc.group(sc.arg("g"), sc.arg("t"), sc.arg("h")), name="sd_where_nested")
  def nested(x: sc.Expr) -> tuple[sc.Expr, sc.Expr, sc.Expr]:
    chain = sc.where(x > 0.0, (x.sqrt() * 2.0).sin(), 0.0)
    inner = sc.where(x > 0.0, sc.where(x > -2.0, x.sqrt(), 0.0), 0.0)
    cost = (chain + inner + sc.where(x <= 0.0, x * x, (1.0 - x).log())).sum()
    return sc.gradient(cost, x), sc.jvp(cost, x, sc.const(np.ones(4))), sc.hessian(cost, x)

  grad, tangent, hess = (np.asarray(v) for v in nested(WHERE_POINT))
  root = np.sqrt(4.0)
  expected = np.array([0.0, -2.0, 0.0, np.cos(2.0 * root) / root + 0.5 / root - 1.0 / (1.0 - 4.0)])
  np.testing.assert_allclose(grad, expected, rtol=1e-15)
  np.testing.assert_allclose(tangent, grad.sum(), rtol=1e-15)
  assert np.all(np.isfinite(hess))


def test_where_with_broadcast_operands_differentiates_like_its_unrolled_form() -> None:
  lam = np.linspace(-1.0, 1.5, 6)

  @sc.function(sc.arg("x", 3), outputs=sc.group(sc.arg("y"), sc.arg("j"), sc.arg("r"), sc.arg("s"), sc.arg("h")), name="sd_where_bcast")
  def picked(x: sc.Expr) -> tuple[sc.Expr, sc.Expr, sc.Expr, sc.Expr, sc.Expr]:
    cond = (x[:2] > 0.5).reshape((2, 1))
    out = sc.where(cond, x.sin(), x[0] * 2.0).reshape((6,))  # (2, 1) condition, (3,) and () branches
    (rev,) = sc.vjp((out,), (x,), (sc.const(lam),))
    return out, sc.jacobian(out, x), rev, sc.sparse_jacobian(out, x).to_dense(), sc.hessian((out * sc.const(lam)).sum(), x)

  xv = np.array([0.2, 0.9, -0.4])
  value, jac, rev, sparse_dense, hess = (np.asarray(v) for v in picked(xv))
  np.testing.assert_allclose(value, np.where((xv[:2] > 0.5)[:, None], np.sin(xv), xv[0] * 2.0).reshape(-1), rtol=1e-15)
  np.testing.assert_allclose(jac, finite_difference(lambda v: np.asarray(picked(v)[0]), xv), rtol=1e-7, atol=1e-9)
  np.testing.assert_allclose(rev, lam @ jac, rtol=1e-14)
  np.testing.assert_array_equal(sparse_dense, jac)
  np.testing.assert_allclose(hess, finite_difference(lambda v: np.asarray(picked(v)[2]), xv), rtol=1e-6, atol=1e-8)


def test_predicates_and_integer_casts_carry_no_derivative_and_float_casts_do() -> None:
  @sc.function(sc.arg("x", 4), outputs=sc.group(*(sc.arg(f"o{k}") for k in range(6))), name="sd_pred_cast")
  def derivatives(x: sc.Expr) -> tuple[sc.Expr, ...]:
    flat = (
      sc.cast(x < 0.5, "float64")
      + sc.cast(sc.isfinite(1.0 / x) & ~sc.equal(x, 2.0), "float64")
      + sc.cast(sc.cast(x, "int64") // 2, "float64")
      + sc.cast(sc.cast(x, "bool"), "float64")
    )
    gated = x * sc.cast(x > 0.0, "float64")
    outs: list[sc.Expr] = []
    for y in (flat, gated):
      outs += [sc.gradient(y.sum(), x), sc.jvp(y, x, sc.const(np.ones(4))), sc.jacobian(y, x)]
    return tuple(outs)

  got = [np.asarray(g) for g in derivatives(np.array([-1.3, 0.25, 0.75, 2.5]))]
  for g in got[:3]:
    np.testing.assert_array_equal(g, np.zeros_like(g))
  gate = np.array([0.0, 1.0, 1.0, 1.0])
  for g in (got[3], got[4], np.diag(got[5])):
    np.testing.assert_array_equal(g, gate)
  x = sc.sym("x", 4)
  assert sc.jacobian_sparsity(sc.cast(x < 0.5, "float64"), x).nnz == 0
  assert not (x < 0.5).type.diff and not sc.cast(x, "int64").type.diff and sc.where(x > 0.0, x, 0.0).type.diff


def test_select_pattern_is_the_union_of_its_branches() -> None:
  x = sc.sym("x", 4)
  picked = sc.where(x[:2] > x[2:], x[:2], x[3] * 2.0)
  pattern = sc.jacobian_sparsity(picked, x)
  assert sorted(zip(pattern.rows, pattern.cols, strict=True)) == [(0, 0), (0, 3), (1, 1), (1, 3)]


def test_derivatives_through_a_float32_cast_come_back_in_the_source_dtype() -> None:
  @sc.function(sc.arg("x", 3), outputs=sc.group(sc.arg("y"), sc.arg("j"), sc.arg("g")), name="sd_f32_cast")
  def through32(x: sc.Expr) -> tuple[sc.Expr, sc.Expr, sc.Expr]:
    narrow = sc.cast(x, "float32")
    y = sc.cast(sc.where(narrow > 0.0, narrow.sin(), narrow * narrow), "float64") + 3.0 * sc.cast(narrow, "float64")
    return y, sc.jacobian(y, x), sc.gradient(y.sum(), x)

  xv = np.array([0.3, -0.7, 1.1])
  value, jac, grad = (np.asarray(v) for v in through32(xv))
  slope = np.where(xv > 0.0, np.cos(xv), 2.0 * xv) + 3.0
  np.testing.assert_allclose(value, np.where(xv > 0.0, np.sin(xv), xv * xv) + 3.0 * xv, rtol=1e-6)
  np.testing.assert_allclose(jac, np.diag(slope), rtol=1e-6, atol=0.0)
  np.testing.assert_allclose(grad, slope, rtol=1e-6)


@sc.function(sc.arg("u", 2), outputs=sc.arg("r"), name="sd_root")
def _root(u: sc.Expr) -> sc.Expr:
  return u.sqrt()


# name -> (cost, point, gradient). Each unchosen branch has an infinite derivative at the point, and
# reaches the select through a reduction, a slice, a broadcast, a reshape, a cast or a second select.
MASKED_PATHS = {
  "norm": (lambda x: sc.where(sc.norm_2(x) > 0.0, x / sc.norm_2(x), 0.0).sum(), np.zeros(3), np.zeros(3)),
  "slice": (lambda x: sc.where(x[0] > 0.0, x.sqrt()[0], 0.0), np.array([0.0, 4.0]), np.zeros(2)),
  "broadcast": (lambda x: sc.where(x[1:] > 0.0, x[0].sqrt(), 0.0).sum(), np.array([0.0, -1.0, -2.0]), np.zeros(3)),
  "two_conditions": (
    lambda x: (sc.where(x > 0.0, x.sqrt(), 0.0) + sc.where(x > 1.0, x.sqrt(), 0.0)).sum(),
    np.array([0.0, 4.0]),
    np.array([0.0, 0.5]),
  ),
  "reshape": (
    lambda x: sc.where(x.reshape((2, 2)) > 0.0, x.sqrt().reshape((2, 2)), 0.0).sum(),
    np.array([0.0, -1.0, 4.0, 9.0]),
    np.array([0.0, 0.0, 0.25, 1.0 / 6.0]),
  ),
  "cast": (
    lambda x: sc.where(x > 0.0, sc.cast(sc.cast(x, "float32").sqrt(), "float64"), 0.0).sum(),
    np.array([0.0, 4.0]),
    np.array([0.0, 0.25]),
  ),
}


def test_reverse_mode_keeps_the_mask_through_reductions_slices_broadcasts_reshapes_and_casts() -> None:
  for name, (cost, point, expected) in MASKED_PATHS.items():

    @sc.function(sc.arg("x", point.size), outputs=sc.group(sc.arg("g"), sc.arg("t")), name=f"sd_masked_{name}")
    def derivatives(x: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
      return sc.gradient(cost(x), x), sc.jvp(cost(x), x, sc.const(np.ones(point.size)))

    grad, tangent = (np.asarray(v) for v in derivatives(point))
    np.testing.assert_allclose(grad, expected, rtol=1e-6, err_msg=name)
    np.testing.assert_allclose(tangent, expected.sum(), rtol=1e-6, err_msg=name)


def test_reverse_mode_does_not_carry_the_mask_into_a_function_call() -> None:
  """A called Function's derivative is built once for every caller, so its own NaN partial stays."""

  @sc.function(sc.arg("x", 2), outputs=sc.group(sc.arg("g"), sc.arg("t")), name="sd_masked_call")
  def derivatives(x: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
    cost = sc.where(x > 0.0, _root(x), 0.0).sum()
    return sc.gradient(cost, x), sc.jvp(cost, x, sc.const(np.ones(2)))

  grad, tangent = (np.asarray(v) for v in derivatives(np.array([0.0, 4.0])))
  assert np.isnan(grad[0]) and grad[1] == 0.25 and tangent == 0.25


def test_reverse_mode_refuses_an_integer_or_bool_output() -> None:
  x = sc.sym("x", 3)
  for value in (sc.cast(x, "int64")[0], (x > 0.0)[0]):
    with pytest.raises(TypeError, match="no tangent"):
      sc.gradient(value, x)


def test_reverse_mode_through_a_float32_sum_stays_float32() -> None:
  @sc.function(sc.arg("x", 3), outputs=sc.arg("g"), name="sd_f32_sum")
  def grad(x: sc.Expr) -> sc.Expr:
    return sc.gradient(sc.cast((sc.cast(x, "float32") ** 2).sum(), "float64"), x)

  xv = np.array([0.3, -0.7, 1.1])
  np.testing.assert_allclose(np.asarray(grad(xv)), 2.0 * xv, rtol=1e-6)
