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
from scaly.function.sugar import _mapped_call
from scaly.ir.types import CanonicalPattern

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
    return out, sc.jacobian(out, x), rev, sc.to_dense(sc.sparse_jacobian(out, x)), sc.hessian((out * sc.const(lam)).sum(), x)

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


@pytest.mark.parametrize("mapped", [False, True])
def test_reverse_mode_carries_the_mask_into_a_function_call(mapped: bool) -> None:
  size = 4 if mapped else 2

  @sc.function(sc.arg("x", size), outputs=sc.group(*(sc.arg(k) for k in ("g", "j", "h", "rr"))), name="sd_masked_call")
  def derivatives(x: sc.Expr) -> tuple[sc.Expr, ...]:
    roots = sc.vmap(_root, 2)(x.reshape((2, 2))).vec() if mapped else _root(x)
    chosen = sc.where(x > 0.0, roots, 0.0)
    grad = sc.gradient(chosen.sum(), x)
    return grad, sc.jacobian(chosen, x), sc.hessian(chosen.sum(), x), sc.gradient(grad.sum(), x)

  grad, jac, hess, reverse_twice = (np.asarray(v) for v in derivatives(np.tile([0.0, 4.0], size // 2)))
  expected = np.tile([0.0, 0.25], size // 2)
  second = np.tile([0.0, -1.0 / 32.0], size // 2)
  np.testing.assert_array_equal(grad, expected)
  np.testing.assert_array_equal(jac, np.diag(expected))
  np.testing.assert_array_equal(hess, np.diag(second))
  np.testing.assert_array_equal(reverse_twice, second)


@pytest.mark.parametrize("mapped", [False, True])
def test_called_mask_uses_the_condition_when_the_cotangent_is_zero(mapped: bool) -> None:
  @sc.function(sc.arg("x", 2), outputs=sc.group(sc.arg("g"), sc.arg("h")), name="sd_zero_cot_call")
  def derivatives(x: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
    roots = sc.vmap(_root, 1)(x.reshape((1, 2))).vec() if mapped else _root(x)
    cost = sc.where(x > 0.0, (roots - 2.0) ** 2, 0.0).sum()
    return sc.gradient(cost, x), sc.hessian(cost, x)

  grad, hess = (np.asarray(v) for v in derivatives(np.array([0.0, 4.0])))
  np.testing.assert_array_equal(grad, [0.0, 0.0])
  np.testing.assert_array_equal(hess, np.diag([0.0, 1.0 / 8.0]))


@pytest.mark.parametrize("mapped", [False, True])
def test_nested_calls_keep_the_callers_condition_with_shared_formal_symbols(mapped: bool) -> None:
  @sc.function(sc.arg("u", 2), outputs=sc.arg("r"), name="sd_shifted_root")
  def shifted(u: sc.Expr) -> sc.Expr:
    return _root(u + 1.0)

  @sc.function(sc.arg("u", 4 if mapped else 2), outputs=sc.group(sc.arg("g"), sc.arg("j")), name="sd_nested_calls")
  def derivatives(u: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
    roots = sc.vmap(shifted, 2)(u.reshape((2, 2))).vec() if mapped else shifted(u)
    chosen = sc.where(u < 5.0, sc.where(u > 0.0, roots, 0.0), 0.0)
    return sc.gradient(chosen.sum(), u), sc.jacobian(chosen, u)

  for pair in ([-1.0, 3.0], [-0.5, 3.0], [-1.0, 8.0]):
    point = np.tile(pair, 2 if mapped else 1)
    expected = np.tile([0.0, 0.25 if pair[1] == 3.0 else 0.0], 2 if mapped else 1)
    grad, jac = (np.asarray(v) for v in derivatives(point))
    np.testing.assert_array_equal(grad, expected)
    np.testing.assert_array_equal(jac, np.diag(expected))


def test_mapped_adjoint_cache_separates_masked_and_unmasked_calls_and_reuses_conditions() -> None:
  @sc.function(sc.arg("u", 2), outputs=sc.arg("r"), name="sd_cached_root")
  def root(u: sc.Expr) -> sc.Expr:
    return u.sqrt()

  @sc.function(sc.arg("x", 4), outputs=sc.group(*(sc.arg(k) for k in ("plain", "positive", "large"))), name="sd_cached_masks")
  def derivatives(x: sc.Expr) -> tuple[sc.Expr, ...]:
    roots = sc.vmap(root, 2)(x.reshape((2, 2))).vec()
    return tuple(sc.gradient(cost, x) for cost in (roots.sum(), sc.where(x > 0.0, roots, 0.0).sum(), sc.where(x > 4.0, roots, 0.0).sum()))

  plain, positive, large = (np.asarray(v) for v in derivatives(np.array([0.0, 4.0, 0.0, 4.0])))
  np.testing.assert_array_equal(plain, [np.inf, 0.25, np.inf, 0.25])
  np.testing.assert_array_equal(positive, [0.0, 0.25, 0.0, 0.25])
  np.testing.assert_array_equal(large, np.zeros(4))
  helpers = [(key, entry[0]) for key, entry in root.instantiate()._memo.helpers.items() if key.role == "adjoint" and isinstance(entry, tuple)]
  assert len(helpers) == 2
  assert {key.masked for key, _ in helpers} == {False, True}
  assert len({helper.name for _, helper in helpers}) == 2


def test_called_mask_survives_a_mapped_call_before_a_singular_partial() -> None:
  @sc.function(sc.arg("u", 2), outputs=sc.arg("r"), name="sd_twice")
  def twice(u: sc.Expr) -> sc.Expr:
    return 2.0 * u

  @sc.function(sc.arg("u", 4), outputs=sc.arg("r"), name="sd_mapped_roots")
  def roots(u: sc.Expr) -> sc.Expr:
    return sc.vmap(twice, 2)(u.sqrt().reshape((2, 2))).vec()

  @sc.function(sc.arg("x", 4), outputs=sc.group(sc.arg("g"), sc.arg("j")), name="sd_called_mapped_roots")
  def derivatives(x: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
    chosen = sc.where(x > 0.0, roots(x), 0.0)
    return sc.gradient(chosen.sum(), x), sc.jacobian(chosen, x)

  grad, jac = (np.asarray(v) for v in derivatives(np.array([0.0, 4.0, -1.0, 9.0])))
  expected = np.array([0.0, 0.5, 0.0, 1.0 / 3.0])
  np.testing.assert_array_equal(grad, expected)
  np.testing.assert_array_equal(jac, np.diag(expected))


@pytest.mark.parametrize("stride", [0, 1])
def test_mapped_masks_accumulate_through_shared_and_overlapping_arguments(stride: int) -> None:
  @sc.function(sc.arg("u", 2), outputs=sc.arg("r"), name="sd_window_twice")
  def twice(u: sc.Expr) -> sc.Expr:
    return 2.0 * u

  indices = np.array([it * stride + j for it in range(3) for j in range(2)])
  point = np.array([0.0, 4.0, 9.0, -1.0]) if stride else np.array([0.0, 4.0])

  @sc.function(sc.arg("x", point.size), outputs=sc.group(sc.arg("g"), sc.arg("j")), name="sd_window_masks")
  def derivatives(x: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
    roots = _mapped_call(twice, 3, [(x.sqrt(), 0, stride)])
    chosen = sc.where(sc.gather(x, indices) > 0.0, roots, 0.0)
    return sc.gradient(chosen.sum(), x), sc.jacobian(chosen, x)

  grad, jac = (np.asarray(v) for v in derivatives(point))
  expected = np.array([0.0, 1.0, 2.0 / 3.0, 0.0]) if stride else np.array([0.0, 1.5])
  np.testing.assert_array_equal(grad, expected)
  np.testing.assert_array_equal(grad, jac.sum(axis=0))


def test_two_mapped_calls_merge_their_masks_before_a_singular_partial() -> None:
  @sc.function(sc.arg("u", 2), outputs=sc.arg("r"), name="sd_double")
  def twice(u: sc.Expr) -> sc.Expr:
    return 2.0 * u

  @sc.function(sc.arg("u", 2), outputs=sc.arg("r"), name="sd_triple")
  def thrice(u: sc.Expr) -> sc.Expr:
    return 3.0 * u

  @sc.function(sc.arg("x", 4), outputs=sc.group(sc.arg("g"), sc.arg("j")), name="sd_merged_maps")
  def derivatives(x: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
    roots = x.sqrt().reshape((2, 2))
    chosen = sc.where(x > 0.0, sc.vmap(twice, 2)(roots).vec(), 0.0) + sc.where(x > 4.0, sc.vmap(thrice, 2)(roots).vec(), 0.0)
    return sc.gradient(chosen.sum(), x), sc.jacobian(chosen, x)

  grad, jac = (np.asarray(v) for v in derivatives(np.array([0.0, 4.0, -1.0, 9.0])))
  np.testing.assert_allclose(grad, [0.0, 0.5, 0.0, 5.0 / 6.0], rtol=1e-15)
  np.testing.assert_allclose(grad, jac.sum(axis=0), rtol=1e-15)


@pytest.mark.parametrize("mapped", [False, True])
def test_called_masks_use_stored_coordinates_for_sparse_outputs(mapped: bool) -> None:
  pattern = CanonicalPattern((2, 3), (0, 1), (0, 2))

  @sc.function(sc.arg("u", sc.TensorType(pattern.shape, pattern=pattern)), outputs=sc.arg("r"), name="sd_sparse_sin")
  def stage(u: sc.Expr) -> sc.Expr:
    return u.sqrt()

  size = 4 if mapped else 2

  @sc.function(sc.arg("x", size), outputs=sc.group(sc.arg("g"), sc.arg("j")), name="sd_sparse_mask")
  def derivatives(x: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
    sparse = sc.pack(x.reshape((2, 2)), pattern) if mapped else sc.pack(x, pattern)
    chosen = sc.where(x > 0.0, sc.values(sc.vmap(stage, 2)(sparse) if mapped else stage(sparse)).vec(), 0.0)
    return sc.gradient(chosen.sum(), x), sc.jacobian(chosen, x)

  point = np.array([0.0, 4.0, -1.0, 9.0]) if mapped else np.array([0.0, 4.0])
  grad, jac = (np.asarray(v) for v in derivatives(point))
  expected = np.zeros_like(point)
  active = point > 0.0
  expected[active] = 0.5 / np.sqrt(point[active])
  np.testing.assert_allclose(grad, expected, rtol=1e-15)
  np.testing.assert_allclose(grad, jac.sum(axis=0), rtol=1e-15)


@pytest.mark.parametrize("mapped", [False, True])
def test_called_masks_keep_sparse_adjoint_activity_before_singular_inputs(mapped: bool) -> None:
  pattern = CanonicalPattern((2, 3), (0, 1), (0, 2))

  @sc.function(sc.arg("u", sc.TensorType(pattern.shape, pattern=pattern)), outputs=sc.arg("r"), name="sd_sparse_twice")
  def stage(u: sc.Expr) -> sc.Expr:
    return 2.0 * u

  size = 4 if mapped else 2

  @sc.function(sc.arg("x", size), outputs=sc.group(sc.arg("g"), sc.arg("j")), name="sd_sparse_input_mask")
  def derivatives(x: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
    roots = x.sqrt()
    sparse = sc.pack(roots.reshape((2, 2)), pattern) if mapped else sc.pack(roots, pattern)
    chosen = sc.where(x > 0.0, sc.values(sc.vmap(stage, 2)(sparse) if mapped else stage(sparse)).vec(), 0.0)
    return sc.gradient(chosen.sum(), x), sc.jacobian(chosen, x)

  point = np.array([0.0, 4.0, -1.0, 9.0]) if mapped else np.array([0.0, 4.0])
  grad, jac = (np.asarray(v) for v in derivatives(point))
  expected = np.zeros_like(point)
  active = point > 0.0
  expected[active] = 1.0 / np.sqrt(point[active])
  np.testing.assert_allclose(grad, expected, rtol=1e-15)
  np.testing.assert_array_equal(grad, jac.sum(axis=0))


@pytest.mark.parametrize("mapped", [False, True])
def test_called_masks_preserve_broadcast_shapes_in_a_second_reverse_derivative(mapped: bool) -> None:
  @sc.function(sc.arg("u", 1), outputs=sc.arg("r"), name="sd_bcast_sin")
  def stage(u: sc.Expr) -> sc.Expr:
    return u.sin() + np.zeros((2, 2))

  @sc.function(sc.arg("x", 1), outputs=sc.arg("cost"), name="sd_bcast_loss")
  def loss(x: sc.Expr) -> sc.Expr:
    y = sc.vmap(stage, 1)(x.reshape((1, 1))) if mapped else stage(x)
    return sc.where(x > 0.0, y, 0.0).sum()

  second = sc.gradient(sc.gradient(loss))
  np.testing.assert_allclose(second(np.array([1.0])), [-4.0 * np.sin(1.0)], rtol=1e-15)
  np.testing.assert_array_equal(second(np.array([0.0])), [0.0])


def test_a_broadcast_cotangent_keeps_the_output_shape() -> None:
  x, p = sc.sym("x", 3), sc.sym("p")
  (adjoint,) = sc.vjp((x + 1.0,), (x,), (sc.where(p > 0.0, 2.0 * p, np.zeros(3)),))
  assert adjoint.shape == (3,)

  @sc.function(sc.arg("x", 3), sc.arg("p", ()), outputs=sc.arg("g", 3), name="sd_broadcast_cot")
  def derivatives(x: sc.Expr, p: sc.Expr) -> sc.Expr:
    (grad,) = sc.vjp((x + 1.0,), (x,), (sc.where(p > 0.0, 2.0 * p, np.zeros(3)),))
    return grad

  np.testing.assert_array_equal(derivatives(np.ones(3), np.array(1.5)), np.full(3, 3.0))
  np.testing.assert_array_equal(derivatives(np.ones(3), np.array(-1.5)), np.zeros(3))


@pytest.mark.parametrize("form", ["mapped", "called"])
@pytest.mark.parametrize("squared", [False, True])
def test_a_select_inside_a_map_guards_the_arguments_partial(form: str, squared: bool) -> None:
  @sc.function(sc.arg("u", ()), outputs=sc.arg("y"), name="sd_inner_select")
  def selected(u: sc.Expr) -> sc.Expr:
    return sc.where(u > 0.0, u, 0.0)

  @sc.function(sc.arg("x", 4), outputs=sc.group(*(sc.arg(k) for k in ("y", "g", "t", "h", "rr"))), name=f"sd_inner_{form}")
  def derivatives(x: sc.Expr) -> tuple[sc.Expr, ...]:
    roots = x.sqrt()
    y = sc.vmap(selected, 4)(roots) if form == "mapped" else sc.stack([selected(roots[i]) for i in range(4)])
    cost = ((y - 2.0) ** 2).sum() if squared else y.sum()
    grad = sc.gradient(cost, x)
    return cost, grad, sc.jvp(cost, x, sc.const(np.ones(4))), sc.hessian(cost, x), sc.gradient(grad.sum(), x)

  for point in (np.array([0.0, 4.0, -1.0, 9.0]), np.array([-2.0, 4.0, -1.0, 9.0])):
    _, grad, tangent, hess, twice = (np.asarray(v) for v in derivatives(point))
    expected = np.array([0.0, 0.0, 0.0, 1.0 / 3.0]) if squared else np.array([0.0, 0.25, 0.0, 1.0 / 6.0])
    second = np.array([0.0, 1.0 / 8.0, 0.0, 1.0 / 27.0]) if squared else np.array([0.0, -1.0 / 32.0, 0.0, -1.0 / 108.0])
    np.testing.assert_allclose(grad, expected, rtol=1e-14)
    np.testing.assert_allclose(tangent, expected.sum(), rtol=1e-14)
    np.testing.assert_allclose(hess, np.diag(second), rtol=1e-14)
    np.testing.assert_allclose(twice, second, rtol=1e-14)
    if np.all(point != 0):
      np.testing.assert_allclose(grad, finite_difference(lambda v: np.asarray(derivatives(v)[0]), point).reshape(-1), rtol=1e-7, atol=1e-9)
      np.testing.assert_allclose(hess, finite_difference(lambda v: np.asarray(derivatives(v)[1]), point), rtol=1e-7, atol=1e-9)


@pytest.mark.parametrize(
  ("runtime", "nested", "squared", "form"),
  [(True, False, False, "inline"), (False, True, True, "called"), (True, True, False, "mapped")],
)
def test_selected_zero_tangents_guard_downstream_singular_partials(runtime: bool, nested: bool, squared: bool, form: str) -> None:
  seed_values = np.array([[1.0, 1.0, 1.0, 1.0], [0.0, -2.0, 3.0, 0.0]])

  def body(x: sc.Expr) -> sc.Expr:
    chosen = sc.where(x > 0.0, x, 0.0)
    if nested:
      chosen = sc.where(x < 8.0, chosen, 0.0)
    root = chosen.sqrt()
    return (root - 2.0) ** 2 if squared else root

  stage = sc.function(sc.arg("u", 4 if form == "called" else ()), outputs=sc.arg("y"), name="sd_selected_root")(body)

  @sc.function(sc.arg("x", 4), sc.arg("seeds", (2, 4)), outputs=sc.group(*(sc.arg(k) for k in ("y", "t", "g", "h"))), name="sd_selected_tangent")
  def derivatives(x: sc.Expr, seeds: sc.Expr) -> tuple[sc.Expr, ...]:
    y = body(x) if form == "inline" else stage(x) if form == "called" else sc.vmap(stage, 4)(x)
    return (
      y,
      sc.jvp_many(y, x, seeds if runtime else sc.const(seed_values)),
      sc.gradient(y.sum(), x),
      sc.hessian(y.sum(), x),
    )

  point = np.array([-1.0, 4.0, -2.0, 9.0])
  y, tangents, grad, hess = (np.asarray(v) for v in derivatives(point, seed_values))
  expected = np.array([0.0, 0.25, 0.0, 0.0 if nested else 1.0 / 6.0])
  roots = np.array([0.0, 2.0, 0.0, 0.0 if nested else 3.0])
  second = np.array([0.0, -1.0 / 32.0, 0.0, 0.0 if nested else -1.0 / 108.0])
  if squared:
    expected *= 2.0 * (roots - 2.0)
    second = np.array([0.0, 1.0 / 8.0, 0.0, 0.0 if nested else 1.0 / 27.0])
  np.testing.assert_array_equal(y, (roots - 2.0) ** 2 if squared else roots)
  np.testing.assert_allclose(tangents, seed_values * expected, rtol=1e-14)
  np.testing.assert_allclose(grad, expected, rtol=1e-14)
  np.testing.assert_allclose(hess, np.diag(second), rtol=1e-14)
  np.testing.assert_allclose(np.diag(grad), finite_difference(lambda v: np.asarray(derivatives(v, seed_values)[0]), point), rtol=1e-7, atol=1e-9)


@pytest.mark.parametrize("stride", [0, 1])
def test_inner_mapped_masks_accumulate_shared_and_overlapping_inputs(stride: int) -> None:
  @sc.function(sc.arg("u", 2), outputs=sc.arg("y"), name="sd_window_select")
  def selected(u: sc.Expr) -> sc.Expr:
    return sc.where(u > 0.0, u, 0.0)

  size = 2 + 2 * stride
  point = np.array([0.0, 4.0, -1.0, 9.0, 0.0, 16.0])[:size]
  indices = np.array([it * stride + j for it in range(3) for j in range(2)])

  @sc.function(sc.arg("x", size), outputs=sc.group(sc.arg("g"), sc.arg("t"), sc.arg("h")), name="sd_window_inner_masks")
  def derivatives(x: sc.Expr) -> tuple[sc.Expr, sc.Expr, sc.Expr]:
    y = _mapped_call(selected, 3, [(x.sqrt(), 0, stride)]).sum()
    return sc.gradient(y, x), sc.jvp(y, x, sc.const(np.ones(size))), sc.hessian(y, x)

  counts = np.bincount(indices, minlength=size)
  expected = np.zeros(size)
  second = np.zeros(size)
  active = point > 0.0
  expected[active] = counts[active] * 0.5 / np.sqrt(point[active])
  second[active] = -counts[active] * 0.25 / point[active] ** 1.5
  grad, tangent, hess = (np.asarray(v) for v in derivatives(point))
  np.testing.assert_allclose(grad, expected, rtol=1e-14)
  np.testing.assert_allclose(tangent, expected.sum(), rtol=1e-14)
  np.testing.assert_allclose(hess, np.diag(second), rtol=1e-14)


@pytest.mark.parametrize(
  "operation",
  ["mul", "div", "cast", "sum", "other_branch", "merged", "nested_active", "broadcast", "selected_broadcast", "stack"],
)
def test_forward_mask_survives_operations_before_a_singular_partial(operation: str) -> None:
  @sc.function(sc.arg("x", 2), outputs=sc.group(sc.arg("t"), sc.arg("g")), name=f"sd_forward_mask_{operation}")
  def derivatives(x: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
    selected = sc.where(x > 0.0, x, 0.0)
    if operation == "mul":
      y = (selected * 2.0).sqrt().sum()
    elif operation == "div":
      y = (selected / 2.0).sqrt().sum()
    elif operation == "cast":
      narrow = sc.cast(selected, "float32")
      assert sc.jvp(narrow, x, sc.const(np.ones(2))).type.dtype == sc.dtypes.float32
      y = sc.cast(narrow, "float64").sqrt().sum()
    elif operation == "sum":
      y = selected.reshape((2, 1)).sum(axis=1).sqrt().sum()
    elif operation == "other_branch":
      y = sc.where(x <= 0.0, 0.0, x).sqrt().sum()
    elif operation == "nested_active":
      y = sc.where(x > 0.0, sc.where(x > 1.0, x, 0.0), sc.where(x < -1.0, -x, 0.0)).sqrt().sum()
    elif operation == "broadcast":
      y = (selected + np.zeros((2, 2))).sqrt().sum() / 2.0
    elif operation == "selected_broadcast":
      y = sc.where(x > 0.0, x[1], 0.0).sqrt().sum()
    elif operation == "stack":
      y = sc.stack([selected, selected]).sqrt().sum() / 2.0
    else:
      y = (selected + sc.where(x > 1.0, x, 0.0)).sqrt().sum()
    return sc.jvp(y, x, sc.const(np.ones(2))), sc.gradient(y, x)

  tangent, grad = (np.asarray(v) for v in derivatives(np.array([-1.0, 4.0])))
  factor = np.sqrt(2.0) if operation in {"mul", "merged"} else 1.0 / np.sqrt(2.0) if operation == "div" else 1.0
  np.testing.assert_allclose(grad, [0.0, 0.25 * factor], rtol=1e-7)
  np.testing.assert_allclose(tangent, grad.sum(), rtol=1e-7)


def test_inner_mapped_masks_keep_each_inputs_activity_separate() -> None:
  @sc.function(sc.arg("u", 2), sc.arg("v", 2), outputs=sc.arg("y"), name="sd_two_input_select")
  def selected(u: sc.Expr, v: sc.Expr) -> sc.Expr:
    return sc.where(u > 0.0, u, 2.0 * v)

  @sc.function(sc.arg("x", 8), outputs=sc.group(sc.arg("g"), sc.arg("t"), sc.arg("h")), name="sd_two_input_masks")
  def derivatives(x: sc.Expr) -> tuple[sc.Expr, sc.Expr, sc.Expr]:
    roots = x.sqrt()
    y = _mapped_call(selected, 2, [(roots[:4], 0, 2), (roots[4:], 0, 2)]).sum()
    return sc.gradient(y, x), sc.jvp(y, x, sc.const(np.ones(8))), sc.hessian(y, x)

  grad, tangent, hess = (np.asarray(v) for v in derivatives(np.array([0.0, 4.0, -1.0, 9.0, 4.0, 0.0, 9.0, -1.0])))
  expected = [0.0, 0.25, 0.0, 1.0 / 6.0, 0.5, 0.0, 1.0 / 3.0, 0.0]
  np.testing.assert_allclose(grad, expected, rtol=1e-14)
  np.testing.assert_allclose(tangent, sum(expected), rtol=1e-14)
  np.testing.assert_allclose(hess, np.diag([0.0, -1.0 / 32.0, 0.0, -1.0 / 108.0, -1.0 / 16.0, 0.0, -1.0 / 54.0, 0.0]), rtol=1e-14)


def test_inner_mapped_masks_keep_mixed_formal_dtypes() -> None:
  @sc.function(sc.arg("u", sc.TensorType((), dtype=sc.dtypes.float32)), sc.arg("v", ()), outputs=sc.arg("y"), name="sd_mixed_formal_select")
  def stage(u: sc.Expr, v: sc.Expr) -> sc.Expr:
    return sc.where(u > 1.0, sc.cast(u, "float64"), 2.0 * v)

  @sc.function(sc.arg("x", 4), outputs=sc.group(sc.arg("y"), sc.arg("g"), sc.arg("h")), name="sd_mixed_formal_masks")
  def derivatives(x: sc.Expr) -> tuple[sc.Expr, sc.Expr, sc.Expr]:
    y = sc.vmap(stage, 2)(sc.cast(x[:2].sqrt(), "float32"), x[2:].sqrt()).sum()
    return y, sc.gradient(y, x), sc.hessian(y, x)

  point = np.array([0.0, 4.0, 4.0, 0.0])
  value, grad, hess = (np.asarray(v) for v in derivatives(point))
  np.testing.assert_array_equal(value, 6.0)
  np.testing.assert_array_equal(grad, [0.0, 0.25, 0.5, 0.0])
  np.testing.assert_array_equal(hess, np.diag([0.0, -1.0 / 32.0, -1.0 / 16.0, 0.0]))


def test_inner_mapped_masks_use_stored_input_coordinates() -> None:
  pattern = CanonicalPattern((2, 3), (0, 1), (0, 2))

  @sc.function(sc.arg("u", sc.TensorType(pattern.shape, pattern=pattern)), outputs=sc.arg("y"), name="sd_stored_select")
  def selected(u: sc.Expr) -> sc.Expr:
    compact = sc.values(u)
    return sc.pack(sc.where(compact > 0.0, compact, 0.0), pattern)

  @sc.function(sc.arg("x", 4), outputs=sc.group(sc.arg("g"), sc.arg("t"), sc.arg("h")), name="sd_inner_stored_mask")
  def derivatives(x: sc.Expr) -> tuple[sc.Expr, sc.Expr, sc.Expr]:
    roots = sc.pack(x.sqrt().reshape((2, 2)), pattern)
    y = sc.values(sc.vmap(selected, 2)(roots)).sum()
    return sc.gradient(y, x), sc.jvp(y, x, sc.const(np.ones(4))), sc.hessian(y, x)

  grad, tangent, hess = (np.asarray(v) for v in derivatives(np.array([0.0, 4.0, -1.0, 9.0])))
  np.testing.assert_allclose(grad, [0.0, 0.25, 0.0, 1.0 / 6.0], rtol=1e-14)
  np.testing.assert_allclose(tangent, grad.sum(), rtol=1e-14)
  np.testing.assert_allclose(hess, np.diag([0.0, -1.0 / 32.0, 0.0, -1.0 / 108.0]), rtol=1e-14)


def test_reverse_mode_refuses_an_integer_or_bool_output() -> None:
  x = sc.sym("x", 3)
  for value in (sc.cast(x, "int64")[0], (x > 0.0)[0]):
    with pytest.raises(TypeError, match="no tangent"):
      sc.gradient(value, x)


@pytest.mark.parametrize("runtime", [False, True])
def test_masked_quotient_retains_stable_contraction(runtime: bool) -> None:
  @sc.function(sc.arg("x", ()), sc.arg("seed", ()), outputs=sc.arg("t"), name=f"sd_div_{runtime}")
  def f(x: sc.Expr, seed: sc.Expr) -> sc.Expr:
    a = sc.where(x > 0, x, 0)
    b = sc.where(x > 0, 2 * x, 0)
    return sc.jvp(a / b, x, seed if runtime else sc.const(1.0))

  np.testing.assert_array_equal(f(np.array(1e-310), np.array(1.0)), 0.0)


@pytest.mark.parametrize("operation", ["sum", "transpose", "stack"])
def test_sparse_selected_tangent_survives_movement(operation: str) -> None:
  p = CanonicalPattern((2, 2), (0, 1), (0, 1))

  @sc.function(sc.arg("x", 2), outputs=sc.group(sc.arg("t"), sc.arg("g")), name=f"sd_sparse_{operation}")
  def f(x: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
    sp = sc.pack(x, p)
    sel = sc.where(sc.pack(x > 0, p), sp, sc.pack(sc.const(np.zeros(2)), p))
    moved = sel.sum() if operation == "sum" else sel.T if operation == "transpose" else sc.stack([sel, sel])
    y = moved.sqrt().sum()
    return sc.jvp(y, x, sc.const(np.ones(2))), sc.gradient(y, x)

  for point in (np.array([-1.0, -2.0]), np.array([-1.0, 4.0])):
    t, g = (np.asarray(v) for v in f(point))
    factor = 2.0 if operation == "stack" else 1.0
    expected = [0.0, factor * 0.25 if point[1] > 0 else 0.0]
    np.testing.assert_allclose(g, expected, rtol=1e-14)
    np.testing.assert_allclose(t, sum(expected), rtol=1e-14)


def test_nested_mapped_adjoint_preserves_full_branch_activity() -> None:
  @sc.function(sc.arg("u", ()), outputs=sc.arg("y"), name="sd_nested_stage")
  def stage(u: sc.Expr) -> sc.Expr:
    return sc.where(u > 0, u, sc.where(u < -2, 2 * u, 0))

  @sc.function(sc.arg("x", 2), outputs=sc.group(sc.arg("value"), sc.arg("t"), sc.arg("g")), name="sd_nested_map")
  def f(x: sc.Expr) -> tuple[sc.Expr, sc.Expr, sc.Expr]:
    y = sc.vmap(stage, 2)(x.sqrt()).sum()
    return y, sc.jvp(y, x, sc.const(np.ones(2))), sc.gradient(y, x)

  point = np.array([-1.0, 4.0])
  value, t, g = (np.asarray(v) for v in f(point))
  assert value == 2.0
  assert t == 0.25
  step = 1e-5
  finite = np.array([(f(point + np.eye(2)[i] * step)[0] - f(point - np.eye(2)[i] * step)[0]) / (2 * step) for i in range(2)])
  np.testing.assert_allclose(finite, [0.0, 0.25], rtol=1e-7)
  np.testing.assert_allclose(g, [0.0, 0.25])


def test_zero_adjoint_term_does_not_erase_map_condition() -> None:
  @sc.function(sc.arg("u", ()), outputs=sc.arg("y"), name="sd_zero_stage")
  def stage(u: sc.Expr) -> sc.Expr:
    return sc.where(u > 0, u, 0) + 0 * u

  @sc.function(sc.arg("x", 2), outputs=sc.group(sc.arg("t"), sc.arg("g")), name="sd_zero_map")
  def f(x: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
    y = sc.vmap(stage, 2)(x.sqrt()).sum()
    return sc.jvp(y, x, sc.const(np.ones(2))), sc.gradient(y, x)

  t, g = (np.asarray(v) for v in f(np.array([0.0, 4.0])))
  assert t == 0.25
  np.testing.assert_allclose(g, [0.0, 0.25])


@pytest.mark.parametrize("form", ["call", "map", "two_maps"])
def test_zero_adjoints_of_other_results_keep_the_shared_arguments_mask(form: str) -> None:
  @sc.function(
    sc.arg("u", (2,) if form == "call" else ()),
    sc.arg("w", (2,) if form == "call" else ()),
    outputs=sc.group(sc.arg("a"), sc.arg("b")),
    name="sd_independent_results",
  )
  def two(u: sc.Expr, w: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
    return sc.where(u > 0.0, u, 0.0), w * w

  @sc.function(sc.arg("x", 2), outputs=sc.group(*(sc.arg(k) for k in ("y", "g", "j", "h", "inline"))), name=f"sd_zero_adjoint_{form}")
  def derivatives(x: sc.Expr) -> tuple[sc.Expr, ...]:
    roots = x.sqrt()
    if form == "call":
      a, b = two(roots, x + 1.0)
    else:
      a, b = sc.vmap(two, 2)(roots, x + 1.0)
      if form == "two_maps":
        a, _ = sc.vmap(two, 2)(roots, x + 2.0)
    y = a.sum() + b.sum()
    inline = sc.where(roots > 0.0, roots, 0.0).sum() + sc.sumsqr(x + 1.0)
    return y, sc.gradient(y, x), sc.jacobian(y, x), sc.hessian(y, x), sc.gradient(inline, x)

  for point in (np.array([0.0, 4.0]), np.array([-1.0, 4.0])):
    _, grad, jac, hess, inline = (np.asarray(v) for v in derivatives(point))
    expected = 2.0 * (point + 1.0) + [0.0, 0.25]
    np.testing.assert_array_equal(grad, expected)
    np.testing.assert_array_equal(grad, jac.reshape(-1))
    np.testing.assert_array_equal(grad, inline)
    np.testing.assert_array_equal(hess, np.diag([2.0, 2.0 - 1.0 / 32.0]))
  numerical = finite_difference(lambda v: np.asarray(derivatives(v)[0]), point).reshape(-1)
  np.testing.assert_allclose(grad, numerical, rtol=1e-7, atol=1e-9)


def test_selected_tangent_survives_constant_matmul() -> None:
  matrix = np.array([[1.0, 2.0], [3.0, 4.0]])

  @sc.function(sc.arg("x", 2), outputs=sc.group(sc.arg("t"), sc.arg("g")), name="sd_masked_matmul")
  def f(x: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
    sel = sc.where(x > 0, x, 0)
    y = (sel @ sc.const(matrix)).sqrt().sum()
    return sc.jvp(y, x, sc.const(np.ones(2))), sc.gradient(y, x)

  for point in (np.array([-1.0, -2.0]), np.array([-1.0, 4.0])):
    t, g = (np.asarray(v) for v in f(point))
    value = np.where(point > 0, point, 0.0) @ matrix
    slope = np.zeros(2)
    slope[value > 0] = 0.5 / np.sqrt(value[value > 0])
    expected = (matrix @ slope) * (point > 0)
    np.testing.assert_allclose(g, expected, rtol=1e-14)
    np.testing.assert_allclose(t, expected.sum(), rtol=1e-14)


@pytest.mark.parametrize("wrapped", [False, True])
def test_forward_selected_activity_uses_constant_matrix_support(wrapped: bool) -> None:
  matrix = np.array([[0.0, 1.0], [0.0, 0.0]])

  @sc.function(sc.arg("x", 2), outputs=sc.group(sc.arg("y"), sc.arg("t")), name="sd_matrix_support")
  def f(x: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
    coefficient = sc.const(matrix) * 2.0 / 2.0 if wrapped else sc.const(matrix)
    y = (sc.where(x > 0, x, 0) @ coefficient).sqrt().sum()
    return y, sc.jvp(y, x, sc.const(np.ones(2)))

  for point in (np.array([-1.0, -2.0]), np.array([-1.0, 4.0])):
    value, tangent = (np.asarray(v) for v in f(point))
    expected = np.sqrt(np.where(point > 0, point, 0.0) @ matrix).sum()
    np.testing.assert_array_equal(value, expected)
    difference = finite_difference(lambda v: np.asarray(f(v)[0]), point).sum()
    np.testing.assert_allclose(tangent, difference, rtol=1e-7, atol=1e-9)


def test_selected_zero_from_maximum_guards_the_square_root_partial() -> None:
  seeds = np.array([[1.0, 1.0], [0.0, -2.0]])

  @sc.function(sc.arg("x", 2), sc.arg("seeds", (2, 2)), outputs=sc.group(sc.arg("y"), sc.arg("t"), sc.arg("g"), sc.arg("h")), name="sd_maximum_root")
  def f(x: sc.Expr, seed: sc.Expr) -> tuple[sc.Expr, sc.Expr, sc.Expr, sc.Expr]:
    y = sc.maximum(x, 0.0).sqrt()
    return y, sc.jvp_many(y, x, seed), sc.gradient(y.sum(), x), sc.hessian(y.sum(), x)

  point = np.array([-1.0, 4.0])
  value, tangent, grad, hess = (np.asarray(v) for v in f(point, seeds))
  expected = np.array([0.0, 0.25])
  np.testing.assert_array_equal(value, [0.0, 2.0])
  np.testing.assert_array_equal(tangent, seeds * expected)
  np.testing.assert_array_equal(grad, expected)
  np.testing.assert_array_equal(hess, np.diag([0.0, -1.0 / 32.0]))
  np.testing.assert_allclose(np.diag(grad), finite_difference(lambda v: np.asarray(f(v, seeds)[0]), point), rtol=1e-7, atol=1e-9)


def test_second_forward_derivative_uses_the_select_condition_at_a_zero_tangent() -> None:
  @sc.function(sc.arg("x", ()), outputs=sc.group(sc.arg("t"), sc.arg("second")), name="sd_second_forward")
  def f(x: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
    y = (sc.where(x > 0, x, 0).sqrt() - 2.0) ** 2
    first = sc.jvp(y, x, sc.const(1.0))
    return first, sc.jvp(first, x, sc.const(1.0))

  np.testing.assert_array_equal(np.asarray(f(np.array(-1.0))), [0.0, 0.0])
  np.testing.assert_array_equal(np.asarray(f(np.array(4.0))), [0.0, 1.0 / 8.0])


@pytest.mark.parametrize("dense", [False, True])
def test_sparse_false_branch_mask_covers_entries_absent_from_the_condition(dense: bool) -> None:
  full = CanonicalPattern((2, 2), (0, 1), (0, 1))
  small = CanonicalPattern((2, 2), (0,), (0,))

  @sc.function(sc.arg("x", 2), outputs=sc.arg("g"), name="sd_sparse_false_seed")
  def f(x: sc.Expr) -> sc.Expr:
    v = sc.pack(x, full)
    condition = sc.pack(x[:1] > 0, small)
    other = sc.pack(sc.const(np.ones(2)), full)
    cot = sc.where(condition, sc.pack(sc.const(np.zeros(1)), small), sc.to_dense(other) if dense else other)
    return sc.vjp((v.sin(),), (x,), (cot,))[0]

  for point in (np.array([-1.0, 4.0]), np.array([1.0, 4.0])):
    expected = np.cos(point)
    expected[0] *= point[0] <= 0
    np.testing.assert_allclose(np.asarray(f(point)), expected, rtol=1e-14)


def test_selected_dot_product_keeps_activity_on_either_operand() -> None:
  @sc.function(sc.arg("x", 2), outputs=sc.group(sc.arg("t"), sc.arg("g")), name="sd_selected_dot")
  def f(x: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
    selected = sc.where(x > 0, x, 0.0)
    coefficient = sc.const(np.ones(2))
    y = (coefficient @ selected).sqrt()
    return sc.jvp(y, x, sc.const(np.ones(2))), sc.gradient(y, x)

  for point in (np.array([-1.0, -2.0]), np.array([-1.0, 4.0])):
    tangent, grad = (np.asarray(v) for v in f(point))
    expected = np.array([0.0, 0.25 if point[1] > 0 else 0.0])
    np.testing.assert_array_equal(tangent, expected.sum())
    np.testing.assert_array_equal(grad, expected)


@pytest.mark.parametrize(("form", "extremum"), [("sparse_vector", False), ("matrix_sparse", True), ("sparse_sparse", False)])
def test_nested_sparse_tangent_conditions_survive_products(form: str, extremum: bool) -> None:
  pattern = CanonicalPattern((4, 4), (0, 1, 2, 3), (0, 1, 2, 3))
  scale = 2 if form == "matrix_sparse" else 1

  def body(x: sc.Expr) -> sc.Expr:
    inner = sc.maximum(x, 0.0) if extremum else sc.where(x > 0, x, 0.0)
    chosen = sc.pack(sc.where(x < 5, inner, 0.0), pattern)
    if form == "sparse_vector":
      result = chosen @ sc.const(np.ones(4))
    elif form == "matrix_sparse":
      result = sc.const(np.ones((2, 4))) @ chosen
    else:
      result = chosen @ sc.pack(sc.const(np.ones(4)), pattern)
    return result.sqrt().sum()

  primal = sc.function(sc.arg("x", 4), outputs=sc.arg("y"), name="sd_nested_sparse_product")(body)

  @sc.function(
    sc.arg("x", 4), sc.arg("seeds", (4, 4)), outputs=sc.group(sc.arg("t"), sc.arg("runtime"), sc.arg("baked")), name="sd_nested_sparse_products"
  )
  def products(x: sc.Expr, seeds: sc.Expr) -> tuple[sc.Expr, sc.Expr, sc.Expr]:
    y = body(x)
    return sc.jvp(y, x, sc.const(np.ones(4))), sc.jvp_many(y, x, seeds), sc.jvp_many(y, x, sc.const(np.eye(4)))

  for point in (np.array([1.0, 2.0, 3.0, 4.0]), np.array([-1.0, 2.0, 6.0, 4.0])):
    active = (point > 0) & (point < 5)
    expected = np.zeros(4)
    expected[active] = scale * 0.5 / np.sqrt(point[active])
    tangent, runtime, baked = (np.asarray(v) for v in products(point, np.eye(4)))
    np.testing.assert_allclose(tangent, expected.sum(), rtol=1e-14)
    np.testing.assert_allclose(runtime, expected, rtol=1e-14)
    np.testing.assert_allclose(baked, expected, rtol=1e-14)
    np.testing.assert_allclose(finite_difference(primal, point).reshape(-1), expected, rtol=1e-7, atol=1e-9)
  point = np.array([1.0, 2.0, 3.0, 4.0])
  np.testing.assert_allclose(np.asarray(sc.hessian(primal)(point)), np.diag(-0.25 * scale / point**1.5), rtol=1e-14)


def test_shared_outer_mask_keeps_the_union_of_inner_mapped_conditions() -> None:
  @sc.function(sc.arg("u", ()), outputs=sc.arg("y"), name="sd_shared_outer")
  def stage(u: sc.Expr) -> sc.Expr:
    return sc.where(u > -2, sc.where(u > 1, u, 0.0), 0.0) + sc.where(u > -2, sc.where(u < -1, 2 * u, 0.0), 0.0)

  @sc.function(sc.arg("x", 2), outputs=sc.group(sc.arg("g"), sc.arg("h")), name="sd_shared_outer_map")
  def f(x: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
    y = sc.vmap(stage, 2)(x.sqrt()).sum()
    return sc.gradient(y, x), sc.hessian(y, x)

  grad, hess = (np.asarray(v) for v in f(np.array([0.0, 4.0])))
  np.testing.assert_array_equal(grad, [0.0, 0.25])
  np.testing.assert_array_equal(hess, np.diag([0.0, -1.0 / 32.0]))


@pytest.mark.parametrize(("mapped", "nested"), [(False, False), (True, False), (True, True)])
def test_forward_calls_carry_selected_activity_in_both_directions(mapped: bool, nested: bool) -> None:
  shape = () if mapped else (2,)

  @sc.function(sc.arg("u", shape), outputs=sc.arg("y"), name="sd_boundary_root")
  def root(u: sc.Expr) -> sc.Expr:
    return u.sqrt()

  @sc.function(sc.arg("u", shape), outputs=sc.arg("y"), name="sd_boundary_selected")
  def selected(u: sc.Expr) -> sc.Expr:
    return sc.where(u > 0.0, u, 0.0)

  @sc.function(sc.arg("u", shape), outputs=sc.arg("y"), name="sd_boundary_nested_root")
  def nested_root(u: sc.Expr) -> sc.Expr:
    return root(u)

  @sc.function(sc.arg("u", shape), outputs=sc.arg("y"), name="sd_boundary_nested_selected")
  def nested_selected(u: sc.Expr) -> sc.Expr:
    return selected(u)

  @sc.function(sc.arg("u", shape), outputs=sc.arg("y"), name="sd_boundary_stationary")
  def stationary(u: sc.Expr) -> sc.Expr:
    return (root(u) - 2.0) ** 2

  seeds = np.array([[1.0, 1.0], [0.0, -2.0]])

  @sc.function(
    sc.arg("x", 2),
    sc.arg("seeds", (2, 2)),
    outputs=sc.group(*(sc.arg(k) for k in ("y", "baked", "runtime", "g", "h", "second", "plain", "stationary_second"))),
    name="sd_boundaries",
  )
  def derivatives(x: sc.Expr, seed: sc.Expr) -> tuple[sc.Expr, ...]:
    root_fn = nested_root if nested else root
    selected_fn = nested_selected if nested else selected
    chosen = sc.where(x > 0.0, x, 0.0)
    into = sc.vmap(root_fn, 2)(chosen) if mapped else root_fn(chosen)
    outof = (sc.vmap(selected_fn, 2)(x) if mapped else selected_fn(x)).sqrt()
    before = sc.vmap(selected_fn, 2)(x.sqrt()) if mapped else selected_fn(x.sqrt())
    plain = sc.vmap(root_fn, 2)(x * x + 1.0) if mapped else root_fn(x * x + 1.0)
    ys = [into, outof, before]
    losses = [((y - 2.0) ** 2).sum() for y in ys]
    first = [sc.jvp(loss, x, sc.const(np.ones(2))) for loss in losses]
    stationary_loss = (sc.vmap(stationary, 2)(chosen) if mapped else stationary(chosen)).sum()
    stationary_first = sc.jvp(stationary_loss, x, sc.const(np.ones(2)))
    return (
      sc.stack(ys),
      sc.stack([sc.jvp_many(y, x, sc.const(seeds)) for y in ys]),
      sc.stack([sc.jvp_many(y, x, seed) for y in ys]),
      sc.stack([sc.gradient(loss, x) for loss in losses]),
      sc.stack([sc.hessian(loss, x) for loss in losses]),
      sc.stack([sc.jvp(t, x, sc.const(np.ones(2))) for t in first]),
      sc.jvp_many(plain, x, seed),
      sc.jvp(stationary_first, x, sc.const(np.ones(2))),
    )

  derivatives.instantiate()
  called_root = (nested_root if nested else root).instantiate()
  assert any(
    any(inp.type.dtype.is_bool for inp in helper[0].inputs) and any(out.type.dtype.is_bool for out in helper[0].outputs)
    for key, helper in called_root._memo.helpers.items()
    if key.role == "forward" and key.masked and isinstance(helper, tuple)
  )
  for point in (np.array([-1.0, 4.0]), np.array([-2.0, 9.0])):
    value, baked, runtime, grad, hess, second, plain, stationary_second = (np.asarray(v) for v in derivatives(point, seeds))
    slope = np.array([0.0, 0.5 / np.sqrt(point[1])])
    expected_grad = np.array([0.0, 1.0 - 2.0 / np.sqrt(point[1])])
    curvature = 1.0 / point[1] ** 1.5
    np.testing.assert_allclose(value, np.tile([0.0, np.sqrt(point[1])], (3, 1)))
    np.testing.assert_allclose(baked, np.broadcast_to(seeds * slope, (3, 2, 2)), rtol=1e-14)
    np.testing.assert_allclose(runtime, baked, rtol=1e-14)
    np.testing.assert_allclose(grad, np.tile(expected_grad, (3, 1)), rtol=1e-14)
    np.testing.assert_allclose(hess, np.broadcast_to(np.diag([0.0, curvature]), (3, 2, 2)), rtol=1e-14)
    np.testing.assert_allclose(second, curvature, rtol=1e-14)
    np.testing.assert_allclose(stationary_second, curvature, rtol=1e-14)
    np.testing.assert_allclose(plain, seeds * (point / np.sqrt(point * point + 1.0)), rtol=1e-14)
    expected_jacobian = np.tile(np.diag(slope), (3, 1))
    np.testing.assert_allclose(finite_difference(lambda v: np.asarray(derivatives(v, seeds)[0]), point), expected_jacobian, rtol=1e-7, atol=1e-9)


@pytest.mark.parametrize("kind", ["jacobian", "hessian"])
def test_selected_map_seeds_keep_local_coloring_code_size(kind: str) -> None:
  from scaly.codegen import render_c_source

  @sc.function(sc.arg("x", 4), sc.arg("u", 2), outputs=sc.arg("y"), name="sd_colored_stage")
  def stage(x: sc.Expr, u: sc.Expr) -> sc.Expr:
    velocity = x[2:]
    return sc.concat([x[:2] + 0.05 * velocity, velocity + 0.05 * (u - 0.1 * sc.sumsqr(velocity) * velocity)])

  length = 40
  sources = []
  for clipped in (False, True):

    @sc.function(sc.arg("z", 6 * length), outputs=sc.arg("d"), name=f"sd_colored_{kind}_{clipped}")
    def derivatives(z: sc.Expr) -> sc.Expr:
      x, u = z[: 4 * length].reshape((length, 4)), z[4 * length :].reshape((length, 2))
      u = sc.maximum(sc.minimum(u, 1.0), -1.0) if clipped else u
      y = sc.vmap(stage, length)(x, u)
      return sc.jacobian(y, z) if kind == "jacobian" else sc.hessian(sc.sumsqr(y), z)

    sources.append(render_c_source(derivatives))
  assert len(sources[1]) < 8 * len(sources[0]), (len(sources[0]), len(sources[1]))


def test_forward_map_keeps_activity_when_another_formal_has_zero_partial() -> None:
  @sc.function(sc.arg("a", ()), sc.arg("b", ()), outputs=sc.arg("y"), name="sd_forward_zero_formal")
  def stage(a: sc.Expr, b: sc.Expr) -> sc.Expr:
    return a + 0.0 * b

  @sc.function(
    sc.arg("x", 2), sc.arg("seed", 2), outputs=sc.group(sc.arg("baked"), sc.arg("runtime"), sc.arg("g")), name="sd_forward_zero_formal_map"
  )
  def f(x: sc.Expr, seed: sc.Expr) -> tuple[sc.Expr, sc.Expr, sc.Expr]:
    y = sc.vmap(stage, 2)(sc.where(x > 0.0, x, 0.0), x).sqrt().sum()
    return sc.jvp(y, x, sc.const(np.ones(2))), sc.jvp(y, x, seed), sc.gradient(y, x)

  baked, runtime, grad = (np.asarray(v) for v in f(np.array([-1.0, 4.0]), np.ones(2)))
  np.testing.assert_array_equal([baked, runtime], [0.25, 0.25])
  np.testing.assert_array_equal(grad, [0.0, 0.25])


def test_forward_call_conditions_use_sparse_stored_coordinates() -> None:
  pattern = CanonicalPattern((2, 2), (0, 1), (0, 1))
  type_ = sc.TensorType((2, 2), pattern=pattern)

  @sc.function(sc.arg("u", type_), outputs=sc.arg("y"), name="sd_sparse_call_root")
  def root(u: sc.Expr) -> sc.Expr:
    return u.sqrt()

  @sc.function(sc.arg("u", type_), outputs=sc.arg("y"), name="sd_sparse_call_selected")
  def selected(u: sc.Expr) -> sc.Expr:
    condition = sc.pack(sc.values(u) > 0.0, pattern)
    return sc.where(condition, u, sc.pack(sc.const(np.zeros(2)), pattern))

  seeds = np.array([[1.0, 1.0], [0.0, -2.0]])

  @sc.function(
    sc.arg("x", 2), sc.arg("seeds", (2, 2)), outputs=sc.group(sc.arg("y"), sc.arg("baked"), sc.arg("runtime")), name="sd_sparse_call_masks"
  )
  def f(x: sc.Expr, seed: sc.Expr) -> tuple[sc.Expr, sc.Expr, sc.Expr]:
    ys = [sc.values(root(sc.pack(sc.where(x > 0.0, x, 0.0), pattern))), sc.values(selected(sc.pack(x, pattern)).sqrt())]
    return sc.stack(ys), sc.stack([sc.jvp_many(y, x, sc.const(seeds)) for y in ys]), sc.stack([sc.jvp_many(y, x, seed) for y in ys])

  value, baked, runtime = (np.asarray(v) for v in f(np.array([-1.0, 4.0]), seeds))
  np.testing.assert_array_equal(value, [[0.0, 2.0], [0.0, 2.0]])
  expected = np.broadcast_to(seeds * [0.0, 0.25], (2, 2, 2))
  np.testing.assert_array_equal(baked, expected)
  np.testing.assert_array_equal(runtime, expected)
