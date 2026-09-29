"""Derivatives of ``atan2``, ``copysign``, ``maximum``/``minimum``, ``where``, the predicates and ``cast`` at their edges.

Forward, reverse, multi-seed forward and second derivatives against closed forms and finite differences: ``atan2``
around the circle, on its branch cut and at the origin; ``copysign`` by the sign bit of a zero or a NaN; the tie
conventions of ``sc.options(nonsmooth=...)`` and when they are read; a NaN or infinite derivative in the branch
``where`` does not choose; predicates and casts; and derivatives through float32. Values are in
``tests/core/codegen/test_elementwise_edge_cases.py``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

import scaly as sc
from scaly.ad import finite_difference


def _fn(name: str, inputs: list[sc.Expr], outputs: list[sc.Expr]) -> sc.Function:
  return sc.Function.from_exprs(name, inputs, outputs, [str(e.name) for e in inputs], [f"out{i}" for i in range(len(outputs))])


# (y, x) around the circle: every quadrant, both axes with either sign of zero (y = ±0 with x < 0 is the
# branch cut), and radii far from one.
ATAN2_Y = np.array([0.3, 2.0, -1.5, -0.2, 0.0, -0.0, 0.0, -0.0, 1.0, -1.0, 3e-3, -4e2])
ATAN2_X = np.array([0.4, -1.0, -0.5, 0.9, 1.0, 1.0, -2.0, -2.0, 0.0, -0.0, 5e-3, 2e2])
ON_CUT = (ATAN2_Y == 0) & (ATAN2_X < 0)


def test_atan2_derivatives_match_the_closed_form_around_the_circle() -> None:
  n = ATAN2_Y.size
  v = sc.sym("v", 2 * n)
  y, x = v[:n], v[n:]
  angle = sc.atan2(y, x)
  (rev,) = sc.vjp((angle,), (v,), (sc.const(np.ones(n)),))
  outs = [angle, sc.jvp(angle, v, sc.const(np.r_[np.ones(n), np.zeros(n)])), sc.jvp(angle, v, sc.const(np.r_[np.zeros(n), np.ones(n)])), rev]
  outs += [sc.jacobian(angle, v), sc.hessian(angle.sum(), v)]
  f = _fn("eed_atan2", [v], outs)
  point = np.r_[ATAN2_Y, ATAN2_X]
  value, d_y, d_x, rev_v, jac, hess = f(point)
  yv, xv = ATAN2_Y, ATAN2_X
  r2 = xv * xv + yv * yv
  np.testing.assert_array_max_ulp(value, np.arctan2(yv, xv), maxulp=1)
  np.testing.assert_allclose(d_y, xv / r2, rtol=1e-14)
  np.testing.assert_allclose(d_x, -yv / r2, rtol=1e-14)
  np.testing.assert_allclose(rev_v, np.r_[xv / r2, -yv / r2], rtol=1e-14)
  np.testing.assert_allclose(jac, np.hstack([np.diag(xv / r2), np.diag(-yv / r2)]), rtol=1e-14)
  expected_hess = np.block(
    [[np.diag(-2 * xv * yv / r2**2), np.diag((yv * yv - xv * xv) / r2**2)], [np.diag((yv * yv - xv * xv) / r2**2), np.diag(2 * xv * yv / r2**2)]]
  )
  np.testing.assert_allclose(hess, expected_hess, rtol=1e-12, atol=0.0)
  # On the branch cut the derivative is the one-sided limit: the angle jumps by 2 pi, its slope does not.
  assert d_y[ON_CUT].tolist() == [-0.5, -0.5]
  off_cut = np.r_[~ON_CUT, ~ON_CUT]
  fd = finite_difference(lambda p: f(p)[0], point, eps=1e-7)
  np.testing.assert_allclose(jac[~ON_CUT][:, off_cut], fd[~ON_CUT][:, off_cut], rtol=1e-6, atol=1e-6)


def test_atan2_derivative_at_the_origin_is_not_finite_in_either_mode() -> None:
  """No convention is documented at the origin; the rule evaluates 0 / 0 there. This pins that the result is
  not silently a number, in forward and reverse mode alike."""
  y, x = sc.sym("y", 4), sc.sym("x", 4)
  angle = sc.atan2(y, x)
  g_y, g_x = sc.vjp((angle.sum(),), (y, x), (sc.const(1.0),))
  f = _fn("eed_atan2_origin", [y, x], [g_y, g_x, sc.jvp(angle, y, sc.const(np.ones(4))), sc.jvp(angle, x, sc.const(np.ones(4)))])
  for out in f((np.array([0.0, -0.0, 0.0, -0.0]), np.array([0.0, 0.0, -0.0, -0.0]))):
    assert not np.isfinite(out).any()


COPYSIGN_X = np.array([-2.0, -0.0, 0.0, 3.0])
COPYSIGN_Y = np.array([-np.inf, -1.0, -5e-324, -0.0, 0.0, 5e-324, 1.0, np.inf, np.nan, -np.nan])


def test_copysign_derivative_follows_the_sign_bits_and_ignores_the_sign_operand() -> None:
  xv, yv = (m.reshape(-1) for m in np.meshgrid(COPYSIGN_X, COPYSIGN_Y, indexing="ij"))
  n = xv.size
  x, y = sc.sym("x", n), sc.sym("y", n)
  out = sc.copysign(x, y)
  g_x, g_y = sc.vjp((out,), (x, y), (sc.const(np.ones(n)),))
  outs = [sc.jvp(out, x, sc.const(np.ones(n))), sc.jvp(out, y, sc.const(np.ones(n))), g_x, g_y, sc.jacobian(out, x), sc.jacobian(out, y)]
  t_x, t_y, r_x, r_y, j_x, j_y = _fn("eed_copysign", [x, y], outs)((xv, yv))
  # The slope is sign(x) * sign(y), read from the sign bits as C's copysign reads them, NaN included.
  slope = np.copysign(1.0, xv) * np.copysign(1.0, yv)
  for got in (t_x, r_x, np.diag(j_x)):
    np.testing.assert_array_equal(got, slope)
  np.testing.assert_array_equal(j_x, np.diag(slope))
  for got in (t_y, r_y, j_y):
    np.testing.assert_array_equal(got, np.zeros_like(got))
  assert r_x[28:30].tolist() == [1.0, -1.0]  # x = +0.0 beside NaN and -NaN: the sign of a NaN counts


# One entry of x per tie, each term summed into one cost: the tie at every entry, its gradient under
# "split" and under "first".
TIE_POINT = np.array([1.0, 0.0, 0.0, -0.0, np.inf, 2.0, 2.0, 1.0, 1.0, 0.0, 1.0])
TIE_GRAD = {
  "split": [0.5, 1.5, 1.5, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 2.0],
  "first": [1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
}


def _tie_cost(x: sc.Expr) -> sc.Expr:
  return sc.stack(
    [
      sc.minimum(x[0], 1.0),  # a constant operand
      sc.maximum(x[1], 2.0 * x[1]),  # both operands move, at different rates
      sc.minimum(2.0 * x[2], x[2]),
      sc.maximum(x[3], -x[3]),  # -0.0 ties with +0.0
      sc.maximum(x[4], np.inf),  # an infinite tie
      sc.minimum(x[5], x[6]),
      sc.maximum(x[7:10], x[10]).sum(),  # a broadcast operand collects its share from every entry
    ]
  ).sum()


@pytest.mark.parametrize("mode", ["split", "first"])
def test_tie_conventions_in_every_mode_and_the_convention_is_fixed_when_the_graph_is_built(mode: str) -> None:
  x = sc.sym("x", TIE_POINT.size)
  seed = sc.sym("seed", TIE_POINT.size)
  with sc.options(nonsmooth=mode):
    cost = _tie_cost(x)
    outs = [sc.gradient(cost, x), sc.jacobian(cost, x), sc.jvp(cost, x, seed), sc.hessian(cost, x)]
  f = _fn(f"eed_ties_{mode}", [x, seed], outs)
  seed_v = np.linspace(-1.0, 2.0, TIE_POINT.size)
  # Compiled under another convention: what was built above must not change.
  with sc.options(nonsmooth="error" if mode == "split" else "split"):
    grad, jac, tangent, hess = f((TIE_POINT, seed_v))
  np.testing.assert_array_equal(grad, TIE_GRAD[mode])
  np.testing.assert_array_equal(jac.reshape(-1), TIE_GRAD[mode])
  np.testing.assert_allclose(tangent, np.dot(TIE_GRAD[mode], seed_v), rtol=1e-15)
  np.testing.assert_array_equal(hess, np.zeros((TIE_POINT.size, TIE_POINT.size)))


def test_error_mode_refuses_every_derivative_of_maximum_and_minimum_but_not_the_values() -> None:
  x, p = sc.sym("x", 3), sc.sym("p", 3)
  with sc.options(nonsmooth="error"):
    low = sc.minimum(x, 0.5)
    fun = _fn("eed_error_primal", [x], [low, sc.maximum(x, x.sin())])
    derivatives: list[Callable[[], object]] = [
      lambda: sc.jvp(low, x, sc.const(np.ones(3))),
      lambda: sc.vjp((low,), (x,), (sc.const(np.ones(3)),)),
      lambda: sc.gradient(low.sum(), x),
      lambda: sc.jacobian(low, x),
      lambda: sc.hessian(sc.maximum(x * x, 1.0).sum(), x),
      lambda: sc.jvp_many(low, x, sc.const(np.eye(3))),
      lambda: sc.jacobian(fun, "out1", "x"),
    ]
    for build in derivatives:
      with pytest.raises(NotImplementedError, match="nonsmooth='error'"):
        build()
    # A nonsmooth term that does not depend on the variable needs no convention.
    smooth = (x * x).sum() + sc.maximum(p, 0.0).sum()
    grads = [sc.gradient(smooth, x), sc.jvp(smooth, x, sc.const(np.ones(3)))]
    values = fun(np.array([0.2, 0.5, 0.9]))
  np.testing.assert_array_equal(values[0], [0.2, 0.5, 0.5])
  np.testing.assert_array_equal(values[1], np.fmax([0.2, 0.5, 0.9], np.sin([0.2, 0.5, 0.9])))
  grad, tangent = _fn("eed_error_smooth", [x, p], grads)((np.array([1.0, -2.0, 0.5]), np.array([-1.0, 0.0, 1.0])))
  np.testing.assert_array_equal(grad, [2.0, -4.0, 1.0])
  assert tangent == -1.0


WHERE_POINT = np.array([0.0, -1.0, -0.0, 4.0])


def _guarded(x: sc.Expr) -> sc.Expr:
  """Each unchosen branch has an infinite or NaN derivative at some entry of ``WHERE_POINT``."""
  return sc.where(x > 0.0, x.sqrt(), 0.0) + sc.where(x > 0.0, x.log(), x) + sc.where(sc.not_equal(x, 0.0), 1.0 / x, 0.0)


WHERE_GRAD = np.array([1.0, 0.0, 1.0, 0.25 + 0.25 - 1.0 / 16.0])


def test_forward_mode_ignores_a_nan_or_infinite_derivative_in_the_unchosen_branch(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  x = sc.sym("x", WHERE_POINT.size)
  y = _guarded(x)
  value, tangent, jac = _fn("eed_where_fwd", [x], [y, sc.jvp(y, x, sc.const(np.ones(WHERE_POINT.size))), sc.jacobian(y, x)])(WHERE_POINT)
  np.testing.assert_array_equal(value, [0.0, -2.0, 0.0, 2.0 + np.log(4.0) + 0.25])
  np.testing.assert_array_equal(tangent, WHERE_GRAD)
  np.testing.assert_array_equal(jac, np.diag(WHERE_GRAD))


def test_reverse_mode_ignores_a_nan_or_infinite_derivative_in_the_unchosen_branch() -> None:
  x = sc.sym("x", WHERE_POINT.size)
  cost = _guarded(x).sum()
  grad, hess = _fn("eed_where_rev", [x], [sc.gradient(cost, x), sc.hessian(cost, x)])(WHERE_POINT)
  np.testing.assert_array_equal(grad, WHERE_GRAD)
  np.testing.assert_array_equal(np.diag(hess), [0.0, -2.0, 0.0, -1.0 / 32.0 - 1.0 / 16.0 + 2.0 / 64.0])


def test_reverse_mode_keeps_a_nested_mask_through_a_chain_of_elementwise_ops() -> None:
  """The unchosen entries stay exactly zero through ``sin(2 sqrt(x))``, through a ``where`` inside a
  ``where`` where only the outer one leaves ``sqrt`` unchosen (at 0 and -1), and in the one ``sqrt`` node
  both read, whose cotangent sums two masked ones: as forward mode gives them."""
  x = sc.sym("x", WHERE_POINT.size)
  chain = sc.where(x > 0.0, (x.sqrt() * 2.0).sin(), 0.0)
  nested = sc.where(x > 0.0, sc.where(x > -2.0, x.sqrt(), 0.0), 0.0)
  cost = (chain + nested + sc.where(x <= 0.0, x * x, (1.0 - x).log())).sum()
  seed = sc.const(np.ones(WHERE_POINT.size))
  grad, tangent, hess = _fn("eed_where_nested", [x], [sc.gradient(cost, x), sc.jvp(cost, x, seed), sc.hessian(cost, x)])(WHERE_POINT)
  root = np.sqrt(4.0)
  expected = np.array([0.0, -2.0, 0.0, np.cos(2.0 * root) / root + 0.5 / root - 1.0 / (1.0 - 4.0)])
  np.testing.assert_allclose(grad, expected, rtol=1e-15)
  assert tangent == grad.sum()
  assert np.all(np.isfinite(hess))


def test_a_masked_cotangent_keeps_a_constant_zero_adjoint_constant() -> None:
  """``copysign``'s sign operand has a constant zero adjoint. Under a ``where`` it must stay that constant,
  not become ``where(c, 0, 0)``, or reverse mode walks back into the sign's producer: here a ``floor``
  under ``nonsmooth="error"``, which refuses a derivative."""
  x = sc.sym("x", 2)
  with sc.options(nonsmooth="error"):
    sign = (x * 2.0 - x[::-1]).floor()
    cost = sc.where(x > 0.0, sc.copysign(x, sign), 0.0).sum()
    grad = _fn("eed_where_zero_adjoint", [x], [sc.gradient(cost, x)])(np.array([0.5, -0.25]))
  np.testing.assert_array_equal(grad, [1.0, 0.0])


def test_where_copysign_and_atan2_with_broadcast_operands_differentiate_like_their_unrolled_form() -> None:
  x = sc.sym("x", 3)
  cond = (x[:2] > 0.5).reshape((2, 1))
  picked = sc.where(cond, x.sin(), x[0] * 2.0)  # (2, 1) condition, (3,) and () branches
  signed = sc.copysign(x * x, x[1] - 0.3)  # a scalar sign operand, which carries no derivative
  angle = sc.atan2(x[:2].reshape((2, 1)) + 1.0, x)  # (2, 1) against (3,)
  out = sc.concat([picked.reshape((6,)), signed, angle.reshape((6,))])
  lam = np.linspace(-1.0, 1.5, 15)
  (rev,) = sc.vjp((out,), (x,), (sc.const(lam),))
  sparse = sc.sparse_jacobian(out, x)
  f = _fn("eed_where_bcast", [x], [out, sc.jacobian(out, x), rev, sparse.to_dense(), sc.hessian((out * sc.const(lam)).sum(), x)])
  xv = np.array([0.2, 0.9, -0.4])
  value, jac, rev_v, sparse_dense, hess = f(xv)
  expected_picked = np.where((xv[:2] > 0.5)[:, None], np.sin(xv), xv[0] * 2.0)
  expected_angle = np.arctan2(xv[:2, None] + 1.0, xv)
  np.testing.assert_allclose(value, np.r_[expected_picked.reshape(-1), np.copysign(xv * xv, xv[1] - 0.3), expected_angle.reshape(-1)], rtol=1e-15)
  fd = finite_difference(lambda v: f(v)[0], xv)
  np.testing.assert_allclose(jac, fd, rtol=1e-7, atol=1e-9)
  np.testing.assert_allclose(rev_v, lam @ jac, rtol=1e-14)
  np.testing.assert_array_equal(sparse_dense, jac)
  rows, cols = np.asarray(sparse.sparsity.rows), np.asarray(sparse.sparsity.cols)
  signs = (rows >= 6) & (rows < 9)
  assert sorted(zip(rows[signs].tolist(), cols[signs].tolist(), strict=True)) == [(6, 0), (7, 1), (8, 2)]  # no column for the sign
  grad_fd = finite_difference(lambda v: f(v)[2], xv)
  np.testing.assert_allclose(hess, grad_fd, rtol=1e-6, atol=1e-8)


def test_predicates_and_non_floating_casts_carry_no_derivative_and_float_casts_do() -> None:
  x = sc.sym("x", 4)
  flat = (
    sc.cast(x < 0.5, "float64")
    + sc.cast(sc.isfinite(1.0 / x) & ~sc.equal(x, 2.0), "float64")
    + sc.cast(sc.cast(x, "int64"), "float64")
    + sc.cast(sc.cast(x, "bool"), "float64")
  )
  gated = x * sc.cast(x > 0.0, "float64")
  through32 = sc.cast(sc.cast(x, "float32").sin(), "float64") + sc.cast(sc.cast(x, "float32"), "float64") * 3.0
  outs = []
  for y in (flat, gated, through32):
    outs += [sc.gradient(y.sum(), x), sc.jvp(y, x, sc.const(np.ones(4))), sc.jacobian(y, x)]
  xv = np.array([-1.3, 0.25, 0.75, 2.5])
  got = _fn("eed_pred_cast", [x], outs)(xv)
  for g in got[:3]:
    np.testing.assert_array_equal(g, np.zeros_like(g))
  gate = (xv > 0.0) * 1.0
  for g in (got[3], got[4], np.diag(got[5])):
    np.testing.assert_array_equal(g, gate)
  x32 = xv.astype(np.float32)
  slope = np.cos(x32).astype(np.float64) + 3.0
  fd = np.diag(finite_difference(lambda v: np.sin(v) + 3.0 * v, xv))
  for g in (got[6], got[7], np.diag(got[8])):
    np.testing.assert_allclose(g, slope, rtol=1e-7)
    np.testing.assert_allclose(g, fd, rtol=1e-6)


def test_multi_seed_forward_is_structural_through_where_logic_and_casts(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  x = sc.sym("x", 4)
  inner = sc.where((x > 0.0) & sc.isfinite(x.log()) | ~(x > -1.0), x * x, sc.cast(sc.cast(x, "float32").sin(), "float64"))
  y = sc.where(sc.cast(x, "int64") > 0, inner, -inner) + sc.cast(x < 1.0, "float64") * x
  xv = np.array([-1.5, -0.5, 0.5, 2.0])
  jac = _fn("eed_many_strict", [x], [sc.jacobian(y, x)])(xv)
  inner_d = np.where((xv > 0.0) | ~(xv > -1.0), 2 * xv, np.cos(xv))
  expected = np.where(xv.astype(np.int64) > 0, inner_d, -inner_d) + (xv < 1.0)
  np.testing.assert_allclose(np.diag(jac), expected, rtol=1e-6)
  np.testing.assert_array_equal(jac - np.diag(np.diag(jac)), np.zeros((4, 4)))
  for build in (lambda: sc.atan2(x, 2.0), lambda: sc.maximum(x, 0.5), lambda: sc.minimum(x, 0.5), lambda: sc.copysign(x, x - 0.5)):
    with pytest.raises(NotImplementedError, match="structural jvp_many does not support"):
      sc.jacobian(build(), x)


def _c32(value: float) -> sc.Expr:
  return sc.const(value, dtype="float32")


# name -> (graph on a float32 h, the same function in float64 NumPy)
F32_OPS: dict[str, tuple[Callable[[sc.Expr], sc.Expr], Callable[[np.ndarray], np.ndarray]]] = {
  "cast": (lambda h: h, lambda v: v),
  "sin": (lambda h: h.sin(), np.sin),
  "compare": (lambda h: sc.cast(h < 0.0, "float32"), lambda v: (v < 0.0) * 1.0),
  "mul_const": (lambda h: h * _c32(2.0), lambda v: v * 2.0),
  "add_const": (lambda h: h + _c32(0.5), lambda v: v + 0.5),
  "mul_self": (lambda h: h * h, lambda v: v * v),
  "div_const": (lambda h: h / _c32(4.0), lambda v: v / 4.0),
  "sum": (lambda h: h.sum(), lambda v: np.sum(v, keepdims=True)),
  "maximum": (lambda h: sc.maximum(h, _c32(0.2)), lambda v: np.fmax(v, 0.2)),
  "minimum": (lambda h: sc.minimum(h, _c32(0.2)), lambda v: np.fmin(v, 0.2)),
  "where": (lambda h: sc.where(h < 0.0, h, h.sin()), lambda v: np.where(v < 0.0, v, np.sin(v))),
  "copysign": (lambda h: sc.copysign(h, h.cos()), lambda v: np.copysign(v, np.cos(v))),
  "atan2": (lambda h: sc.atan2(h, _c32(0.5)), lambda v: np.arctan2(v, 0.5)),
}
F32_MODES = ("jvp", "jacobian", "vjp", "gradient")


@pytest.mark.parametrize(("op", "mode"), [(op, mode) for op in F32_OPS for mode in F32_MODES])
def test_derivatives_through_float32_match_finite_differences(op: str, mode: str) -> None:
  build, reference = F32_OPS[op]
  x = sc.sym("x", 3)
  y = sc.cast(build(sc.cast(x, "float32")), "float64")
  if mode == "jvp":
    d = sc.jvp(y, x, sc.const(np.ones(3)))
  elif mode == "jacobian":
    d = sc.jacobian(y, x)
  elif mode == "vjp":
    (d,) = sc.vjp((y,), (x,), (sc.const(np.ones(y.shape)),))
  else:
    d = sc.gradient(y.sum(), x)
  xv = np.array([0.3, -0.7, 1.1])
  got = _fn(f"eed_f32_{op}_{mode}", [x], [d])(xv)
  jac = finite_difference(reference, xv)
  expected = {"jvp": jac.sum(axis=1), "jacobian": jac, "vjp": jac.sum(axis=0), "gradient": jac.sum(axis=0)}[mode]
  np.testing.assert_allclose(got.reshape(expected.shape), expected, rtol=1e-6, atol=1e-6)
