from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import dblquad, quad
from scipy.interpolate import NdBSpline, PchipInterpolator

import scaly as sc
from scaly import interp
from scaly.codegen import render_c_source
from scaly.interp.grid import Extrap

from .helpers import evaluate, inside_points, scale
from .test_spline import random_knots

EXTRAPS = ["extend", "linear", "clamp", "periodic", "fill"]


def spline_1d(k: int, extrap: Extrap = "linear", fill: float = 0.0) -> interp.BSpline:
  rng = np.random.default_rng(k)
  t = random_knots(rng, k, 7, lo=0.0, hi=2.0)
  return interp.BSpline(t, rng.normal(size=(t.size - k - 1)), k, extrap=extrap, fill=fill)


@pytest.mark.parametrize("k", [1, 2, 3, 5])
def test_derivative_splines_are_scipys(k: int) -> None:
  f = spline_1d(k)
  ref = f.to_scipy()
  rng = np.random.default_rng(10 + k)
  x = inside_points((f.axes[0].edges,), rng, 800)
  for nu in range(1, k + 1):
    d = f.derivative(nu)
    assert d.degree == (k - nu,) and np.array_equal(d.axes[0].edges, f.axes[0].edges)
    np.testing.assert_allclose(evaluate(d, x)["y"], ref(x, nu=nu), rtol=0, atol=1e-11 * scale(ref(x, nu=nu)))
  assert f.derivative(0) is f
  with pytest.raises(ValueError, match="nu"):
    f.derivative(k + 1)


@pytest.mark.parametrize("extrap", EXTRAPS)
def test_a_derivative_spline_is_the_derivative_of_the_spline_everywhere(extrap: Extrap) -> None:
  """Outside the base interval too: the derivative's extrapolation is the one the original's implies,
  so it equals differentiating the original's graph at every point but the knots."""
  f = spline_1d(3, extrap, fill=1.5)
  x = np.concatenate([np.linspace(-3.0, 5.0, 161) + 1e-7, [np.nan]])
  got = evaluate(f, x, derivatives=True)
  first, second = evaluate(f.derivative(), x)["y"], evaluate(f.derivative(2), x)["y"]
  np.testing.assert_allclose(first, got["grad"], rtol=0, atol=1e-10 * scale(first))
  np.testing.assert_allclose(second, got["hess"], rtol=0, atol=1e-9 * scale(second))


def test_a_derivative_shares_the_splines_search() -> None:
  f = interp.interpolant(np.linspace(0.0, 1.0, 33), np.sin(np.linspace(0.0, 4.0, 33)), kind="cubic", search="uniform")
  x = sc.sym("x")
  i = f.index(x)
  d, dd = f.derivative(), f.derivative(2)
  fn = sc.Function.from_exprs("with_derivs", [x], [f(x, index=i), d(x, index=i), dd(x, index=i)], ["x"], ["f", "d", "dd"])
  assert render_c_source(fn).count("floor(") == 1
  xs = np.array(0.61)
  np.testing.assert_allclose(fn(xs), [f.to_scipy()(xs), f.to_scipy()(xs, 1), f.to_scipy()(xs, 2)], rtol=1e-12)


def test_derivative_in_nd_is_along_one_axis() -> None:
  rng = np.random.default_rng(3)
  knots = (random_knots(rng, 3, 4, repeat=False), random_knots(rng, 2, 3, repeat=False))
  c = rng.normal(size=(knots[0].size - 4, knots[1].size - 3, 2))
  f = interp.BSpline(knots, c, (3, 2))
  ref = NdBSpline(knots, c, (3, 2))
  x = inside_points(tuple(ax.edges for ax in f.axes), rng, 200)
  np.testing.assert_allclose(evaluate(f.derivative(1, axis=1), x)["y"], ref(x, nu=(0, 1)), rtol=0, atol=1e-11 * scale(c))
  np.testing.assert_allclose(evaluate(f.derivative(2, axis=0), x)["y"], ref(x, nu=(2, 0)), rtol=0, atol=1e-10 * scale(c))


@pytest.mark.parametrize("extrap", ["extend", "linear", "clamp", "fill"])
def test_antiderivative_starts_at_zero_and_differentiates_back(extrap: Extrap) -> None:
  f = spline_1d(3, extrap, fill=0.0)
  F = f.antiderivative()
  assert F.degree == (4,)
  x = np.linspace(0.0, 2.0, 81)
  np.testing.assert_allclose(evaluate(F, np.array([0.0]))["y"], [0.0], atol=1e-15)
  exact = f.to_scipy().antiderivative()
  np.testing.assert_allclose(evaluate(F, x)["y"], exact(x) - exact(0.0), rtol=0, atol=1e-13)
  outside = np.array([-1.5, -0.2, 2.3, 4.0])
  got = evaluate(F, outside, derivatives=True)
  if extrap in ("extend", "clamp", "fill"):  # the antiderivative integrates the extrapolation exactly
    np.testing.assert_allclose(got["grad"], evaluate(f, outside)["y"], rtol=0, atol=1e-10)
  else:  # linear: the continued end polynomials of F, whose slope agrees with f's to first order at the ends
    assert F.axes[0].extrap == "extend"
  assert F.axes[0].extrap == {"extend": "extend", "linear": "extend", "clamp": "linear", "fill": "clamp"}[extrap]


def test_antiderivative_of_unclamped_knots_starts_at_zero() -> None:
  """A P-spline's knots run past its data, so the integral from the base interval's start is not the
  B-spline antiderivative's own zero, which sits at the first knot."""
  x = np.linspace(0.3, 1.9, 40)
  f = interp.smoothing(x, np.cos(2 * x), segments=6, lam=1e-4)
  F = f.antiderivative()
  pts = np.array([0.3, 0.8, 1.9])
  exact = f.to_scipy().antiderivative()
  np.testing.assert_allclose(evaluate(F, pts)["y"], exact(pts) - exact(0.3), rtol=0, atol=1e-13)


def test_antiderivative_refuses_what_has_no_extrapolation() -> None:
  with pytest.raises(ValueError, match="not a spline extrapolation"):
    spline_1d(3, "periodic").antiderivative()
  with pytest.raises(ValueError, match="not a spline extrapolation"):
    spline_1d(3, "fill", fill=2.0).antiderivative()
  assert spline_1d(3, "fill", fill=np.nan).antiderivative().axes[0].extrap == "fill"


def reference_1d(f: interp.BSpline):
  """The spline with its extrapolation, in NumPy, for ``quad``."""
  ax, s = f.axes[0], f.to_scipy()

  def value(x: float) -> float:
    if ax.extrap == "extend" or ax.lo <= x <= ax.hi:
      return float(s(x))
    if ax.extrap == "periodic":
      return float(s(ax.lo + np.mod(x - ax.lo, ax.hi - ax.lo)))
    end = ax.lo if x < ax.lo else ax.hi
    if ax.extrap == "fill":
      return ax.fill
    return float(s(end) + (s(end, 1) * (x - end) if ax.extrap == "linear" else 0.0))

  return value


@pytest.mark.parametrize("extrap", EXTRAPS)
def test_numeric_integrals_are_quads_in_every_mode(extrap: Extrap) -> None:
  f = spline_1d(3, extrap, fill=-0.75)
  ref = reference_1d(f)
  for a, b in ((0.3, 1.7), (-1.2, 0.4), (1.1, 3.9), (-2.5, 4.5), (1.9, 0.2), (0.5, 0.5), (2.5, 4.0), (-3.0, -1.0), (-3.0, -0.5), (5.0, 3.0)):
    breaks = np.concatenate([f.axes[0].edges + 2.0 * j for j in range(-3, 4)])  # knots, wrapped copies for periodic
    points = [p for p in breaks if min(a, b) < p < max(a, b)]
    want = quad(ref, a, b, points=points or None, limit=200, epsabs=1e-12, epsrel=1e-12)[0]
    assert abs(f.integrate(a, b) - want) <= 1e-10 * max(1.0, abs(want)), (a, b)


@pytest.mark.parametrize("extrap", EXTRAPS)
def test_expression_bounds_integrate_as_numbers_do_in_every_mode(extrap: Extrap) -> None:
  """In 1-D an ``Expr`` bound gives the numeric integral, beyond either end too (the tangent's
  quadratic, a fill's line, whole periods), and its derivative in the upper bound is the spline
  there, continuation included."""
  f = spline_1d(3, extrap, fill=-0.75)
  if extrap == "periodic":
    rng = np.random.default_rng(3)
    t = random_knots(rng, 3, 7, lo=0.0, hi=2.0, repeat=False)
    c = rng.normal(size=t.size - 4)
    c[-3:] = c[:3]  # C2 across the period: the same end coefficients on equally clamped ends
    f = interp.BSpline(t, c, 3, extrap="periodic")
  ref = reference_1d(f)
  a, b = sc.sym("a"), sc.sym("b")
  fn = sc.Function.from_exprs(f"integral_{extrap}", [a, b], [f.integrate(a, b), sc.gradient(f.integrate(a, b), b)], ["a", "b"], ["I", "dIdb"])
  for lo, hi in ((0.3, 1.7), (-3.0, -1.0), (2.5, 4.0), (-2.5, 4.5), (1.9, 0.2)):
    value, slope = fn((np.array(lo), np.array(hi)))
    assert abs(value - f.integrate(lo, hi)) <= 1e-12 * max(1.0, abs(value)), (lo, hi)
    assert abs(slope - ref(hi)) <= 1e-12 * max(1.0, abs(slope)), (lo, hi)


def test_expression_bounds_integrate_through_the_antiderivative() -> None:
  f = spline_1d(3)
  a, b = sc.sym("a"), sc.sym("b")
  fn = sc.Function.from_exprs("integral", [a, b], [f.integrate(a, b), sc.gradient(f.integrate(a, b), b)], ["a", "b"], ["I", "dIdb"])
  value, slope = fn((np.array(0.25), np.array(1.6)))
  np.testing.assert_allclose(value, f.integrate(0.25, 1.6), rtol=1e-12)
  np.testing.assert_allclose(slope, f.to_scipy()(1.6), rtol=1e-12)


def test_nd_integrals() -> None:
  rng = np.random.default_rng(4)
  knots = (random_knots(rng, 2, 3, 0.0, 1.0, repeat=False), random_knots(rng, 3, 4, -1.0, 1.0, repeat=False))
  c = rng.normal(size=(knots[0].size - 3, knots[1].size - 4))
  f = interp.BSpline(knots, c, (2, 3))
  ref = NdBSpline(knots, c, (2, 3))
  want = dblquad(lambda y, x: float(ref([x, y])), 0.1, 0.8, -0.6, 0.9, epsabs=1e-12, epsrel=1e-12)[0]
  assert abs(f.integrate([0.1, -0.6], [0.8, 0.9]) - want) < 1e-10
  lo, hi = sc.sym("lo", 2), sc.sym("hi", 2)
  fn = sc.Function.from_exprs("box", [lo, hi], [f.integrate(lo, hi)], ["lo", "hi"], ["I"])
  np.testing.assert_allclose(fn((np.array([0.1, -0.6]), np.array([0.8, 0.9]))), want, rtol=1e-10)
  with pytest.raises(ValueError, match="inside the base box"):
    f.integrate([0.1, -2.0], [0.8, 0.9])


def test_inverse_round_trips_and_its_derivative_is_one_over_the_slope() -> None:
  rng = np.random.default_rng(5)
  x = np.cumsum(rng.uniform(0.1, 1.0, 14))
  y = -np.cumsum(rng.uniform(0.05, 2.0, 14))  # decreasing
  f = interp.interpolant(x, y, kind="pchip")
  inv = f.inverse()
  assert isinstance(inv, interp.Inverse)
  ref = PchipInterpolator(x, y)
  ys = np.concatenate([y, np.linspace(y[-1], y[0], 301)])
  v = sc.sym("v", ys.size)
  out = inv(v)
  fn = sc.Function.from_exprs(
    "inverse", [v], [out, sc.jvp(out, v, sc.const(np.ones(ys.size))), sc.gradient(out.sum(), v)], ["v"], ["x", "fwd", "rev"]
  )
  xs, fwd, rev = fn(ys)
  np.testing.assert_allclose(ref(xs), ys, rtol=0, atol=1e-12 * scale(ys))
  np.testing.assert_allclose(fwd, 1.0 / ref(xs, 1), rtol=1e-10)
  np.testing.assert_allclose(rev, 1.0 / ref(xs, 1), rtol=1e-10)
  point = sc.sym("p")
  second = sc.Function.from_exprs("inverse_curvature", [point], [sc.hessian(inv(point), point)], ["p"], ["h"])
  p0 = 0.5 * (y[3] + y[4])
  x0 = float(fn(np.full(ys.size, p0))[0][0])
  np.testing.assert_allclose(second(np.array(p0)), -ref(x0, 2) / ref(x0, 1) ** 3, rtol=1e-8)
  beyond = sc.Function.from_exprs("inverse_outside", [point], [inv(point), sc.gradient(inv(point), point)], ["p"], ["x", "g"])
  slope_end = ref(x[-1], 1)
  np.testing.assert_allclose(beyond(np.array(y[-1] - 1.0)), [x[-1] - 1.0 / slope_end, 1.0 / slope_end], rtol=1e-12)
  assert np.isnan(beyond(np.array(np.nan))[0])


def test_inverse_holds_a_flat_end_and_takes_the_data_end_as_inside() -> None:
  """PCHIP's slope is zero at an end whose last secant is much flatter than the one before; the
  spline's value there differs from the data's by an ulp. Neither may send the inverse along a
  near-vertical tangent."""
  x = np.arange(6.0)
  y = np.array([0.0, 2.0, 3.0, 4.0, 4.001, 4.0011])
  inv = interp.interpolant(x, y, kind="pchip").inverse()
  assert PchipInterpolator(x, y)(x[-1], 1) < 1e-12
  point = sc.sym("p")
  fn = sc.Function.from_exprs("inverse_flat_end", [point], [inv(point), sc.gradient(inv(point), point)], ["p"], ["x", "g"])
  for target, want, slope in ((y[-1], x[-1], None), (np.nextafter(y[-1], np.inf), x[-1], None), (y[-1] + 1.0, x[-1], 0.0)):
    got, g = fn(np.array(target))
    assert abs(got - want) < 1e-9, target
    if slope is not None:
      assert g == slope


def test_inverse_bisects_where_newton_would_leave_its_cell() -> None:
  """A quintic cell whose coefficients rise by steps from 1e-7 to 1: from the linear guess, plain
  Newton steps out of the cell and diverges on the polynomial's continuation (off by 5e23). The
  bisection keeps every iterate in the cell."""
  rng = np.random.default_rng(3)
  k = int(rng.choice([3, 4, 5]))
  steps = np.where(rng.uniform(size=k) < 0.5, 10.0 ** rng.uniform(-7, -3, k), rng.exponential(size=k))
  c = np.concatenate([[0.0], np.cumsum(steps)])
  assert k == 5
  f = interp.BSpline(np.concatenate([np.zeros(6), np.ones(6)]), c, 5)
  u = np.concatenate([np.linspace(0.0, 1.0, 200), 1.0 - np.logspace(-9, -1, 30), np.logspace(-9, -1, 30)])
  targets = c[0] + (c[-1] - c[0]) * u
  v = sc.sym("v", targets.size)
  xs = sc.Function.from_exprs("inverse_quintic", [v], [f.inverse()(v)], ["v"], ["x"])(targets)
  assert np.all((xs >= 0.0) & (xs <= 1.0))
  np.testing.assert_allclose(f.to_scipy()(xs), targets, rtol=0, atol=1e-12 * c[-1])


def test_a_linear_tables_inverse_is_the_swapped_table() -> None:
  x = np.array([0.0, 1.0, 2.5, 4.0])
  y = np.array([1.0, 3.0, 3.5, 7.0])
  inv = interp.interpolant(x, y).inverse()
  assert isinstance(inv, interp.BSpline) and inv.degree == (1,)
  pts = np.array([0.0, 1.0, 2.0, 3.25, 5.0, 7.0, 9.0])
  np.testing.assert_allclose(
    evaluate(inv, pts)["y"], np.interp(pts, y, x) + np.where(pts > 7, (pts - 7) * 1.5 / 3.5, 0) + np.where(pts < 1, (pts - 1) / 2, 0), rtol=1e-14
  )


def test_inverse_refuses_what_is_not_monotone() -> None:
  x = np.arange(8.0)
  with pytest.raises(ValueError, match="monotone"):
    interp.interpolant(x, np.sin(x)).inverse()
  with pytest.raises(ValueError, match="derivative changes sign"):
    interp.interpolant(x, np.array([0.0, 0.1, 0.2, 3.0, 3.1, 3.2, 6.0, 6.1]), kind="cubic").inverse()
  with pytest.raises(ValueError, match="1-D scalar"):
    interp.interpolant(x, np.column_stack([x, x])).inverse()
  with pytest.raises(ValueError, match="not invertible"):
    interp.interpolant(x, x, kind="zoh").inverse()


def _fn(inputs: dict[str, sc.Expr], outputs: dict[str, sc.Expr], name: str) -> sc.Function:
  return sc.Function.from_exprs(name, list(inputs.values()), list(outputs.values()), list(inputs), list(outputs))


def test_expression_antiderivative_of_unclamped_knots_starts_at_zero() -> None:
  xs = np.linspace(0.3, 1.9, 40)
  ys = np.cos(2 * xs)
  d = sc.sym("d", 40)
  F = interp.smoothing(xs, d, segments=6, lam=1e-4).antiderivative()
  ref = interp.smoothing(xs, ys, segments=6, lam=1e-4).to_scipy().antiderivative()
  p = sc.sym("p", 3)
  pts = np.array([0.3, 0.8, 1.9])
  np.testing.assert_allclose(_fn({"p": p, "d": d}, {"y": F(p)}, "expr_antiderivative")((pts, ys)), ref(pts) - ref(0.3), rtol=0, atol=1e-12)


def test_inverse_refuses_a_cubic_that_turns_inside_a_cell() -> None:
  t = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
  f = interp.BSpline(t, np.array([0.0, 1.0, -0.5, 0.6]), 3)  # f(0) = 0 < f(1) = 0.6, but f' < 0 around the middle
  s = f.to_scipy()
  assert s(1.0) > s(0.0) and s(np.linspace(0, 1, 101), 1).min() < 0 and s(np.array([0.0, 1.0]), 1).min() > 0
  with pytest.raises(ValueError, match="changes sign"):
    f.inverse()


def test_inverse_refuses_a_non_monotone_curve_at_any_scale() -> None:
  """The allowance for a negative slope is relative to the largest slope, with no floor: a curve
  scaled by 1e-13 is refused as the curve itself is."""
  x = np.linspace(0.0, 2.1, 8)
  y = np.array([0.0, 0.1, 0.2, 3.0, 3.1, 3.2, 6.0, 6.1])
  for size in (1.0, 1e-6, 1e-16):
    with pytest.raises(ValueError, match="changes sign"):
      interp.interpolant(x, size * y, kind="cubic").inverse()


def test_a_clamped_linear_tables_inverse_clamps() -> None:
  x = np.array([0.0, 1.0, 2.5, 4.0])
  y = np.array([1.0, 3.0, 3.5, 7.0])
  inv = interp.interpolant(x, y, extrap="clamp").inverse()
  v = sc.sym("v", 3)
  np.testing.assert_allclose(_fn({"v": v}, {"x": inv(v)}, "linear_inverse_clamp")(np.array([-5.0, 3.25, 20.0])), [0.0, 1.75, 4.0], rtol=1e-15)


def test_two_inverses_of_one_spline_keep_their_own_tolerance() -> None:
  xs = np.linspace(0.0, 2.0, 5)
  f = interp.interpolant(xs, np.exp(2 * xs), kind="cubic")
  loose, strict = f.inverse(tol=1e-1, max_iter=1), f.inverse()
  y = sc.sym("y")
  _, b = _fn({"y": y}, {"a": loose(y), "b": strict(y)}, "two_inverses")(np.array(10.3))
  assert abs(float(f.to_scipy()(b)) - 10.3) < 1e-12


@pytest.mark.parametrize(("x_scale", "y_scale", "y_shift"), [(1.0, 1e-10, 0.0), (1.0, 1e6, 0.0), (1e-16, 1.0, 0.0), (1e6, 1.0, 0.0), (1.0, 1.0, 1e6)])
def test_the_inverse_stops_relative_to_the_cell_whatever_the_scales(x_scale: float, y_scale: float, y_shift: float) -> None:
  """Newton stops once a step moves x by no more than ``tol`` times the larger of |x| and the cell
  width: the round trip is at rounding for any scale of x or y."""
  x = x_scale * np.linspace(0.0, 3.0, 9)
  f = interp.interpolant(x, y_shift + y_scale * np.sinh(x / x_scale), kind="pchip")
  xs = x_scale * np.linspace(0.05, 2.95, 57)
  ys = f.to_scipy()(xs)
  back = evaluate(f.inverse(), ys)["y"]  # ty: ignore[invalid-argument-type]
  np.testing.assert_allclose(back, xs, rtol=0, atol=1e-13 * x_scale * 3.0 + 4 * np.spacing(np.abs(y_shift) / max(y_scale, 1e-300)) * x_scale)


def test_the_inverse_is_nan_with_a_nan_derivative_at_nan() -> None:
  f = interp.interpolant(np.linspace(0.0, 1.0, 6), np.linspace(0.0, 1.0, 6) ** 3 + np.linspace(0.0, 1.0, 6), kind="cubic")
  inv, y = f.inverse(), sc.sym("y", 2)
  x = inv(y)
  value, slope = _fn({"y": y}, {"x": x, "dxdy": sc.gradient(x.sum(), y)}, "inverse_at_nan")(np.array([np.nan, 0.5]))
  assert np.isnan(value[0]) and np.isnan(slope[0]) and np.isfinite(slope[1])
