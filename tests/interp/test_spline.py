from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.interpolate import BSpline as SciBSpline
from scipy.interpolate import NdBSpline

import scaly as sc
from scaly import interp
from scaly.interp.spline import Strategy
from scaly.codegen import render_c_source

from .helpers import evaluate, inside_points, scale


def random_knots(rng: np.random.Generator, k: int, inner: int, lo: float = -1.0, hi: float = 2.0, repeat: bool = True) -> np.ndarray:
  """A clamped knot vector with ``inner`` random interior knots, one of them doubled when ``repeat``
  (a drop in continuity the evaluation has to respect)."""
  t = np.sort(rng.uniform(lo, hi, inner))
  if repeat and k >= 2 and inner >= 3:
    t = np.sort(np.append(t, t[inner // 2]))
  return np.concatenate([np.full(k + 1, lo), t, np.full(k + 1, hi)])


@pytest.mark.parametrize("strategy", ["pp", "basis"])
@pytest.mark.parametrize("k", [0, 1, 2, 3, 4, 5])
def test_values_and_derivatives_match_scipy_in_1d(k: int, strategy: Strategy) -> None:
  rng = np.random.default_rng(k)
  t = random_knots(rng, k, 9)
  c = rng.normal(size=t.size - k - 1)
  f = interp.BSpline(t, c, k, strategy=strategy)
  ref = SciBSpline(t, c, k)
  x = inside_points((f.axes[0].edges,), rng, 2000)
  got = evaluate(f, x, derivatives=k > 0)
  np.testing.assert_allclose(got["y"], ref(x), rtol=0, atol=1e-13 * scale(ref(x)))
  if k:
    d1, d2 = ref(x, nu=1), ref(x, nu=2)
    for key in ("grad", "jvp"):
      np.testing.assert_allclose(got[key], d1, rtol=0, atol=1e-11 * scale(d1), err_msg=key)
    for key in ("hess", "hess_fwd"):
      np.testing.assert_allclose(got[key], d2, rtol=0, atol=1e-11 * scale(d2), err_msg=key)


@pytest.mark.parametrize("strategy", ["pp", "basis"])
@pytest.mark.parametrize("degrees", [(1, 1), (3, 3), (3, 1, 2), (0, 3), (2, 1, 3, 1)])
def test_values_and_derivatives_match_ndbspline(degrees: tuple[int, ...], strategy: Strategy) -> None:
  rng = np.random.default_rng(sum(degrees) + len(degrees))
  knots = tuple(random_knots(rng, k, 5, lo=-0.5 * d, hi=1.0 + d, repeat=False) for d, k in enumerate(degrees))
  c = rng.normal(size=tuple(t.size - k - 1 for t, k in zip(knots, degrees, strict=True)))
  f = interp.BSpline(knots, c, degrees, strategy=strategy)
  ref = NdBSpline(knots, c, degrees)
  x = inside_points(tuple(ax.edges for ax in f.axes), rng, 300)
  got = evaluate(f, x, derivatives=True)
  value = ref(x)
  np.testing.assert_allclose(got["y"], value, rtol=0, atol=1e-13 * scale(value))
  d = len(degrees)
  unit = np.eye(d, dtype=int)
  grad = np.stack([ref(x, nu=tuple(unit[i])) for i in range(d)], axis=1)
  hess = np.stack([np.stack([ref(x, nu=tuple(unit[i] + unit[j])) for j in range(d)], axis=1) for i in range(d)], axis=1)
  for key in ("grad", "jvp"):
    np.testing.assert_allclose(got[key], grad, rtol=0, atol=1e-11 * scale(grad), err_msg=key)
  for key in ("hess", "hess_fwd"):
    np.testing.assert_allclose(got[key], hess, rtol=0, atol=1e-11 * scale(hess), err_msg=key)


@pytest.mark.parametrize("strategy", ["pp", "basis"])
def test_vector_and_matrix_outputs(strategy: Strategy) -> None:
  rng = np.random.default_rng(7)
  t = random_knots(rng, 3, 6)
  c = rng.normal(size=(t.size - 4, 2, 3))
  f = interp.BSpline(t, c, 3, strategy=strategy)
  assert f.out_shape == (2, 3)
  x = inside_points((f.axes[0].edges,), rng, 200)
  np.testing.assert_allclose(evaluate(f, x)["y"], SciBSpline(t, c, 3)(x), rtol=0, atol=1e-13 * scale(c))
  point = sc.sym("p")
  fn = sc.Function._from_exprs(f"matrix_point_{strategy}", [point], [f(point)], ["p"], ["y"])
  np.testing.assert_allclose(fn(np.array(0.37)), SciBSpline(t, c, 3)(0.37), rtol=0, atol=1e-13 * scale(c))
  t2 = (random_knots(rng, 1, 3, repeat=False), random_knots(rng, 2, 4, repeat=False))
  c2 = rng.normal(size=(t2[0].size - 2, t2[1].size - 3, 4))
  g = interp.BSpline(t2, c2, (1, 2), strategy=strategy)
  x2 = inside_points(tuple(ax.edges for ax in g.axes), rng, 100)
  np.testing.assert_allclose(evaluate(g, x2)["y"], NdBSpline(t2, c2, (1, 2))(x2), rtol=0, atol=1e-13 * scale(c2))


def _spline_1d(extrap: str, fill: float = math.nan, k: int = 3) -> tuple[interp.BSpline, SciBSpline]:
  rng = np.random.default_rng(11)
  t = random_knots(rng, k, 6, lo=0.0, hi=1.0)
  c = rng.normal(size=t.size - k - 1)
  return interp.BSpline(t, c, k, extrap=extrap, fill=fill), SciBSpline(t, c, k)  # ty: ignore[invalid-argument-type]


OUTSIDE = np.array([-1e3, -1.5, -1e-9, 0.0, 0.4, 1.0, 1.0 + 1e-9, 2.25, 1e3])


def test_extend_continues_the_end_polynomials() -> None:
  f, ref = _spline_1d("extend")
  got = evaluate(f, OUTSIDE, derivatives=True)
  np.testing.assert_allclose(got["y"], ref(OUTSIDE), rtol=1e-12)
  np.testing.assert_allclose(got["grad"], ref(OUTSIDE, nu=1), rtol=1e-12)


def test_clamp_holds_the_end_values_with_a_zero_slope() -> None:
  f, ref = _spline_1d("clamp")
  got = evaluate(f, OUTSIDE, derivatives=True)
  clipped = np.clip(OUTSIDE, 0.0, 1.0)
  np.testing.assert_allclose(got["y"], ref(clipped), rtol=0, atol=1e-14)
  inside = (OUTSIDE >= 0.0) & (OUTSIDE <= 1.0)
  np.testing.assert_allclose(got["grad"], np.where(inside, ref(clipped, nu=1), 0.0), rtol=0, atol=1e-12)


def test_linear_continues_along_the_end_tangents() -> None:
  f, ref = _spline_1d("linear")
  got = evaluate(f, OUTSIDE, derivatives=True)
  clipped = np.clip(OUTSIDE, 0.0, 1.0)
  np.testing.assert_allclose(got["y"], ref(clipped) + ref(clipped, nu=1) * (OUTSIDE - clipped), rtol=1e-13, atol=1e-13)
  np.testing.assert_allclose(got["grad"], ref(clipped, nu=1), rtol=0, atol=1e-11)
  np.testing.assert_allclose(got["jvp"], ref(clipped, nu=1), rtol=0, atol=1e-11)
  outside = (OUTSIDE < 0.0) | (OUTSIDE > 1.0)
  np.testing.assert_allclose(got["hess"], np.where(outside, 0.0, ref(clipped, nu=2)), rtol=0, atol=1e-10)


def test_linear_extrapolation_in_2d_is_the_tensor_product_of_the_axes() -> None:
  """Outside in both axes the value is ``p + d1 p_1 + d2 p_2 + d1 d2 p_12`` at the corner: the
  product of the axes' linear continuations, which keeps the function C1 across the lines where a
  point leaves the second axis."""
  rng = np.random.default_rng(12)
  knots = (random_knots(rng, 3, 4, 0.0, 1.0, repeat=False), random_knots(rng, 2, 3, 0.0, 1.0, repeat=False))
  c = rng.normal(size=(knots[0].size - 4, knots[1].size - 3))
  f = interp.BSpline(knots, c, (3, 2))
  ref = NdBSpline(knots, c, (3, 2))
  pts = np.array([[1.3, 1.6], [-0.4, 0.5], [0.5, -2.0], [-1.0, -1.0], [1.0 + 1e-12, 0.7]])
  clipped = np.clip(pts, 0.0, 1.0)
  d = pts - clipped
  want = ref(clipped) + d[:, 0] * ref(clipped, nu=(1, 0)) + d[:, 1] * ref(clipped, nu=(0, 1)) + d[:, 0] * d[:, 1] * ref(clipped, nu=(1, 1))
  np.testing.assert_allclose(evaluate(f, pts)["y"], want, rtol=1e-12, atol=1e-12)


def test_periodic_wraps_the_point() -> None:
  f, ref = _spline_1d("periodic")
  got = evaluate(f, OUTSIDE, derivatives=True)
  wrapped = np.mod(OUTSIDE, 1.0)
  np.testing.assert_allclose(got["y"], ref(wrapped), rtol=0, atol=1e-10)
  np.testing.assert_allclose(got["grad"], ref(wrapped, nu=1), rtol=0, atol=1e-8)


def test_fill_gives_the_constant_outside_and_the_spline_on_the_closed_interval() -> None:
  f, ref = _spline_1d("fill", fill=-7.5)
  got = evaluate(f, OUTSIDE, derivatives=True)
  inside = (OUTSIDE >= 0.0) & (OUTSIDE <= 1.0)
  np.testing.assert_allclose(got["y"], np.where(inside, ref(np.clip(OUTSIDE, 0.0, 1.0)), -7.5), rtol=0, atol=1e-14)
  np.testing.assert_allclose(got["grad"], np.where(inside, ref(np.clip(OUTSIDE, 0.0, 1.0), nu=1), 0.0), rtol=0, atol=1e-12)
  nan_fill, _ = _spline_1d("fill")
  assert np.isnan(evaluate(nan_fill, np.array([-1.0, 2.0]))["y"]).all()


@pytest.mark.parametrize("extrap", ["extend", "linear", "clamp", "periodic", "fill"])
@pytest.mark.parametrize("k", [0, 1, 3])
def test_non_finite_points(extrap: str, k: int) -> None:
  """NaN gives NaN in every mode. An infinity gives the end value (clamp), the fill value, NaN
  (periodic: inf - inf), or the IEEE value of the continuation: an infinity signed by the leading
  coefficient (extend) or by the end slope (linear, the end value where that slope is zero)."""
  f, ref = _spline_1d(extrap, fill=4.0, k=k)
  got = evaluate(f, np.array([np.nan, np.inf, -np.inf]))["y"]
  assert np.isnan(got[0])
  hi, lo = got[1], got[2]
  if extrap == "clamp" or (k == 0 and extrap in ("linear", "extend")):
    np.testing.assert_allclose([hi, lo], [ref(1.0), ref(0.0)], rtol=0, atol=1e-14)
  elif extrap == "fill":
    assert (hi, lo) == (4.0, 4.0)
  elif extrap == "periodic":
    assert np.isnan(hi) and np.isnan(lo)
  elif extrap == "linear":
    assert (hi, lo) == (math.copysign(math.inf, ref(1.0, nu=1)), math.copysign(math.inf, -ref(0.0, nu=1)))
  else:  # extend: the end polynomial at infinity, signed by its leading coefficient
    assert np.isinf(hi) and np.isinf(lo)
    assert (np.sign(hi), np.sign(lo)) == (np.sign(ref(1.0, nu=k)), np.sign(ref(0.0, nu=k)) * (-1) ** k)


def test_an_index_is_shared_and_found_once() -> None:
  """Value, gradient and Hessian at the same point search once: the index path carries no derivative,
  so every derivative graph reuses the value's search node, and the program has one ``floor``."""
  f = interp.interpolant(np.linspace(0.0, 1.0, 17), np.sin(np.linspace(0.0, 3.0, 17)), kind="cubic", search="uniform")
  x = sc.sym("x")
  y = f(x)
  fn = sc.Function._from_exprs("shared_search", [x], [y, sc.gradient(y, x), sc.hessian(y, x)], ["x"], ["y", "g", "h"])
  assert render_c_source(fn).count("floor(") == 1
  i = f.index(x)
  shared = sc.Function._from_exprs("shared_index", [x], [f(x, index=i), f(x)], ["x"], ["a", "b"])
  a, b = shared(np.array(0.4321))
  assert a == b
  with pytest.raises(ValueError, match="index="):
    f(sc.sym("xs", 3), index=i)


def test_function_names_follow_the_content() -> None:
  t = np.array([0.0, 0.0, 0.5, 1.0, 1.0])
  f = interp.BSpline(t, np.array([0.0, 1.0, 0.0]), 1, name="tbl")
  g = interp.BSpline(t, np.array([0.0, 2.0, 0.0]), 1, name="tbl")
  assert f.function().name != g.function().name and f.function().name.startswith("tbl_")
  assert f.function() is interp.BSpline(t, np.array([0.0, 1.0, 0.0]), 1, name="tbl").function()  # one Function per content
  x = sc.sym("x")
  F, G = f.function(), g.function()
  both = sc.Function._from_exprs("two_tables", [x], [F(x) + F(x + 0.1) + G(x)], ["x"], ["y"])
  np.testing.assert_allclose(both(np.array(0.25)), 0.5 + 0.7 + 1.0, rtol=1e-15)


def test_to_scipy_is_the_same_spline() -> None:
  rng = np.random.default_rng(13)
  t = random_knots(rng, 2, 5)
  c = rng.normal(size=t.size - 3)
  s = interp.BSpline(t, c, 2).to_scipy()
  assert isinstance(s, SciBSpline) and s.k == 2 and np.array_equal(s.t, t) and np.array_equal(s.c, c)
  t2 = (t, random_knots(rng, 1, 2, repeat=False))
  c2 = rng.normal(size=(t.size - 3, t2[1].size - 2))
  assert isinstance(interp.BSpline(t2, c2, (2, 1)).to_scipy(), NdBSpline)


def test_auto_strategy_leaves_large_tables_to_the_basis() -> None:
  g = np.linspace(0.0, 1.0, 300)
  big = interp.interpolant((g, g), np.zeros((300, 300)), kind="cubic")
  assert big.strategy == "basis"
  assert interp.interpolant((g[:20], g[:20]), np.zeros((20, 20)), kind="cubic").strategy == "pp"


def test_bspline_validation() -> None:
  t = np.array([0.0, 0.0, 1.0, 1.0])
  with pytest.raises(ValueError, match="coeffs must have shape"):
    interp.BSpline(t, np.zeros(3), 1)
  with pytest.raises(ValueError, match="finite"):
    interp.BSpline(t, np.array([0.0, np.nan]), 1)
  with pytest.raises(ValueError, match="strategy"):
    interp.BSpline(t, np.zeros(2), 1, strategy="table")  # ty: ignore[invalid-argument-type]
  with pytest.raises(ValueError, match="entries for 2 axes"):
    interp.BSpline((t, t), np.zeros((2, 2)), (1, 1, 1))
  f = interp.BSpline((t, t), np.zeros((2, 2)), 1)
  with pytest.raises(ValueError, match="point"):
    f(sc.sym("x", 3))
  with pytest.raises(ValueError, match="scalar or a batch"):
    interp.BSpline(t, np.zeros(2), 1)(sc.sym("x", (2, 2)))
  assert "BSpline('interp'" in repr(f)


def test_a_batch_is_one_map_with_a_block_diagonal_jacobian() -> None:
  rng = np.random.default_rng(14)
  g = (np.linspace(0.0, 1.0, 9), np.sort(rng.uniform(-1.0, 1.0, 6)))
  f = interp.interpolant(g, rng.normal(size=(9, 6, 2)), kind=("cubic", "linear"))
  n = 40
  x = sc.sym("x", (n, 2))
  y = f(x)
  assert y.shape == (n, 2)
  sj = sc.sparse_jacobian(y.reshape((2 * n,)), x.reshape((2 * n,)))
  rows, cols = np.asarray(sj.sparsity.rows), np.asarray(sj.sparsity.cols)
  assert np.array_equal(rows // 2, cols // 2) and sj.sparsity.nnz == 4 * n  # each output reads its own point
  i = f.index(x)
  fn = sc.Function._from_exprs("batch_index", [x], [y, f(x, index=i), sj.values], ["x"], ["y", "yi", "sj"])
  pts = np.column_stack([rng.uniform(-0.2, 1.2, n), rng.uniform(-1.2, 1.2, n)])
  yv, yiv, sjv = fn(pts)
  np.testing.assert_array_equal(yv, yiv)
  single = sc.sym("p", 2)
  point = sc.Function._from_exprs("batch_point", [single], [f(single), sc.jacobian(f(single), single)], ["p"], ["y", "j"])
  dense = np.zeros((2 * n, 2 * n))
  dense[rows, cols] = sjv
  for k in range(n):
    value, jac = point(pts[k])
    np.testing.assert_allclose(yv[k], value, rtol=0, atol=1e-15)
    np.testing.assert_allclose(dense[2 * k : 2 * k + 2, 2 * k : 2 * k + 2], jac, rtol=1e-14, atol=1e-14)
  with pytest.raises(ValueError, match="index="):
    f(sc.sym("x2", (n + 1, 2)), index=i)
