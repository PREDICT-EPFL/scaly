from __future__ import annotations

import itertools

import numpy as np
import pytest
from scipy.interpolate import CubicSpline, NdBSpline, RegularGridInterpolator, make_interp_spline

from scaly import interp
from scaly.interp.spline import Strategy

from .helpers import evaluate, inside_points, scale


def sites(rng: np.random.Generator, n: int, uniform: bool, lo: float = -1.0, hi: float = 2.0) -> np.ndarray:
  if uniform:
    return np.linspace(lo, hi, n)
  g = np.sort(rng.uniform(lo, hi, n))
  g[0], g[-1] = lo, hi
  return g


@pytest.mark.parametrize("uniform", [True, False])
def test_linear_is_np_interp_and_continues_the_end_segments(uniform: bool) -> None:
  rng = np.random.default_rng(1)
  g = sites(rng, 23, uniform)
  y = rng.normal(size=g.size)
  f = interp.interpolant(g, y)
  assert f.degree == (1,) and f.axes[0].extrap == "extend"  # the default "linear", which is "extend" at degree 1
  x = inside_points((g,), rng, 3000)
  got = evaluate(f, x, derivatives=True)
  np.testing.assert_allclose(got["y"], np.interp(x, g, y), rtol=0, atol=1e-14 * scale(y))
  out = np.array([-5.0, g[0] - 1e-3, g[-1] + 1e-3, 40.0])
  extended = RegularGridInterpolator((g,), y, bounds_error=False, fill_value=None)(out[:, None])
  np.testing.assert_allclose(evaluate(f, out)["y"], extended, rtol=1e-13)
  clamped = interp.interpolant(g, y, extrap="clamp")
  np.testing.assert_allclose(evaluate(clamped, out)["y"], np.interp(out, g, y), rtol=0, atol=1e-15)
  cell = np.clip(np.searchsorted(g, x, side="right") - 1, 0, g.size - 2)
  np.testing.assert_allclose(got["grad"], np.diff(y)[cell] / np.diff(g)[cell], rtol=1e-12)


@pytest.mark.parametrize("bc", ["not-a-knot", "natural", "clamped", "periodic"])
@pytest.mark.parametrize("uniform", [True, False])
def test_cubic_is_cubicspline(bc: str, uniform: bool) -> None:
  rng = np.random.default_rng(len(bc) + uniform)
  g = sites(rng, 14, uniform)
  y = rng.normal(size=g.size)
  if bc == "periodic":
    y[-1] = y[0]
  ref = CubicSpline(g, y, bc_type=bc, extrapolate="periodic" if bc == "periodic" else True)
  for strategy in ("pp", "basis"):
    f = interp.interpolant(g, y, kind="cubic", bc=bc, strategy=strategy)  # ty: ignore[invalid-argument-type]
    assert f.axes[0].search == ("uniform" if uniform else "binary")
    x = inside_points((g,), rng, 2000)
    got = evaluate(f, x, derivatives=True)
    np.testing.assert_allclose(got["y"], ref(x), rtol=0, atol=1e-13 * scale(ref(x)))
    for key, nu in (("grad", 1), ("jvp", 1), ("hess", 2), ("hess_fwd", 2)):
      np.testing.assert_allclose(got[key], ref(x, nu), rtol=0, atol=1e-11 * scale(ref(x, nu)), err_msg=f"{strategy} {key}")
  if bc == "periodic":
    assert f.axes[0].extrap == "periodic"
    far = np.array([-11.3, -1.0 - 1e-9, 2.0 + 1e-9, 7.7, 1e3])
    np.testing.assert_allclose(evaluate(f, far)["y"], ref(far), rtol=0, atol=1e-10)


@pytest.mark.parametrize("k", [1, 2, 3, 4, 5])
def test_spline_of_each_degree_is_make_interp_spline(k: int) -> None:
  rng = np.random.default_rng(k)
  g = sites(rng, 12, uniform=k % 2 == 1)
  y = rng.normal(size=(g.size, 2))
  ref = make_interp_spline(g, y, k=k)
  f = interp.interpolant(g, y, kind="spline", degree=k)
  x = inside_points((f.axes[0].edges,), rng, 1500)
  got = evaluate(f, x, derivatives=True)
  np.testing.assert_allclose(got["y"], ref(x), rtol=0, atol=1e-13 * scale(ref(x)))
  np.testing.assert_allclose(got["jvp"], ref(x, 1).sum(axis=1), rtol=0, atol=1e-11 * scale(ref(x, 1)))
  np.testing.assert_allclose(evaluate(f, g)["y"], y, rtol=0, atol=1e-13 * scale(y))  # it interpolates


def test_nearest_is_regulargridinterpolator_and_ties_go_to_the_lower_site() -> None:
  rng = np.random.default_rng(5)
  g = sites(rng, 15, uniform=False)
  y = rng.normal(size=g.size)
  f = interp.interpolant(g, y, kind="nearest")
  assert f.axes[0].side == "left" and f.axes[0].extrap == "clamp"
  mids = 0.5 * (g[:-1] + g[1:])
  x = rng.uniform(-2.0, 3.0, 4000)
  x = x[np.min(np.abs(x[:, None] - mids[None, :]), axis=1) > 1e-9]
  ref = RegularGridInterpolator((g,), y, method="nearest", bounds_error=False, fill_value=None)
  np.testing.assert_array_equal(evaluate(f, x)["y"], ref(x[:, None]))
  np.testing.assert_array_equal(evaluate(f, mids)["y"], y[:-1])  # the float midpoint is the lower site's
  np.testing.assert_array_equal(evaluate(f, np.nextafter(mids, np.inf))["y"], y[1:])


def test_zoh_holds_the_last_value_at_or_before_the_point() -> None:
  rng = np.random.default_rng(6)
  g = sites(rng, 24, uniform=False, lo=0.0, hi=23.0)
  y = rng.normal(size=g.size)
  f = interp.interpolant(g, y, kind="zoh")
  x = np.concatenate([g, np.nextafter(g, -np.inf), rng.uniform(-3.0, 30.0, 3000)])
  want = y[np.clip(np.searchsorted(g, x, side="right") - 1, 0, g.size - 1)]
  np.testing.assert_array_equal(evaluate(f, x)["y"], want)
  np.testing.assert_array_equal(evaluate(f, np.array([np.nan]))["y"], [np.nan])
  hourly = interp.interpolant(np.arange(24.0), y, kind="zoh", extrap="periodic", period=24.0)
  t = np.array([-0.5, 23.5, 24.0, 47.25, 100.0])
  np.testing.assert_array_equal(evaluate(hourly, t)["y"], y[np.floor(np.mod(t, 24.0)).astype(int)])


@pytest.mark.parametrize("strategy", ["pp", "basis"])
def test_2d_and_3d_fits_are_the_per_axis_tensor_fits(strategy: Strategy) -> None:
  rng = np.random.default_rng(8)
  g = (sites(rng, 9, False), sites(rng, 7, True, 0.0, 1.0))
  v = rng.normal(size=(9, 7))
  f = interp.interpolant(g, v, kind="cubic", strategy=strategy)
  c = np.moveaxis(make_interp_spline(g[1], make_interp_spline(g[0], v, k=3).c, k=3, axis=1).c, 0, 1)
  ref = NdBSpline((make_interp_spline(g[0], v, k=3).t, make_interp_spline(g[1], v.T, k=3).t), c, 3)
  x = inside_points(g, rng, 600)
  np.testing.assert_allclose(evaluate(f, x)["y"], ref(x), rtol=0, atol=1e-13 * scale(v))
  g3 = (sites(rng, 5, True), sites(rng, 6, False), sites(rng, 4, True))
  v3 = rng.normal(size=(5, 6, 4))
  lin = interp.interpolant(g3, v3, strategy=strategy)
  x3 = inside_points(g3, rng, 500)
  np.testing.assert_allclose(evaluate(lin, x3)["y"], RegularGridInterpolator(g3, v3)(x3), rtol=0, atol=1e-14 * scale(v3))


def test_a_4d_table_of_mixed_kinds_interpolates_its_data() -> None:
  rng = np.random.default_rng(9)
  g = (sites(rng, 5, True), sites(rng, 4, False), sites(rng, 6, True), sites(rng, 5, False))
  v = rng.normal(size=(5, 4, 6, 5, 2))
  f = interp.interpolant(g, v, kind=("cubic", "linear", "zoh", "nearest"))
  nodes = np.array(list(itertools.product(*g)))
  np.testing.assert_allclose(evaluate(f, nodes)["y"], v.reshape(-1, 2), rtol=0, atol=1e-13 * scale(v))
  x = inside_points(g, rng, 200)
  ref = f.to_scipy()
  inner = np.all([(x[:, d] != g[3][i]) for d in (3,) for i in range(g[3].size)], axis=0)  # nearest ties aside
  np.testing.assert_allclose(evaluate(f, x[inner])["y"], ref(x[inner]), rtol=0, atol=1e-13 * scale(v))


def test_vector_valued_cubic_is_cubicspline_along_axis_0() -> None:
  rng = np.random.default_rng(10)
  theta = np.linspace(0.0, 2 * np.pi, 33)
  xy = np.column_stack([np.cos(theta), np.sin(2 * theta), theta * 0.0 + 1.0, np.cos(3 * theta)])
  f = interp.interpolant(theta, xy, kind="cubic", bc="periodic")
  ref = CubicSpline(theta, xy, bc_type="periodic", extrapolate="periodic")
  x = np.concatenate([inside_points((theta,), rng, 500), [-7.0, 20.0]])
  np.testing.assert_allclose(evaluate(f, x)["y"], ref(x), rtol=0, atol=1e-10)


def test_interpolant_validation() -> None:
  g = np.linspace(0.0, 1.0, 5)
  with pytest.raises(ValueError, match="kind"):
    interp.interpolant(g, g, kind="quadratic")  # ty: ignore[invalid-argument-type]
  with pytest.raises(ValueError, match="bc"):
    interp.interpolant(g, g, kind="cubic", bc="free")  # ty: ignore[invalid-argument-type]
  with pytest.raises(ValueError, match="degree must be 1 to 5"):
    interp.interpolant(g, g, kind="spline", degree=6)
  with pytest.raises(ValueError, match="takes bc"):
    interp.interpolant(g, g, kind="spline", bc="natural")
  with pytest.raises(ValueError, match="at least 4 points"):
    interp.interpolant(g[:3], g[:3], kind="cubic")
  with pytest.raises(ValueError, match="first and last values equal"):
    interp.interpolant(g, g, kind="cubic", bc="periodic")
  with pytest.raises(ValueError, match="first and last values equal"):
    interp.interpolant(g, g, extrap="periodic")
  with pytest.raises(ValueError, match="period must exceed"):
    interp.interpolant(g, g, kind="zoh", period=0.5)
  with pytest.raises(ValueError, match="entries along axis 0"):
    interp.interpolant(g, g[:4])
  with pytest.raises(ValueError, match="finite"):
    interp.interpolant(g, np.array([0.0, 1.0, np.inf, 0.0, 0.0]))
  with pytest.raises(ValueError, match="at least 2 axes"):
    interp.interpolant((g, g), g)
  with pytest.raises(ValueError, match="strictly increasing"):
    interp.interpolant(g[::-1], g)


def steps_and_plateau(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
  """Data with a jump, a plateau and a monotone run: where C2 splines ring."""
  x = np.sort(rng.uniform(0.0, 10.0, n))
  y = np.concatenate([np.zeros(n // 3), np.ones(n // 3), np.linspace(1.0, 3.0, n - 2 * (n // 3))]) + 0.01 * rng.normal(size=n)
  return x, y


@pytest.mark.parametrize("kind", ["pchip", "akima", "makima"])
@pytest.mark.parametrize("uniform", [True, False])
def test_shape_preserving_kinds_are_scipys(kind: str, uniform: bool) -> None:
  from scipy.interpolate import Akima1DInterpolator, PchipInterpolator

  rng = np.random.default_rng(len(kind) + uniform)
  x, y = steps_and_plateau(rng, 16)
  if uniform:
    x = np.linspace(0.0, 10.0, 16)
  values = np.column_stack([y, np.sin(x)])
  ref = PchipInterpolator(x, values) if kind == "pchip" else Akima1DInterpolator(x, values, method=kind, extrapolate=True)  # ty: ignore[invalid-argument-type]
  f = interp.interpolant(x, values, kind=kind, extrap="extend")  # ty: ignore[invalid-argument-type]
  assert f.degree == (3,) and f.axes[0].search == ("uniform" if uniform else "binary")
  pts = np.concatenate([inside_points((x,), rng, 1500), [-2.0, 12.5]])
  got = evaluate(f, pts, derivatives=True)
  np.testing.assert_allclose(got["y"], ref(pts), rtol=0, atol=1e-13 * scale(ref(pts)))
  np.testing.assert_allclose(got["jvp"], ref(pts, 1).sum(axis=1), rtol=0, atol=1e-11 * scale(ref(pts, 1)))
  np.testing.assert_allclose(got["hess"], ref(pts, 2).sum(axis=1), rtol=0, atol=1e-10 * scale(ref(pts, 2)))
  two = interp.interpolant(x[:2], values[:2], kind=kind, extrap="extend")  # ty: ignore[invalid-argument-type]
  beyond = 2 * x[1] - x[0]  # two points: the line through them, continued
  np.testing.assert_allclose(
    evaluate(two, np.array([x[0], 0.5 * (x[0] + x[1]), beyond]))["y"],
    np.array([values[0], values[:2].mean(0), 2 * values[1] - values[0]]),
    rtol=1e-12,
    atol=1e-14,
  )


# Data that reach the rarely taken branches: PCHIP's end limiter (at the start the three-point slope
# lies between two and three times the first secant, so only the threshold 3 decides; at the end it is
# more than three times, so the limiter acts), Akima's break (the outer secants differ by 1e-12 on one side and not at all on
# the other: below 1e-9 of the largest weight, so the slope is the fill value), Steffen's end limit
# (the parabola through the first three points more than twice as steep as the first secant).
EDGE_CASES = {
  "pchip": (np.arange(8.0), np.array([0.0, 0.1, -0.1, 1.0, 2.0, 4.0, 4.5, 4.4])),
  "akima": (np.arange(9.0), np.array([0.0, 0.0, 0.0, 1.0, 2.0 + 1e-12, 3.0 + 2e-12, -2.0, 5.0, 0.0])),
  "makima": (np.arange(9.0), np.array([0.0, 0.0, 0.0, 1.0, 2.0 + 1e-12, 3.0 + 2e-12, -2.0, 5.0, 0.0])),
  "steffen": (np.arange(6.0), np.array([0.0, 1.0, -9.0, -8.0, 2.0, 3.0])),
}


@pytest.mark.parametrize("kind", ["pchip", "akima", "makima"])
def test_shape_preserving_edge_branches_are_scipys(kind: str) -> None:
  from scipy.interpolate import Akima1DInterpolator, PchipInterpolator

  x, y = EDGE_CASES[kind]
  ref = PchipInterpolator(x, y) if kind == "pchip" else Akima1DInterpolator(x, y, method=kind, extrapolate=True)  # ty: ignore[invalid-argument-type]
  pts = np.linspace(x[0], x[-1], 241)
  got = evaluate(interp.interpolant(x, y, kind=kind), pts, derivatives=True)  # ty: ignore[invalid-argument-type]
  np.testing.assert_allclose(got["y"], ref(pts), rtol=0, atol=1e-13 * scale(y))
  np.testing.assert_allclose(got["jvp"], ref(pts, 1), rtol=0, atol=1e-12 * scale(ref(pts, 1)))


def steffen_reference(x: np.ndarray, y: np.ndarray) -> np.ndarray:
  """Steffen (1990), eqs. 11 and 26-27, one site at a time."""
  n = x.size
  h = [x[i + 1] - x[i] for i in range(n - 1)]
  s = [(y[i + 1] - y[i]) / h[i] for i in range(n - 1)]
  d = np.zeros(n)
  for i in range(1, n - 1):
    p = (s[i - 1] * h[i] + s[i] * h[i - 1]) / (h[i - 1] + h[i])
    d[i] = (np.sign(s[i - 1]) + np.sign(s[i])) * min(abs(s[i - 1]), abs(s[i]), 0.5 * abs(p))
  for i, (h0, h1, s0, s1) in ((0, (h[0], h[1], s[0], s[1])), (n - 1, (h[-1], h[-2], s[-1], s[-2]))):
    p = s0 * (1 + h0 / (h0 + h1)) - s1 * h0 / (h0 + h1)
    d[i] = 0.0 if p * s0 <= 0 else 2 * s0 if abs(p) > 2 * abs(s0) else p
  return d


def test_steffen_follows_the_paper_and_is_monotone_without_overshoot() -> None:
  from scipy.interpolate import CubicHermiteSpline

  from scaly.interp.fit import hermite_slopes

  rng = np.random.default_rng(3)
  x, y = steps_and_plateau(rng, 14)
  np.testing.assert_allclose(hermite_slopes("steffen", x, y), steffen_reference(x, y), rtol=1e-15, atol=1e-15)
  np.testing.assert_allclose(hermite_slopes("steffen", *EDGE_CASES["steffen"]), steffen_reference(*EDGE_CASES["steffen"]), rtol=1e-15, atol=1e-15)
  f = interp.interpolant(x, y, kind="steffen")
  pts = inside_points((x,), rng, 1000)
  np.testing.assert_allclose(evaluate(f, pts)["y"], CubicHermiteSpline(x, y, steffen_reference(x, y))(pts), rtol=0, atol=1e-13 * scale(y))
  u = np.linspace(0.0, 1.0, 41)
  for trial in range(10_000):  # monotone data: a monotone interpolant that never leaves an interval's range
    n = int(rng.integers(3, 12))
    xs = np.cumsum(rng.uniform(0.01, 1.0, n))
    sign = 1.0 if trial % 2 else -1.0
    ys = sign * np.cumsum(rng.exponential(size=n) * (rng.uniform(size=n) > 0.3))
    slopes, secants = hermite_slopes("steffen", xs, ys), np.diff(ys) / np.diff(xs)
    ratio = np.divide(np.stack([slopes[:-1], slopes[1:]]), secants, out=np.zeros((2, n - 1)), where=secants != 0)
    assert np.all((ratio >= 0) & (ratio <= 3)), trial  # Fritsch and Carlson's sufficient square
    assert np.all(slopes[:-1][secants == 0] == 0) and np.all(slopes[1:][secants == 0] == 0), trial
    dense = interp.interpolant(xs, ys, kind="steffen").to_scipy()(xs[:-1, None] + u[None, :] * np.diff(xs)[:, None])
    assert np.all(sign * np.diff(dense, axis=1) >= -1e-12), trial
    lo, hi = np.minimum(ys[:-1], ys[1:]), np.maximum(ys[:-1], ys[1:])
    assert np.all(dense >= lo[:, None] - 1e-12) and np.all(dense <= hi[:, None] + 1e-12), trial


@pytest.mark.parametrize("kind", ["pchip", "akima", "steffen"])
def test_shape_preserving_kinds_do_not_ring_where_a_cubic_does(kind: str) -> None:
  x = np.arange(10.0)
  y = np.where(x < 5, 0.0, 1.0)
  pts = np.linspace(0.0, 9.0, 901)
  flat = evaluate(interp.interpolant(x, y, kind=kind), pts)["y"]  # ty: ignore[invalid-argument-type]
  assert flat.min() >= -1e-15 and flat.max() <= 1.0 + 1e-15
  assert evaluate(interp.interpolant(x, y, kind="cubic"), pts)["y"].min() < -0.05


def test_shape_preserving_kinds_are_1d_only() -> None:
  with pytest.raises(ValueError, match="1-D only"):
    interp.interpolant((np.arange(4.0), np.arange(3.0)), np.zeros((4, 3)), kind=("pchip", "linear"))
  with pytest.raises(ValueError, match="bc applies"):
    interp.interpolant(np.arange(4.0), np.zeros(4), kind="pchip", bc="natural")


def test_smooth_linear_is_linear_away_from_the_sites() -> None:
  rng = np.random.default_rng(4)
  g = np.sort(rng.uniform(0.0, 5.0, 8))
  y = rng.normal(size=8)
  f = interp.interpolant(g, y, kind="smooth_linear", frac=0.2)
  step = 0.2 * np.min(np.diff(g))
  mid = 0.5 * (g[:-1] + g[1:])
  away = np.concatenate([np.linspace(g[i] + step, g[i + 1] - step, 7) for i in range(7)])
  np.testing.assert_allclose(evaluate(f, away)["y"], np.interp(away, g, y), rtol=0, atol=1e-13)
  assert np.abs(evaluate(f, g[1:-1])["y"] - y[1:-1]).max() > 1e-3  # rounded at the sites, not through them
  np.testing.assert_allclose(evaluate(f, mid)["y"], np.interp(mid, g, y), rtol=0, atol=1e-13)
  with pytest.raises(ValueError, match="frac"):
    interp.interpolant(g, y, kind="smooth_linear", frac=0.5)


def test_pspline_is_the_penalized_least_squares_fit() -> None:
  from scipy.interpolate import BSpline

  from scaly.interp.fit import pspline_fit

  rng = np.random.default_rng(5)
  x = np.sort(rng.uniform(0.0, 1.0, 120))
  y = np.column_stack([np.sin(6 * x), np.cos(3 * x)]) + 0.1 * rng.normal(size=(120, 2))
  lam = 0.37
  f = interp.smoothing(x, y, degree=2, segments=12, penalty=3, lam=lam)
  t = f.knots[0]
  B = BSpline.design_matrix(x, t, 2).toarray()
  D = np.diff(np.eye(B.shape[1]), n=3, axis=0)
  want = np.linalg.lstsq(np.vstack([B, np.sqrt(lam) * D]), np.vstack([y, np.zeros((D.shape[0], 2))]), rcond=None)[0]
  np.testing.assert_allclose(f.coeffs, want, rtol=1e-9, atol=1e-10)
  np.testing.assert_allclose(evaluate(f, x)["y"], B @ want, rtol=0, atol=1e-9)
  assert (t[2], t[-3]) == (x.min(), x.max()) and f.axes[0].search == "uniform"
  # GCV: the weight chosen is a minimum of the criterion along log lam
  coeffs, chosen = pspline_fit([t], [2], x[:, None], y, 3, "gcv")

  def gcv(weight: float) -> float:
    c = np.linalg.solve(B.T @ B + weight * D.T @ D, B.T @ y)
    trace = np.trace(np.linalg.solve(B.T @ B + weight * D.T @ D, B.T @ B))
    return 120 * np.sum((y - B @ c) ** 2) / (120 - trace) ** 2

  assert all(gcv(chosen) <= gcv(chosen * factor) for factor in (0.7, 1.4, 0.1, 10.0))
  assert chosen > 0


def test_pspline_knots_span_the_data_exactly() -> None:
  """``lo + segments * step`` can round below the largest site, which would then lie outside the base
  interval; the knots are laid so that it is exactly ``[min x, max x]``."""
  rng = np.random.default_rng(8)
  x = np.sort(rng.uniform(0.0, 1.0, 2000))
  for segments in (20, 80, 97):
    f = interp.smoothing(x, np.sin(6 * x), segments=segments, lam=1e-3)
    assert (f.axes[0].lo, f.axes[0].hi) == (x[0], x[-1]) and f.axes[0].search == "uniform"


def test_pspline_smooths_in_2d_and_from_scattered_points() -> None:
  rng = np.random.default_rng(6)
  g = (np.linspace(0.0, 1.0, 17), np.linspace(-1.0, 1.0, 13))
  truth = np.sin(3 * g[0])[:, None] * np.cos(2 * g[1])[None, :]
  noisy = truth + 0.05 * rng.normal(size=truth.shape)
  f = interp.smoothing(g, noisy, segments=(8, 6))
  nodes = np.stack(np.meshgrid(*g, indexing="ij"), axis=-1).reshape(-1, 2)
  err = np.sqrt(np.mean((evaluate(f, nodes)["y"] - truth.reshape(-1)) ** 2))
  assert err < 0.5 * np.sqrt(np.mean((noisy - truth) ** 2))
  scattered = interp.smoothing(nodes, noisy.reshape(-1), segments=(8, 6))
  np.testing.assert_allclose(scattered.coeffs, f.coeffs, rtol=1e-10, atol=1e-12)


def test_cubic_smoothing_is_make_smoothing_spline() -> None:
  from scipy.interpolate import make_smoothing_spline

  rng = np.random.default_rng(7)
  x = np.sort(rng.uniform(0.0, 2.0, 60))
  y = np.exp(-x) + 0.02 * rng.normal(size=60)
  for lam in ("gcv", 1e-3):
    f = interp.smoothing(x, y, method="cubic", lam=lam)
    ref = make_smoothing_spline(x, y, lam=None if lam == "gcv" else lam)
    pts = inside_points((f.axes[0].edges,), rng, 500)
    np.testing.assert_allclose(evaluate(f, pts)["y"], ref(pts), rtol=0, atol=1e-13)


def test_smoothing_validation() -> None:
  x = np.linspace(0.0, 1.0, 20)
  with pytest.raises(ValueError, match="method"):
    interp.smoothing(x, x, method="loess")  # ty: ignore[invalid-argument-type]
  with pytest.raises(ValueError, match="lam"):
    interp.smoothing(x, x, lam=-1.0)
  with pytest.raises(ValueError, match="1-D"):
    interp.smoothing(np.column_stack([x, x]), x, method="cubic")
  with pytest.raises(ValueError, match="points"):
    interp.smoothing(x, x[:5])
  with pytest.raises(ValueError, match="more than 2 coefficients"):
    interp.smoothing(x, x, degree=0, segments=2)
