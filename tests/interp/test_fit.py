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
