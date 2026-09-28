"""Parity with CasADi 3.8's ``interpolant`` wherever it has the feature, and its quirks pinned as
expected behaviour, so that an upgrade changing them is noticed."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from scaly import interp

from .helpers import evaluate, inside_points, numbers, scale

ca = pytest.importorskip("casadi")


def ca_values(values: np.ndarray) -> list[float]:
  """CasADi's data order: the grid's first axis varies fastest."""
  return list(np.ravel(values, order="F"))


def ca_eval(fn, points: np.ndarray) -> np.ndarray:
  pts = points.reshape(points.shape[0], -1)
  return np.asarray(fn.map(pts.shape[0])(pts.T)).reshape(-1)


def grid(rng: np.random.Generator, dims: tuple[int, ...]) -> tuple[np.ndarray, ...]:
  """Non-uniform but well spaced: sites a random fifth of a cell off a uniform grid. Close sites
  make a high-degree fit ill-conditioned, and two solvers of it then differ far above rounding."""
  out = []
  for d, n in enumerate(dims):
    g = np.linspace(0.0, 1.0 + d, n)
    g[1:-1] += rng.uniform(-0.2, 0.2, n - 2) * (g[1] - g[0])
    out.append(g)
  return tuple(out)


@pytest.mark.parametrize("dims", [(17,), (9, 7), (6, 5, 4)])
def test_linear_matches_inside_and_extended(dims: tuple[int, ...]) -> None:
  rng = np.random.default_rng(len(dims))
  g = grid(rng, dims)
  v = rng.normal(size=dims)
  f = interp.interpolant(g if len(g) > 1 else g[0], v, kind="linear")
  F = ca.interpolant("lin", "linear", [list(a) for a in g], ca_values(v))
  x = inside_points(g, rng, 400)
  x = x if x.ndim > 1 else x
  outside = np.array([[a[0] - 0.7 * (a[-1] - a[0]) for a in g], [a[-1] + 1.3 for a in g], [a[0] - 0.1 for a in g]])
  pts = np.concatenate([x.reshape(-1, len(g)), outside])
  pts = pts[:, 0] if len(g) == 1 else pts
  np.testing.assert_allclose(evaluate(f, pts)["y"], ca_eval(F, pts), rtol=0, atol=1e-13 * scale(v))


@pytest.mark.parametrize("k", [1, 3, 5])
@pytest.mark.parametrize("dims", [(13,), (8, 9), (7, 6, 6)])
def test_bspline_evaluation_matches_casadis_bspline_node(k: int, dims: tuple[int, ...]) -> None:
  """The same knots and coefficients through CasADi's ``bspline`` node: the evaluation alone."""
  rng = np.random.default_rng(10 * k + len(dims))
  g = grid(rng, dims)
  f = interp.interpolant(g if len(g) > 1 else g[0], rng.normal(size=dims), kind="spline", degree=k)
  x = ca.MX.sym("x", len(g))
  F = ca.Function("bsn", [x], [ca.bspline(x, ca.DM(ca_values(numbers(f.coeffs))), [list(t) for t in f.knots], list(f.degree), 1, {})])
  pts = inside_points(g, rng, 300)
  np.testing.assert_allclose(evaluate(f, pts)["y"], ca_eval(F, pts), rtol=0, atol=1e-13 * scale(numbers(f.coeffs)))


@pytest.mark.parametrize("k", [1, 3, 5])
@pytest.mark.parametrize("dims", [(13,), (8, 9), (7, 6, 6)])
def test_bspline_interpolant_matches_within_casadis_own_fit_error(k: int, dims: tuple[int, ...]) -> None:
  """``interpolant("bspline", degree=k)`` fits the same not-a-knot spline, but CasADi's fit is the
  less accurate: it misses its own data by up to 1.6e-10 (3-D, degree 5) where SciPy's per-axis
  solves miss by 4e-14. The two agree to 1e-13, or to ten times CasADi's miss where that is larger."""
  rng = np.random.default_rng(10 * k + len(dims))
  g = grid(rng, dims)
  v = rng.normal(size=dims)
  f = interp.interpolant(g if len(g) > 1 else g[0], v, kind="spline", degree=k)
  F = ca.interpolant("bs", "bspline", [list(a) for a in g], ca_values(v), {"degree": [k] * len(g)})
  nodes = np.array(list(itertools.product(*g))) if len(g) > 1 else g[0]
  ours = np.abs(evaluate(f, nodes)["y"].reshape(-1) - v.reshape(-1)).max()
  theirs = np.abs(ca_eval(F, nodes) - v.reshape(-1)).max()
  assert ours <= max(theirs, 1e-13 * scale(v))
  pts = inside_points(g, rng, 300)
  np.testing.assert_allclose(evaluate(f, pts)["y"], ca_eval(F, pts), rtol=0, atol=max(1e-13 * scale(v), 10 * theirs))
  if k == 3:
    cubic = interp.interpolant(g if len(g) > 1 else g[0], v, kind="cubic")
    np.testing.assert_allclose(evaluate(cubic, pts)["y"], evaluate(f, pts)["y"], rtol=0, atol=1e-13 * scale(v))


@pytest.mark.parametrize("frac", [0.1, 0.35])
@pytest.mark.parametrize("dims", [(11,), (8, 6), (5, 4, 6)])
def test_smooth_linear_matches(frac: float, dims: tuple[int, ...]) -> None:
  """CasADi's ``smooth_linear`` builds no fit to solve (the coefficients are the linear interpolant at
  the Greville points), so the two agree to rounding."""
  rng = np.random.default_rng(len(dims))
  g = tuple(np.sort(rng.uniform(0.0, 1.0 + d, n)) for d, n in enumerate(dims))
  v = rng.normal(size=dims)
  f = interp.interpolant(g if len(g) > 1 else g[0], v, kind="smooth_linear", frac=frac)
  opts = {"algorithm": "smooth_linear", "smooth_linear_frac": frac}
  F = ca.interpolant("sl", "bspline", [list(a) for a in g], ca_values(v), opts)
  pts = inside_points(g, rng, 400)
  np.testing.assert_allclose(evaluate(f, pts)["y"], ca_eval(F, pts), rtol=0, atol=1e-12 * scale(v))


def test_casadi_quirks_are_what_the_comparisons_assume() -> None:
  """The bspline interpolant is zero outside its grid, its derivative with respect to the data is
  zero unless inlined, and it refuses degrees 2 and 4."""
  g = np.linspace(0.0, 1.0, 8)
  v = np.sin(3 * g) + 2.0
  F = ca.interpolant("bs", "bspline", [list(g)], list(v))
  assert float(F(1.5)) == 0.0 and float(F(-0.5)) == 0.0
  for k in (2, 4):
    with pytest.raises(RuntimeError):
      ca.interpolant("bs", "bspline", [list(g)], list(v), {"degree": [k]})
  x, data = ca.MX.sym("x"), ca.MX.sym("v", g.size)
  lookup = ca.interpolant("par", "linear", [list(g)], 1)
  jac = ca.Function("j", [x, data], [ca.jacobian(lookup(x, data), data)])
  assert np.all(np.asarray(jac(0.37, v)) == 0.0)
  inlined = ca.interpolant("par_inline", "linear", [list(g)], 1, {"inline": True})
  jac_inline = ca.Function("ji", [x, data], [ca.jacobian(inlined(x, data), data)])
  assert np.count_nonzero(np.asarray(jac_inline(0.37, v))) == 2
