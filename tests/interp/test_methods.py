"""The interp methods: every registered method fits the spline its shorthand fits, a kind per axis
mixes them, and a method refuses the data it cannot fit, naming why."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

import scaly as sc
from scaly import interp

RNG = np.random.default_rng(7)
X = np.cumsum(RNG.uniform(0.2, 1.0, 12))
Y = np.sin(X) + 0.1 * X
POINTS = RNG.uniform(0.0, 1.0, (60, 2))
VALUES = np.sin(3 * POINTS[:, 0]) * np.cos(2 * POINTS[:, 1]) + 0.01 * RNG.standard_normal(60)

KINDS = {
  "nearest": interp.Nearest(),
  "zoh": interp.ZOH(),
  "linear": interp.Linear(),
  "cubic": interp.Cubic(),
  "spline": interp.Spline(),
  "pchip": interp.PCHIP(),
  "akima": interp.Akima(),
  "makima": interp.Makima(),
  "steffen": interp.Steffen(),
  "smooth_linear": interp.SmoothLinear(),
}


def test_every_method_is_registered() -> None:
  assert sorted(interp.REGISTRY.installed()) == sorted([*KINDS, "smoothing", "constrained"])
  for key, method in KINDS.items():
    assert interp.REGISTRY.get(key) is type(method)
  assert isinstance(interp.REGISTRY.auto(interp.Fit(X, Y)), interp.Linear)
  assert isinstance(interp.REGISTRY.auto(interp.Fit(POINTS, VALUES)), interp.Smoothing)


@pytest.mark.parametrize("kind", sorted(KINDS))
def test_a_kind_fits_its_shorthands_spline(kind: str) -> None:
  fit = interp.Fit(X, Y, extrap="clamp", name="table")
  spline = interp.solver(fit, KINDS[kind])
  assert spline.name == "table"
  assert spline.digest == interp.interpolant(X, Y, cast(Any, kind), extrap="clamp", name="table").digest


def test_options_pass_through() -> None:
  pairs: list[tuple[Any, dict[str, Any]]] = [
    (interp.Cubic(bc="natural"), dict(kind="cubic", bc="natural")),
    (interp.Spline(degree=5), dict(kind="spline", degree=5)),
    (interp.ZOH(period=X[-1] - X[0] + 1.0), dict(kind="zoh", period=X[-1] - X[0] + 1.0)),
    (interp.SmoothLinear(frac=0.3), dict(kind="smooth_linear", frac=0.3)),
  ]
  for method, options in pairs:
    assert interp.solver(interp.Fit(X, Y), method).digest == interp.interpolant(X, Y, **options).digest


def test_a_kind_per_axis_and_values_in_the_graph() -> None:
  grid = (np.linspace(0.0, 1.0, 6), np.linspace(-1.0, 2.0, 9))
  table = np.add.outer(np.sin(grid[0]), grid[1] ** 2)
  mixed = interp.solver(interp.Fit(grid, table), interp.PerAxis(interp.Linear(), interp.Cubic(bc="natural")))
  assert mixed.digest == interp.interpolant(grid, table, ("linear", "cubic"), bc=("not-a-knot", "natural")).digest
  values = sc.sym("values", table.shape)
  symbolic = interp.solver(interp.Fit(grid, values), "linear")
  point = sc.sym("point", 2)
  fn = sc.Function.from_exprs("linear_in_graph", [values, point], [symbolic(point)], ["values", "point"], ["y"])
  np.testing.assert_allclose(
    fn((table, np.array([0.3, 0.7]))),
    sc.Function.from_exprs("linear_const", [point], [interp.interpolant(grid, table)(point)], ["point"], ["y"])(np.array([0.3, 0.7])),
  )


def test_smoothing_fits_its_shorthands_spline() -> None:
  smooth = interp.solver(interp.Fit(POINTS, VALUES), interp.Smoothing(segments=(6, 5), lam=1e-3))
  assert smooth.digest == interp.smoothing(POINTS, VALUES, segments=(6, 5), lam=1e-3).digest


@pytest.mark.method("opt.piqp")
def test_constrained_fits_its_shorthands_spline_by_any_qp_method() -> None:
  noisy = np.cumsum(RNG.uniform(0.0, 1.0, 40)) + RNG.standard_normal(40)
  sites = np.linspace(0.0, 1.0, 40)
  knots = np.linspace(0.1, 0.9, 7)
  method = interp.Constrained(knots=knots, monotone="increasing", bounds=(0.0, 25.0))
  default = interp.solver(interp.Fit(sites, noisy), method)
  assert default.digest == interp.constrained(sites, noisy, knots=knots, monotone="increasing", bounds=(0.0, 25.0)).digest
  sparse = interp.solver(
    interp.Fit(sites, noisy), interp.Constrained(knots=knots, monotone="increasing", bounds=(0.0, 25.0), qp=sc.opt.PIQP(sparse=True))
  )
  grid = np.linspace(0.0, 1.0, 101)
  np.testing.assert_allclose(sparse.to_scipy()(grid), default.to_scipy()(grid), atol=1e-8)
  assert default.derivative().to_scipy()(grid).min() >= -1e-9


def test_what_a_method_refuses() -> None:
  scattered, grid2d = interp.Fit(POINTS, VALUES), interp.Fit((X, X), np.outer(Y, Y))
  with pytest.raises(ValueError, match="scattered points; an interpolant needs them on a grid"):
    interp.solver(scattered, "linear")
  with pytest.raises(ValueError, match="pchip is 1-D only"):
    interp.solver(grid2d, "pchip")
  with pytest.raises(ValueError, match="the data are an Expr"):
    interp.solver(interp.Fit(X, sc.sym("y", X.size)), "constrained")
  with pytest.raises(ValueError, match="2 methods for 1 axes"):
    interp.solver(interp.Fit(X, Y), interp.PerAxis(interp.Linear(), interp.Linear()))
  with pytest.raises(ValueError, match="bc must be one of"):
    interp.Cubic(bc="free")
  with pytest.raises(ValueError, match="degree must be 1 to 5"):
    interp.Spline(degree=7)
  with pytest.raises(TypeError, match="one interpolating method"):
    interp.PerAxis(interp.Smoothing())  # ty: ignore[invalid-argument-type]
