from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.interpolate import BSpline, PPoly

import scaly as sc
from scaly.interp.grid import Axis, Search, Side, basis_derivatives, check_sites, derivative_matrix, uniform_step


def adversarial(edges: np.ndarray, rng: np.random.Generator, n_random: int) -> np.ndarray:
  """Every edge, the floats on both sides of it, both ends far outside, the non-finite values and
  random points in and around the partition."""
  width = edges[-1] - edges[0]
  return np.concatenate(
    [
      edges,
      np.nextafter(edges, -np.inf),
      np.nextafter(edges, np.inf),
      edges[0] - width * np.array([1e3, 1.0, 1e-9]),
      edges[-1] + width * np.array([1e3, 1.0, 1e-9]),
      [np.nan, np.inf, -np.inf, 1e300, -1e300],
      rng.uniform(edges[0] - 0.1 * width, edges[-1] + 0.1 * width, n_random),
    ]
  )


def expected_cell(edges: np.ndarray, x: np.ndarray, side: Side) -> np.ndarray:
  cell = np.clip(np.searchsorted(edges, x, side=side) - 1, 0, edges.size - 2).astype(np.float64)
  cell[np.isnan(x)] = 0.0
  return cell


def search_fn(axis: Axis, n: int) -> sc.Function:
  x = sc.sym("x", n)
  return sc.Function._from_exprs(f"cell_{axis.search}_{axis.side}_{n}", [x], [axis.cell(x)], ["x"], ["j"])


def grids(rng: np.random.Generator) -> dict[str, np.ndarray]:
  return {
    "linspace": np.linspace(-0.3, 0.7, 41),
    "linspace_long": np.linspace(0.0, 1.0, 1001),
    "arange": np.arange(-5.0, 6.0) * 0.1,
    "clustered": np.cumsum(np.concatenate([[0.0], rng.exponential(size=300) ** 3 + 1e-6])),
    "two_cells": np.array([1.0, 1.5, 3.0]),
    "tiny": np.array([1e-12, 2e-12, 3.5e-12, 4e-12]),
  }


@pytest.mark.parametrize("side", ["right", "left"])
@pytest.mark.parametrize("search", ["uniform", "count", "binary"])
def test_every_search_finds_the_searchsorted_cell(search: Search, side: Side) -> None:
  """1e6 adversarial and random points over six partitions, against ``searchsorted``: the tie at an
  edge follows the side, points outside get the end cells and NaN the first."""
  rng = np.random.default_rng(0)
  checked = 0
  for name, edges in grids(rng).items():
    if search == "uniform" and uniform_step(edges) is None:
      continue
    axis = Axis(edges, 0, side=side, search=search)  # degree 0 on the partition: the knots are the edges
    x = adversarial(edges, rng, 170_000 - 3 * edges.size)
    got = search_fn(axis, x.size)(x)
    np.testing.assert_array_equal(got, expected_cell(edges, x, side), err_msg=name)
    checked += x.size
  assert checked >= (1_000_000 if search != "uniform" else 500_000)


def test_a_scalar_point_searches_like_a_batch() -> None:
  edges = np.array([0.0, 0.3, 1.0, 1.1, 2.5])
  for search in ("count", "binary"):
    axis = Axis(edges, 0, search=search)
    x = sc.sym("x")
    fn = sc.Function._from_exprs(f"cell_scalar_{search}", [x], [axis.cell(x)], ["x"], ["j"])
    for v in (-1.0, 0.0, 0.3, 1.05, 2.5, 7.0, np.nan):
      assert fn(np.array(v)) == expected_cell(edges, np.array([v]), "right")[0]


def test_uniform_step_accepts_every_linspace_and_rejects_non_uniform_partitions() -> None:
  """``floor((x - e_0) / h)`` is off by one at many ``linspace`` edges (``0.30000000000000004 / 0.1``);
  the correction makes the search exact, so every one of these grids is uniform."""
  accepted = 0
  for n in range(3, 102, 7):
    for lo, hi in ((0.0, 1.0), (-0.3, 0.7), (1e3, 1e3 + 0.1), (-7.0, 13.0)):
      assert uniform_step(np.linspace(lo, hi, n)) is not None
      accepted += 1
  assert accepted == 60
  assert uniform_step(np.array([0.0, 1.0, 2.0, 4.0])) is None
  assert uniform_step(np.array([0.0, 1.0])) is None  # one cell needs no search
  for off, accepted in ((0.2, True), (0.3, False)):  # ends fixed, edges off by up to `off` cells
    wobble = np.linspace(0.0, 1.0, 11) + off * 0.1 * np.sin(np.arange(11) * 0.3 * np.pi) / np.sin(0.3 * np.pi * 5)
    assert (uniform_step(wobble) is not None) is accepted
  # The correction is exact for edges off by anything under a cell, so the uniform search, forced
  # onto a partition whose edges alternate 0.45 cell either side, still finds every cell.
  zigzag = np.linspace(0.0, 1.0, 11) + 0.045 * np.where(np.arange(11) % 2, -1.0, 1.0) * (np.arange(11) % 10 != 0)
  for side in ("right", "left"):
    axis = Axis(zigzag, 0, side=side)
    axis._uniform, axis.search = (0.0, 10.0), "uniform"
    x = adversarial(zigzag, np.random.default_rng(1), 20_000)
    np.testing.assert_array_equal(search_fn(axis, x.size)(x), expected_cell(zigzag, x, side))


def test_auto_search_is_uniform_else_binary() -> None:
  assert Axis(np.linspace(0, 1, 500), 0).search == "uniform"
  clustered = np.cumsum(np.arange(1.0, 30.0))
  assert Axis(clustered, 0).search == "binary"
  assert Axis(clustered, 0, search="count").search == "count"
  with pytest.raises(ValueError, match="uniform"):
    Axis(clustered, 0, search="uniform")


def test_derivative_matrix_and_basis_derivatives_match_scipy() -> None:
  rng = np.random.default_rng(2)
  for k in range(0, 6):
    inner = np.sort(rng.uniform(0.0, 1.0, 9))
    t = np.concatenate([np.zeros(k + 1), inner, np.ones(k + 1)])
    n = t.size - k - 1
    c = rng.normal(size=n)
    x = np.concatenate([t[k : n + 1], rng.uniform(0.0, 1.0, 50)])
    for order in range(k + 1):
      np.testing.assert_allclose(basis_derivatives(t, k, x, order) @ c, BSpline(t, c, k)(x, nu=order), rtol=1e-11, atol=1e-11)
    if k:
      dt = BSpline(t, c, k).derivative()
      np.testing.assert_allclose(derivative_matrix(t, k) @ c, dt.c[: n - 1], rtol=1e-12, atol=1e-12)


def test_local_polynomials_rebuild_the_spline_on_every_cell() -> None:
  rng = np.random.default_rng(3)
  for k in (0, 1, 2, 3, 5):
    inner = np.repeat(np.sort(rng.uniform(0.0, 1.0, 5)), [1, 2, 1, k or 1, 1])  # multiplicities
    t = np.concatenate([np.zeros(k + 1), inner, np.ones(k + 1)])
    axis = Axis(t, k)
    c = rng.normal(size=axis.n)
    cells = np.repeat(np.arange(axis.cells), 7)
    x = axis.edges[cells] + rng.uniform(0.0, 1.0, cells.size) * np.diff(axis.edges)[cells]
    powers = (x - axis.centers[cells])[:, None] ** np.arange(k + 1)[None, :]
    local = np.einsum("pam,pm->pa", axis.local[cells], powers)
    value = np.sum(local * c[axis.offsets[cells][:, None] + np.arange(k + 1)[None, :]], axis=1)
    np.testing.assert_allclose(value, BSpline(t, c, k)(x), rtol=1e-11, atol=1e-11)
    pp = PPoly.from_spline(BSpline(t, c, k))
    taylor = axis.taylor(c)
    for j in range(axis.cells):
      piece = np.searchsorted(pp.x, axis.edges[j], side="right") - 1
      shift = axis.centers[j] - pp.x[piece]  # SciPy expands at the left edge
      left = [np.polyval(np.polyder(pp.c[:, piece], m), shift) / math.factorial(m) for m in range(k + 1)]
      np.testing.assert_allclose(taylor[j], left, rtol=1e-10, atol=1e-10)


def test_offsets_name_the_first_basis_function_of_each_cell() -> None:
  t = np.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0])
  axis = Axis(t, 3)
  assert axis.edges.tolist() == [0.0, 0.5, 1.0] and axis.offsets.tolist() == [0, 2]
  refined = Axis(t, 3, edges=np.array([0.0, 0.25, 0.5, 0.75, 1.0]))
  assert refined.offsets.tolist() == [0, 0, 2, 2]


def test_check_sites_and_axis_validation() -> None:
  with pytest.raises(ValueError, match="strictly increasing"):
    check_sites([0.0, 1.0, 1.0], "grid")
  with pytest.raises(ValueError, match="finite"):
    check_sites([0.0, np.nan], "grid")
  with pytest.raises(ValueError, match="at least 2"):
    check_sites([0.0], "grid")
  with pytest.raises(ValueError, match="vector"):
    check_sites(np.zeros((2, 2)), "grid")
  with pytest.raises(ValueError, match="non-decreasing"):
    Axis(np.array([0.0, 0.0, 1.0, 0.5, 1.0, 1.0]), 1)
  with pytest.raises(ValueError, match="base interval"):
    Axis(np.array([0.0, 0.0, 0.0, 0.0]), 1)
  with pytest.raises(ValueError, match="degree"):
    Axis(np.array([0.0, 1.0]), -1)
  with pytest.raises(ValueError, match="contain every distinct knot"):
    Axis(np.array([0.0, 0.0, 0.5, 1.0, 1.0]), 1, edges=np.array([0.0, 0.4, 1.0]))
  with pytest.raises(ValueError, match="extrap"):
    Axis(np.array([0.0, 1.0]), 0, extrap="wrap")  # ty: ignore[invalid-argument-type]
  with pytest.raises(ValueError, match="search"):
    Axis(np.array([0.0, 1.0, 2.0]), 0, search="hash")  # ty: ignore[invalid-argument-type]
  assert Axis(np.array([0.0, 1.0, 2.0]), 0, extrap="linear").extrap == "clamp"  # a constant's continuation


def test_wrap_moves_by_whole_periods() -> None:
  axis = Axis(np.array([-1.0, 0.0, 2.0]), 0, extrap="periodic")
  x = sc.sym("x", 6)
  xs = np.array([-1.0, 2.0, 5.5, -4.25, 0.3, -7.0])
  got = sc.Function._from_exprs("wrap", [x], [axis.wrap(x)], ["x"], ["w"])(xs)
  np.testing.assert_allclose(got, -1.0 + np.mod(xs + 1.0, 3.0), rtol=0, atol=1e-15)
  assert math.isclose(axis.hi - axis.lo, 3.0)
