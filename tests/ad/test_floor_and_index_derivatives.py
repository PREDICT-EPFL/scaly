from __future__ import annotations

import numpy as np
import pytest

import scaly as sc

T = 2.0


def _wrap(x: sc.Expr, rounding: str) -> sc.Expr:
  return x - T * getattr(x / T, rounding)()


def _np_wrap(x: np.ndarray, rounding: str) -> np.ndarray:
  return x - T * getattr(np, rounding)(x / T)


@pytest.mark.parametrize("rounding", ["floor", "ceil"])
def test_periodic_wrap_differentiates_through_floor_and_ceil(rounding: str) -> None:
  """``sin(x - T floor(x / T))``: forward, reverse and second derivatives are those of ``sin`` at the
  wrapped point, since the rounding contributes a zero derivative off its jumps."""
  x = sc.sym("x", 5)
  f = _wrap(x, rounding).sin()
  fn = sc.Function._from_exprs(
    f"wrap_{rounding}",
    [x],
    [f, sc.gradient(f.sum(), x), sc.jacobian(f, x), sc.hessian(f.sum(), x)],
    ["x"],
    ["f", "grad", "jac", "hess"],
  )
  for xv in (np.array([0.3, 1.7, -0.4, 5.1, -13.9]), np.array([1e3 + 0.25, -1e3 - 0.75, 0.999, 2.001, -2.001])):
    w = _np_wrap(xv, rounding)
    fv, gv, jv, hv = fn(xv)
    np.testing.assert_allclose(fv, np.sin(w), rtol=0, atol=1e-14)
    np.testing.assert_allclose(gv, np.cos(w), rtol=0, atol=1e-14)
    np.testing.assert_allclose(jv, np.diag(np.cos(w)), rtol=0, atol=1e-14)
    np.testing.assert_allclose(hv, np.diag(-np.sin(w)), rtol=0, atol=1e-14)


def test_floor_derivative_is_refused_under_nonsmooth_error() -> None:
  x = sc.sym("x", 3)
  f = _wrap(x, "floor").sin().sum()
  with sc.options(nonsmooth="error"):
    for build in (
      lambda: sc.gradient(f, x),
      lambda: sc.jacobian(f.reshape((1,)), x),
      lambda: sc.jvp(f, x, sc.sym("seed", 3)),
      lambda: sc.sparse_jacobian(_wrap(x, "ceil"), x),
    ):
      with pytest.raises(NotImplementedError, match="nonsmooth"):
        build()


def test_floor_records_no_dependence_in_the_sparsity_pattern() -> None:
  x = sc.sym("x", 3)
  y = sc.stack([x[0] * x[1].floor(), x[2].ceil() + 1.0, x[1] - T * (x[1] / T).floor()])

  def entries() -> list[tuple[int, int]]:
    pattern = sc.jacobian_sparsity(y, x)
    return sorted(zip(pattern.rows, pattern.cols, strict=True))

  assert entries() == [(0, 0), (2, 1)]
  with sc.options(nonsmooth="error"):  # kept, so that differentiating reaches the refusal
    assert entries() == [(0, 0), (0, 1), (1, 2), (2, 1)]


def test_floor_type_is_differentiable() -> None:
  x = sc.sym("x", 2)
  assert x.floor().type.diff and x.ceil().type.diff
  assert not sc.cast(x.floor(), "int64").type.diff


# A 1-D cubic in piecewise-polynomial form, as a flat cell-major table of 4 coefficients per cell on
# the cells [0, 1), [1, 2), [2, 3]: the index arithmetic of an interpolant.
_CELLS = 3
_COEF = np.random.default_rng(0).normal(size=(_CELLS, 4))


def _cell(x: sc.Expr) -> sc.Expr:
  return sc.cast(sc.minimum(sc.maximum(x.floor(), 0.0), float(_CELLS - 1)), "int64")


def _np_cubic(xv: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  cell = np.clip(np.floor(xv), 0, _CELLS - 1).astype(int)
  c, s = _COEF[cell], xv - cell
  value = c[:, 0] + s * (c[:, 1] + s * (c[:, 2] + s * c[:, 3]))
  return value, c[:, 1] + s * (2 * c[:, 2] + 3 * s * c[:, 3]), 2 * c[:, 2] + 6 * s * c[:, 3]


def test_broadcast_integer_index_build_differentiates_for_one_point() -> None:
  """The coefficient indices ``arange(4) + [cell] * 4`` are ``int64`` arithmetic broadcast against a
  constant: forward mode must give them no tangent rather than form a mixed-dtype product."""
  x = sc.sym("x")
  cell = _cell(x)
  c = sc.take(sc.const(_COEF.reshape(-1)), sc.const(np.arange(4), dtype="int64") + sc.stack([cell]) * 4, in_range=True)
  s = x - sc.cast(cell, "float64")
  f = c[0] + s * (c[1] + s * (c[2] + s * c[3]))
  fn = sc.Function._from_exprs(
    "int_index_point", [x], [f, sc.gradient(f, x), sc.jacobian(f, x), sc.hessian(f, x), sc.jvp(f, x, sc.const(1.0))], ["x"], ["f", "g", "j", "h", "t"]
  )
  for xv in (0.25, 1.0, 2.75, -0.5, 3.5):
    value, slope, curvature = (v[0] for v in _np_cubic(np.array([xv])))
    fv, gv, jv, hv, tv = fn(np.array(xv))
    np.testing.assert_allclose([fv, gv, jv.item(), hv.item(), tv], [value, slope, slope, curvature, slope], rtol=1e-14, atol=1e-14)


def test_broadcast_integer_index_build_differentiates_for_a_batch() -> None:
  n = 6
  x = sc.sym("x", n)
  cell = _cell(x)
  idx = (cell.reshape((n, 1)) * 4 + sc.const(np.arange(4), dtype="int64").reshape((1, 4))).reshape((4 * n,))
  c = sc.take(sc.const(_COEF.reshape(-1)), idx, in_range=True).reshape((n, 4))
  s = x - sc.cast(cell, "float64")
  f = c[:, 0] + s * (c[:, 1] + s * (c[:, 2] + s * c[:, 3]))
  sj = sc.sparse_jacobian(f, x)
  fn = sc.Function._from_exprs("int_index_batch", [x], [f, sc.jacobian(f, x), sc.hessian(f.sum(), x), sj.values], ["x"], ["f", "j", "h", "sj"])
  xv = np.array([0.1, 0.9, 1.5, 2.2, 2.999, -0.3])
  value, slope, curvature = _np_cubic(xv)
  fv, jv, hv, sjv = fn(xv)
  np.testing.assert_allclose(fv, value, rtol=1e-14, atol=1e-14)
  np.testing.assert_allclose(jv, np.diag(slope), rtol=1e-14, atol=1e-14)
  np.testing.assert_allclose(hv, np.diag(curvature), rtol=1e-13, atol=1e-13)
  assert sorted(zip(sj.sparsity.rows, sj.sparsity.cols, strict=True)) == [(i, i) for i in range(n)]
  np.testing.assert_allclose(np.asarray(sjv)[np.argsort(sj.sparsity.rows)], slope, rtol=1e-14, atol=1e-14)
