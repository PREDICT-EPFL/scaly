"""The Root and LeastSquares suite: a system of equations, a bracketed scalar equation, and least-squares problems with and without a zero residual, and the contract a method of ``scaly.roots`` meets on those it takes."""

from __future__ import annotations

from functools import cache
from typing import Any

import numpy as np

import scaly as sc

TOL = 1e-8
"""The residual a root reaches, and the gradient a least-squares point reaches, relative to the data."""

_T = np.linspace(0.0, 2.0, 12)
_Y = 1.5 * np.exp(-0.8 * _T) + 0.02 * np.sin(7.0 * _T)  # an exponential decay, not exactly one


@cache
def problems() -> dict[str, tuple[Any, np.ndarray, np.ndarray]]:
  """The problems by name, each with a start and a parameter value: ``system`` (three unknowns, no
  bounds), ``bracketed`` (one unknown between bounds), ``zero_residual`` and ``fit`` (least squares
  whose optimum leaves a residual)."""

  @sc.roots.root(vars=sc.L("z", 3), params=sc.L("p", 3), name="conformance_system")
  def system(z: sc.Expr, p: sc.Expr) -> sc.Expr:
    a = sc.const(np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 5.0]]))
    return a @ z + 0.5 * z.sin() - p

  @sc.roots.root(vars=sc.L("z", 1), params=sc.L("p", 1), name="conformance_bracketed")
  def bracketed(z: sc.Expr, p: sc.Expr) -> sc.roots.RootSpec:
    return sc.roots.RootSpec(z * z * z - 2.0 * z - p, lb=sc.const(np.array([1.0])), ub=sc.const(np.array([3.0])))

  @sc.roots.least_squares(vars=sc.L("z", 2), params=sc.L("p", 1), name="conformance_zero_residual")
  def zero_residual(z: sc.Expr, p: sc.Expr) -> sc.Expr:
    return sc.stack([p[0] * (z[1] - z[0] * z[0]), 1.0 - z[0]])  # Rosenbrock's, its minimum (1, 1)

  @sc.roots.least_squares(vars=sc.L("z", 2), params=sc.L("y", _T.size), name="conformance_fit")
  def fit(z: sc.Expr, y: sc.Expr) -> sc.Expr:
    return z[0] * (-z[1] * sc.const(_T)).exp() - y

  return {
    "system": (system, np.zeros(3), np.array([1.0, -2.0, 3.0])),
    "bracketed": (bracketed, np.array([2.0]), np.array([5.0])),
    "zero_residual": (zero_residual, np.array([-1.2, 1.0]), np.array([10.0])),
    "fit": (fit, np.array([1.0, 1.0]), _Y),
  }


def _residual_np(name: str, z: np.ndarray, p: np.ndarray) -> np.ndarray:
  if name == "system":
    a = np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 5.0]])
    return a @ z + 0.5 * np.sin(z) - p
  if name == "bracketed":
    return z**3 - 2.0 * z - p
  if name == "zero_residual":
    return np.array([p[0] * (z[1] - z[0] ** 2), 1.0 - z[0]])
  return z[0] * np.exp(-z[1] * _T) - p


def _gradient_np(name: str, z: np.ndarray, p: np.ndarray, h: float = 1e-7) -> np.ndarray:
  """``J' r`` by central differences of the NumPy residual."""
  r = _residual_np(name, z, p)
  jac = np.column_stack([(_residual_np(name, z + h * e, p) - _residual_np(name, z - h * e, p)) / (2 * h) for e in np.eye(z.size)])
  return jac.T @ r


def check(method: Any, name: str, *, label: str) -> None:
  """``method`` solves the problem ``name``: ``OK`` in at least one iteration, a root's residual or
  a least-squares point's gradient ``J' r`` below ``TOL`` relative to the data, a bracketed root
  inside its bounds; again at a second parameter value on the same compiled solver."""
  problem, z0, p = problems()[name]
  solve_fn = sc.roots.solver(problem, method, name=f"{problem.name}_{label}")
  for scale in (1.0, 1.1):
    pv = p * scale
    z, info = solve_fn(z0, pv)
    z = np.asarray(z, dtype=np.float64)
    assert sc.Status(int(info.status)).ok, sc.Status(int(info.status))
    assert int(info.iter) >= 1
    size = 1.0 + float(np.abs(pv).max())
    if name in ("system", "bracketed"):
      assert float(np.abs(_residual_np(name, z, pv)).max()) <= TOL * size, "residual"
    else:
      assert float(np.abs(_gradient_np(name, z, pv)).max()) <= 1e-6 * size, "stationarity"  # differences limit the check
    if name == "bracketed":
      assert 1.0 <= float(z[0]) <= 3.0


__all__ = ["TOL", "check", "problems"]
