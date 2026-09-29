# /// script
# requires-python = ">=3.12"
# dependencies = ["casadi"]
#
# [tool.ty.environment]
# extra-paths = ["."]  # the modules beside this script, which it imports
# ///
"""Lookup tables evaluated at batches of points: values, gradients and a Hessian (CasADi).

Three tables built by ``interpolant`` from gridded data:

- a cubic spline through 1 000 non-uniform sites (``bspline``, ``lookup_mode="binary"``);
- a bicubic spline on a uniform 64 x 64 grid (``bspline``, ``lookup_mode="binary"``: ``"exact"``
  is refused, since a not-a-knot spline's knots are not the uniform grid), with its Hessian;
- a trilinear table on a uniform 20 x 20 x 20 grid (``linear``, ``lookup_mode="exact"``).

Each is evaluated at 10 000 random points and at every grid node, one ``map`` per batch. CasADi's
``bspline`` is zero outside its grid, so its points stay inside; the trilinear table is also
evaluated at 1 000 points up to half its width outside, where both sides continue it linearly.
"""

import casadi as ca
import numpy as np

from _common import as_arrays, casadi_jit, show
from _data import lut_nodes, lut_points, lut_tables

N = 10_000


def table(name, grid, values, method, lookup):
  opts = {"lookup_mode": [lookup] * len(grid)}
  return ca.interpolant(name, method, [list(g) for g in grid], np.ravel(values, order="F"), opts)


def evaluator(name, F, dim, n, hessian=False):
  """One mapped call of the table's value, gradient and (if asked) Hessian at a point."""
  x = ca.MX.sym("x", dim)
  y = F(x)
  outs = [y, ca.gradient(y, x)] + ([ca.hessian(y, x)[0]] if hessian else [])
  point = ca.Function(f"{name}_point", [x], outs)
  X = ca.MX.sym("X", dim, n)
  return ca.Function(name, [X], point.map(n).call([X]), casadi_jit(expand=False))


def build(verbose: bool = False):
  tables = lut_tables()
  (g1,), y1 = tables["1d"]
  grid2, y2 = tables["2d"]
  grid3, y3 = tables["3d"]
  F1 = table("cubic_1d", (g1,), y1, "bspline", "binary")
  F2 = table("bicubic_2d", grid2, y2, "bspline", "binary")
  F3 = table("trilinear_3d", grid3, y3, "linear", "exact")
  points = {
    "1d": lut_points((g1,), N, 1),
    "1d_nodes": lut_nodes((g1,)),
    "2d": lut_points(grid2, N, 2),
    "2d_nodes": lut_nodes(grid2),
    "3d": lut_points(grid3, N, 3),
    "3d_nodes": lut_nodes(grid3),
    "3d_out": lut_points(grid3, 1000, 4, margin=0.5),
  }
  evaluators = {
    key: evaluator(f"lut_{key}", {"1": F1, "2": F2, "3": F3}[key[0]], int(key[0]), p.shape[0], hessian=key == "2d")
    for key, p in points.items()
  }

  def run():
    out = {}
    for key, fn in evaluators.items():
      pts = points[key]
      y, g, *h = fn(pts.T)
      out[f"y_{key}"] = np.asarray(y).reshape(-1)
      out[f"g_{key}"] = np.asarray(g).T  # (n, D), a row per point
      if h:
        out[f"h_{key}"] = np.asarray(h[0]).reshape(2, -1, 2).transpose(1, 0, 2)  # (n, 2, 2)
    return as_arrays(out)

  return run


if __name__ == "__main__":
  show(build(verbose=True)())
