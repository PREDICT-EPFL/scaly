# /// script
# requires-python = ">=3.12"
# dependencies = ["scaly"]
#
# [tool.ty.environment]
# extra-paths = ["."]  # the modules beside this script, which it imports
# ///
"""Lookup tables evaluated at batches of points: values, gradients and a Hessian (Scaly).

Three tables built by ``interp.interpolant`` from gridded data:

- a cubic spline through 1 000 non-uniform sites (``kind="cubic"``, not-a-knot as CasADi's);
- a bicubic spline on a uniform 64 x 64 grid, with its Hessian;
- a trilinear table on a uniform 20 x 20 x 20 grid (``kind="linear"``).

Each is evaluated at 10 000 random points and at every grid node, and the trilinear table also at
1 000 points up to half its width outside, where it continues linearly. A table called with a batch
``(n, D)`` is one mapped call of the point's graph, and the gradients are reverse mode through that
map: each point's value depends on its own row alone, so the gradient of the sum is the stack of the
points' gradients. The Hessian maps the point's Hessian. The search and the table layout are chosen
by ``"auto"``.
"""

import numpy as np

import scaly as sc
from _common import show
from _data import lut_nodes, lut_points, lut_tables
from scaly import interp

N = 10_000


def evaluator(name, f, dim, n, hessian=False):
  """One Function of a batch of points: values, gradients and (if asked) Hessians."""

  @sc.function(sc.L("X", (n, dim)), output=sc.G("y", "g", *(("h",) if hessian else ())), name=name)
  def batch(X):
    y = f(X[:, 0] if dim == 1 else X)
    outs = [y, sc.gradient(y.sum(), X)]
    if hessian:
      point = sc.function(sc.L("p", dim), output="h", name=f"{name}_point")(lambda p: sc.hessian(f(p), p))
      outs.append(sc.vmap(point, n, [(X.reshape((n * dim,)), 0, dim)]).reshape((n, dim, dim)))
    return tuple(outs)

  return batch


def build(verbose: bool = False):
  tables = lut_tables()
  (g1,), y1 = tables["1d"]
  grid2, y2 = tables["2d"]
  grid3, y3 = tables["3d"]
  f = {"1": interp.interpolant(g1, y1, kind="cubic"), "2": interp.interpolant(grid2, y2, kind="cubic"), "3": interp.interpolant(grid3, y3, kind="linear")}
  points = {
    "1d": lut_points((g1,), N, 1),
    "1d_nodes": lut_nodes((g1,)),
    "2d": lut_points(grid2, N, 2),
    "2d_nodes": lut_nodes(grid2),
    "3d": lut_points(grid3, N, 3),
    "3d_nodes": lut_nodes(grid3),
    "3d_out": lut_points(grid3, 1000, 4, margin=0.5),
  }
  evaluators = {key: evaluator(f"lut_{key}", f[key[0]], int(key[0]), p.shape[0], hessian=key == "2d") for key, p in points.items()}
  for key, fn in evaluators.items():
    fn(points[key])  # generate and compile

  def run():
    out = {}
    for key, fn in evaluators.items():
      y, g, *h = fn(points[key])
      out[f"y_{key}"], out[f"g_{key}"] = y, g
      if h:
        out[f"h_{key}"] = h[0]
    return out

  return run


if __name__ == "__main__":
  show(build(verbose=True)())
