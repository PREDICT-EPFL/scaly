"""SP0: is a spline evaluation composed from existing ops fast enough? Timed in C against CasADi.

    uv run internal/notes/perf_2026_09_28_interp/bench_spike.py [--rounds 7] [--reps 300]

The feasibility spike behind the kill criterion of `../interp_plan_2026_09_28.md` (section 10). A
prototype of the composite evaluation, written with `floor`, `cast`, `take`, `where` and arithmetic
and no interpolation code in the library, against CasADi 3.8's `interpolant` at the matching
lookup mode. Three cells, each one point per call:

- `cubic1d`: a not-a-knot cubic on 1 000 clustered knots, value and derivative. Scaly: a
  branch-free binary search (10 halvings on the knots padded to 1 024 with +inf), the cell's four
  piecewise-polynomial coefficients by one `take`, Horner. CasADi: `bspline`, `lookup_mode` `binary`
  and `linear`.
- `bicubic2d`: a not-a-knot bicubic on a uniform 64 x 64 grid, value and gradient. Scaly: the uniform
  search per axis, the cell's 16 coefficients by one `take`, nested Horner. CasADi: `bspline`,
  `lookup_mode` `linear` (its default at this size) and `binary`. `exact` is refused: CasADi applies
  it to the B-spline's knot vector, which not-a-knot leaves unequally spaced.

The CasADi column is the faster of its lookup modes, named (`docs/results/fairness.md`).
- `trilinear3d`: a trilinear table on a uniform 20 x 20 x 20 grid, value. Scaly: the uniform search
  per axis, the 8 corner values by one `take`, the weights. CasADi: `linear`, `lookup_mode="exact"`.

Each cell is compiled to a shared library with the JIT's flags and timed by
`../perf_2026_09_27_integrators/time_entry.c`, rounds interleaved, fastest sample kept. Values are
checked against SciPy before timing.
"""

from __future__ import annotations

import argparse
import json
import sys
from math import factorial

import numpy as np

from _harness import BUILD as ROOT_BUILD
from _harness import ROOT, build_casadi, build_scaly, time_cell

BUILD = ROOT_BUILD / "spike"


def clustered_knots(n: int) -> np.ndarray:
  u = np.linspace(0.0, 1.0, n)
  return u + 0.35 * np.sin(np.pi * u) ** 2 * (u - 0.5)  # denser in the middle, monotone


def scaly_cubic1d(knots_data: np.ndarray, ydata: np.ndarray):
  import scaly as sc
  from scipy.interpolate import CubicSpline

  cs = CubicSpline(knots_data, ydata)
  cells = knots_data.size - 1
  table = np.ascontiguousarray(cs.c[::-1].T).reshape(-1)  # cell-major, ascending powers
  levels = int(np.ceil(np.log2(cells)))
  padded = np.full(1 << levels, np.inf)
  padded[:cells] = knots_data[:cells]
  x = sc.sym("x")
  lo = sc.const(np.zeros(1), dtype="int64")
  for m in reversed(range(levels)):
    step = sc.const(np.full(1, 1 << m), dtype="int64")
    hit = x >= sc.take(sc.const(padded), lo + step, in_range=True)[0]
    lo = lo + sc.where(hit, step, sc.const(np.zeros(1), dtype="int64"))
  s = x - sc.take(sc.const(knots_data), lo, in_range=True)[0]
  c = sc.take(sc.const(table), lo * 4 + sc.const(np.arange(4), dtype="int64"), in_range=True)
  f = c[0] + s * (c[1] + s * (c[2] + s * c[3]))
  fn = sc.Function._from_exprs("spike_cubic1d", [x], [f, sc.gradient(f, x)], ["x"], ["f", "g"])
  return fn, lambda xv: np.array([cs(xv), cs(xv, 1)])


def _uniform_cell(x, t0: float, h: float, cells: int):
  """The uniform search with its one-step correction: ``floor((x - t0) / h)`` clamped, then moved
  down a cell when rounding put ``x`` below the cell's left edge (and up when at or past the next)."""
  import scaly as sc

  j = sc.minimum(sc.maximum(((x - t0) * (1.0 / h)).floor(), 0.0), float(cells - 1))
  left = t0 + j * h
  j = j - sc.cast((x < left) & (j > 0.0), "float64") + sc.cast((x >= left + h) & (j < cells - 1.0), "float64")
  return j


def scaly_bicubic2d(grid: np.ndarray, z: np.ndarray):
  import scaly as sc
  from scipy.interpolate import NdBSpline, make_interp_spline

  n = grid.size
  s1 = make_interp_spline(grid, z, k=3, axis=0)
  s2 = make_interp_spline(grid, s1.c, k=3, axis=1)
  nd = NdBSpline((s1.t, s1.t), np.moveaxis(s2.c, 0, 1), 3)  # make_interp_spline moves its axis first
  cells = n - 1
  # The piecewise-polynomial coefficients of each grid cell, 16 per cell, cell-major: the Taylor
  # coefficients at its lower-left corner (a knot, where NdBSpline takes the piece to the right).
  corners = np.stack(np.meshgrid(grid[:-1], grid[:-1], indexing="ij"), axis=-1).reshape(-1, 2)
  pp = np.stack([nd(corners, nu=(a, b)) / (factorial(a) * factorial(b)) for a in range(4) for b in range(4)], axis=-1)
  table = pp.reshape(-1)
  h, t0 = grid[1] - grid[0], grid[0]
  p = sc.sym("p", 2)
  ji = _uniform_cell(p[0], t0, h, cells)
  jj = _uniform_cell(p[1], t0, h, cells)
  si, sj = p[0] - (t0 + ji * h), p[1] - (t0 + jj * h)
  base = sc.cast(ji * float(cells) + jj, "int64")
  c = sc.take(sc.const(table), sc.stack([base]) * 16 + sc.const(np.arange(16), dtype="int64"), in_range=True).reshape((4, 4))
  rows = [c[a, 0] + sj * (c[a, 1] + sj * (c[a, 2] + sj * c[a, 3])) for a in range(4)]
  f = rows[0] + si * (rows[1] + si * (rows[2] + si * rows[3]))
  fn = sc.Function._from_exprs("spike_bicubic2d", [p], [f, sc.gradient(f, p)], ["p"], ["f", "g"])
  return fn, lambda pv: np.array([nd(pv), nd(pv, nu=(1, 0)), nd(pv, nu=(0, 1))]).ravel()


def scaly_trilinear3d(grid: np.ndarray, v: np.ndarray):
  import scaly as sc
  from scipy.interpolate import RegularGridInterpolator

  n, cells = grid.size, grid.size - 1
  h, t0 = grid[1] - grid[0], grid[0]
  p = sc.sym("p", 3)
  js = [_uniform_cell(p[d], t0, h, cells) for d in range(3)]
  us = [(p[d] - (t0 + js[d] * h)) * (1.0 / h) for d in range(3)]
  base = sc.cast((js[0] * n + js[1]) * n + js[2], "int64")
  offsets = np.array([(a * n + b) * n + c for a in (0, 1) for b in (0, 1) for c in (0, 1)])
  corner = sc.take(sc.const(v.reshape(-1)), sc.stack([base]) + sc.const(offsets, dtype="int64"), in_range=True)
  w = [sc.stack([1.0 - u, u]) for u in us]
  weights = (w[0].reshape((2, 1, 1)) * w[1].reshape((1, 2, 1)) * w[2].reshape((1, 1, 2))).reshape((8,))
  f = (corner * weights).sum()
  fn = sc.Function._from_exprs("spike_trilinear3d", [p], [f], ["p"], ["f"])
  rgi = RegularGridInterpolator((grid, grid, grid), v)
  return fn, lambda pv: rgi(pv)


def casadi_cell(name: str, method: str, grid: list[np.ndarray], values: np.ndarray, mode: str, grad: bool):
  import casadi as ca

  itp = ca.interpolant(name + "_itp", method, [list(g) for g in grid], values.ravel(order="F"), {"lookup_mode": [mode] * len(grid)})
  x = ca.MX.sym("x", len(grid))
  y = itp(x)
  outs = [y, ca.jacobian(y, x)] if grad else [y]
  return ca.Function(name, [x], outs)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--rounds", type=int, default=7)
  parser.add_argument("--reps", type=int, default=300, help="samples per cell and round")
  args = parser.parse_args()
  rng = np.random.default_rng(0)
  t1 = clustered_knots(1000)
  g2 = np.arange(64) / 64.0  # exactly equally spaced, which CasADi's "exact" lookup demands
  z2 = np.sin(3 * g2)[:, None] * np.cos(2 * g2)[None, :] + rng.normal(scale=0.01, size=(64, 64))
  g3 = np.arange(20) / 16.0
  v3 = rng.normal(size=(20, 20, 20))
  cells = {
    "cubic1d": (scaly_cubic1d(t1, np.sin(6 * t1)), ("bspline", [t1], np.sin(6 * t1), ("binary", "linear"), True), np.array([0.4137])),
    "bicubic2d": (scaly_bicubic2d(g2, z2), ("bspline", [g2, g2], z2, ("linear", "binary"), True), np.array([0.3712, 0.6093])),
    "trilinear3d": (scaly_trilinear3d(g3, v3), ("linear", [g3, g3, g3], v3, ("exact", "linear"), False), np.array([0.371, 0.911, 0.107])),
  }
  metas: dict[tuple[str, str], dict] = {}
  sides: dict[str, list[str]] = {}
  for name, ((fn, ref), (method, grid, values, modes, grad), point) in cells.items():
    metas[name, "scaly"] = build_scaly(fn, [point], BUILD / name / "scaly")
    for mode in modes:
      metas[name, mode] = build_casadi(casadi_cell(f"ca_{name}_{mode}", method, grid, values, mode, grad), [point], BUILD / name / mode)
    sides[name] = ["scaly", *modes]
    metas[name, "ref"] = {"values": np.ravel(ref(point)).tolist()}
  best: dict[tuple[str, str], float] = {}
  for _ in range(args.rounds):
    for name in cells:
      for side in sides[name]:
        t, values = time_cell(BUILD / name / side, metas[name, side], args.reps, 200)
        best[name, side] = min(best.get((name, side), np.inf), t)
        np.testing.assert_allclose(values, metas[name, "ref"]["values"], rtol=1e-11, atol=1e-11, err_msg=f"{name} {side}")
  print("| cell | Scaly ns | CasADi ns (lookup) | Scaly/CasADi | C lines Scaly / CasADi | lowering s | compile s Scaly / CasADi |")
  print("| --- | --- | --- | --- | --- | --- | --- |")
  rows = []
  for name in cells:
    mode = min(sides[name][1:], key=lambda m: best[name, m])
    s, c = metas[name, "scaly"], metas[name, mode]
    ratio = best[name, "scaly"] / best[name, mode]
    print(
      f"| {name} | {best[name, 'scaly']:.1f} | {best[name, mode]:.1f} ({mode}) | {ratio:.2f} | {s['c_lines']} / {c['c_lines']} | {s['generate_s']:.3f} | {s['compile_s']:.2f} / {c['compile_s']:.2f} |"
    )
    rows.append({"cell": name, **{f"{side}_ns": best[name, side] for side in sides[name]}, "casadi_mode": mode, **{side: metas[name, side] for side in sides[name]}})
  (BUILD / "results.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
  sys.path.insert(0, str(ROOT))
  main()
