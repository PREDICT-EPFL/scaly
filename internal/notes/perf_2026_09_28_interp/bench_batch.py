"""Batches: one call evaluating many random points, per point, against CasADi's map and SciPy.

    uv run internal/notes/perf_2026_09_28_interp/bench_batch.py [--rounds 5]

A batch of `N` uniformly random points per call (so the table is read at random, unlike the one
repeated point of `bench_eval.py`), values only:

- a 1-D cubic on 1 000 clustered sites and a 2-D bicubic on 64 x 64, `N` = 1e3, 1e4, 1e5, 1e6:
  Scaly's batch (one map of the point's Function), CasADi's `interpolant` mapped serially (`map`,
  and `map(..., "unroll")` up to 1e4), and SciPy called from Python for context;
- a 2-D bicubic on 256 x 256 at `N` = 1e5 with each strategy: per-cell polynomials (8.4 MB of
  table, past the caches) against local bases (0.5 MB), which sets `PP_BUDGET`.

Scaly and CasADi are timed in C (`time_entry.c`), one call per sample; ns per point reported.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from _harness import BUILD, ROOT, build_casadi, build_scaly, time_cell

OUT = BUILD / "batch"


def clustered(n: int) -> np.ndarray:
  u = np.linspace(0.0, 1.0, n)
  return u + 0.35 * np.sin(np.pi * u) ** 2 * (u - 0.5)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--rounds", type=int, default=5)
  args = parser.parse_args()
  import casadi as ca

  import scaly as sc
  from scaly import interp
  from scipy.interpolate import RegularGridInterpolator, make_interp_spline

  rng = np.random.default_rng(0)
  g1 = clustered(1000)
  y1 = np.sin(6 * g1)
  g2 = np.arange(64) / 64.0
  z2 = np.sin(3 * g2)[:, None] * np.cos(2 * g2)[None, :]
  g3 = np.linspace(0.0, 1.0, 256)
  z3 = rng.normal(size=(256, 256))
  cases: list[tuple[str, tuple[np.ndarray, ...], np.ndarray, str, int, str]] = []
  for n in (1_000, 10_000, 100_000, 1_000_000):
    cases.append(("1d_cubic_1000", (g1,), y1, "cubic", n, "auto"))
    cases.append(("2d_cubic_64", (g2, g2), z2, "cubic", n, "auto"))
  for strategy in ("pp", "basis"):
    cases.append(("2d_cubic_256", (g3, g3), z3, "cubic", 100_000, strategy))
  cells: dict[str, dict] = {}
  scipy_ns: dict[str, float] = {}
  for base, grid, values, kind, n, strategy in cases:
    label = f"{base}_{n}_{strategy}"
    pts = np.column_stack([rng.uniform(g[0], g[-1], n) for g in grid])
    arg = pts[:, 0] if len(grid) == 1 else pts
    f = interp.interpolant(grid if len(grid) > 1 else grid[0], values, kind=kind, strategy=strategy)  # ty: ignore[invalid-argument-type]
    x = sc.sym("x", arg.shape)
    fn = sc.Function._from_exprs(f"bb_{label}", [x], [f(x)], ["x"], ["y"])
    cells[f"{label}/scaly"] = {"meta": build_scaly(fn, [arg], OUT / label / "scaly"), "n": n, "folder": OUT / label / "scaly"}
    if strategy == "auto":
      method = "bspline"
      itp = ca.interpolant(f"ca_{base}", method, [list(g) for g in grid], np.ravel(values, order="F"))
      xs = ca.MX.sym("x", len(grid), n)
      for mode in ("serial", "unroll") if n <= 10_000 else ("serial",):
        mapped = ca.Function(f"cb_{label}_{mode}", [xs], [itp.map(n, mode)(xs)])
        cells[f"{label}/casadi_{mode}"] = {"meta": build_casadi(mapped, [pts.T.reshape(-1, order="F").copy()], OUT / label / f"casadi_{mode}"), "n": n, "folder": OUT / label / f"casadi_{mode}"}
      ref = make_interp_spline(grid[0], values, k=3) if len(grid) == 1 else RegularGridInterpolator(grid, values, method="cubic_legacy")
      t0 = time.perf_counter()
      ref(arg)
      scipy_ns[label] = (time.perf_counter() - t0) * 1e9 / n
  best: dict[str, float] = {}
  for _ in range(args.rounds):
    for key, cell in cells.items():
      t, _ = time_cell(cell["folder"], cell["meta"], 3, 1)
      best[key] = min(best.get(key, np.inf), t / cell["n"])
  print("| batch | Scaly ns/point | CasADi serial | CasADi unroll | SciPy (Python) |")
  print("| --- | --- | --- | --- | --- |")
  rows = []
  for base, _, _, _, n, strategy in cases:
    label = f"{base}_{n}_{strategy}"
    row = [f"{best[f'{label}/scaly']:.1f}"]
    for mode in ("serial", "unroll"):
      row.append(f"{best[f'{label}/casadi_{mode}']:.1f}" if f"{label}/casadi_{mode}" in best else "—")
    row.append(f"{scipy_ns[label]:.0f}" if label in scipy_ns else "—")
    print(f"| {base} N={n} ({strategy}) | " + " | ".join(row) + " |")
    rows.append({"label": label, **{k.split("/")[1]: v for k, v in best.items() if k.startswith(label + "/")}, "scipy": scipy_ns.get(label)})
  (OUT / "results.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
  sys.path.insert(0, str(ROOT))
  main()
