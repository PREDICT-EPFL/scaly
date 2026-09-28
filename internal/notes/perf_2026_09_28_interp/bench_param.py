"""Derivatives in the coefficients and in the data, timed in C against CasADi.

    uv run internal/notes/perf_2026_09_28_interp/bench_param.py [--rounds 7] [--reps 100]

Two studies, each a Function of the coefficients (or data) evaluated at points fixed now:

- `coeff_jac`: a 1-D cubic B-spline whose `n` coefficients are the input, at `m` points: the values
  and the Jacobian in the coefficients, (a) at symbolic points, a batch map whose Jacobian rows are
  structurally dense (`sc.jacobian`, an `m x n` output), (b) at points known now through `at()`, a
  sparse product whose Jacobian has `4m` entries (`sc.sparse_jacobian`), and (c) CasADi 3.8's
  parametric `bspline` interpolant, inlined (its default derivative in the data is zero), mapped
  over the points, with its dense Jacobian.
- `fit_jac`: a cubic interpolant whose 300 data are the input, at one point, value and gradient in
  the data: the fit as a dense constant map (`DENSE_FIT` raised) against the tridiagonal solve in
  two scans (the default above 256 sites).
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from _harness import BUILD, ROOT, build_casadi, build_scaly, time_cell

OUT = BUILD / "param"


def coeff_cells(n: int, m: int) -> dict:
  import casadi as ca

  import scaly as sc
  from scaly import interp

  rng = np.random.default_rng(n)
  t = np.concatenate([np.zeros(4), np.linspace(0.0, 1.0, n - 2)[1:-1], np.ones(4)])
  pts = np.sort(rng.uniform(0.0, 1.0, m))
  c = sc.sym("c", n)
  f = interp.BSpline(t, c, 3)
  y_sym = f(sc.const(pts))
  symbolic = sc.Function._from_exprs(f"cj_sym_{n}_{m}", [c], [y_sym, sc.jacobian(y_sym, c)], ["c"], ["y", "J"])
  at = f.at(pts)
  sj = sc.sparse_jacobian(at, c)
  static = sc.Function._from_exprs(f"cj_at_{n}_{m}", [c], [at, sj.values], ["c"], ["y", "J"])
  x, coeffs = ca.MX.sym("x"), ca.MX.sym("c", n)
  node = ca.Function("node", [x, coeffs], [ca.bspline(x, coeffs, [list(t)], [3], 1, {"inline": True})])
  ys = node.map(m)(ca.DM(pts).T, ca.repmat(coeffs, 1, m)).T
  casadi = ca.Function(f"cj_ca_{n}_{m}", [coeffs], [ys, ca.jacobian(ys, coeffs)])
  return {"scaly_symbolic": symbolic, "scaly_at": static, "casadi_inline": casadi, "_input": rng.normal(size=n)}


def fit_cells(n: int) -> dict:
  import scaly as sc
  from scaly import interp
  from scaly.interp import fit

  g = np.linspace(0.0, 10.0, n)
  out = {}
  for label, threshold in (("dense_map", 10**9), ("scans", fit.DENSE_FIT)):
    saved, fit.DENSE_FIT = fit.DENSE_FIT, threshold
    try:
      y = sc.sym("y", n)
      value = interp.interpolant(g, y, kind="cubic", name=f"fj_{label}")(sc.const(3.14159))
      out[f"scaly_{label}"] = sc.Function._from_exprs(f"fj_{label}_{n}", [y], [value, sc.gradient(value, y)], ["y"], ["v", "g"])
    finally:
      fit.DENSE_FIT = saved
  out["_input"] = np.sin(g)
  return out


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--rounds", type=int, default=7)
  parser.add_argument("--reps", type=int, default=100)
  args = parser.parse_args()
  studies = {f"coeff_jac_{n}x{m}": coeff_cells(n, m) for n, m in ((20, 200), (100, 2000))} | {"fit_jac_300": fit_cells(300)}
  metas: dict[tuple[str, str], dict] = {}
  for study, cells in studies.items():
    arg = cells["_input"]
    for label, fn in cells.items():
      if label.startswith("_"):
        continue
      build = build_casadi if label.startswith("casadi") else build_scaly
      metas[study, label] = build(fn, [arg], OUT / study / label)
  best: dict[tuple[str, str], float] = {}
  for _ in range(args.rounds):
    for (study, label), meta in metas.items():
      t, values = time_cell(OUT / study / label, meta, args.reps, 10)
      best[study, label] = min(best.get((study, label), np.inf), t)
  print("| study | variant | ns per call | C lines |")
  print("| --- | --- | --- | --- |")
  rows = []
  for (study, label), t in best.items():
    print(f"| {study} | {label} | {t:.0f} | {metas[study, label]['c_lines']} |")
    rows.append({"study": study, "variant": label, "ns": t, **metas[study, label]})
  (OUT / "results.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
  sys.path.insert(0, str(ROOT))
  main()
