"""Fitting costs, paid when the graph is built: `interpolant` per kind and `smoothing`, against SciPy.

    uv run internal/notes/perf_2026_09_28_interp/bench_fit.py [--repeat 5] [--only kinds|smoothing|constrained]

For each kind and size, the time `interp.interpolant` takes to return (the fit, the axis and its
search), the time to trace one evaluation (the per-cell tables are built then), and SciPy's
constructor for the same interpolant; then `interp.smoothing` (P-spline with GCV, and
`method="cubic"`) against `make_smoothing_spline`; then `interp.constrained` (SP5): the first call
at a problem size, which generates and compiles its PIQP solver into an empty cache, the calls after,
and the QP's own share, against `lsq_linear` (bounds alone) and `trust-constr` (small problems).
Python wall time, fastest of `--repeat`. The in-graph fits of SP3 are in `bench_param.py`.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time

import numpy as np

from _harness import ROOT


def best(fn, repeat: int) -> float:
  times = []
  for _ in range(repeat):
    t0 = time.perf_counter()
    fn()
    times.append(time.perf_counter() - t0)
  return min(times)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--repeat", type=int, default=5)
  parser.add_argument("--only", choices=("kinds", "smoothing", "constrained"))
  args = parser.parse_args()
  if args.only in (None, "kinds"):
    kinds(args.repeat)
  if args.only in (None, "smoothing"):
    smoothing(args.repeat)
  if args.only in (None, "constrained"):
    constrained(args.repeat)


def kinds(repeat: int) -> None:
  import scaly as sc
  from scaly import interp
  from scipy import interpolate as si

  rng = np.random.default_rng(0)
  scipy_fit = {
    "linear": lambda x, y: si.make_interp_spline(x, y, k=1),
    "cubic": lambda x, y: si.CubicSpline(x, y),
    "spline": lambda x, y: si.make_interp_spline(x, y, k=5),
    "pchip": lambda x, y: si.PchipInterpolator(x, y),
    "akima": lambda x, y: si.Akima1DInterpolator(x, y),
    "steffen": None,
    "smooth_linear": None,
  }
  print("| kind | n | interpolant ms | first trace ms | SciPy constructor ms |")
  print("| --- | --- | --- | --- | --- |")
  for n in (100, 10_000, 1_000_000):
    x = np.cumsum(rng.uniform(0.1, 1.0, n))
    y = np.cumsum(rng.uniform(0.0, 1.0, n))
    for kind, ref in scipy_fit.items():
      kw = {"degree": 5} if kind == "spline" else {}
      fit = best(lambda: interp.interpolant(x, y, kind=kind, **kw), repeat)  # noqa: B023
      point = sc.sym("p")
      trace = best(lambda: interp.interpolant(x, y, kind=kind, **kw)(point), 1) - fit  # noqa: B023
      scipy_ms = f"{1e3 * best(lambda: ref(x, y), repeat):.2f}" if ref else "—"  # noqa: B023
      print(f"| {kind} | {n} | {1e3 * fit:.2f} | {1e3 * trace:.1f} | {scipy_ms} |")
  print()


def smoothing(repeat: int) -> None:
  from scaly import interp
  from scipy import interpolate as si

  rng = np.random.default_rng(1)
  print("| smoothing | m | segments | scaly ms | make_smoothing_spline ms |")
  print("| --- | --- | --- | --- | --- |")
  for m in (200, 2_000, 20_000):
    x = np.sort(rng.uniform(0.0, 1.0, m))
    y = np.sin(6 * x) + 0.1 * rng.normal(size=m)
    for segments in (20, 80):
      t = best(lambda: interp.smoothing(x, y, segments=segments), max(1, repeat // 2))  # noqa: B023
      print(f"| pspline gcv | {m} | {segments} | {1e3 * t:.1f} | — |")
    try:
      t = best(lambda: interp.smoothing(x, y, method="cubic"), max(1, repeat // 2))  # noqa: B023
      ref = best(lambda: si.make_smoothing_spline(x, y), max(1, repeat // 2))  # noqa: B023
      print(f"| cubic gcv | {m} | — | {1e3 * t:.1f} | {1e3 * ref:.1f} |")
    except ValueError as exc:  # SciPy's own GCV gives up on dense data ("ill-posed")
      print(f"| cubic gcv | {m} | — | SciPy fails: {exc} | SciPy fails |")
  print()


def constrained(repeat: int) -> None:
  from scaly import interp
  from scipy.interpolate import BSpline as SciBSpline
  from scipy.optimize import LinearConstraint, lsq_linear, minimize

  module = sys.modules["scaly.interp.constrained"]

  built: set[tuple[int, int]] = set()

  def cold() -> None:  # an empty cache, so a first call generates and compiles its solver
    module._SOLVERS.clear()
    os.environ["SCALY_CACHE_DIR"] = tempfile.mkdtemp(prefix="bench_fit_")

  def first_s(seconds: float, n: int, rows: int) -> str:  # a size built earlier in the process is loaded, not built
    fresh = (n, rows) not in built
    built.add((n, rows))
    return f"{seconds:.2f}" if fresh else "(built above)"

  rng = np.random.default_rng(2)

  def ocv(m: int) -> tuple[np.ndarray, np.ndarray]:
    soc = np.sort(rng.uniform(0.0, 1.0, m))
    return soc, 3.0 + 0.9 * soc + 0.25 * np.tanh(12 * (soc - 0.08)) - 0.2 * np.exp(-30 * (1 - soc)) + 0.03 * rng.normal(size=m)

  def qp_ms(call) -> float:
    call()
    return 1e3 * min(s.solver_stats().t_total for s in module._SOLVERS.values() if s.solver_stats().t_total > 0)

  print("| constrained | m | n | constraints | first call s | later calls ms | QP ms | lsq_linear ms | trust-constr ms |")
  print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
  for m, knots in ((1_000, 16), (1_000, 64), (10_000, 64)):
    x, y = ocv(m)
    for label, kw in (("bounds", {"bounds": (2.6, 4.0)}), ("monotone + bounds", {"monotone": "increasing", "bounds": (2.6, 4.0)})):
      cold()
      call = lambda: interp.constrained(x, y, knots=knots, **kw)  # noqa: B023
      first = best(call, 1)
      later = best(call, repeat)
      n = knots + 3
      rows = n + (n - 1 if "monotone" in kw else 0)
      t = call().knots[0]
      B = SciBSpline.design_matrix(x, t, 3)
      lsq = f"{1e3 * best(lambda: lsq_linear(B.toarray(), y, bounds=(2.6, 4.0), method='bvls', tol=1e-12), repeat):.1f}" if label == "bounds" else "—"  # noqa: B023
      trust = "—"
      if knots == 16:
        dense = B.toarray()
        cons = [LinearConstraint(np.eye(n), 2.6, 4.0)] + ([LinearConstraint(np.diff(np.eye(n), axis=0), 0.0, np.inf)] if "monotone" in kw else [])
        run = lambda: minimize(lambda c: np.sum((dense @ c - y) ** 2), np.full(n, 3.3), jac=lambda c: 2 * dense.T @ (dense @ c - y), method="trust-constr", constraints=cons, options={"gtol": 1e-10, "xtol": 1e-12})  # noqa: B023
        trust = f"{1e3 * best(run, 1):.0f}"
      print(f"| 1-D {label} | {m} | {n} | {rows} | {first_s(first, n, rows)} | {1e3 * later:.1f} | {qp_ms(call):.2f} | {lsq} | {trust} |", flush=True)
  for knots in (8, 16):
    speed, torque = rng.uniform(0.0, 1.0, 4_000), rng.uniform(0.0, 1.0, 4_000)
    eta = 0.95 - 0.3 * (speed - 0.6) ** 2 - 0.2 * (torque - 0.5) ** 2 + 0.04 * rng.normal(size=4_000)
    cold()
    call = lambda: interp.constrained(np.column_stack([speed, torque]), eta, knots=(knots, knots), bounds=(0.0, 0.95), monotone=("increasing", "decreasing"), lam=1e-4)  # noqa: B023
    first = best(call, 1)
    later = best(call, repeat)
    n, rows = (knots + 3) ** 2, (knots + 3) ** 2 + 2 * (knots + 2) * (knots + 3)
    print(f"| 2-D monotone + bounds | 4000 | {n} | {rows} | {first_s(first, n, rows)} | {1e3 * later:.1f} | {qp_ms(call):.2f} | — | — |", flush=True)


if __name__ == "__main__":
  sys.path.insert(0, str(ROOT))
  main()
