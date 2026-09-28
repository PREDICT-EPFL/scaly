"""Fitting costs, paid when the graph is built: `interpolant` per kind and `smoothing`, against SciPy.

    uv run internal/notes/perf_2026_09_28_interp/bench_fit.py [--repeat 5]

For each kind and size, the time `interp.interpolant` takes to return (the fit, the axis and its
search), the time to trace one evaluation (the per-cell tables are built then), and SciPy's
constructor for the same interpolant; then `interp.smoothing` (P-spline with GCV, and
`method="cubic"`) against `make_smoothing_spline`. Python wall time, fastest of `--repeat`.
Later phases add the in-graph fits of SP3 and the constrained QP of SP5.
"""

from __future__ import annotations

import argparse
import sys
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
  args = parser.parse_args()
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
      fit = best(lambda: interp.interpolant(x, y, kind=kind, **kw), args.repeat)  # noqa: B023
      point = sc.sym("p")
      trace = best(lambda: interp.interpolant(x, y, kind=kind, **kw)(point), 1) - fit  # noqa: B023
      scipy_ms = f"{1e3 * best(lambda: ref(x, y), args.repeat):.2f}" if ref else "—"  # noqa: B023
      print(f"| {kind} | {n} | {1e3 * fit:.2f} | {1e3 * trace:.1f} | {scipy_ms} |")
  print()
  print("| smoothing | m | segments | scaly ms | make_smoothing_spline ms |")
  print("| --- | --- | --- | --- | --- |")
  for m in (200, 2_000, 20_000):
    x = np.sort(rng.uniform(0.0, 1.0, m))
    y = np.sin(6 * x) + 0.1 * rng.normal(size=m)
    for segments in (20, 80):
      t = best(lambda: interp.smoothing(x, y, segments=segments), max(1, args.repeat // 2))  # noqa: B023
      print(f"| pspline gcv | {m} | {segments} | {1e3 * t:.1f} | — |")
    try:
      t = best(lambda: interp.smoothing(x, y, method="cubic"), max(1, args.repeat // 2))  # noqa: B023
      ref = best(lambda: si.make_smoothing_spline(x, y), max(1, args.repeat // 2))  # noqa: B023
      print(f"| cubic gcv | {m} | — | {1e3 * t:.1f} | {1e3 * ref:.1f} |")
    except ValueError as exc:  # SciPy's own GCV gives up on dense data ("ill-posed")
      print(f"| cubic gcv | {m} | — | SciPy fails: {exc} | SciPy fails |")


if __name__ == "__main__":
  sys.path.insert(0, str(ROOT))
  main()
