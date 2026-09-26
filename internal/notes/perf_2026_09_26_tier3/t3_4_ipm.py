"""T3-4 to T3-6: the generated interior-point solver end to end, against vendored PIQP.

Per problem and backend: status and iterations of each; the generated solve's time through the JIT
(equilibration, initial point, loop and unscaling; inputs already NumPy arrays); PIQP's own solve
time (its C timer, the fastest of 51 solves in one process) and its setup (one run: it includes the
symbolic analysis the generated code does at build time) plus that solve time; time per
iteration of each; and the generated solver's graph-build and first-call (render and compile)
times. Both sides are warmed up and take the minimum of equally many samples. Ends with the
geometric mean of the time ratios per backend. Usage: ``uv run
internal/notes/perf_2026_09_26_tier3/t3_4_ipm.py [names...]`` from the repository root; set
``SCALY_CACHE_DIR`` to an empty directory for cold first calls.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "perf_2026_09_26_tier2"))

import scaly as sc  # noqa: E402
from scaly.solvers.ipm import QPValues, Solver  # noqa: E402
from tests.ipm import piqp_trace  # noqa: E402
from tests.ipm.problems import ipm_inputs, maros_meszaros, maros_meszaros_names, mpc_qp  # noqa: E402

ORDER = ("P", "c", "A", "b", "G", "h_l", "h_u", "x_l", "x_u")


SAMPLES = 50  # per side: PIQP's repeated solves and generated calls, so neither minimum has more chances


def min_us(fn, warmup: float = 0.2) -> float:
  """The fastest of ``SAMPLES`` calls after ``warmup`` seconds of calls, in microseconds. Apple
  Silicon runs the first calls of a burst on a slower core or clock: cold, a solve takes up to
  1.6x as long."""
  t_end = time.perf_counter() + warmup
  while time.perf_counter() < t_end:
    fn()
  best = float("inf")
  for _ in range(SAMPLES):
    t0 = time.perf_counter()
    fn()
    best = min(best, time.perf_counter() - t0)
  return best * 1e6


def main() -> None:
  names = sys.argv[1:]
  problems = [maros_meszaros(n) for n in (names or maros_meszaros_names())]
  if not names:
    problems += [mpc_qp(*a) for a in ((4, 2, 10), (12, 4, 20), (27, 6, 30))]
  problems.sort(key=lambda q: q.n + q.A.shape[0] + q.G.shape[0])
  print(
    "| problem | n / p / m | backend | status | iter | Scaly (us) | PIQP solve (us) | PIQP setup + solve (us) | Scaly / PIQP solve | per iter Scaly / PIQP (us) | build (s) | first call (s) |"
  )
  print("| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |")
  ratios: dict[str, list[tuple[float, float, float]]] = {"sparse": [], "dense": []}
  for qp in problems:
    s, values = ipm_inputs(qp)
    for backend in ("sparse", "dense"):
      runs = [piqp_trace.run(qp, dense=backend == "dense", repeat=SAMPLES)]
      pq_solve = runs[0].info["solve_time_min"] * 1e6
      pq_total = runs[0].info["setup_time"] * 1e6 + pq_solve
      pq_iter = int(runs[0].info["iter"])
      t0 = time.perf_counter()
      syms = {k: sc.sym(k, np.shape(values[k])) for k in ORDER}
      out = Solver(s, backend, name=f"b4_{qp.name}").solve(QPValues.preprocess(s, **syms))
      keys = ["x", "status", "iter"]
      fn = sc.Function._from_exprs(f"b4_{qp.name}_{backend}", [syms[k] for k in ORDER], [out[k] for k in keys], list(ORDER), keys)
      build = time.perf_counter() - t0
      args = tuple(values[k] for k in ORDER)
      t0 = time.perf_counter()
      _, status, iters = fn(args)
      first = time.perf_counter() - t0
      us = min_us(lambda: fn(args))
      it = int(iters)
      ratios[backend].append((us / pq_solve, us / pq_total, pq_solve))
      print(
        f"| {qp.name} | {s.n} / {s.p} / {s.m} | {backend} | {int(status)} / {runs[0].status} | {it} / {pq_iter} | {us:.0f} | {pq_solve:.0f} | {pq_total:.0f} "
        f"| {us / pq_solve:.2f} | {us / max(it, 1):.1f} / {pq_solve / max(pq_iter, 1):.1f} | {build:.2f} | {first:.2f} |",
        flush=True,
      )
  for backend, rs in ratios.items():
    for label, subset in (("all", rs), ("PIQP solve >= 50 us", [r for r in rs if r[2] >= 50.0])):
      if subset:
        solve_gm = float(np.exp(np.mean(np.log([a for a, _, _ in subset]))))
        total_gm = float(np.exp(np.mean(np.log([b for _, b, _ in subset]))))
        print(
          f"{backend} ({label}): geometric mean Scaly / PIQP solve {solve_gm:.2f}, Scaly / PIQP setup + solve {total_gm:.2f} over {len(subset)} problems"
        )


if __name__ == "__main__":
  main()
