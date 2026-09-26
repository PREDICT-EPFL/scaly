"""T3-4/T3-5: the generated interior-point solver end to end, against vendored PIQP.

Per problem and backend: status and iterations of each, the generated solve's median wall time
through the JIT (equilibration, initial point, loop and unscaling; inputs already NumPy arrays),
PIQP's own solve time and setup plus solve time (median of 5, C-side timers: its setup includes the
symbolic analysis the generated code does once at build time), time per iteration of each, and the
generated solver's graph-build and first-call (render and compile) times. Ends with the geometric
mean of the time ratios per backend. Usage: ``uv run internal/notes/perf_2026_09_26_tier3/t3_4_ipm.py
[names...]`` from the repository root.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "perf_2026_09_26_tier2"))

import scaly as sc  # noqa: E402
from bench_common import median_us  # noqa: E402
from scaly.solvers.ipm import QPValues, Solver  # noqa: E402
from tests.ipm import piqp_trace  # noqa: E402
from tests.ipm.problems import ipm_inputs, maros_meszaros, maros_meszaros_names, mpc_qp  # noqa: E402

ORDER = ("P", "c", "A", "b", "G", "h_l", "h_u", "x_l", "x_u")


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
  ratios: dict[str, list[tuple[float, float]]] = {"sparse": [], "dense": []}
  for qp in problems:
    s, values = ipm_inputs(qp)
    for backend in ("sparse", "dense"):
      runs = [piqp_trace.run(qp, dense=backend == "dense") for _ in range(5)]
      pq_solve = float(np.median([r.info["solve_time"] for r in runs])) * 1e6
      pq_total = float(np.median([r.info["setup_time"] + r.info["solve_time"] for r in runs])) * 1e6
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
      us = median_us(lambda: fn(args), repeat=11)
      it = int(iters)
      ratios[backend].append((us / pq_solve, us / pq_total))
      print(
        f"| {qp.name} | {s.n} / {s.p} / {s.m} | {backend} | {int(status)} / {runs[0].status} | {it} / {pq_iter} | {us:.0f} | {pq_solve:.0f} | {pq_total:.0f} "
        f"| {us / pq_solve:.2f} | {us / max(it, 1):.1f} / {pq_solve / max(pq_iter, 1):.1f} | {build:.2f} | {first:.2f} |",
        flush=True,
      )
  for backend, rs in ratios.items():
    if rs:
      solve_gm = float(np.exp(np.mean(np.log([a for a, _ in rs]))))
      total_gm = float(np.exp(np.mean(np.log([b for _, b in rs]))))
      print(f"\n{backend}: geometric mean Scaly / PIQP solve {solve_gm:.2f}, Scaly / PIQP setup + solve {total_gm:.2f} over {len(rs)} problems")


if __name__ == "__main__":
  main()
