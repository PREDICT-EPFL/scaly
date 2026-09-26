"""T3-3: the KKT backends as generated code.

Per problem and backend: one factorization and the two solves of an iteration (predictor and
corrector) through the JIT, from already equilibrated data, against PIQP's average time per iteration with the same backend (which
includes the factorization, two solves and the rest of the iteration), and the first-call time.
Usage: ``uv run internal/notes/perf_2026_09_26_tier3/t3_3_kkt.py`` from the repository root.
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
from scaly.solvers.ipm import KKT, Iterate, Kernels, QPValues, ScaledQP, Scaling  # noqa: E402
from tests.ipm import piqp_trace  # noqa: E402
from tests.ipm.problems import ipm_inputs, maros_meszaros, mpc_qp  # noqa: E402

ORDER = ("P", "c", "A", "b", "G", "h_l", "h_u", "x_l", "x_u")


def main() -> None:
  print("| problem | n / p / m | backend | factor + 2 solves (us) | PIQP per iteration (us) | first call (s) |")
  print("| --- | --- | --- | ---: | ---: | ---: |")
  problems = [maros_meszaros(n) for n in ("HS21", "QAFIRO", "CVXQP1_S", "QPCBLEND", "QSC205", "QBORE3D", "QCAPRI")] + [mpc_qp(12, 4, 20)]
  for qp in problems:
    s, values = ipm_inputs(qp)
    pq = {backend: piqp_trace.run(qp, dense=backend == "dense") for backend in ("dense", "sparse")}
    for backend in ("dense", "sparse"):
      # Scaled data as inputs, so that only the factorization and the solves are timed.
      scaled_shapes = {
        "P": s.P_rows.size,
        "c": s.n,
        "A": s.A_rows.size,
        "b": s.p,
        "G": s.G_rows.size,
        "h_l": s.m,
        "h_u": s.m,
        "x_l": s.x_l_idx.size,
        "x_u": s.x_u_idx.size,
      }
      syms = {k: sc.sym(k, scaled_shapes[k]) for k in ORDER}
      xb = sc.sym("xb", s.n)
      unit = Scaling(sc.const(np.ones(s.n + s.p + s.m)), sc.const(np.ones(s.n)), sc.const(1.0))
      kkt = KKT(Kernels(s, backend, name=f"bk_{qp.name}"), ScaledQP(QPValues(**syms), xb, unit))
      size = sum(Iterate.sizes(s))
      it, rhs = sc.sym("it", (size,)), sc.sym("rhs", (size,))
      f = kkt.factor(sc.const(1e-6), sc.const(1e-4), Iterate.unflat(s, it))
      one = f.solve(Iterate.unflat(s, rhs))
      two = f.solve(Iterate.unflat(s, rhs * 0.5 + one.flat() * 0.0 + 1.0))
      fn = sc.Function._from_exprs(
        f"bk_{qp.name}_{backend}", [*(syms[k] for k in ORDER), xb, it, rhs], [one.flat(), two.flat()], [*ORDER, "xb", "it", "rhs"], ["a", "b"]
      )
      packed = dict(
        values, x_l=values["x_l"][s.x_l_idx], x_u=values["x_u"][s.x_u_idx], h_l=np.nan_to_num(values["h_l"]), h_u=np.nan_to_num(values["h_u"])
      )
      args = [np.ascontiguousarray(packed[k], dtype=np.float64) for k in ORDER] + [np.ones(s.n), np.ones(size), np.linspace(-1, 1, size)]
      t0 = time.perf_counter()
      fn._flat_numerical_call(*args)
      first = time.perf_counter() - t0
      us = median_us(lambda: fn._flat_numerical_call(*args))
      per_iter = pq[backend].info["solve_time"] / max(1, pq[backend].info["iter"]) * 1e6
      print(f"| {qp.name} | {s.n} / {s.p} / {s.m} | {backend} | {us:.1f} | {per_iter:.1f} | {first:.2f} |", flush=True)


if __name__ == "__main__":
  main()
