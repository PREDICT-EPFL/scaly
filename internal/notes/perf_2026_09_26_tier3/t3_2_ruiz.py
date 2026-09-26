"""T3-2: Ruiz equilibration as generated code.

Per problem: the generated equilibration's call time (JIT, median), the NumPy reference's, PIQP's
whole setup time (equilibration plus KKT symbolic analysis and allocation, so an upper bound for
its Ruiz), the number of passes, and the generated code's first-call time (render and compile).
Usage: ``uv run internal/notes/perf_2026_09_26_tier3/t3_2_ruiz.py`` from the repository root.
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
from scaly.solvers.ipm import QPValues, ruiz  # noqa: E402
from tests.ipm import piqp_trace  # noqa: E402
from tests.ipm import reference as ref  # noqa: E402
from tests.ipm.problems import ipm_inputs, maros_meszaros  # noqa: E402

ORDER = ("P", "c", "A", "b", "G", "h_l", "h_u", "x_l", "x_u")


def main() -> None:
  print("| problem | n + p + m | nnz | generated (us) | NumPy reference (us) | PIQP setup (us) | first call (s) |")
  print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
  for name in ("HS21", "QAFIRO", "DUAL1", "CVXQP1_S", "QPCBLEND", "QSC205", "PRIMAL1", "QBORE3D", "QCAPRI"):
    qp = maros_meszaros(name)
    s, values = ipm_inputs(qp)
    syms = {k: sc.sym(k, np.shape(values[k])) for k in ORDER}
    scaling = ruiz(s, QPValues.preprocess(s, **syms))
    fn = sc.Function._from_exprs(f"bench_ruiz_{name}", [syms[k] for k in ORDER], [scaling.delta, scaling.delta_b], list(ORDER), ["d", "db"])
    args = [np.ascontiguousarray(values[k], dtype=np.float64) for k in ORDER]
    t0 = time.perf_counter()
    fn._flat_numerical_call(*args)
    first = time.perf_counter() - t0
    gen = median_us(lambda: fn._flat_numerical_call(*args))
    data = ref.setup_data(qp)

    def numpy_ruiz() -> None:
      d = ref.setup_data(qp)
      ref.Ruiz(d.n, d.p, d.m).scale_data(d, False, 10)

    numpy_us = median_us(numpy_ruiz, repeat=11)
    setup = np.median([piqp_trace.run(qp).info["setup_time"] for _ in range(5)]) * 1e6
    nnz = s.P_rows.size + s.A_rows.size + s.G_rows.size
    print(f"| {name} | {data.n + data.p + data.m} | {nnz} | {gen:.1f} | {numpy_us:.0f} | {setup:.0f} | {first:.2f} |", flush=True)


if __name__ == "__main__":
  main()
