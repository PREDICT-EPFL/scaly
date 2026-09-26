"""T3-0a: vendored PIQP 0.6.2 as the Tier 3 baseline.

For every stored Maros–Mészáros problem and three linear-MPC sizes, run the trace driver with the
sparse and the dense backend and print status, iterations and PIQP's own setup and solve times
(median of 5 runs; process start-up is not included). Usage: ``uv run <this file>`` from the
repository root.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tests.solvers.ipm import piqp_trace  # noqa: E402
from tests.solvers.ipm.problems import maros_meszaros, maros_meszaros_names, mpc_qp  # noqa: E402


def row(qp) -> str:
  cells = [qp.name, str(qp.n), str(qp.A.shape[0]), str(qp.G.shape[0])]
  for dense in (False, True):
    runs = [piqp_trace.run(qp, dense=dense) for _ in range(5)]
    status, iters = runs[0].status, int(runs[0].info["iter"])
    solve = np.median([r.info["solve_time"] for r in runs]) * 1e6
    setup = np.median([r.info["setup_time"] for r in runs]) * 1e6
    cells += [str(status), str(iters), f"{setup:.0f}", f"{solve:.0f}"]
  return "| " + " | ".join(cells) + " |"


def main() -> None:
  print("| problem | n | p | m | sparse status | iter | setup (us) | solve (us) | dense status | iter | setup (us) | solve (us) |")
  print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
  problems = [maros_meszaros(name) for name in maros_meszaros_names()] + [mpc_qp(*a) for a in ((4, 2, 10), (12, 4, 20), (27, 6, 30))]
  for qp in sorted(problems, key=lambda q: q.n + q.A.shape[0] + q.G.shape[0]):
    print(row(qp), flush=True)


if __name__ == "__main__":
  main()
