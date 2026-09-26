"""T3-0b: the NumPy PIQP reference against vendored PIQP, problem by problem.

For every stored Maros–Mészáros problem, the infeasible set and the MPC and random families: PIQP's
iterations with the sparse and the dense backend, the reference's, whether the reference takes the
same decisions as each backend (status, iteration count, rho and delta to 1e-3), whether the two
backends follow the same path (the test for rounding sensitivity), and the reference's wall time.
Usage: ``uv run <this file>`` from the repository root.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tests.ipm import piqp_trace  # noqa: E402
from tests.ipm import reference as ref  # noqa: E402
from tests.ipm.problems import infeasible_problems, maros_meszaros, maros_meszaros_names, mpc_qp, random_qp  # noqa: E402
from tests.ipm.test_reference import _backends_agree, _decisions_match  # noqa: E402


def main() -> None:
  problems = [maros_meszaros(n) for n in maros_meszaros_names()]
  problems += [q for q, _ in infeasible_problems().values()] + [mpc_qp(4, 2, 10), mpc_qp(12, 4, 20), mpc_qp(27, 6, 30)]
  problems += [random_qp(30, 20, 5, seed=s) for s in range(4)] + [random_qp(20, 15, 5, seed=s, lp=True) for s in (7, 35)]
  print("| problem | status | PIQP sparse iter | PIQP dense iter | reference iter | = sparse | = dense | backends agree | reference (ms) |")
  print("| --- | ---: | ---: | ---: | ---: | :---: | :---: | :---: | ---: |")
  totals = np.zeros(3, dtype=int)
  for i, qp in enumerate(problems):
    s, d = piqp_trace.run(qp), piqp_trace.run(qp, dense=True)
    t0 = time.perf_counter()
    r = ref.solve(qp)
    ms = (time.perf_counter() - t0) * 1e3
    flags = (_decisions_match(s, r), _decisions_match(d, r), _backends_agree(s, d))
    if i < 48:
      totals += np.array(flags, dtype=int)
    mark = ["yes" if f else "**no**" for f in flags]
    print(f"| {qp.name} | {r.status} | {int(s.info['iter'])} | {int(d.info['iter'])} | {r.info.iter} | {mark[0]} | {mark[1]} | {mark[2]} | {ms:.0f} |", flush=True)
  print(f"\nMaros–Mészáros: reference = sparse on {totals[0]}/48, = dense on {totals[1]}/48; the backends agree on {totals[2]}/48.")


if __name__ == "__main__":
  main()
