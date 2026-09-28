"""The smallest parametric QP for PIQP, small enough to read all of its generated C.

    minimize    x0^2 + x1^2 + x0 x1 - (2 + p) x0 - 4 x1
    subject to  x0 + x1 = 1,  0 <= x <= 0.8

One variable leaf, one scalar parameter, box bounds and one equality. Eliminating ``x1 = 1 - x0``
leaves ``x0^2 + (1 - p) x0 - 3`` on ``0.2 <= x0 <= 0.8``, so the solution is
``x0 = clip((p - 1) / 2, 0.2, 0.8)``: both bounds are active at some ``p`` and neither is in between.
The example sweeps ``p`` across both regimes and checks the solver against that closed form and
against the stationarity condition ``P x + c + A^T lam_eq + lam_box = 0``.

Run as a script, it writes the oracle and the PIQP wrapper to ``examples/generated/tiny_qp/``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import scaly as sc
from scaly.codegen import write_module

GENERATED = Path(__file__).resolve().parent / "generated" / "tiny_qp"
P_SWEEP = (-1.0, 0.5, 2.0, 2.4, 4.0)


@sc.problem(vars=sc.L("x", 2), params=sc.L("p", ()))
def tiny_qp(x, p):
  return sc.ProblemSpec(
    minimize=x[0] ** 2 + x[1] ** 2 + x[0] * x[1] - (2.0 + p) * x[0] - 4.0 * x[1],
    eq=(x.sum() - 1.0,),
    lb=sc.const(np.zeros(2)),
    ub=sc.const(np.full(2, 0.8)),
  )


solve = sc.solver(tiny_qp, "piqp", name="tiny_qp_solve", options={"sparse": True})


def closed_form(p: float) -> np.ndarray:
  x0 = np.clip((p - 1.0) / 2.0, 0.2, 0.8)
  return np.array([x0, 1.0 - x0])


def main() -> dict:
  hess = np.array([[2.0, 1.0], [1.0, 2.0]])
  rows = []
  for p in P_SWEEP:
    x, lam_box, lam_eq, _ = solve(np.zeros(2), np.zeros(2), np.zeros(1), np.zeros(0), np.array(p))
    reference = closed_form(p)
    stationarity = hess @ x + np.array([-(2.0 + p), -4.0]) + lam_eq[0] + lam_box
    rows.append((p, x, reference, lam_eq[0], float(np.abs(x - reference).max()), float(np.abs(stationarity).max())))
  return {"rows": rows}


if __name__ == "__main__":
  print(f"{'p':>5}  {'x':>16}  {'closed form':>16}  {'lam_eq':>8}  {'error':>8}  {'kkt':>8}")
  for p, x, reference, lam_eq, err, kkt in main()["rows"]:
    print(f"{p:5.1f}  {np.array2string(x, precision=4):>16}  {np.array2string(reference, precision=4):>16}  {lam_eq:8.4f}  {err:8.1e}  {kkt:8.1e}")
  write_module(solve, GENERATED)
  print(f"generated C in {GENERATED}")
