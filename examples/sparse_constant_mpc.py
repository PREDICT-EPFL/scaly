"""Solve a small linear MPC with fixed sparse dynamics and a changing target."""

import numpy as np

import scaly as sc


@sc.problem(vars=sc.arg("w", 7), params=sc.arg("target", ()))
def tracking_mpc(w: sc.Expr, target: sc.Expr) -> sc.ProblemSpec[sc.Expr]:
  states, controls = w[:4], w[4:]
  return sc.ProblemSpec(
    minimize=sc.sumsqr(states - target) + 0.1 * sc.sumsqr(controls),
    eq=(states[:1] - 1.0, states[1:] - states[:-1] - controls),
  )


solve = sc.solver(tracking_mpc, "piqp", options={"sparse": True})

for target in (0.0, 2.0):
  result = solve(np.array(target))
  assert solve.stats().to_solver_status().ok
  states, controls = result[0][:4], result[0][4:]
  print(f"target={target}: states={states.round(3)}, controls={controls.round(3)}")
