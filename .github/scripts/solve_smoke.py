"""Solve one small problem with every solver backend, from installed wheels."""

import numpy as np
import scaly as sc


@sc.problem(vars=sc.arg("x", 2), params=sc.arg("target", 2))
def tracking(x, target):
  return sc.ProblemSpec(minimize=sc.sumsqr(x - target), lb=sc.const(0.0))


for backend in ("piqp", "ipopt", "sqp"):
  solve = sc.solver(tracking, backend)
  x, *_ = solve(np.array([-1.0, 2.0]), x0=np.ones(2))
  assert solve.stats().to_solver_status().ok, backend
  np.testing.assert_allclose(x, [0.0, 2.0], atol=1e-4)
  print(backend, "ok", x)
