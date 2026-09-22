import numpy as np

import scaly as sc

N = 20  # the decision vector w stacks N + 1 states of size 2, then N controls


@sc.function(sc.G(sc.L("z", 2), sc.L("u", 1), sc.L("znext", 2)), sc.L("defect", ...))
def defect(inputs: tuple[sc.Expr, sc.Expr, sc.Expr]) -> sc.Expr:
  z, u, znext = inputs
  return z + 0.1 * sc.concat([z[1:], u]) - znext


@sc.problem(vars=sc.L("w", 3 * N + 2), params=sc.L("z0", 2))
def multiple_shooting(w: sc.Expr, z0: sc.Expr) -> sc.ProblemSpec:
  zs, us = w[: 2 * N + 2], w[2 * N + 2 :]
  defects = sc.vmap(defect, N, {"z": zs[:-2], "u": us, "znext": zs[2:]})  # one loop, not N copies
  return sc.ProblemSpec(minimize=sc.sumsqr(zs) + 0.1 * sc.sumsqr(us), eq=(zs[:2] - z0, defects))


solve = sc.solver(multiple_shooting, "ipopt")
w_opt, *_ = solve(np.array([1.0, 0.0]))
