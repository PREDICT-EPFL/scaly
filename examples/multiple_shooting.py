import numpy as np

import scaly as sc

N = 20  # the decision vector w stacks N + 1 states of size 2, then N controls


@sc.function(sc.arg("z", 2), sc.arg("u", 1), sc.arg("znext", 2), outputs=sc.arg("defect"))
def defect(z: sc.Expr, u: sc.Expr, znext: sc.Expr) -> sc.Expr:
  return z + 0.1 * sc.concat([z[1:], u]) - znext


@sc.problem(vars=sc.arg("w", 3 * N + 2), params=sc.arg("z0", 2))
def multiple_shooting(w: sc.Expr, z0: sc.Expr) -> sc.ProblemSpec:
  zs, us = w[: 2 * N + 2], w[2 * N + 2 :]
  defects = sc.vmap(defect, N)(zs[:-2], us, zs[2:]).vec()  # one loop, not N copies
  return sc.ProblemSpec(minimize=sc.sumsqr(zs) + 0.1 * sc.sumsqr(us), eq=(zs[:2] - z0, defects))


solve = sc.solver(multiple_shooting, "ipopt")
w_opt, *_ = solve(np.array([1.0, 0.0]))
