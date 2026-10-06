from pathlib import Path

import numpy as np

import scaly as sc

N = 20  # the decision vector w stacks N + 1 states of size 2, then N controls


@sc.function(sc.arg("x", 2), sc.arg("u", 1), sc.arg("xnext", 2))
def defect(x: sc.Expr, u: sc.Expr, xnext: sc.Expr) -> sc.Expr:
  return x + 0.1 * sc.concat([x[1:], u]) - xnext


@sc.problem(vars=sc.arg("w", 3 * N + 2), params=sc.arg("x0", 2))
def multiple_shooting(w: sc.Expr, x0: sc.Expr) -> sc.ProblemSpec:
  xs, us = w[: 2 * N + 2].reshape((N + 1, 2)), w[2 * N + 2 :].reshape((N, 1))
  defects = sc.vmap(defect, N)(xs[:-1], us, xs[1:]).vec()  # one loop, not N copies
  return sc.ProblemSpec(minimize=sc.sumsqr(xs) + 0.1 * sc.sumsqr(us), eq=(xs[0] - x0, defects))


solve = sc.solver(multiple_shooting, "ipopt")
w_opt, *_ = solve(np.array([1.0, 0.0]))

sc.codegen.write_module(solve, Path(__file__).parent / "generated")  # generated/multiple_shooting_ipopt.h and .c
