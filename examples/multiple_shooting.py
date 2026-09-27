from pathlib import Path

import numpy as np
import scaly as sc
from scaly.codegen import write_module

N = 20  # the decision vector w stacks N + 1 states of size 2, then N controls

@sc.function(2, 1, 2)
def defect(z, u, znext):
    return z + 0.1 * sc.concat([z[1:], u]) - znext

@sc.problem(vars=sc.L("w", 3 * N + 2), params=sc.L("z0", 2))
def multiple_shooting(w, z0):
    zs, us = w[: 2 * N + 2], w[2 * N + 2 :]
    defects = sc.vmap(defect, N, [(zs, 0, 2), (us, 0, 1), (zs, 2, 2)])  # one loop, not N copies
    return sc.ProblemSpec(minimize=sc.sumsqr(zs) + 0.1 * sc.sumsqr(us), eq=(zs[:2] - z0, defects))

solve = sc.solver(multiple_shooting, "ipopt")
w_opt, *_ = solve(np.zeros(3 * N + 2), np.zeros(3 * N + 2), np.zeros(2 * N + 2), np.zeros(0), np.array([1.0, 0.0]))

write_module(solve, Path(__file__).resolve().parent / "generated" / "multiple_shooting")  # the solver as C, next to this file
