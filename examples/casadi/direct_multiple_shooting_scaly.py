"""Van der Pol optimal control by direct multiple shooting, RK4 integrator (Scaly).

The problem of ``direct_single_shooting``; now the states at the interval boundaries are variables
too, and continuity is imposed as equality constraints (the "gaps"). The variables are laid out as
the CasADi original lays them out, [x_0, u_0, x_1, u_1, ..., x_N], so one ``vmap`` with stride 3
reads every interval's (x_k, u_k, x_k+1).

After casadi/docs/examples/python/direct_multiple_shooting.py (Joel Andersson, 2016).
"""

import numpy as np

import scaly as sc
from scaly import integrators as si
from _common import scaly_ipopt_options, show

T = 10.0  # horizon
N = 20  # control intervals
M = 4  # RK4 steps per interval
NW = 3 * N + 2  # [x_0, u_0, x_1, u_1, ..., x_N]
LBW = np.tile([-0.25, -np.inf, -1.0], N + 1)[:NW]
UBW = np.tile([np.inf, np.inf, 1.0], N + 1)[:NW]
LBW[:2] = UBW[:2] = [0.0, 1.0]  # x_0 fixed
W0 = np.zeros(NW)
W0[:2] = [0.0, 1.0]


@sc.function(3, 1, output="xqdot")
def f(xq, u):  # the state x and, as a third entry, the cost q it accumulates: q' = x1^2 + x2^2 + u^2
  x1, x2 = xq[0], xq[1]
  return sc.stack([(1 - x2**2) * x1 - x2 + u[0], x1, x1**2 + x2**2 + u[0] ** 2])


rk4 = si.rk4(f, dt=T / N, steps=M)  # rk4(xq, u) -> xqnext over one interval


@sc.function(sc.L("x0", 2), sc.L("p", 1), output=sc.G("xf", "qf"))
def F(X, U):
  XQ = rk4(sc.concat([X, sc.const(np.zeros(1))]), U)
  return XQ[:2], XQ[2:]


@sc.function(2, 1, 2, output="gap_and_q")
def interval(x, u, xnext):
  xf, qf = F(x, u)
  return sc.concat([xf - xnext, qf])


@sc.opt.problem(vars=sc.L("w", NW))
def multiple_shooting(w):
  out = sc.vmap(interval, N, [(w, 0, 3), (w, 2, 3), (w, 3, 3)]).reshape((N, 3))
  return sc.opt.ProblemSpec(minimize=out[:, 2].sum(), eq=(out[:, 0:2].reshape((2 * N,)),), lb=sc.const(LBW), ub=sc.const(UBW))


def build(verbose: bool = False):
  solve = sc.opt.solver(multiple_shooting, sc.opt.IPOPT(options=scaly_ipopt_options(verbose)))

  def run():
    w, _, lam_g, _, _ = solve(W0, np.zeros(NW), np.zeros(2 * N), np.zeros(0), ())
    stats = sc.opt.solver_stats(solve)
    return {"f": np.array([stats.obj]), "x1": w[0::3], "x2": w[1::3], "u": w[2::3], "lam_g": lam_g, "iter": np.array([stats.iter])}

  return run


if __name__ == "__main__":
  show(build(verbose=True)())
