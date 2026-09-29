"""Van der Pol optimal control by direct single shooting, RK4 integrator (Scaly).

    minimize  int_0^10 x1^2 + x2^2 + u^2 dt
    subject to  x1' = (1 - x2^2) x1 - x2 + u,  x2' = x1,  x(0) = (0, 1),
                -1 <= u <= 1,  x1 >= -0.25 at the end of every interval

The controls on N = 20 intervals are the only variables; each interval is 4 RK4 steps, and the
N intervals are a ``scan`` that stacks each interval's end state and cost.

After casadi/docs/examples/python/direct_single_shooting.py (Joel Andersson, 2016).
"""

import numpy as np

import scaly as sc
from scaly import integrators as si
from _common import scaly_ipopt_options, show

T = 10.0  # horizon
N = 20  # control intervals
M = 4  # RK4 steps per interval
X_START = np.array([0.0, 1.0])


@sc.function(3, 1, output="xqdot")
def f(xq, u):  # the state x and, as a third entry, the cost q it accumulates: q' = x1^2 + x2^2 + u^2
  x1, x2 = xq[0], xq[1]
  return sc.stack([(1 - x2**2) * x1 - x2 + u[0], x1, x1**2 + x2**2 + u[0] ** 2])


rk4 = si.rk4(f, dt=T / N, steps=M)  # rk4(xq, u) -> xqnext over one interval


@sc.function(sc.L("x0", 2), sc.L("p", 1), output=sc.G("xf", "qf"))
def F(X, U):
  XQ = rk4(sc.concat([X, sc.const(np.zeros(1))]), U)
  return XQ[:2], XQ[2:]


@sc.function(2, 1, output=sc.G("xnext", "x_and_q"))
def interval(x, u):
  xf, qf = F(x, u)
  return xf, sc.concat([xf, qf])


@sc.function(N, output="x_and_q")
def rollout(u):
  _, x_and_q = sc.scan(interval, sc.const(X_START), [(u, 0, 1)], length=N)
  return x_and_q.reshape((N, 3))


@sc.opt.problem(vars=sc.L("u", N))
def single_shooting(u):
  x_and_q = rollout(u)
  return sc.opt.ProblemSpec(minimize=x_and_q[:, 2].sum(), ineq=(sc.opt.bounded(x_and_q[:, 0], lo=-0.25),), lb=sc.const(-1.0), ub=sc.const(1.0))


def build(verbose: bool = False):
  solve = sc.opt.solver(single_shooting, sc.opt.IPOPT(options=scaly_ipopt_options(verbose)))

  def run():
    xf_test, qf_test = F(np.array([0.2, 0.3]), np.array([0.4]))  # the original's check of one interval
    u, *_ = solve(np.zeros(N), np.zeros(N), np.zeros(0), np.zeros(N), ())
    stats = sc.opt.solver_stats(solve)
    x = np.vstack([X_START, rollout(u)[:, :2]])
    return {"xf_test": xf_test, "qf_test": qf_test, "f": np.array([stats.obj]), "u": u, "x1": x[:, 0], "x2": x[:, 1], "iter": np.array([stats.iter])}

  return run


if __name__ == "__main__":
  show(build(verbose=True)())
