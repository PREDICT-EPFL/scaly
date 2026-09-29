# /// script
# requires-python = ">=3.12"
# dependencies = ["scaly", "scaly-ipopt"]
#
# [tool.ty.environment]
# extra-paths = ["."]  # the modules beside this script, which it imports
# ///
"""Minimum-effort rocket flight: reach position 10 at rest after 50 thrust intervals (Scaly).

Each interval is 20 explicit Euler steps of s' = v, v' = (u - 0.05 v^2) / m, m' = -0.1 u^2; the
controls are the only variables (single shooting). The 50 intervals are a ``scan``, one loop in C.

After casadi/docs/examples/python/rocket.py.
"""

import numpy as np

import scaly as sc
from scaly import integrators as si
from _common import scaly_ipopt_options, show

NU = 50  # control intervals
DT = 0.01  # Euler step, 20 per interval


@sc.function(3, 1, output="xdot")
def f(x, u):
  v, m = x[1], x[2]  # x = (position, speed, mass)
  return sc.stack([v, (u[0] - 0.05 * v * v) / m, -0.1 * u[0] * u[0]])


F = si.explicit(f, "euler", dt=20 * DT, steps=20, name="F")  # F(x, u) -> xnext over one interval


@sc.opt.problem(vars=sc.L("U", NU))
def rocket(U):
  (X,) = sc.scan(F, sc.const(np.array([0.0, 0.0, 1.0])), [(U, 0, 1)], length=NU)
  return sc.opt.ProblemSpec(minimize=sc.sumsqr(U), eq=(X[0:2] - sc.const(np.array([10.0, 0.0])),), lb=sc.const(-0.5), ub=sc.const(0.5))


def build(verbose: bool = False):
  solve = sc.opt.solver(rocket, sc.opt.IPOPT(options=scaly_ipopt_options(verbose, tol=1e-10)))

  def run():
    U, lam_U, lam_g, _, _ = solve(np.full(NU, 0.4), np.zeros(NU), np.zeros(2), np.zeros(0), ())
    stats = sc.opt.solver_stats(solve)
    return {"f": np.array([stats.obj]), "u": U, "lam_u": lam_U, "lam_g": lam_g, "iter": np.array([stats.iter])}

  return run


if __name__ == "__main__":
  show(build(verbose=True)())
