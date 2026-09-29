# /// script
# requires-python = ">=3.12"
# dependencies = ["scaly", "scaly-ipopt"]
#
# [tool.ty.environment]
# extra-paths = ["."]  # the modules beside this script, which it imports
# ///
"""Minimum-time race on a track with a position-dependent speed limit (Scaly).

    minimize T   subject to  p' = v,  v' = u - v  (RK4 on N intervals of T/N),
                             v <= 1 - sin(2 pi p) / 2,  0 <= u <= 1,  p(0) = v(0) = 0,  p(T) = 1

The N continuity constraints are one ``vmap`` of an RK4 defect, with T broadcast (stride 0).

After casadi/docs/examples/python/race_car.py.
"""

import numpy as np

import scaly as sc
from scaly import integrators as si
from _common import scaly_ipopt_options, show

N = 100  # control intervals


@sc.function(2, 1, output="xdot")
def f(x, u):
  return sc.stack([x[1], u[0] - x[1]])


rk4 = si.rk4(f, dt=None)  # rk4(x, u, dt) -> xnext, the interval an input


@sc.function(2, 1, 2, 1)
def defect(z, u, znext, T):
  return rk4(z, u, T[0] / N) - znext


@sc.opt.problem(vars=sc.G(sc.L("X", 2 * (N + 1)), sc.L("U", N), sc.L("T", 1)))
def race_car(variables):
  X, U, T = variables  # X stacks (position, speed) at the N + 1 grid points
  pos, speed = X.reshape((N + 1, 2))[:, 0], X.reshape((N + 1, 2))[:, 1]
  gaps = sc.vmap(defect, N, [(X, 0, 2), (U, 0, 1), (X, 2, 2), (T, 0, 0)])
  return sc.opt.ProblemSpec(
    minimize=T[0],
    eq=(gaps, pos[0:1], speed[0:1], pos[N : N + 1] - 1.0),
    ineq=(sc.opt.bounded(speed - (1 - (2 * np.pi * pos).sin() / 2), hi=0.0),),
    lb=(sc.opt.NO_LB, sc.const(0.0), sc.const(0.0)),
    ub=(sc.opt.NO_UB, sc.const(1.0), sc.opt.NO_UB),
  )


def build(verbose: bool = False):
  solve = sc.opt.solver(race_car, sc.opt.IPOPT(options=scaly_ipopt_options(verbose)))
  X0 = np.zeros(2 * (N + 1))
  X0[1::2] = 1.0  # speed 1

  def run():
    (X, U, T), *_ = solve((X0, np.zeros(N), np.ones(1)), (np.zeros(2 * (N + 1)), np.zeros(N), np.zeros(1)), np.zeros(2 * N + 3), np.zeros(N + 1), ())
    return {"T": T, "pos": X[0::2], "speed": X[1::2], "u": U, "iter": np.array([sc.opt.solver_stats(solve).iter])}

  return run


if __name__ == "__main__":
  show(build(verbose=True)())
