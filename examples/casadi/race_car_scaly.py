"""Minimum-time race on a track with a position-dependent speed limit (Scaly).

    minimize T   subject to  p' = v,  v' = u - v  (RK4 on N intervals of T/N),
                             v <= 1 - sin(2 pi p) / 2,  0 <= u <= 1,  p(0) = v(0) = 0,  p(T) = 1

The N continuity constraints are one ``vmap`` of an RK4 defect, with T broadcast (stride 0).

After casadi/docs/examples/python/race_car.py.
"""

import numpy as np

import scaly as sc
from _common import scaly_ipopt_options, show

N = 100  # control intervals


@sc.function(sc.G(sc.L("z", 2), sc.L("u", 1), sc.L("znext", 2), sc.L("T", 1)), sc.L("defect", ...))
def defect(inputs):
  z, u, znext, T = inputs
  dt = T[0] / N

  def f(x):
    return sc.stack([x[1], u[0] - x[1]])

  k1 = f(z)
  k2 = f(z + dt / 2 * k1)
  k3 = f(z + dt / 2 * k2)
  k4 = f(z + dt * k3)
  return z + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4) - znext


@sc.problem(vars=sc.G(sc.L("X", 2 * (N + 1)), sc.L("U", N), sc.L("T", 1)))
def race_car(variables):
  X, U, T = variables  # X stacks (position, speed) at the N + 1 grid points
  pos, speed = X.reshape((N + 1, 2))[:, 0], X.reshape((N + 1, 2))[:, 1]
  gaps = sc.vmap(defect, N, [(X, 0, 2), (U, 0, 1), (X, 2, 2), (T, 0, 0)])
  return sc.ProblemSpec(
    minimize=T[0],
    eq=(gaps, pos[0:1], speed[0:1], pos[N : N + 1] - 1.0),
    ineq=(sc.bounded(speed - (1 - (2 * np.pi * pos).sin() / 2), hi=0.0),),
    lb=(sc.NO_LB, sc.const(0.0), sc.const(0.0)),
    ub=(sc.NO_UB, sc.const(1.0), sc.NO_UB),
  )


def build(verbose: bool = False):
  solve = sc.solver(race_car, "ipopt", options=scaly_ipopt_options(verbose))
  X0 = np.zeros(2 * (N + 1))
  X0[1::2] = 1.0  # speed 1

  def run():
    (X, U, T), *_ = solve(
      ((X0, np.zeros(N), np.ones(1)), (np.zeros(2 * (N + 1)), np.zeros(N), np.zeros(1)), np.zeros(2 * N + 3), np.zeros(N + 1), ())
    )
    return {"T": T, "pos": X[0::2], "speed": X[1::2], "u": U, "iter": np.array([solve.solver_stats().iter])}

  return run


if __name__ == "__main__":
  show(build(verbose=True)())
