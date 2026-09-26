"""Minimum-time race on a track with a position-dependent speed limit, with ``Opti`` (CasADi).

    minimize T   subject to  p' = v,  v' = u - v  (RK4 on N intervals of T/N),
                             v <= 1 - sin(2 pi p) / 2,  0 <= u <= 1,  p(0) = v(0) = 0,  p(T) = 1

After casadi/docs/examples/python/race_car.py.
"""

import casadi as ca
from numpy import pi

from _common import as_arrays, casadi_jit, show

N = 100  # control intervals


def build(verbose: bool = False):
  opti = ca.Opti()
  X = opti.variable(2, N + 1)  # state trajectory
  pos, speed = X[0, :], X[1, :]
  U = opti.variable(1, N)  # throttle
  T = opti.variable()  # final time
  opti.minimize(T)

  def f(x, u):
    return ca.vertcat(x[1], u - x[1])

  dt = T / N
  for k in range(N):
    k1 = f(X[:, k], U[:, k])
    k2 = f(X[:, k] + dt / 2 * k1, U[:, k])
    k3 = f(X[:, k] + dt / 2 * k2, U[:, k])
    k4 = f(X[:, k] + dt * k3, U[:, k])
    opti.subject_to(X[:, k + 1] == X[:, k] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4))

  opti.subject_to(speed <= 1 - ca.sin(2 * pi * pos) / 2)
  opti.subject_to(opti.bounded(0, U, 1))
  opti.subject_to(pos[0] == 0)
  opti.subject_to(speed[0] == 0)
  opti.subject_to(pos[-1] == 1)
  opti.subject_to(T >= 0)
  opti.set_initial(speed, 1)
  opti.set_initial(T, 1)
  opti.solver("ipopt", {"print_time": verbose, **casadi_jit()}, {"print_level": 5 if verbose else 0, "sb": "yes"})

  def run():
    sol = opti.solve()
    return as_arrays({"T": sol.value(T), "pos": sol.value(pos), "speed": sol.value(speed), "u": sol.value(U), "iter": sol.stats()["iter_count"]})

  return run


if __name__ == "__main__":
  show(build(verbose=True)())
