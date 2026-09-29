# /// script
# requires-python = ">=3.12"
# dependencies = ["casadi"]
#
# [tool.ty.environment]
# extra-paths = ["."]  # the modules beside this script, which it imports
# ///
"""Model predictive contouring control around a Formula Student track (CasADi).

A kinematic bicycle ``(X, Y, psi, v)`` with its progress ``theta`` along the track as a fifth state,
driven by acceleration, steering and the progress speed. CasADi has no periodic spline, so the track's
centre line is ``interpolant("bspline")`` through the lap repeated three times, ``theta`` running in
the middle one, where the ends' conditions have died out. The stage cost weighs the contouring error
(across the track) and the lag error (along it) against the progress, and a constraint keeps the car
inside the track. ``Opti`` transcribes it by multiple shooting (RK4, 40 intervals of 50 ms) for
IPOPT; the first solve from rest, then 200 closed-loop steps, each warm-started from the last
solution shifted.
"""

import casadi as ca
import numpy as np

from _common import as_arrays, casadi_ipopt_options, casadi_jit, show
from _data import MPCC, mpcc_start, track


def centre_line():
  t = track()
  s, xy, length = t.s[:-1], t.xy[:-1], t.length
  s3 = np.concatenate([s - length, s, s + length, [2 * length]])
  xy3 = np.vstack([xy, xy, xy, xy[:1]])
  return [ca.interpolant(f"centre_{c}", "bspline", [list(s3)], list(xy3[:, i])) for i, c in enumerate("xy")], t.width - MPCC.margin


def build(verbose: bool = False):
  (cx, cy), width = centre_line()
  x, u = ca.MX.sym("x", 5), ca.MX.sym("u", 3)
  psi, v, theta = x[2], x[3], x[4]
  bicycle = ca.Function("bicycle", [x, u], [ca.vertcat(v * ca.cos(psi), v * ca.sin(psi), v * ca.tan(u[1]) / MPCC.wheelbase, u[0], u[2])])
  dt = MPCC.dt
  k1 = bicycle(x, u)
  k2 = bicycle(x + dt / 2 * k1, u)
  k3 = bicycle(x + dt / 2 * k2, u)
  k4 = bicycle(x + dt * k3, u)
  step = ca.Function("step", [x, u], [x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)])
  th = ca.MX.sym("theta")
  on_track = ca.vertcat(cx(th), cy(th))
  centre, tangent = ca.Function("centre_line", [th], [on_track, ca.jacobian(on_track, th)])(theta)
  norm = ca.norm_2(tangent)
  dx, dy = x[0] - centre[0], x[1] - centre[1]
  contour = (tangent[1] * dx - tangent[0] * dy) / norm
  lag = -(tangent[0] * dx + tangent[1] * dy) / norm
  r = MPCC.r
  stage_cost = MPCC.q_contour * contour**2 + MPCC.q_lag * lag**2 - MPCC.q_progress * u[2] + r[0] * u[0] ** 2 + r[1] * u[1] ** 2 + r[2] * u[2] ** 2
  stage = ca.Function("stage", [x, u], [stage_cost, contour])

  n = MPCC.horizon
  opti = ca.Opti()
  X, U, x0 = opti.variable(5, n + 1), opti.variable(3, n), opti.parameter(5)
  opti.subject_to(X[:, 0] == x0)
  cost = 0
  for k in range(n):
    opti.subject_to(X[:, k + 1] == step(X[:, k], U[:, k]))
    l_k, e_k = stage(X[:, k], U[:, k])
    cost += dt * l_k
    opti.subject_to(opti.bounded(-width, e_k, width))
    opti.subject_to(opti.bounded(np.array(MPCC.u_lo), U[:, k], np.array(MPCC.u_hi)))
    opti.subject_to(opti.bounded(0.0, X[3, k + 1], MPCC.v_max))
  opti.minimize(cost)
  opti.solver("ipopt", {**casadi_ipopt_options(verbose, tol=1e-10), **casadi_jit(expand=False)})
  start = mpcc_start()

  def solve(state, x_guess, u_guess):
    opti.set_value(x0, state)
    opti.set_initial(X, x_guess)
    opti.set_initial(U, u_guess)
    sol = opti.solve()
    return np.asarray(sol.value(X)), np.asarray(sol.value(U)).reshape(3, n), sol.value(cost), sol.stats()["iter_count"]

  def run():
    x_guess, u_guess = np.tile(start[:, None], (1, n + 1)), np.zeros((3, n))
    first_xs, first_us, first_cost, iter_first = solve(start, x_guess, u_guess)
    state, xs, us, iters = start, [start], [], []
    for _ in range(MPCC.steps):
      x_opt, u_opt, _, it = solve(state, x_guess, u_guess)
      state = step(state, u_opt[:, 0]).full().reshape(-1)
      x_guess = np.hstack([x_opt[:, 1:], x_opt[:, -1:]])  # shifted by one interval, the last repeated
      u_guess = np.hstack([u_opt[:, 1:], u_opt[:, -1:]])
      xs.append(state)
      us.append(u_opt[:, 0])
      iters.append(it)
    return as_arrays(
      {
        "first_xs": first_xs.T,
        "first_us": first_us.T,
        "first_cost": first_cost,
        "progress": np.array(xs)[:, 4],
        "loop_us": np.array(us),
        "iter_first": iter_first,
        "iter_loop": np.array(iters),
      }
    )

  return run


if __name__ == "__main__":
  show(build(verbose=False)())
