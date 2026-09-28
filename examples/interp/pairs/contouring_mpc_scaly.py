"""Model predictive contouring control around a Formula Student track (Scaly).

A kinematic bicycle ``(X, Y, psi, v)`` with its progress ``theta`` along the track as a fifth state,
driven by acceleration, steering and the progress speed. The track's centre line is a periodic cubic
spline in chord length (``interp.interpolant(bc="periodic")``, which also wraps ``theta`` past a
lap); the stage cost weighs the contouring error (across the track) and the lag error (along it)
against the progress, and a path constraint keeps the car inside the track. ``scaly.mpc`` transcribes
it by multiple shooting (RK4, 40 intervals of 50 ms) for IPOPT; the first solve from rest, then 200
closed-loop steps, each warm-started from the last solution shifted.
"""

import numpy as np

import scaly as sc
from _common import scaly_ipopt_options, show
from _data import MPCC, mpcc_start, track
from scaly import integrators as si
from scaly import interp
from scaly import mpc

TRACK = track()
CENTRE = interp.interpolant(TRACK.s, TRACK.xy, kind="cubic", bc="periodic")
TANGENT = CENTRE.derivative()
WIDTH = TRACK.width - MPCC.margin


@sc.function(5, 3, output="xdot")
def bicycle(x, u):
  psi, v = x[2], x[3]
  return sc.stack([v * psi.cos(), v * psi.sin(), v * u[1].tan() / MPCC.wheelbase, u[0], u[2]])


def errors(x):
  """The contouring and lag errors: the car's offset from the centre line at ``theta``, across and
  along the unit tangent."""
  centre, tangent = CENTRE(x[4]), TANGENT(x[4])
  tx, ty = tangent[0], tangent[1]
  norm = (tx**2 + ty**2).sqrt()
  dx, dy = x[0] - centre[0], x[1] - centre[1]
  return (ty * dx - tx * dy) / norm, -(tx * dx + ty * dy) / norm


@sc.function(5, 3, output="l")
def stage(x, u):
  contour, lag = errors(x)
  r = MPCC.r
  return MPCC.q_contour * contour**2 + MPCC.q_lag * lag**2 - MPCC.q_progress * u[2] + r[0] * u[0] ** 2 + r[1] * u[1] ** 2 + r[2] * u[2] ** 2


@sc.function(5, 3, output="contour")
def contour(x, u):
  return sc.stack([errors(x)[0]])


def build(verbose: bool = False):
  x_lo = np.array([-np.inf, -np.inf, -np.inf, 0.0, -np.inf])
  x_hi = np.array([np.inf, np.inf, np.inf, MPCC.v_max, np.inf])
  ocp = mpc.OCP(
    ode=bicycle,
    horizon=MPCC.horizon,
    dt=MPCC.dt,
    stage_cost=stage,
    x_bounds=(x_lo, x_hi),
    u_bounds=(np.array(MPCC.u_lo), np.array(MPCC.u_hi)),
    constraints=(mpc.Path(contour, -WIDTH, WIDTH),),
    name="mpcc",
  )
  ctrl = mpc.MPC(ocp, "ipopt", options=scaly_ipopt_options(verbose, tol=1e-10))
  plant = si.rk4(bicycle, dt=MPCC.dt)
  x0 = mpcc_start()

  def run():
    first = ctrl.solve(x0, guess=ctrl.initial_guess(x0))
    iter_first = ctrl.solver.solver_stats().iter
    ctrl.reset()
    loop = mpc.simulate(ctrl, plant, x0, MPCC.steps)
    assert first.status.ok and all(s in ("OK", "ACCEPTABLE") for s in loop.statuses)
    return {
      "first_xs": first.xs,
      "first_us": first.us,
      "first_cost": np.array([first.cost]),
      "progress": loop.xs[:, 4],
      "loop_us": loop.us,
      "iter_first": np.array([iter_first]),
      "iter_loop": loop.iterations,
    }

  return run


if __name__ == "__main__":
  show(build(verbose=False)())
