"""Model predictive contouring control around a Formula Student track (Scaly).

A kinematic bicycle ``(X, Y, psi, v)`` with its progress ``theta`` along the track as a fifth state,
driven by acceleration, steering and the progress speed. The track's centre line is a periodic cubic
spline in chord length (``interp.interpolant(bc="periodic")``, which also wraps ``theta`` past a
lap); the stage cost weighs the contouring error (across the track) and the lag error (along it)
against the progress, and a path constraint keeps the car inside the track. ``scaly.ocp`` transcribes
it by multiple shooting (RK4, 40 intervals of 50 ms) for IPOPT; the first solve from rest, then 200
closed-loop steps, each warm-started from the last solution shifted (``ocp.shift``).
"""

import numpy as np

import scaly as sc
from _common import scaly_ipopt_options, show
from _data import MPCC, mpcc_start, track
from scaly import integrators as si
from scaly import interp
from scaly import ocp

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
  continuous = ocp.ContinuousOCP(
    bicycle,
    T=MPCC.horizon * MPCC.dt,
    stage_cost=stage,
    x_bounds=(x_lo, x_hi),
    u_bounds=(np.array(MPCC.u_lo), np.array(MPCC.u_hi)),
    constraints=(ocp.Path(contour, -WIDTH, WIDTH),),
    name="mpcc",
  )
  problem = ocp.transcribe(continuous, N=MPCC.horizon)
  method = ocp.Direct(sc.opt.IPOPT(options=scaly_ipopt_options(verbose, tol=1e-10)))
  solve, shift = ocp.solver(problem, method), ocp.shift(problem, method)
  plant = si.rk4(bicycle, dt=MPCC.dt)
  x0 = mpcc_start()

  def run():
    first_xs, first_us, _, first = solve(x0, ocp.initial_guess(problem, method, x0))
    x, warm, progress, loop_us, iter_loop, ok = x0, ocp.initial_guess(problem, method, x0), [x0[4]], [], [], []
    for _ in range(MPCC.steps):
      _, us, point, info = solve(x, warm)
      warm = shift(point)
      x = np.asarray(plant(x, us[0]))
      progress.append(x[4])
      loop_us.append(us[0])
      iter_loop.append(int(info.iter))
      ok.append(sc.Status(int(info.status)).ok)
    assert sc.Status(int(first.status)).ok and all(ok)
    return {
      "first_xs": first_xs,
      "first_us": first_us,
      "first_cost": np.array([float(first.objective)]),
      "progress": np.array(progress),
      "loop_us": np.array(loop_us),
      "iter_first": np.array([int(first.iter)]),
      "iter_loop": np.array(iter_loop),
    }

  return run


if __name__ == "__main__":
  show(build(verbose=False)())
