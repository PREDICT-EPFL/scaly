"""The ALTRO case study's problems in Scaly: the IROS 2019 parallel park and cartpole, as `al_ilqr.OCP`s.

Each is the paper-era definition from TrajectoryOptimization.jl (`problems/parallel_park.jl`,
`problems/cartpole.jl` at 320dbaca, 2019-07-18), the same as `baseline/run_altro.jl` builds on current
Altro.jl: the model, Kutta's third-order Runge-Kutta, the horizon, the weights (the stage cost carries
the factor dt), the bounds, the goal constraint and the initial controls.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import scaly as sc

sys.path.insert(0, str(Path(__file__).resolve().parent))
from al_ilqr import OCP, Options, solver  # noqa: E402


def dubins_car(x: sc.Expr, u: sc.Expr) -> sc.Expr:
  """RobotZoo's `DubinsCar`, the 2019 `car_model`: `x = (px, py, heading)`, `u = (speed, turn rate)`."""
  return sc.stack([u[0] * x[2].cos(), u[0] * x[2].sin(), u[1]])


def cartpole_dynamics(mc: float = 1.0, mp: float = 0.2, l: float = 0.5, g: float = 9.81):  # noqa: E741
  """RobotZoo's `Cartpole(1.0, 0.2, 0.5, 9.81)`: `x = (cart, angle, cart rate, angle rate)`, `u = force`,
  with the 2x2 mass matrix solved in closed form."""

  def f(x: sc.Expr, u: sc.Expr) -> sc.Expr:
    s, c = x[1].sin(), x[1].cos()
    a, b, d = mc + mp, mp * l * c, mp * l * l
    r1 = mp * l * s * x[3] * x[3] + u[0]
    r2 = -mp * g * l * s
    det = a * d - b * b
    return sc.stack([x[2], x[3], (d * r1 - b * r2) / det, (a * r2 - b * r1) / det])

  return f


def parallel_park() -> OCP:
  n_knots, dt = 51, 0.06
  K = n_knots - 1

  def ineq(x: sc.Expr, u: sc.Expr) -> sc.Expr:
    # Altro's BoundConstraint rows for the finite bounds: upper then lower.
    return sc.concat([u - 2.0, -2.0 - u, sc.stack([x[0] - 0.25, x[1] - 1.001, -0.25 - x[0], -0.001 - x[1]])])

  masks = np.ones((K, 8))
  masks[0, 4:] = 0.0  # the first knot bounds the controls only
  return OCP(
    name="parallel_park",
    nx=3,
    nu=2,
    n_knots=n_knots,
    dt=dt,
    dynamics=dubins_car,
    q=np.full(3, 1e-2),
    r=np.full(2, 1e-2),
    qf=np.full(3, 100.0),
    x0=np.zeros(3),
    xf=np.array([0.0, 1.0, 0.0]),
    u0=np.ones((K, 2)),
    ineq=ineq,
    nc=8,
    masks=masks,
  )


def cartpole() -> OCP:
  n_knots, tf = 101, 5.0
  K = n_knots - 1

  def ineq(x: sc.Expr, u: sc.Expr) -> sc.Expr:
    return sc.concat([u - 3.0, -3.0 - u])

  return OCP(
    name="cartpole",
    nx=4,
    nu=1,
    n_knots=n_knots,
    dt=tf / K,
    dynamics=cartpole_dynamics(),
    q=np.full(4, 1e-2),
    r=np.full(1, 1e-1),
    qf=np.full(4, 100.0),
    x0=np.zeros(4),
    xf=np.array([0.0, np.pi, 0.0, 0.0]),
    u0=np.full((K, 1), 0.01),
    ineq=ineq,
    nc=2,
    masks=np.ones((K, 2)),
  )


PROBLEMS = {"parallel_park": parallel_park, "cartpole": cartpole}


def build(name: str, constraint_tolerance: float = 1e-6, projected_newton: bool = False) -> tuple[OCP, sc.Function]:
  """The problem and its solver: the augmented Lagrangian alone, or ALTRO (with the projected Newton phase)."""
  ocp = PROBLEMS[name]()
  return ocp, solver(ocp, Options(constraint_tolerance=constraint_tolerance, projected_newton=projected_newton))
