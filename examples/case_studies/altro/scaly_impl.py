"""The ALTRO case study's problems in Scaly: the IROS 2019 parallel park and cartpole, solved by `sc.ocp.ALTRO`.

Each is the paper-era definition from TrajectoryOptimization.jl (`problems/parallel_park.jl`,
`problems/cartpole.jl` at 320dbaca, 2019-07-18), the same as `baseline/run_altro.jl` builds on current
Altro.jl: the model, Kutta's third-order Runge-Kutta, the horizon, the weights (the stage cost carries
the factor dt), the bounds, the goal constraint and the initial controls. `OCP` keeps that data in
Altro's terms; `OCP.discrete()` states it as the `DiscreteOCP` that `sc.ocp.ALTRO` solves.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

import scaly as sc
from scaly import integrators as si
from scaly import ocp as sco


@dataclass
class OCP:
  """One problem in Altro's terms: `n_knots` states `x_0 .. x_{N-1}` and `n_knots - 1` controls, the
  continuous-time model integrated by Kutta's RK3 over `dt`, the stage cost `dt (1/2 e'Qe + 1/2 u'Ru)`
  with `e = x - x_f` and diagonal weights `q`, `r`, the terminal cost `1/2 e_N' Q_f e_N`, the goal
  `x_N = x_f`, the initial controls `u0`, and box bounds on the controls and (after the first knot) the
  states, `None` or a side of `+-inf` for none."""

  name: str
  nx: int
  nu: int
  n_knots: int
  dt: float
  dynamics: Callable[[sc.Expr, sc.Expr], sc.Expr]
  q: np.ndarray
  r: np.ndarray
  qf: np.ndarray
  x0: np.ndarray
  xf: np.ndarray
  u0: np.ndarray  # (N - 1, nu)
  u_bounds: tuple[np.ndarray, np.ndarray]
  x_bounds: tuple[np.ndarray, np.ndarray] | None = None

  def _rows(self) -> tuple[int, int]:
    count = lambda bounds: 0 if bounds is None else int(sum(np.isfinite(np.asarray(side, dtype=float)).sum() for side in bounds))  # noqa: E731
    return count(self.u_bounds), count(self.x_bounds)

  @property
  def nc(self) -> int:
    """Altro's BoundConstraint rows per stage: the finite control bounds, upper then lower, then the state's."""
    return sum(self._rows())

  @property
  def masks(self) -> np.ndarray:
    """Which rows apply at each of the `n_knots - 1` stages: the first knot bounds the controls only."""
    n_u, _ = self._rows()
    masks = np.ones((self.n_knots - 1, self.nc))
    masks[0, n_u:] = 0.0
    return masks

  def discrete(self) -> sco.DiscreteOCP:
    model = sc.function(sc.L("x", self.nx), sc.L("u", self.nu), output="xdot", name=f"{self.name}_dynamics")(self.dynamics)
    continuous = sco.ContinuousOCP(
      model,
      T=(self.n_knots - 1) * self.dt,
      stage_cost=sco.Quadratic(np.diag(self.q) / 2, np.diag(self.r) / 2, x_ref=self.xf),
      terminal_cost=sco.Quadratic(np.diag(self.qf) / 2, x_ref=self.xf),
      u_bounds=self.u_bounds,
      x_bounds=self.x_bounds,
      terminal=sco.TerminalEquality(self.xf),
      name=self.name,
    )
    return sco.transcribe(continuous, sco.MultipleShooting(si.RK3()), N=self.n_knots - 1)


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
    u0=np.ones((n_knots - 1, 2)),
    u_bounds=(np.full(2, -2.0), np.full(2, 2.0)),
    x_bounds=(np.array([-0.25, -0.001, -np.inf]), np.array([0.25, 1.001, np.inf])),
  )


def cartpole() -> OCP:
  n_knots, tf = 101, 5.0
  return OCP(
    name="cartpole",
    nx=4,
    nu=1,
    n_knots=n_knots,
    dt=tf / (n_knots - 1),
    dynamics=cartpole_dynamics(),
    q=np.full(4, 1e-2),
    r=np.full(1, 1e-1),
    qf=np.full(4, 100.0),
    x0=np.zeros(4),
    xf=np.array([0.0, np.pi, 0.0, 0.0]),
    u0=np.full((n_knots - 1, 1), 0.01),
    u_bounds=(np.full(1, -3.0), np.full(1, 3.0)),
  )


PROBLEMS = {"parallel_park": parallel_park, "cartpole": cartpole}


def build(name: str, constraint_tolerance: float = 1e-6, projected_newton: bool = False) -> tuple[OCP, sc.Function]:
  """The problem and its solver, `solve(x0) -> (us, xs, cost, outer, inner, violation, projections,
  projection_steps)` from the problem's initial controls: the augmented Lagrangian alone, or ALTRO
  (with the projected Newton phase)."""
  problem = PROBLEMS[name]()
  method = sco.ALTRO(constraint_tolerance=constraint_tolerance, projected_newton=projected_newton)
  altro = method.solver(problem.discrete(), name=f"{name}_altro")

  @sc.function(
    sc.L("x0", problem.nx), output=sc.G("us", "xs", "cost", "outer", "inner", "violation", "projections", "projection_steps"), name=f"{name}_al_ilqr"
  )
  def solve(x0):
    return altro(x0, sc.const(problem.u0.reshape(-1)))

  return problem, solve
