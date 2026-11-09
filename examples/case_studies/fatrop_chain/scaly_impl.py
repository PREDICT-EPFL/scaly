"""Two-dimensional hanging-chain control with Fatrop.

Vanroye et al., "FATROP", IROS 2023 (https://arxiv.org/abs/2303.16746).
The example uses six free masses, an actuated end mass, 25 shooting intervals
and one fourth-order Runge-Kutta step per interval over four seconds.
"""

from __future__ import annotations

import numpy as np

import scaly as sc

SPRING, REST_LENGTH, MASS, GRAVITY = 1.6, 0.0055, 0.03, 9.81


def sizes(no_masses: int = 6) -> tuple[int, int]:
  """State and control sizes for the free masses and actuated end mass."""
  return 2 * (2 * no_masses + 1), 2


def chain_model(no_masses: int = 6) -> sc.Function:
  """Spring forces and gravity for free masses, and the end mass's commanded velocity."""
  nx, nu = sizes(no_masses)

  @sc.function(sc.arg("left", 2), sc.arg("right", 2))
  def force(left: sc.Expr, right: sc.Expr) -> sc.Expr:
    displacement = right - left
    return SPRING * (1.0 - REST_LENGTH / sc.norm_2(displacement)) * displacement

  @sc.function(sc.arg("x", nx), sc.arg("u", nu), outputs=sc.arg("xdot"), name=f"paper_chain_M{no_masses}")
  def model(x: sc.Expr, u: sc.Expr) -> sc.Expr:
    positions = sc.concat([sc.const(np.zeros(2)), x[: 2 * (no_masses + 1)]])
    forces = sc.vmap(force, no_masses + 1)(sc.window(positions, 0, 2), sc.window(positions, 2, 2)).vec()
    gravity = sc.const(np.tile([0.0, -GRAVITY], no_masses))
    acceleration = (forces[2:] - forces[: 2 * no_masses]) / MASS + gravity
    return sc.concat([x[2 * (no_masses + 1) :], u, acceleration])

  return model


def chain_step(no_masses: int = 6, dt: float = 0.16) -> sc.Function:
  """One fourth-order Runge-Kutta step with constant control."""
  nx, nu = sizes(no_masses)
  model = chain_model(no_masses)

  @sc.function(sc.arg("x", nx), sc.arg("u", nu), name=f"paper_chain_step_M{no_masses}")
  def step(x: sc.Expr, u: sc.Expr) -> sc.Expr:
    k1 = model(x, u)
    k2 = model(x + 0.5 * dt * k1, u)
    k3 = model(x + 0.5 * dt * k2, u)
    k4 = model(x + dt * k3, u)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

  return step


def build(no_masses: int = 6, horizon: int = 25, duration: float = 4.0) -> sc.Problem:
  """The hand-written multiple-shooting problem, ordered by stage as [x_k, u_k]."""
  nx, nu = sizes(no_masses)
  width = nx + nu
  step = chain_step(no_masses, duration / horizon)

  @sc.function(sc.arg("x", nx), sc.arg("u", nu), sc.arg("xnext", nx))
  def gap(x: sc.Expr, u: sc.Expr, xnext: sc.Expr) -> sc.Expr:
    return xnext - step(x, u)

  @sc.function(sc.arg("x", nx), sc.arg("u", nu))
  def cost(x: sc.Expr, u: sc.Expr) -> sc.Expr:
    end_error = x[2 * no_masses : 2 * (no_masses + 1)] - sc.const(np.array([1.0, 0.0]))
    return 25.0 * sc.sumsqr(end_error) + sc.sumsqr(x[2 * (no_masses + 1) :]) + 0.01 * sc.sumsqr(u)

  @sc.problem(vars=sc.arg("z", horizon * width + nx), params=sc.arg("x0", nx), name=f"paper_chain_M{no_masses}_N{horizon}")
  def problem(z: sc.Expr, x0: sc.Expr) -> sc.ProblemSpec[sc.Expr]:
    gaps = sc.vmap(gap, horizon)(sc.window(z, 0, width), sc.window(z, nx, width), sc.window(z, width, width)).vec()
    costs = sc.vmap(cost, horizon)(sc.window(z, 0, width), sc.window(z, nx, width))
    controls = z[: horizon * width].reshape((horizon, width))[:, nx:].vec()
    return sc.ProblemSpec(minimize=costs.sum(), eq=(gaps, z[:nx] - x0), ineq=(sc.bounded(controls, lo=-1.0, hi=1.0),))

  return problem


def initial_state(no_masses: int = 6) -> np.ndarray:
  """The paper's straight chain after three 0.05-second steps with end velocity [-1.5, 1.5]."""
  line = np.linspace([0.0, 0.0], [1.0, 0.0], no_masses + 2)
  x = np.concatenate([line[1:].reshape(-1), np.zeros(2 * no_masses)])
  step = chain_step(no_masses, 0.05)
  for _ in range(3):
    x = np.asarray(step(x, np.array([-1.5, 1.5])))
  return x


def initial_guess(no_masses: int = 6, horizon: int = 25) -> np.ndarray:
  """The paper's free masses on a line at every node, with zero end position and velocity."""
  nx, nu = sizes(no_masses)
  node = np.zeros(nx + nu)
  line = np.linspace([0.0, 0.0], [1.0, 0.0], no_masses + 2)
  node[: 2 * no_masses] = line[1 : no_masses + 1].reshape(-1)
  return np.concatenate([np.tile(node, horizon), node[:nx]])


if __name__ == "__main__":
  problem = build()
  solve = sc.solver(problem, "fatrop")
  z, *_ = solve(initial_state(), x0=initial_guess())
  stats = solve.stats()
  assert stats.to_solver_status().ok, stats
  nx, nu = sizes()
  print(f"2D chain: {stats.iter} iterations, {stats.status.name}, u_0 = {z[nx : nx + nu]}")
