"""The Fatrop paper's hanging-chain MPC (IROS 2023, Tables III and IV) written in Scaly.

The problem is `hanging_chain/` of the authors' benchmark repository (`baseline/setup.sh` pins it):
`no_masses` masses on springs between a fixed ground point and an actuated end mass whose velocity is
the control, in 2D or 3D, one RK4 step per shooting interval. rockit transcribes it by multiple
shooting; this file writes the same nonlinear program with the same variable order, the same rows in
the same order and the same initial guess, so IPOPT sees the same problem up to the sign convention of
the initial-state row (`x_0 - x0 = 0` here, `x_0 = x0` there).

Variables: `z = [x_0, u_0, x_1, u_1, ..., x_{N-1}, u_{N-1}, x_N]`, with
`x = [p_0 .. p_{no_masses} (dim each), v_0 .. v_{no_masses-1} (dim each)]`, `u` the end mass's velocity.

    minimize   sum_{k<N} 25 |p_end,k - x_end|^2 + sum_i |v_i,k|^2 + 0.01 |u_k|^2
    subject to x_{k+1} = F(x_k, u_k),  x_0 = x0,  -1 <= u_k <= 1
"""

from __future__ import annotations

import numpy as np

import scaly as sc
from scaly import integrators as si

D, L_REST, MASS, GRAVITY = 1.6, 0.0055, 0.03, 9.81
ALPHA, BETA, GAMMA = 25.0, 1.0, 0.01
IPOPT_OPTIONS = {  # the authors' SolveIpopt options, with MUMPS for MA57 (no HSL here)
  "tol": 1e-8,
  "gamma_theta": 1e-12,
  "mu_init": 1e2,
  "kappa_d": 1e-5,
  "min_refinement_steps": 0,
  "residual_ratio_max": 1e-6,
  "linear_solver": "mumps",
}
PAPER = {2: {"T": 4.0, "N": 25}, 3: {"T": 2.0, "N": 25}}  # benchmark_script.py solves both chains at N = 25


def sizes(dim: int, no_masses: int) -> tuple[int, int]:
  return dim * (2 * no_masses + 1), dim


def chain_model(dim: int, no_masses: int) -> sc.Function:
  """`xdot = f(x, u)`: link `i` joins `p_{i-1}` (the ground for `i = 0`) and `p_i` and pulls with
  `D (1 - L / |d|) d`; each free mass feels its two links and gravity, the end mass moves with `u`."""
  nx, nu = sizes(dim, no_masses)
  down = np.zeros(dim)
  down[1] = -GRAVITY  # rockit's model puts gravity on the second axis in 2D and 3D alike

  @sc.function(dim, dim)
  def link_force(left: sc.Expr, right: sc.Expr) -> sc.Expr:
    d = right - left
    ss = sc.sumsqr(d)
    dist = sc.where(sc.greater(ss, 0.0), ss.sqrt(), 0.0)  # the authors' guarded sqrt
    return D * (1.0 - L_REST / dist) * d

  @sc.function(sc.L("x", nx), sc.L("u", nu), output="xdot", name=f"chain{dim}d_M{no_masses}")
  def model(x: sc.Expr, u: sc.Expr) -> sc.Expr:
    positions = sc.concat([sc.const(np.zeros(dim)), x[: dim * (no_masses + 1)]])
    forces = sc.vmap(link_force, no_masses + 1, [(positions, 0, dim), (positions, dim, dim)])
    accel = (1.0 / MASS) * (forces[dim:] - forces[: dim * no_masses] + MASS * sc.const(np.tile(down, no_masses)))
    return sc.concat([x[dim * (no_masses + 1) :], u, accel])

  return model


def build(dim: int, no_masses: int = 6, horizon: int | None = None, T: float | None = None):
  """The problem, its IPOPT solver pieces and the rockit initial guess. Returns a dict."""
  horizon = PAPER[dim]["N"] if horizon is None else horizon
  T = PAPER[dim]["T"] if T is None else T
  nx, nu = sizes(dim, no_masses)
  nz = nx + nu
  n_var = horizon * nz + nx
  model = chain_model(dim, no_masses)
  step = si.rk4(model, dt=T / horizon)
  x_end = np.zeros(dim)
  x_end[0] = 1.0
  end = dim * no_masses  # the end mass p_{no_masses}

  @sc.function(sc.L("x", nx), sc.L("u", nu), sc.L("xnext", nx), name=f"chain{dim}d_M{no_masses}_gap")
  def gap(x: sc.Expr, u: sc.Expr, xnext: sc.Expr) -> sc.Expr:
    return xnext - step(x, u)

  @sc.function(sc.L("x", nx), sc.L("u", nu), name=f"chain{dim}d_M{no_masses}_cost")
  def stage_cost(x: sc.Expr, u: sc.Expr) -> sc.Expr:
    e = x[end : end + dim] - sc.const(x_end)
    return (ALPHA * sc.sumsqr(e) + BETA * sc.sumsqr(x[dim * (no_masses + 1) :]) + GAMMA * sc.sumsqr(u)).reshape((1,))

  @sc.opt.problem(vars=sc.L("z", n_var), params=sc.L("x0", nx), name=f"chain{dim}d_M{no_masses}_N{horizon}")
  def problem(z: sc.Expr, x0: sc.Expr) -> sc.opt.ProblemSpec:
    gaps = sc.vmap(gap, horizon, [(z, 0, nz), (z, nx, nz), (z, nz, nz)])
    costs = sc.vmap(stage_cost, horizon, [(z, 0, nz), (z, nx, nz)])
    us = z[: horizon * nz].reshape((horizon, nz))[:, nx:].reshape((horizon * nu,))
    return sc.opt.ProblemSpec(minimize=costs.sum(), eq=(gaps, z[:nx] - x0), ineq=(sc.opt.bounded(us, lo=-1.0, hi=1.0, name="u_box"),))

  return {"problem": problem, "model": model, "step": step, "n_var": n_var, "nx": nx, "nu": nu, "horizon": horizon, "T": T}


def initial_state(dim: int, no_masses: int = 6) -> np.ndarray:
  """The authors' x0: the chain spread on a line from the ground to `x_end`, at rest, then three RK4
  steps of 0.05 s under `u = 3 (-0.5, 0.5, 0.5)`."""
  nx, nu = sizes(dim, no_masses)
  x_end = np.zeros(dim)
  x_end[0] = 1.0
  line = np.linspace(np.zeros(dim), x_end, no_masses + 2)
  x = np.concatenate([line[1:].reshape(-1), np.zeros(dim * no_masses)])
  u = 3.0 * np.array([-0.5, 0.5, 0.5][:dim])
  step = si.rk4(chain_model(dim, no_masses), dt=0.05, name=f"chain{dim}d_M{no_masses}_sim")
  for _ in range(3):
    x = step(x, u)
  return np.asarray(x)


def initial_guess(dim: int, no_masses: int, horizon: int) -> np.ndarray:
  """rockit's guess: the free masses on the line at every node; the end mass, velocities and u at zero."""
  nx, nu = sizes(dim, no_masses)
  x_end = np.zeros(dim)
  x_end[0] = 1.0
  line = np.linspace(np.zeros(dim), x_end, no_masses + 2)
  node = np.zeros(nx + nu)
  node[: dim * no_masses] = line[1 : no_masses + 1].reshape(-1)
  return np.concatenate([np.tile(node, horizon), node[:nx]])


if __name__ == "__main__":
  for dim in (2, 3):
    b = build(dim)
    solve = sc.opt.solver(b["problem"], sc.opt.IPOPT(options=IPOPT_OPTIONS), name=f"fatrop_chain{dim}d_ipopt")
    x0, z0 = initial_state(dim), initial_guess(dim, 6, b["horizon"])
    p = b["problem"]
    z, *_ = solve(z0, np.zeros(b["n_var"]), np.zeros(p.n_eq), np.zeros(p.n_ineq), x0)
    stats = sc.opt.solver_stats(solve)
    print(f"{dim}D: {stats.iter} IPOPT iterations, status {stats.to_solver_status().name}, u_0 = {z[b['nx'] : b['nx'] + b['nu']]}")
