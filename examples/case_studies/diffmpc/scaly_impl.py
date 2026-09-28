"""The DiffMPC study's learning loop in Scaly: batched LQ MPC rolled out over an episode, and its gradient.

The benchmark of DiffMPC (Table 3; `benchmarking/reinforcement-learning/` in its repository) runs, for a
batch of initial states, a 50-step episode in which every step solves an MPC problem at the current
state, applies the first control to `x+ = A x + B u + b` and adds `|x+|^2 + |u|^2` to the episode's
cost; the backward pass is the gradient of the batch's summed cost with respect to the diagonal of the
MPC's state weight `Q`. Every problem is unconstrained, so each MPC is an LQ problem, with `T` knots
(`T - 1` transitions), cost `1/2 sum_t x_t' Q x_t + 1/2 sum_{t < T-1} u_t' R u_t`, and one solver
iteration solves it exactly: mpc.pytorch's LQR, DiffMPC's SQP step (to its PCG tolerance) and trajax's
iLQR step all return the LQ solution.

`mpc_function` is the MPC as a Scaly Function of the state and the problem data: a Riccati `scan`
backward over the horizon, `Q_uu` factored by the generated Cholesky, and the first control
`u_0 = K_0 x_0 + k_0`. `episode_function` `vmap`s it over the batch inside a `scan` over the episode;
`gradient_function` is `sc.gradient` of that episode, which Scaly differentiates in reverse mode through
the two scans and the `vmap`. Nothing is hoisted by hand: every step of every batch element solves its
own MPC, as each baseline's code asks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import scaly as sc
from scaly import linalg


@dataclass(frozen=True)
class Problem:
  """One row of the paper's Table 3."""

  name: str
  nx: int
  nu: int
  horizon: int  # T knots, as the benchmark scripts' --horizon
  batch: int
  steps: int = 50


PROBLEMS = {
  1: Problem("1", 8, 4, 40, 64),
  2: Problem("2", 8, 4, 30, 16),
  3: Problem("3", 8, 4, 30, 64),
  4: Problem("4", 8, 4, 30, 256),
  5: Problem("5", 16, 8, 30, 16),
  6: Problem("6", 16, 8, 30, 64),
}


def problem_data(nx: int, nu: int, batch: int, seed: int) -> dict[str, np.ndarray]:
  """`utils.generate_problem_data` of the benchmark, with the dimensions as arguments: the same draws
  from `np.random.seed(seed)` in the same order, so the same matrices and initial states."""
  np.random.seed(seed)
  Q, R = np.eye(nx), np.eye(nu)
  A = np.eye(nx) + 0.1 * np.random.randn(nx, nx)
  lam, V = np.linalg.eig(A)
  lam = np.where(np.abs(lam) < 1 - 1e-2, lam, lam / (np.abs(lam) + 1e-2))
  A = (V @ np.diag(lam) @ np.linalg.inv(V)).real
  B = np.random.randn(nx, nu)
  b = 0.01 * np.random.randn(nx)
  x0 = np.random.randn(batch, nx) * 5.0
  return {"Q": Q, "R": R, "A": A, "B": B, "b": b, "x0": x0}


def mpc_function(nx: int, nu: int, horizon: int, rule: str | None = None) -> sc.Function:
  """`mpc(x0, qd, A, B, b, R) -> u0`: the first control of the LQ MPC with `Q = diag(qd)`, row-major
  `A`, `B`, `R`.

  `rule=None` leaves the derivatives to AD through the Riccati `scan`. `rule="implicit"` attaches the
  implicit-function rule of the LQ problem as the reverse derivative (`implicit_rule`): one rollout of
  the solution and of its adjoint with the Riccati gains, instead of reverse mode through the recursion.
  `rule="implicit_no_x0"` is the same rule with the state's cotangent set to zero, which is what
  DiffMPC's and trajax's rules return (their gradients drop du0/dx0)."""
  n_params = nx + nx * nx + nx * nu + nx + nu * nu
  n_value = nx * nx + nx + nu * nx + nu
  EYE = sc.const(np.eye(nx))

  def unpack(params):
    qd, rest = params[:nx], params[nx:]
    A, rest = rest[: nx * nx].reshape((nx, nx)), rest[nx * nx :]
    B, rest = rest[: nx * nu].reshape((nx, nu)), rest[nx * nu :]
    b, R = rest[:nx], rest[nx:].reshape((nu, nu))
    return qd, A, B, b, R

  # One Riccati step backward, t+1 -> t. Carry: (P, p, K, k); broadcast: the problem data. Stacked:
  # the step's gains and the value it started from, (K_t, k_t, P_{t+1}, p_{t+1}).
  @sc.function(sc.L("value", n_value), sc.L("params", n_params), name=f"lq_riccati_{nx}x{nu}")
  def riccati_step(value, params):
    qd, A, B, b, R = unpack(params)
    P, p = value[: nx * nx].reshape((nx, nx)), value[nx * nx : nx * nx + nx]
    pb = P @ b + p
    PA, PB = P @ A, P @ B
    quu = R + B.T @ PB
    qux = B.T @ PA
    chol = linalg.cholesky(quu)
    K = -linalg.cho_solve(chol, qux)
    k = -linalg.cho_solve(chol, B.T @ pb)
    P_new = A.T @ PA + qux.T @ K
    P_new = 0.5 * (P_new + P_new.T) + qd.reshape((nx, 1)) * EYE
    p_new = A.T @ pb + qux.T @ k
    gains = sc.concat([K.reshape((nu * nx,)), k])
    return sc.concat([P_new.reshape((nx * nx,)), p_new, gains]), sc.concat([gains, value[: nx * nx + nx]])

  def riccati(qd, params):
    start = sc.concat([(qd.reshape((nx, 1)) * EYE).reshape((nx * nx,)), sc.const(np.zeros(nx + nu * nx + nu))])
    return sc.scan(riccati_step, start, [(params, 0, 0)], length=horizon - 1)

  @sc.function(
    sc.L("x0", nx),
    sc.L("qd", nx),
    sc.L("A", nx * nx),
    sc.L("B", nx * nu),
    sc.L("b", nx),
    sc.L("R", nu * nu),
    output="u0",
    name=f"lq_mpc_{nx}x{nu}_T{horizon}",
  )
  def mpc(x0, qd, A, B, b, R):
    value, _ = riccati(qd, sc.concat([qd, A, B, b, R]))
    K = value[nx * nx + nx : nx * nx + nx + nu * nx].reshape((nu, nx))
    return K @ x0 + value[nx * nx + nx + nu * nx :]

  if rule is None:
    return mpc
  return sc.custom_derivative(mpc, vjp=implicit_rule(nx, nu, horizon, riccati, unpack, with_x0=rule == "implicit"))


def implicit_rule(nx: int, nu: int, horizon: int, riccati, unpack, with_x0: bool) -> sc.Function:
  """The reverse derivative of `u0 = mpc(x0, theta)` by the implicit-function theorem on the LQ problem's
  KKT system (Amos et al., "Differentiable MPC", NeurIPS 2018).

  With `ubar` the cotangent of `u0`, the adjoint problem is the same LQ problem with the linear cost
  `ubar' du_0`, zero initial state and no offset. Its Riccati recursion has the same gains, and its
  linear terms vanish after the first stage, so its solution is `du_0 = -Q_uu,0^{-1} ubar`,
  `du_t = K_t dx_t` and `dx_{t+1} = A dx_t + B du_t`, with multipliers `dlam_t = P_t dx_t`. With the
  solution `x_t, u_t` and its multipliers `lam_t = P_t x_t + p_t`, the cotangents are
  `xbar0 = K_0' ubar`, `qdbar = sum_t dx_t * x_t`, `Rbar` from `sum_t du_t u_t'`, `bbar = sum_t dlam_{t+1}`,
  `Abar = sum_t (lam_{t+1} dx_t' + dlam_{t+1} x_t')`, `Bbar = sum_t (lam_{t+1} du_t' + dlam_{t+1} u_t')`.
  `Rbar` is on the lower triangle, the one the Function reads."""
  n_acc = nx + nx * nx + nx * nu + nx + nu * nu
  n_stack = nu * nx + nu + nx * nx + nx

  def adjoint_accumulate(x, dx, u, du, x1, dx1, lam1, dlam1, acc):
    o = [0, nx, nx + nx * nx, nx + nx * nx + nx * nu, 2 * nx + nx * nx + nx * nu, n_acc]
    parts = [
      acc[o[0] : o[1]] + dx1 * x1,
      acc[o[1] : o[2]] + (lam1.reshape((nx, 1)) * dx.reshape((1, nx)) + dlam1.reshape((nx, 1)) * x.reshape((1, nx))).reshape((nx * nx,)),
      acc[o[2] : o[3]] + (lam1.reshape((nx, 1)) * du.reshape((1, nu)) + dlam1.reshape((nx, 1)) * u.reshape((1, nu))).reshape((nx * nu,)),
      acc[o[3] : o[4]] + dlam1,
      acc[o[4] : o[5]] + (du.reshape((nu, 1)) * u.reshape((1, nu))).reshape((nu * nu,)),
    ]
    return sc.concat([x1, dx1, *parts])

  # One stage forward, t -> t+1, of the solution and the adjoint. Carry: (x, dx, accumulators).
  # Sliced, in stage order: (K_t, k_t, P_{t+1}, p_{t+1}). Broadcast: the problem data.
  @sc.function(
    sc.L("carry", 2 * nx + n_acc), sc.L("stage", n_stack), sc.L("params", nx + nx * nx + nx * nu + nx + nu * nu), name=f"lq_adjoint_{nx}x{nu}"
  )
  def adjoint_step(carry, stage, params):
    _, A, B, b, _ = unpack(params)
    x, dx, acc = carry[:nx], carry[nx : 2 * nx], carry[2 * nx :]
    K = stage[: nu * nx].reshape((nu, nx))
    k = stage[nu * nx : nu * nx + nu]
    P1 = stage[nu * nx + nu : nu * nx + nu + nx * nx].reshape((nx, nx))
    p1 = stage[nu * nx + nu + nx * nx :]
    u, du = K @ x + k, K @ dx
    x1, dx1 = A @ x + B @ u + b, A @ dx + B @ du
    lam1, dlam1 = P1 @ x1 + p1, P1 @ dx1
    return adjoint_accumulate(x, dx, u, du, x1, dx1, lam1, dlam1, acc)

  @sc.function(
    sc.L("x0", nx),
    sc.L("qd", nx),
    sc.L("A", nx * nx),
    sc.L("B", nx * nu),
    sc.L("b", nx),
    sc.L("R", nu * nu),
    sc.L("u0", nu),
    sc.L("ubar", nu),
    output=sc.G("x0bar", "qdbar", "Abar", "Bbar", "bbar", "Rbar"),
    name=f"lq_mpc_{nx}x{nu}_T{horizon}_vjp{'' if with_x0 else '_no_x0'}",
  )
  def vjp(x0, qd, A, B, b, R, u0, ubar):
    params = sc.concat([qd, A, B, b, R])
    _, Am, Bm, _, Rm = unpack(params)
    value, stages = riccati(qd, params)
    N = horizon - 1
    K0 = value[nx * nx + nx : nx * nx + nx + nu * nx].reshape((nu, nx))
    # Stage 0 carries the adjoint's only linear term: du_0 = -Q_uu,0^{-1} ubar, from P_1 (stacked last).
    last = stages[(N - 1) * n_stack :]
    P1 = last[nu * nx + nu : nu * nx + nu + nx * nx].reshape((nx, nx))
    du0 = -linalg.cho_solve(linalg.cholesky(Rm + Bm.T @ P1 @ Bm), ubar)
    x1, dx1 = Am @ x0 + Bm @ u0 + b, Bm @ du0
    lam1, dlam1 = P1 @ x1 + last[nu * nx + nu + nx * nx :], P1 @ dx1
    zero = sc.const(np.zeros(nx))
    first = adjoint_accumulate(x0, zero, u0, du0, x1, dx1, lam1, dlam1, sc.const(np.zeros(n_acc)))
    (final,) = sc.scan(adjoint_step, first, [(stages, (N - 2) * n_stack, -n_stack), (params, 0, 0)], length=N - 1)
    acc = final[2 * nx :]
    o = [0, nx, nx + nx * nx, nx + nx * nx + nx * nu, 2 * nx + nx * nx + nx * nu, n_acc]
    # The generated Cholesky reads the lower triangle of Q_uu = R + B'PB, so R's cotangent lives there:
    # S_ij + S_ji below the diagonal, S_ii on it, with S = sum_t du_t u_t'.
    rbar = acc[o[4] : o[5]].reshape((nu, nu))
    rbar = rbar * sc.const(np.tril(np.ones((nu, nu)))) + rbar.T * sc.const(np.tril(np.ones((nu, nu)), -1))
    x0bar = K0.T @ ubar if with_x0 else sc.const(np.zeros(nx))
    return x0bar, acc[o[0] : o[1]], acc[o[1] : o[2]], acc[o[2] : o[3]], acc[o[3] : o[4]], rbar.reshape((nu * nu,))

  return vjp


def episode_function(p: Problem, hoist: bool = True, rule: str | None = None) -> sc.Function:
  """`episode(x0s, qd, A, B, b, R) -> cost`: the batch's summed episode cost.

  With `hoist`, the problem data reach every MPC as one broadcast value, as in the benchmark's code.
  The Riccati recursion depends on nothing else, and Scaly's `scan` hoists that loop-invariant part:
  the generated C runs it once before the episode and then applies `u_0 = K_0 x + k_0` 3200 times.
  Without `hoist`, each batch element gets its own copy of the learned weights `qd`, carried through the
  episode's `scan` as it would be if they changed from step to step, so every step of every element
  solves its own MPC: the work the benchmark's solvers do."""
  nx, nu, nb = p.nx, p.nu, p.batch
  mpc = mpc_function(nx, nu, p.horizon, rule)
  n_params = nx + nx * nx + nx * nu + nx + nu * nu
  o = [0, nx, nx + nx * nx, nx + nx * nx + nx * nu, 2 * nx + nx * nx + nx * nu]
  n_carry = nb * nx + 1 + (0 if hoist else nb * nx)

  # One step of the episode. Carry: (the batch's states, the running cost[, each element's data]);
  # broadcast: the problem data.
  @sc.function(
    sc.L("carry", n_carry), sc.L("params", n_params), name=f"episode_step_{p.name}{'' if hoist else '_per_solve'}{'_' + rule if rule else ''}"
  )
  def sim_step(carry, params):
    X, cost = carry[: nb * nx], carry[nb * nx]
    weights = (params, 0, 0) if hoist else (carry[nb * nx + 1 :], 0, nx)
    U = sc.vmap(mpc, nb, [(X, 0, nx), weights, *[(params, o[k], 0) for k in range(1, 5)]])
    A = params[o[1] : o[2]].reshape((nx, nx))
    B = params[o[2] : o[3]].reshape((nx, nu))
    b = params[o[3] : o[4]]
    Xm, Um = X.reshape((nb, nx)), U.reshape((nb, nu))
    Xn = Xm @ A.T + Um @ B.T + b
    parts = [Xn.reshape((nb * nx,)), (cost + (Xn * Xn).sum() + (Um * Um).sum()).reshape((1,))]
    return sc.concat(parts if hoist else [*parts, carry[nb * nx + 1 :]])

  @sc.function(
    sc.L("x0s", nb * nx),
    sc.L("qd", nx),
    sc.L("A", nx * nx),
    sc.L("B", nx * nu),
    sc.L("b", nx),
    sc.L("R", nu * nu),
    output="cost",
    name=f"episode_{p.name}{'' if hoist else '_per_solve'}{'_' + rule if rule else ''}",
  )
  def episode(x0s, qd, A, B, b, R):
    params = sc.concat([qd, A, B, b, R])
    start = [x0s, sc.const(np.zeros(1))] if hoist else [x0s, sc.const(np.zeros(1)), sc.concat([qd] * nb)]
    (final,) = sc.scan(sim_step, sc.concat(start), [(params, 0, 0)], length=p.steps)
    return final[nb * nx]

  return episode


def gradient_function(p: Problem, episode: sc.Function) -> sc.Function:
  """`episode_grad(x0s, qd, A, B, b, R) -> d cost / d qd`, the gradient alone, as `jax.grad` returns it.
  (Returning the cost beside it costs Scaly a second forward pass: the value and the reverse sweep's
  forward are not shared.)"""
  nx, nu, nb = p.nx, p.nu, p.batch

  @sc.function(
    sc.L("x0s", nb * nx),
    sc.L("qd", nx),
    sc.L("A", nx * nx),
    sc.L("B", nx * nu),
    sc.L("b", nx),
    sc.L("R", nu * nu),
    output="grad",
    name=f"{episode.name}_grad",
  )
  def episode_grad(x0s, qd, A, B, b, R):
    return sc.gradient(episode(x0s, qd, A, B, b, R), qd)

  return episode_grad


def arguments(d: dict[str, np.ndarray], qd: np.ndarray | None = None) -> tuple[np.ndarray, ...]:
  """The Functions' arguments from `problem_data`'s arrays."""
  qd = np.diag(d["Q"]).copy() if qd is None else qd
  return d["x0"].reshape(-1), qd, d["A"].reshape(-1), d["B"].reshape(-1), d["b"], d["R"].reshape(-1)


def learning_curve(p: Problem, rule: str = "implicit", steps: int = 20, seed: int = 0, start: float = 3.0) -> dict[str, list]:
  """Descent on the state weights: `qd <- max(qd - 0.1 g / |g|_inf, 0.01)` from `qd = start`, with the
  episode's gradient (`rule="implicit"`) or the truncated one DiffMPC and trajax compute
  (`rule="implicit_no_x0"`). The same update as `baseline/learning_curve.py`."""
  d = problem_data(p.nx, p.nu, p.batch, seed)
  episode = episode_function(p, False, rule)
  grad = gradient_function(p, episode)
  qd, out = np.full(p.nx, start), {"loss": [], "gradient": [], "qd": []}
  for _ in range(steps + 1):
    a = arguments(d, qd)
    g = np.asarray(grad(*a))
    out["loss"].append(float(episode(*a)))
    out["gradient"].append(g.tolist())
    out["qd"].append(qd.tolist())
    qd = np.maximum(qd - 0.1 * g / np.abs(g).max(), 0.01)
  return out
