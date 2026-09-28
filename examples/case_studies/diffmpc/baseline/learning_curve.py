"""A learning curve through a baseline's gradient: descent on the MPC's state weights for Problem 1, seed 0.

    run inside the baseline's environment, from run_baselines.py's copy of the benchmark directory:
    python learning_curve.py --impl mpcpytorch|diffmpc --steps 20 --out curve.json

The rollout is the benchmark script's own (`benchmark_mpcpytorch.py`'s `rollout_pytorch`,
`benchmark_diffmpc.py`'s `rollout`, the latter without its timing and per-seed loop), and the update
is the one `learning_curve` in `../scaly_impl.py` applies: from `qd = start`, a step of 0.1 along the
gradient's direction scaled to unit largest entry, `qd <- max(qd - 0.1 g / |g|_inf, 0.01)`.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from utils import N_CTRL, N_STATE, generate_problem_data

HORIZON, BATCH, STEPS_SIM = 40, 64, 50


def mpcpytorch_loss_and_grad():
  import torch
  from mpc import mpc
  from mpc.dynamics import AffineDynamics
  from mpc.mpc import QuadCost
  from utils import MPC_LQR_ITER, MPC_MAX_LINES, MPC_TOL

  Q, R, A, B, b, x0 = (torch.from_numpy(x) for x in generate_problem_data(BATCH, 0))
  c_batch = torch.zeros(HORIZON, BATCH, N_STATE + N_CTRL, dtype=torch.double)
  dynamics = AffineDynamics(A, B, b)
  solver = mpc.MPC(
    n_state=N_STATE, n_ctrl=N_CTRL, T=HORIZON, u_lower=None, u_upper=None, lqr_iter=MPC_LQR_ITER, eps=MPC_TOL, n_batch=BATCH,
    max_linesearch_iter=MPC_MAX_LINES, verbose=0, backprop=True, grad_method=mpc.GradMethods.ANALYTIC, exit_unconverged=False,
    detach_unconverged=False,
  )  # fmt: skip

  def f(qd: np.ndarray):
    w = torch.tensor(qd, dtype=torch.double, requires_grad=True)
    states, total = x0.clone(), 0.0
    for _ in range(STEPS_SIM):
      H = torch.block_diag(torch.diag(w), R).unsqueeze(0).unsqueeze(0).expand(HORIZON, BATCH, -1, -1)
      _, u, _ = solver(states, QuadCost(H, c_batch), dynamics)
      states = (A @ states.T + B @ u[0].T + b.unsqueeze(1)).T
      total = total + torch.sum(states**2) + torch.sum(u[0] ** 2)
    total.backward()
    return float(total), w.grad.numpy().copy()

  return f


def diffmpc_loss_and_grad():
  from jax import config

  config.update("jax_enable_x64", True)
  import jax
  import jax.numpy as jnp
  from utils import DIFFMPC_SCP_ITER

  from diffmpc.dynamics.linear_dynamics import LinearDynamics
  from diffmpc.problems.optimal_control_problem import OptimalControlProblem
  from diffmpc.solvers.sqp import SQPSolver
  from diffmpc.utils.load_params import load_problem_params, load_solver_params

  Q, R, A, B, b, x0 = (jnp.array(x) for x in generate_problem_data(BATCH, 0))
  N = HORIZON - 1
  pp = load_problem_params("linear.yaml")
  pp.update(
    horizon=N, discretization_resolution=1.0, initial_state=jnp.zeros(N_STATE), final_state=jnp.zeros(N_STATE),
    reference_state_trajectory=jnp.zeros((N + 1, N_STATE)), reference_control_trajectory=jnp.zeros((N + 1, N_CTRL)),
    weights_penalization_reference_state_trajectory=jnp.diag(Q), weights_penalization_control_squared=jnp.diag(R),
    weights_penalization_final_state=jnp.zeros(N_STATE), A=A - jnp.eye(N_STATE), B=B, b=b,
  )  # fmt: skip
  sp = load_solver_params("sqp.yaml")
  sp["num_scp_iteration_max"] = DIFFMPC_SCP_ITER
  sp["pcg"]["tol_epsilon"] = 1e-12
  sp["warm_start_backward"] = True
  sp["linesearch"] = True
  sp["linesearch_alphas"] = [1.0]
  sp["verbose"] = sp["pcg"]["verbose"] = False
  dyn = LinearDynamics(
    {
      "verbose": False,
      "num_states": N_STATE,
      "num_controls": N_CTRL,
      "names_states": [f"x{i}" for i in range(N_STATE)],
      "names_controls": [f"u{i}" for i in range(N_CTRL)],
    }
  )
  solver = SQPSolver(program=OptimalControlProblem(dynamics=dyn, params=pp), params=sp)

  def rollout(state, weights):
    def step(carry, _):
      x, sol, c = carry
      sol_new = solver.solve(sol, {**pp, "initial_state": x}, weights)
      u = sol_new.controls[0]
      x1 = A @ x + B @ u + b
      return (x1, jax.lax.stop_gradient(sol_new), c + jnp.sum(x1**2) + jnp.sum(u**2)), None

    (_, _, c), _ = jax.lax.scan(step, (state, solver.initial_guess({**pp, "initial_state": state}), 0.0), None, length=STEPS_SIM)
    return c

  batch = jax.jit(jax.vmap(rollout, in_axes=(0, None)))
  loss = jax.jit(jax.value_and_grad(lambda w: jnp.sum(batch(x0, {"weights_penalization_reference_state_trajectory": w}))))

  def f(qd: np.ndarray):
    v, g = loss(jnp.array(qd))
    return float(v), np.asarray(g)

  return f


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--impl", choices=("mpcpytorch", "diffmpc"), required=True)
  ap.add_argument("--steps", type=int, default=20)
  ap.add_argument("--start", type=float, default=3.0)
  ap.add_argument("--out", required=True)
  args = ap.parse_args()
  f = mpcpytorch_loss_and_grad() if args.impl == "mpcpytorch" else diffmpc_loss_and_grad()
  qd, losses, grads, weights = np.full(N_STATE, args.start), [], [], []
  for _ in range(args.steps + 1):
    v, g = f(qd)
    losses.append(v)
    grads.append(g.tolist())
    weights.append(qd.tolist())
    qd = np.maximum(qd - 0.1 * g / np.abs(g).max(), 0.01)
  with open(args.out, "w") as fp:
    json.dump({"impl": args.impl, "loss": losses, "gradient": grads, "qd": weights}, fp)


if __name__ == "__main__":
  main()
