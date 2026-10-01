"""A8 (Tier 8): the DiffMPC case study's episode, its forward pass and its gradient, Scaly alone.

Each MPC of the episode is an LQ problem solved by a Riccati recursion, with the implicit-function
rule as its reverse derivative (``examples/case_studies/diffmpc/scaly_impl.py``). The gradient is
reverse mode through the episode's ``scan`` and the batch's ``vmap``. This times both through the
Function's call, best of ``--repeats`` on one seed's data, per variant, and prints their ratio:
what the reverse sweep costs beside the forward pass it differentiates.

``per_solve_outputs`` is the per-solve episode with one change to the model, made here and not in
the case study: the MPC returns its Riccati recursion (the final value and the stacked gains)
beside ``u0``, and the rule reads them from its output arguments where the case study's rule runs
the recursion again. ``custom_derivative`` hands a rule the outputs for that purpose.

  uv run python internal/notes/perf_2026_09_30_gaps/e5_episode.py [--problems 1,5] [--variants ...]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "examples" / "case_studies" / "diffmpc"))
VARIANTS = {
  "per_solve_implicit": (False, "implicit"),
  "per_solve_outputs": (False, "implicit_outputs"),
  "per_solve_ad": (False, None),
  "hoisted_implicit": (True, "implicit"),
}


def mpc_with_outputs(nx: int, nu: int, horizon: int):
  """``scaly_impl.mpc_function`` with the recursion as two more outputs, ``(u0, value, stages)``,
  and ``scaly_impl.implicit_rule`` reading them: the same arithmetic, the recursion run once."""
  import scaly as sc
  from scaly import linalg

  n_params = nx + nx * nx + nx * nu + nx + nu * nu
  n_value = nx * nx + nx + nu * nx + nu
  n_stack = nu * nx + nu + nx * nx + nx
  n_acc = n_params
  N = horizon - 1
  EYE = sc.const(np.eye(nx))
  o = [0, nx, nx + nx * nx, nx + nx * nx + nx * nu, 2 * nx + nx * nx + nx * nu, n_acc]

  def unpack(params):
    qd, rest = params[:nx], params[nx:]
    A, rest = rest[: nx * nx].reshape((nx, nx)), rest[nx * nx :]
    B, rest = rest[: nx * nu].reshape((nx, nu)), rest[nx * nu :]
    return qd, A, B, rest[:nx], rest[nx:].reshape((nu, nu))

  @sc.function(sc.L("value", n_value), sc.L("params", n_params), name=f"lqo_riccati_{nx}x{nu}")
  def riccati_step(value, params):
    qd, A, B, b, R = unpack(params)
    P, pv = value[: nx * nx].reshape((nx, nx)), value[nx * nx : nx * nx + nx]
    pb = P @ b + pv
    PA, PB = P @ A, P @ B
    qux = B.T @ PA
    chol = linalg.cholesky(R + B.T @ PB)
    K = -linalg.cho_solve(chol, qux)
    k = -linalg.cho_solve(chol, B.T @ pb)
    P_new = A.T @ PA + qux.T @ K
    P_new = 0.5 * (P_new + P_new.T) + qd.reshape((nx, 1)) * EYE
    gains = sc.concat([K.reshape((nu * nx,)), k])
    return sc.concat([P_new.reshape((nx * nx,)), A.T @ pb + qux.T @ k, gains]), sc.concat([gains, value[: nx * nx + nx]])

  inputs = [sc.L("x0", nx), sc.L("qd", nx), sc.L("A", nx * nx), sc.L("B", nx * nu), sc.L("b", nx), sc.L("R", nu * nu)]

  @sc.function(*inputs, output=sc.G("u0", "value", "stages"), name=f"lqo_mpc_{nx}x{nu}_T{horizon}")
  def mpc(x0, qd, A, B, b, R):
    params = sc.concat([qd, A, B, b, R])
    start = sc.concat([(qd.reshape((nx, 1)) * EYE).reshape((nx * nx,)), sc.const(np.zeros(nx + nu * nx + nu))])
    value, stages = sc.scan(riccati_step, start, [(params, 0, 0)], length=N)
    K = value[nx * nx + nx : nx * nx + nx + nu * nx].reshape((nu, nx))
    return K @ x0 + value[nx * nx + nx + nu * nx :], value, stages

  def accumulate(x, dx, u, du, x1, dx1, lam1, dlam1, acc):
    parts = [
      acc[o[0] : o[1]] + dx1 * x1,
      acc[o[1] : o[2]] + (lam1.reshape((nx, 1)) * dx.reshape((1, nx)) + dlam1.reshape((nx, 1)) * x.reshape((1, nx))).reshape((nx * nx,)),
      acc[o[2] : o[3]] + (lam1.reshape((nx, 1)) * du.reshape((1, nu)) + dlam1.reshape((nx, 1)) * u.reshape((1, nu))).reshape((nx * nu,)),
      acc[o[3] : o[4]] + dlam1,
      acc[o[4] : o[5]] + (du.reshape((nu, 1)) * u.reshape((1, nu))).reshape((nu * nu,)),
    ]
    return sc.concat([x1, dx1, *parts])

  @sc.function(sc.L("carry", 2 * nx + n_acc), sc.L("stage", n_stack), sc.L("params", n_params), name=f"lqo_adjoint_{nx}x{nu}")
  def adjoint_step(carry, stage, params):
    _, A, B, b, _ = unpack(params)
    x, dx, acc = carry[:nx], carry[nx : 2 * nx], carry[2 * nx :]
    K, k = stage[: nu * nx].reshape((nu, nx)), stage[nu * nx : nu * nx + nu]
    P1 = stage[nu * nx + nu : nu * nx + nu + nx * nx].reshape((nx, nx))
    u, du = K @ x + k, K @ dx
    x1, dx1 = A @ x + B @ u + b, A @ dx + B @ du
    return accumulate(x, dx, u, du, x1, dx1, P1 @ x1 + stage[nu * nx + nu + nx * nx :], P1 @ dx1, acc)

  outputs = [sc.L("u0", nu), sc.L("value", n_value), sc.L("stages", N * n_stack)]
  cotangents = [sc.L("ubar", nu), sc.L("valuebar", n_value), sc.L("stagesbar", N * n_stack)]

  @sc.function(
    *inputs, *outputs, *cotangents, output=sc.G("x0bar", "qdbar", "Abar", "Bbar", "bbar", "Rbar"), name=f"lqo_mpc_{nx}x{nu}_T{horizon}_vjp"
  )
  def vjp(x0, qd, A, B, b, R, u0, value, stages, ubar, valuebar, stagesbar):  # the recursion's cotangents are not read: nothing else reads it
    params = sc.concat([qd, A, B, b, R])
    _, Am, Bm, _, Rm = unpack(params)
    K0 = value[nx * nx + nx : nx * nx + nx + nu * nx].reshape((nu, nx))
    last = stages[(N - 1) * n_stack :]
    P1 = last[nu * nx + nu : nu * nx + nu + nx * nx].reshape((nx, nx))
    du0 = -linalg.cho_solve(linalg.cholesky(Rm + Bm.T @ P1 @ Bm), ubar)
    x1, dx1 = Am @ x0 + Bm @ u0 + b, Bm @ du0
    first = accumulate(x0, sc.const(np.zeros(nx)), u0, du0, x1, dx1, P1 @ x1 + last[nu * nx + nu + nx * nx :], P1 @ dx1, sc.const(np.zeros(n_acc)))
    (final,) = sc.scan(adjoint_step, first, [(stages, (N - 2) * n_stack, -n_stack), (params, 0, 0)], length=N - 1)
    acc = final[2 * nx :]
    rbar = acc[o[4] : o[5]].reshape((nu, nu))
    rbar = rbar * sc.const(np.tril(np.ones((nu, nu)))) + rbar.T * sc.const(np.tril(np.ones((nu, nu)), -1))
    return K0.T @ ubar, acc[o[0] : o[1]], acc[o[1] : o[2]], acc[o[2] : o[3]], acc[o[3] : o[4]], rbar.reshape((nu * nu,))

  return sc.custom_derivative(mpc, vjp=vjp)


def best(call, repeats: int) -> float:
  times = []
  for _ in range(repeats):
    t = time.perf_counter()
    call()
    times.append(time.perf_counter() - t)
  return min(times)


def main() -> None:
  import scaly_impl as si

  case_study = si.mpc_function
  si.mpc_function = lambda nx, nu, horizon, rule=None: (
    mpc_with_outputs(nx, nu, horizon) if rule == "implicit_outputs" else case_study(nx, nu, horizon, rule)
  )
  parser = argparse.ArgumentParser()
  parser.add_argument("--problems", default="1,5")
  parser.add_argument("--variants", default=",".join(VARIANTS))
  parser.add_argument("--repeats", type=int, default=7)
  args = parser.parse_args()
  print(f"{'problem':>7s} {'variant':>20s} {'forward ms':>11s} {'gradient ms':>12s} {'gradient / forward':>19s} {'first calls s':>14s}")
  for number in (int(v) for v in args.problems.split(",")):
    p = si.PROBLEMS[number]
    a = si.arguments(si.problem_data(p.nx, p.nu, p.batch, 0))
    reference = None
    for variant in args.variants.split(","):
      hoist, rule = VARIANTS[variant]
      t0 = time.perf_counter()
      episode = si.episode_function(p, hoist, rule)
      grad = si.gradient_function(p, episode)
      cost, g = float(episode(*a)), np.asarray(grad(*a))
      first = time.perf_counter() - t0
      if reference is None:
        reference = (cost, g)
      np.testing.assert_allclose(cost, reference[0], rtol=1e-9)
      np.testing.assert_allclose(g, reference[1], rtol=1e-6, atol=1e-8 * np.abs(reference[1]).max())
      fwd, bwd = best(lambda: episode(*a), args.repeats), best(lambda: grad(*a), args.repeats)
      print(f"{number:7d} {variant:>20s} {fwd * 1e3:11.2f} {bwd * 1e3:12.2f} {bwd / fwd:19.1f} {first:14.1f}", flush=True)


if __name__ == "__main__":
  main()
