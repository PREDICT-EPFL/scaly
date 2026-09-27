"""Tuning an LQR by differentiating through the Riccati recursion and the closed loop.

A planar quadrotor (position ``x, z``, tilt ``theta``, their rates; two rotor thrusts) must return to
hover from several offsets while a gust pushes it sideways. The controller is finite-horizon LQR:
the gains ``K_k`` come from the backward Riccati recursion on the *nominal* linear model,

    S = R + B^T P B,   K = S^{-1} B^T P A,   P <- Q + A^T P (A - B K),

with the ``2 x 2`` solve done by the generated dense ``cholesky`` and ``cho_solve``. The closed loop
then runs on the *true* vehicle, 20% heavier, with a gust whose strength at step ``k`` is computed
from the step number (``index=True``). Its cost is what the designer cares about: position error,
thrust, a penalty on tilting past 0.25 rad, and the final position error. None of that is the LQR
cost, so the weights ``Q = diag(exp(q))`` and ``R = exp(r) I`` are tuned for it.

Both loops are ``sc.scan``s. The recursion runs backwards over the horizon with ``Q`` and ``R`` read at
every step (a stride-0 input), and the rollout reads the gains in reverse order (a negative
stride). ``sc.gradient`` of the closed-loop cost in the seven log-weights runs reverse mode through
both scans and the factorizations inside them, and SciPy's L-BFGS uses it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import linalg as sla
from scipy import optimize

import scaly as sc
from scaly import linalg
from scaly.codegen import write_module

GENERATED = Path(__file__).resolve().parent / "generated" / "lqr_tuning"

H, N = 0.05, 100  # 5 s horizon
NX, NU = 6, 2
G_ACC, ARM = 9.81, 0.15
TILT_LIMIT = 0.25
STARTS = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -0.5], [0.5, 0.5]])  # (x, z) offsets from hover
NS = len(STARTS)


def model(mass: float, inertia: float) -> tuple[np.ndarray, np.ndarray]:
  """Hover linearization, states ``(x, z, theta, vx, vz, omega)``, discretized exactly."""
  a = np.zeros((NX, NX))
  a[0:3, 3:6] = np.eye(3)
  a[3, 2] = -G_ACC
  b = np.zeros((NX, NU))
  b[4] = 1.0 / mass
  b[5] = [ARM / inertia, -ARM / inertia]
  big = sla.expm(np.block([[a, b], [np.zeros((NU, NX + NU))]]) * H)
  return big[:NX, :NX], big[:NX, NX:]


A_NOM, B_NOM = model(0.5, 0.005)
A_TRUE, B_TRUE = model(0.6, 0.006)


def _riccati_step() -> sc.Function:
  p_flat, weights = sc.sym("P", NX * NX), sc.sym("w", NX + 1)
  a, b = sc.const(A_NOM), sc.const(B_NOM)
  p = p_flat.reshape((NX, NX))
  q = sc.const(np.eye(NX)) * weights[:NX].exp().reshape((1, NX))
  s = sc.const(np.eye(NU)) * weights[NX].exp() + b.T @ p @ b
  k = linalg.cho_solve(linalg.cholesky(s), b.T @ p @ a)
  p_next = q + a.T @ p @ (a - b @ k)
  p_next = 0.5 * (p_next + p_next.T)
  return sc.Function._from_exprs("riccati_step", [p_flat, weights], [p_next.reshape((NX * NX,)), k.reshape((NU * NX,))], ["P", "w"], ["P_prev", "K"])


def _closed_loop_step() -> sc.Function:
  x_flat, step, k_flat = sc.sym("X", NX * NS), sc.sym("k", (), dtype="int64"), sc.sym("K", NU * NX)
  x = x_flat.reshape((NX, NS))
  u = -(k_flat.reshape((NU, NX)) @ x)
  t = H * step.cast("float64")
  gust = 3.0 * (-(((t - 1.5) / 0.3) ** 2)).exp()  # m/s^2 on vx, the same for every start
  x_next = sc.const(A_TRUE) @ x + sc.const(B_TRUE) @ u + sc.const(np.eye(NX)[:, 3:4] * H) * gust
  tilt_excess = sc.maximum(x[2].abs() - TILT_LIMIT, 0.0)
  cost = H * (sc.sumsqr(x[0:2]) + 0.01 * sc.sumsqr(u) + 100.0 * sc.sumsqr(tilt_excess))
  return sc.Function._from_exprs(
    "closed_loop_step", [x_flat, step, k_flat], [x_next.reshape((NX * NS,)), cost.reshape((1,))], ["X", "k", "K"], ["X_next", "cost"]
  )


RICCATI_STEP, CLOSED_LOOP_STEP = _riccati_step(), _closed_loop_step()


def closed_loop_cost(weights: sc.Expr) -> sc.Expr:
  """The closed-loop cost for the log-weights ``(q, r)``."""
  q_final = (sc.const(np.eye(NX)) * weights[:NX].exp().reshape((1, NX))).reshape((NX * NX,))
  # Backwards over the horizon: the gains come out in the order N-1, ..., 0.
  _, gains_backward = sc.scan(RICCATI_STEP, q_final, [(weights, 0, 0)], length=N)
  x0 = np.zeros((NX, NS))
  x0[0:2] = STARTS.T
  last = (N - 1) * NU * NX
  x_final, costs = sc.scan(CLOSED_LOOP_STEP, sc.const(x0.reshape(-1)), [(gains_backward, last, -NU * NX)], length=N, index=True)
  return costs.sum() + 10.0 * sc.sumsqr(x_final.reshape((NX, NS))[0:2])


@sc.function(NX + 1, output=sc.G("cost", "gradient"))
def tuning_objective(weights: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
  cost = closed_loop_cost(weights)
  return cost, sc.gradient(cost, weights)


def reference_cost(weights: np.ndarray) -> float:
  """The same closed-loop cost in NumPy."""
  q, r = np.diag(np.exp(weights[:NX])), np.exp(weights[NX]) * np.eye(NU)
  p, gains = q.copy(), []
  for _ in range(N):
    k = np.linalg.solve(r + B_NOM.T @ p @ B_NOM, B_NOM.T @ p @ A_NOM)
    p = q + A_NOM.T @ p @ (A_NOM - B_NOM @ k)
    p = 0.5 * (p + p.T)
    gains.append(k)
  x = np.zeros((NX, NS))
  x[0:2] = STARTS.T
  total = 0.0
  for step, k in enumerate(reversed(gains)):
    u = -k @ x
    t = H * step
    total += H * (np.sum(x[0:2] ** 2) + 0.01 * np.sum(u**2) + 100.0 * np.sum(np.maximum(np.abs(x[2]) - TILT_LIMIT, 0.0) ** 2))
    x = A_TRUE @ x + B_TRUE @ u + np.eye(NX)[:, 3:4] * H * 3.0 * np.exp(-(((t - 1.5) / 0.3) ** 2))
  return total + 10.0 * np.sum(x[0:2] ** 2)


def main(max_iter: int = 60) -> dict[str, np.ndarray]:
  start = np.zeros(NX + 1)
  history: list[float] = []

  def fun(w: np.ndarray) -> tuple[float, np.ndarray]:
    cost, grad = tuning_objective(w)
    history.append(float(cost))
    return float(cost), grad

  result = optimize.minimize(fun, start, jac=True, method="L-BFGS-B", bounds=[(-8.0, 8.0)] * (NX + 1), options={"maxiter": max_iter})
  return {"start": start, "weights": result.x, "initial_cost": np.array(history[0]), "tuned_cost": np.array(result.fun), "history": np.array(history)}


if __name__ == "__main__":
  out = main()
  print(f"closed-loop cost: Q = I, R = I gives {float(out['initial_cost']):.3f}; tuned {float(out['tuned_cost']):.3f} after {len(out['history'])} evaluations")
  names = ["x", "z", "theta", "vx", "vz", "omega", "r"]
  print("tuned weights: " + ", ".join(f"{n} {np.exp(w):.3g}" for n, w in zip(names, out["weights"], strict=True)))
  write_module(tuning_objective, GENERATED)
  print(f"generated C in {GENERATED}")
