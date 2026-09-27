"""Optimal control of a heated plate: sparse implicit time stepping, sparse matrices between Functions.

A square plate made of two materials (the right half conducts five times worse) is held at ambient
temperature on its edges and warmed by four heaters under it. Choose the heater powers ``U`` over the
horizon so that at the final time a disc on the left is at temperature 1 while the rest of the plate
stays cool, which matters a tenth as much:

    minimize 1/2 |W (T_N - T_target)|^2 + alpha/2 |U|^2   subject to   0 <= U <= u_max,
    (I + dt L(kappa)) T_{k+1} = T_k + dt B u_k,   T_0 = 0,

with ``L(kappa)`` the finite-volume diffusion operator on an ``n x n`` grid. Implicit Euler needs one
sparse symmetric positive definite solve per step with the same matrix.

The work is split in two generated ``Function``s that pass a sparse matrix between them:

- ``system(kappa)`` assembles ``K = I + dt L(kappa)`` and returns it as ``sc.S("K", ...)``: a
  ``SparseMatrix`` in a symbolic call, a SciPy matrix in an evaluation.
- ``simulate((K, U))`` declares its input ``sc.S("K", pattern)``, so only a matrix with exactly that
  pattern is accepted. It factors ``K`` once with the generated sparse ``L D L^T`` and solves once per
  step.

``objective`` calls both symbolically and takes ``sc.gradient`` of the cost in ``U``. Reverse mode
runs back through the time steps using each solve's implicit rule, which is the adjoint heat
equation: one solve with the same factorization per step and no differentiation of the factor.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy import optimize, sparse

import scaly as sc
from scaly import linalg

from scaly.codegen import write_module

GENERATED = Path(__file__).resolve().parent / "generated" / "heat_control"


N_GRID, STEPS, DT = 14, 25, 0.012
N_HEATERS, ALPHA, U_MAX = 4, 1e-6, 400.0
H = 1.0 / (N_GRID + 1)
NODES = N_GRID * N_GRID
XY = np.array([((j + 1) * H, (i + 1) * H) for i in range(N_GRID) for j in range(N_GRID)])


def faces() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Interior faces ``(p, q)`` with ``p < q``, and the number of faces each node has on the edge."""
  grid = np.arange(NODES).reshape(N_GRID, N_GRID)
  p = np.r_[grid[:, :-1].ravel(), grid[:-1, :].ravel()]
  q = np.r_[grid[:, 1:].ravel(), grid[1:, :].ravel()]
  walls = np.zeros((N_GRID, N_GRID))
  walls[0, :] += 1
  walls[-1, :] += 1
  walls[:, 0] += 1
  walls[:, -1] += 1
  return p, q, walls.ravel()


FACE_P, FACE_Q, WALLS = faces()
HEATERS = np.stack([np.exp(-np.sum((XY - c) ** 2, axis=1) / (2 * 0.08**2)) for c in [(0.25, 0.25), (0.25, 0.75), (0.75, 0.25), (0.75, 0.75)]], axis=1)
TARGET = (np.sum((XY - [0.3, 0.55]) ** 2, axis=1) < 0.15**2).astype(float)
WEIGHT = np.where(TARGET > 0, 1.0, np.sqrt(0.1))
KAPPA = np.where(XY[:, 0] < 0.5, 1.0, 0.2)


def assemble(kappa: sc.Expr) -> linalg.SparseMatrix:
  """The lower triangle of ``I + dt L(kappa)``: a face conducts with the mean of its two nodes."""
  k_face = 0.5 * (sc.gather(kappa, FACE_P) + sc.gather(kappa, FACE_Q)) * (DT / H**2)
  diagonal = 1.0 + sc.segment_sum(sc.concat([k_face, k_face]), np.r_[FACE_P, FACE_Q], NODES) + kappa * sc.const(WALLS * DT / H**2)
  rows, cols = np.r_[np.arange(NODES), FACE_Q], np.r_[np.arange(NODES), FACE_P]
  return linalg.SparseMatrix.from_coo(rows, cols, sc.concat([diagonal, -k_face]), (NODES, NODES))


@sc.function(sc.L("kappa", NODES), output=sc.S("K", ...))
def system(kappa: sc.Expr) -> linalg.SparseMatrix:
  return assemble(kappa)


K_PATTERN = system.output_sparsities[0]


@sc.function(sc.G(sc.S("K", K_PATTERN), sc.L("U", (STEPS, N_HEATERS))), output=sc.L("T_final", NODES))
def simulate(inputs: tuple[linalg.SparseMatrix, sc.Expr]) -> sc.Expr:
  k, u = inputs
  fact = linalg.SparseLDL(k, name="heat")
  b = sc.const(HEATERS * DT)
  t = sc.const(np.zeros(NODES))
  for step in range(STEPS):
    t = fact.solve(t + b @ u[step])
  return t


@sc.function(sc.G(sc.L("kappa", NODES), sc.L("U", (STEPS, N_HEATERS))), output=sc.G(sc.L("cost", ()), sc.L("gradient", (STEPS, N_HEATERS))))
def objective(inputs: tuple[sc.Expr, sc.Expr]) -> tuple[sc.Expr, sc.Expr]:
  kappa, u = inputs
  t_final = simulate((system(kappa), u))
  cost = 0.5 * sc.sumsqr(sc.const(WEIGHT) * (t_final - sc.const(TARGET))) + 0.5 * ALPHA * sc.sumsqr(u)
  return cost, sc.gradient(cost, u)


def reference_final(kappa: np.ndarray, u: np.ndarray) -> np.ndarray:
  """The same simulation with SciPy: the full symmetric matrix and ``spsolve``."""
  kf = 0.5 * (kappa[FACE_P] + kappa[FACE_Q]) * DT / H**2
  diag = 1.0 + np.bincount(np.r_[FACE_P, FACE_Q], np.r_[kf, kf], NODES) + kappa * WALLS * DT / H**2
  off = sparse.coo_array((-kf, (FACE_Q, FACE_P)), shape=(NODES, NODES))
  k = sparse.csc_array(sparse.diags_array(diag) + off + off.T)
  t = np.zeros(NODES)
  for step in range(STEPS):
    t = sparse.linalg.spsolve(k, t + HEATERS @ u[step] * DT)
  return t


def main(max_iter: int = 200) -> dict[str, Any]:
  def fun(flat: np.ndarray) -> tuple[float, np.ndarray]:
    cost, grad = objective((KAPPA, flat.reshape(STEPS, N_HEATERS)))
    return float(cost), grad.reshape(-1)

  zero = np.zeros(STEPS * N_HEATERS)
  result = optimize.minimize(fun, zero, jac=True, method="L-BFGS-B", bounds=[(0.0, U_MAX)] * zero.size, options={"maxiter": max_iter})
  u = result.x.reshape(STEPS, N_HEATERS)
  # The two Functions also compose numerically: the SciPy matrix ``system`` returns is what ``simulate`` takes.
  k = system(KAPPA)
  t_final = simulate((k, u))
  return {"u": u, "t_final": t_final, "k": k, "initial_cost": np.array(fun(zero)[0]), "cost": np.array(result.fun), "iterations": np.array(result.nit)}


if __name__ == "__main__":
  out = main()
  inside = TARGET > 0
  print(f"{NODES} nodes, {STEPS} implicit steps, K with {out['k'].nnz} stored entries (lower triangle)")
  print(f"cost {float(out['initial_cost']):.3f} with the heaters off -> {float(out['cost']):.4f} after {int(out['iterations'])} L-BFGS-B iterations")
  print(f"final temperature: {out['t_final'][inside].mean():.2f} on the disc, {np.abs(out['t_final'][~inside]).max():.2f} at most elsewhere")
  print("mean heater power: " + ", ".join(f"{p:.1f}" for p in out["u"].mean(axis=0)))
  for fn in (system, simulate, objective):
    write_module(fn, GENERATED)
  print(f"generated C in {GENERATED}")
