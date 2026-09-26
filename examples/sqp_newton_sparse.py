"""An SQP Newton step with a sparse KKT system, for swinging a pendulum up in multiple-shooting form.

The decision vector ``w`` stacks ``N + 1`` states ``(angle, rate)`` and then ``N`` torques. The
equality constraints are the initial state and the explicit-Euler dynamics. One Newton step on the
KKT conditions solves

    [[H + rho I, J^T], [J, -delta I]] [dw; dlam] = -[grad f + J^T lam; c]

where ``H`` is the Hessian of the Lagrangian and ``J`` the constraint Jacobian, both sparse. The KKT
matrix is assembled as a ``SparseMatrix`` from the lower triangle of ``H`` and the block ``J``,
factored by the generated sparse ``L D L^T`` and solved. The whole step is one generated
``Function``; the loop below it is plain Python.

``H`` itself may be indefinite. The KKT matrix has ``NW`` positive and ``NC`` negative eigenvalues
exactly when its Schur complement ``H + rho I + J^T J / delta`` is positive definite, which for a
small ``delta`` and ``J`` of full row rank is ``H + rho I`` positive definite on the null space of
``J``: the condition for a descent step. By Sylvester's law of inertia these counts are the signs
of the pivots in ``D``, so the step returns ``fact.inertia()`` and the loop raises ``rho`` until
they are right, as SQP and interior-point codes do. ``fact.health(x=step)`` checks that every pivot
is finite and nonzero and the step finite.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import scaly as sc
from scaly import linalg
from scaly.codegen import write_module

GENERATED = Path(__file__).resolve().parent / "generated" / "sqp_newton_sparse"

N, H_STEP = 30, 0.1
NW, NC = 3 * N + 2, 2 * N + 2  # variables: 2 (N + 1) states and N torques; constraints: 2 (N + 1)
TARGET = 2.0  # the final angle, in radians (the pendulum hangs at 0)
DELTA = 1e-10  # the dual regularization, which keeps the lower block negative definite


def dynamics(w: sc.Expr, z0: sc.Expr) -> sc.Expr:
  """The constraint vector: the initial state, then ``z[k+1] - z[k] - h f(z[k], u[k])``."""
  z, u = w[: 2 * N + 2].reshape((N + 1, 2)), w[2 * N + 2 :]
  angle, rate = z[:-1, 0], z[:-1, 1]
  f = sc.stack([rate, -9.81 * angle.sin() - 0.1 * rate + u], axis=1)
  return sc.concat([z[0] - z0, (z[1:] - z[:-1] - H_STEP * f).reshape((2 * N,))])


def cost(w: sc.Expr) -> sc.Expr:
  z, u = w[: 2 * N + 2].reshape((N + 1, 2)), w[2 * N + 2 :]
  target = sc.const(np.array([TARGET, 0.0]))
  return 10.0 * sc.sumsqr(z[-1] - target) + 0.1 * sc.sumsqr(z[:, 1]) + 0.01 * sc.sumsqr(u)


@sc.function(
  sc.G(sc.L("w", NW), sc.L("lam", NC), sc.L("z0", 2), sc.L("rho", ())),
  sc.G(sc.L("w_next", NW), sc.L("lam_next", NC), sc.L("kkt_residual", ...), sc.L("inertia", 3), sc.L("healthy", ...)),
)
def newton_step(inputs: tuple[sc.Expr, sc.Expr, sc.Expr, sc.Expr]) -> tuple[sc.Expr, sc.Expr, sc.Expr, sc.Expr, sc.Expr]:
  w, lam, z0, rho = inputs
  c = dynamics(w, z0)
  lagrangian = cost(w) + (lam * c).sum()
  grad = sc.gradient(lagrangian, w)
  hess = linalg.SparseMatrix.from_sparse_jacobian(sc.sparse_hessian(lagrangian, w, triangle="lower"))
  jac = linalg.SparseMatrix.from_sparse_jacobian(sc.sparse_jacobian(c, w))
  kkt = linalg.SparseMatrix.block([[hess.add_diagonal(rho), None], [jac, linalg.SparseMatrix.identity(NC) * (-DELTA)]])
  fact = linalg.SparseLDL(kkt)  # the lower triangle is all it reads
  rhs = -sc.concat([grad, c])
  step = fact.solve(rhs, refine=1)
  return w + step[:NW], lam + step[NW:], sc.norm_inf(rhs), fact.inertia(), fact.health(x=step)


def main(iterations: int = 20) -> dict[str, np.ndarray]:
  z0 = np.zeros(2)
  w, lam = np.zeros(NW), np.zeros(NC)
  w[2 * N + 2 :] = 1.0  # a nonzero initial torque so the pendulum starts moving
  residuals, rhos = [], []
  rho = 0.0
  for _ in range(iterations):
    # Inertia correction: the smallest rho in 0, 1e-4, 1e-3, ... that gives the expected signs.
    rho = 0.0 if rho < 1e-3 else rho / 10.0
    while True:
      w_next, lam_next, residual, inertia, healthy = newton_step((w, lam, z0, np.array(rho)))
      if healthy and tuple(inertia) == (NW, NC, 0):
        break
      rho = 1e-4 if rho == 0.0 else 10.0 * rho
    residuals.append(float(residual))
    rhos.append(rho)
    if residual < 1e-9:
      break
    w, lam = w_next, lam_next
  return {"w": w, "lam": lam, "residuals": np.array(residuals), "rhos": np.array(rhos)}


if __name__ == "__main__":
  result = main()
  for k, r in enumerate(result["residuals"]):
    print(f"iteration {k:2d}  ||KKT residual||_inf = {r:.2e}  rho = {result['rhos'][k]:.0e}")
  print("final angle", result["w"][2 * N], "rate", result["w"][2 * N + 1])
  write_module(newton_step, GENERATED)
  print(f"generated C in {GENERATED}")
