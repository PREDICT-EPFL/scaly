"""Compressed sensing: recovering a sparse signal from few measurements with the LASSO, solved by ADMM.

A signal ``x`` of length ``N`` with a handful of nonzero entries is measured through a fixed random
matrix, ``b = A x + noise``, with far fewer measurements ``M`` than unknowns. The LASSO estimate

    minimize 1/2 |A x - b|^2 + lam |x|_1

recovers it. ADMM splits the two terms, ``x = z``, and repeats

    x = (A^T A + rho I)^{-1} (A^T b + rho (z - u))
    z = soft(x + u, lam / rho),      soft(v, k) = copysign(max(|v| - k, 0), v)
    u = u + x - z

until the primal residual ``|x - z|`` and the dual residual ``rho |z - z_prev|`` (both ``norm_inf``)
are below a tolerance. The ``x`` step never forms the ``N x N`` matrix: by the Woodbury identity it
needs only the ``M x M`` Cholesky factor of ``rho I + A A^T``, computed once from the run-time
``rho`` by the generated dense ``cholesky`` and applied with ``cho_solve`` at every iteration.

The solve is one generated ``Function`` of ``b``, ``lam`` and ``rho``. Its outputs are the estimate,
the number of iterations and the recovered support as a ``bool`` vector.
"""

from __future__ import annotations

import numpy as np

import scaly as sc
from scaly import linalg

M, N, K_TRUE = 40, 120, 6
TOL = 1e-8
MAX_ITER = 5000
A_CONST = np.random.default_rng(1).standard_normal((M, N)) / np.sqrt(M)


def soft(v: sc.Expr, k: sc.Expr) -> sc.Expr:
  return sc.copysign(sc.maximum(v.abs() - k, 0.0), v)


def x_step(l_factor: sc.Expr, atb: sc.Expr, z: sc.Expr, u: sc.Expr, rho: sc.Expr) -> sc.Expr:
  """``(A^T A + rho I)^{-1} q`` as ``(q - A^T (rho I + A A^T)^{-1} A q) / rho``."""
  a = sc.const(A_CONST)
  q = atb + rho * (z - u)
  return (q - a.T @ linalg.cho_solve(l_factor, a @ q)) / rho


# A loop body reads only its carry, so the factor and the parameters travel in it:
# carry = [L (M*M) | A^T b (N) | lam | rho | z (N) | u (N) | primal residual | dual residual].
_L, _ATB, _LAM, _RHO, _Z, _U, _RES = 0, M * M, M * M + N, M * M + N + 1, M * M + N + 2, M * M + 2 * N + 2, M * M + 3 * N + 2
_carry = sc.sym("carry", _RES + 2)
_l, _atb = _carry[_L:_ATB].reshape((M, M)), _carry[_ATB:_LAM]
_lam, _rho, _z, _u = _carry[_LAM], _carry[_RHO], _carry[_Z:_U], _carry[_U:_RES]
_x = x_step(_l, _atb, _z, _u, _rho)
_z_new = soft(_x + _u, _lam / _rho)
_u_new = _u + _x - _z_new
_residuals = sc.stack([sc.norm_inf(_x - _z_new), _rho * sc.norm_inf(_z_new - _z)])
admm_iteration = sc.Function._from_exprs(
  "admm_iteration", [_carry], [sc.concat([_carry[:_Z], _z_new, _u_new, _residuals])], ["carry"], ["next"]
)
not_converged = sc.Function._from_exprs(
  "admm_not_converged", [_carry], [sc.logical_or(sc.greater(_carry[_RES], TOL), sc.greater(_carry[_RES + 1], TOL))], ["carry"], ["go_on"]
)


@sc.function(
  sc.G(sc.L("b", M), sc.L("lam", ()), sc.L("rho", ())),
  sc.G(sc.L("x", N), sc.L("iterations", ...), sc.L("support", ...)),
)
def lasso(inputs: tuple[sc.Expr, sc.Expr, sc.Expr]) -> tuple[sc.Expr, sc.Expr, sc.Expr]:
  b, lam, rho = inputs
  a = sc.const(A_CONST)
  l_factor = linalg.cholesky(a @ a.T + rho * sc.const(np.eye(M)))
  start = sc.concat([
    l_factor.reshape((M * M,)),
    a.T @ b,
    lam.reshape((1,)),
    rho.reshape((1,)),
    sc.const(np.zeros(2 * N)),
    sc.const(np.full(2, np.inf)),
  ])
  carry, iterations = sc.while_loop(not_converged, admm_iteration, start, max_iter=MAX_ITER)
  z = carry[_Z:_U]
  return z, iterations, sc.not_equal(z, 0.0)


def problem(seed: int = 0, noise: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
  """A ``K_TRUE``-sparse signal and its noisy measurements."""
  rng = np.random.default_rng(seed)
  x = np.zeros(N)
  x[rng.choice(N, K_TRUE, replace=False)] = rng.choice([-1.0, 1.0], K_TRUE) * rng.uniform(1.0, 2.0, K_TRUE)
  return x, A_CONST @ x + noise * rng.standard_normal(M)


def main(seed: int = 0, lam: float = 0.02, rho: float = 1.0) -> dict[str, np.ndarray]:
  truth, b = problem(seed)
  x, iterations, support = lasso((b, np.array(lam), np.array(rho)))
  return {"truth": truth, "b": b, "lam": np.array(lam), "x": x, "iterations": iterations, "support": support}


if __name__ == "__main__":
  out = main()
  true = np.flatnonzero(out["truth"])
  largest = np.sort(np.argsort(-np.abs(out["x"]))[:K_TRUE])
  print(f"{int(out['iterations'])} ADMM iterations, {int(out['support'].sum())} nonzeros")
  print(f"the {K_TRUE} largest at {largest.tolist()}, the true support {true.tolist()}")
  print(f"relative error {np.linalg.norm(out['x'] - out['truth']) / np.linalg.norm(out['truth']):.3f}")
  for rho in (0.3, 3.0):
    print(f"rho = {rho}: {int(main(rho=rho)['iterations'])} iterations")
