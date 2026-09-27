"""A tour of the derivative wrappers on a Lennard-Jones cluster of seven atoms.

The energy of ``N`` atoms at positions ``x`` (``3N`` coordinates) is

    E(x) = sum_{i<j} 4 (r_ij^-12 - r_ij^-6),

written once with static pair tables (``sc.gather``). Everything else is derived from it, and each
derived object is itself a ``Function`` that compiles to C:

* ``sc.gradient``: the forces ``-dE/dx``, one reverse sweep;
* ``sc.hessian``: the vibrational (normal-mode) analysis at the minimum, where six eigenvalues
  vanish for the three translations and three rotations;
* ``sc.forward`` of the gradient: a Hessian-vector product without the Hessian, the kernel of a
  Lanczos or truncated-Newton method;
* ``sc.jacobian`` and ``sc.sparse_jacobian`` of the squared pair lengths: the rigidity matrix of the
  bar framework, 6 nonzeros per row, whose rank ``3N - 6`` says the cluster is rigid;
* ``sc.adjoint`` of the same map: ``R^T t``, the nodal forces of a set of bar tensions ``t``;
* ``Function.factory``: energy, gradient and Hessian in one C function, driving SciPy's
  ``trust-exact`` from a perturbed pentagonal bipyramid to the global minimum, ``E = -16.505384``.

Every derivative is checked against central differences of the energy. The generated C lands in
``examples/generated/derivatives_tour/``.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import optimize

import scaly as sc
from scaly.codegen import write_module

GENERATED = Path(__file__).resolve().parent / "generated" / "derivatives_tour"
N = 7
PAIRS = np.array(list(combinations(range(N), 2)))
P = len(PAIRS)
E_MIN = -16.505384  # the LJ7 global minimum (Wales and Doye 1997)


def _pair_vectors(x: sc.Expr) -> sc.Expr:
  """``x_i - x_j`` for every pair, shape ``(P, 3)``, read with static index tables."""
  xyz = np.arange(3)
  return sc.gather(x, 3 * PAIRS[:, :1] + xyz) - sc.gather(x, 3 * PAIRS[:, 1:] + xyz)


@sc.function(3 * N, output="len2")
def squared_lengths(x: sc.Expr) -> sc.Expr:
  d = _pair_vectors(x)
  return (d * d) @ sc.const(np.ones(3))


@sc.function(3 * N, output="E")
def energy(x: sc.Expr) -> sc.Expr:
  s3 = (1.0 / squared_lengths(x)) ** 3
  return (4.0 * (s3 * s3 - s3)).sum()


# One output and one input each, so the derivatives need not name them.
gradient = sc.gradient(energy)  # (x) -> grad_E_x
hessian = sc.hessian(energy)  # (x) -> hess_E_x_x
hvp = sc.forward(gradient)  # (x, fwd:x) -> H v
rigidity = sc.jacobian(squared_lengths)  # dense (P, 3N)
rigidity_sparse = sc.sparse_jacobian(squared_lengths)  # compact values + pattern
bar_forces = sc.adjoint(squared_lengths)  # (x, lam:len2) -> R^T t
newton_oracle = energy.factory("lj_newton", ["x"], ["E", sc.factory.Grad("E", "x"), sc.factory.Hess("E", "x")])


def bipyramid(seed: int = 0, noise: float = 0.08) -> np.ndarray:
  """A pentagonal bipyramid with unit-ish bonds, perturbed."""
  ring, apex = 0.953, 0.588
  angles = 2 * np.pi * np.arange(5) / 5
  x = np.concatenate([np.stack([ring * np.cos(angles), ring * np.sin(angles), np.zeros(5)], axis=1), [[0, 0, apex], [0, 0, -apex]]])
  return (x + noise * np.random.default_rng(seed).standard_normal(x.shape)).reshape(-1)


def energy_numpy(x: np.ndarray) -> float:
  p = x.reshape(N, 3)
  r2 = np.sum((p[PAIRS[:, 0]] - p[PAIRS[:, 1]]) ** 2, axis=1)
  return float(np.sum(4 * (r2**-6 - r2**-3)))


def central_difference(fun, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
  return np.array([(fun(x + eps * e) - fun(x - eps * e)) / (2 * eps) for e in np.eye(x.size)])


def main() -> dict:
  x0 = bipyramid()
  pattern = rigidity_sparse.output_sparsities[0]
  assert pattern is not None
  rng = np.random.default_rng(1)
  v = rng.standard_normal(3 * N)

  checks = {
    "gradient": np.abs(gradient(x0) - central_difference(energy_numpy, x0)).max(),
    "hessian": np.abs(hessian(x0) - central_difference(lambda y: gradient(y), x0)).max(),
    "hvp": np.abs(hvp(x0, v) - hessian(x0) @ v).max(),
    "sparse = dense rigidity": np.abs(rigidity_sparse(x0).reshape(-1) - rigidity(x0)[pattern.rows, pattern.cols]).max(),
  }
  t = rng.standard_normal(P)
  checks["adjoint = R^T t"] = np.abs(bar_forces(x0, t) - rigidity(x0).T @ t).max()

  result = optimize.minimize(
    lambda x: newton_oracle(x)[0],
    x0,
    jac=lambda x: newton_oracle(x)[1],
    hess=lambda x: newton_oracle(x)[2],
    method="trust-exact",
    options={"gtol": 1e-10},
  )
  modes = np.linalg.eigvalsh(hessian(result.x))
  return {
    "checks": checks,
    "x": result.x,
    "energy": np.array(result.fun),
    "iterations": np.array(result.nit),
    "modes": modes,
    "rigidity_rank": np.array(np.linalg.matrix_rank(rigidity(result.x))),
    "rigidity_nnz": np.array(pattern.nnz),
  }


if __name__ == "__main__":
  out = main()
  for name, err in out["checks"].items():
    print(f"{name:>24}: largest difference {err:.1e}")
  print(
    f"trust-exact on the factory's E, grad, Hess: E = {float(out['energy']):.6f} after {int(out['iterations'])} iterations (global minimum {E_MIN})"
  )
  print(
    f"Hessian eigenvalues at the minimum: 6 near zero (largest |.| {np.abs(out['modes'][:6]).max():.1e}), softest vibration {out['modes'][6]:.3f}"
  )
  print(f"rigidity matrix {P} x {3 * N}: rank {int(out['rigidity_rank'])} = 3N - 6, {int(out['rigidity_nnz'])} nonzeros of {P * 3 * N}")
  for fn in (energy, gradient, hessian, hvp, rigidity_sparse, bar_forces, newton_oracle):
    write_module(fn, GENERATED)
  print(f"generated C for {', '.join(f.name for f in (energy, gradient, hessian, hvp, rigidity_sparse, bar_forces, newton_oracle))} in {GENERATED}")
