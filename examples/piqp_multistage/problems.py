"""Random multistage QPs: a linear MPC problem written stage after stage, with optional parameters shared by every stage."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy import sparse


class QP(NamedTuple):
  """``minimize 1/2 x^T P x + c^T x`` subject to ``A x = b``, ``h_l <= G x <= h_u`` and ``x_l <= x <= x_u``."""

  P: sparse.csc_array
  c: np.ndarray
  A: sparse.csc_array
  b: np.ndarray
  G: sparse.csc_array
  h_l: np.ndarray
  h_u: np.ndarray
  x_l: np.ndarray
  x_u: np.ndarray


def mpc_qp(horizon: int, nx: int, nu: int, n_global: int = 0, density: float = 0.4, seed: int = 0) -> QP:
  """A linear MPC problem over ``horizon`` stages with sparse random stage matrices.

  The variables are ordered ``x_0, u_0, x_1, u_1, ..., x_N`` and then ``n_global`` parameters
  ``theta``. The dynamics are ``x_{k+1} = A_k x_k + B_k u_k + E_k theta``, the initial state is
  fixed, each stage has a sparse positive definite cost on ``(x_k, u_k)`` and the inequalities
  bound ``C_k x_k + D_k u_k``. The parameters couple every stage to the last variables, which
  gives ``Psi`` its arrow.
  """
  rng = np.random.default_rng(seed)
  nz = nx + nu
  n = horizon * nz + nx + n_global
  stage = lambda k: k * nz  # noqa: E731
  globals_ = n - n_global

  def rand(rows: int, cols: int) -> sparse.coo_array:
    return sparse.random_array((rows, cols), density=density, rng=rng, data_sampler=rng.standard_normal)

  # cost: a positive definite block per stage, and on the terminal state and the parameters
  P = sparse.lil_array((n, n))
  for k in range(horizon):
    M = rand(nz, nz).toarray()
    P[stage(k) : stage(k + 1), stage(k) : stage(k + 1)] = M @ M.T + np.eye(nz)
  M = rand(nx, nx).toarray()
  P[stage(horizon) : globals_, stage(horizon) : globals_] = M @ M.T + np.eye(nx)
  if n_global:
    P[globals_:, globals_:] = np.eye(n_global)

  # dynamics and initial state
  A = sparse.lil_array((nx * (horizon + 1), n))
  A[:nx, :nx] = np.eye(nx)
  for k in range(horizon):
    rows = slice(nx * (k + 1), nx * (k + 2))
    A[rows, stage(k) : stage(k) + nx] = rand(nx, nx).toarray() + np.eye(nx)
    A[rows, stage(k) + nx : stage(k + 1)] = rand(nx, nu).toarray()
    A[rows, stage(k + 1) : stage(k + 1) + nx] = -np.eye(nx)
    if n_global:
      A[rows, globals_:] = rand(nx, n_global).toarray()
  b = np.concatenate([rng.standard_normal(nx), np.zeros(nx * horizon)])

  # stage constraints on (x_k, u_k)
  nc = max(1, nz // 2)
  G = sparse.lil_array((nc * horizon, n))
  for k in range(horizon):
    G[k * nc : (k + 1) * nc, stage(k) : stage(k + 1)] = rand(nc, nz).toarray()

  m = G.shape[0]
  return QP(
    P=sparse.csc_array(P),
    c=rng.standard_normal(n),
    A=sparse.csc_array(A),
    b=b,
    G=sparse.csc_array(G),
    h_l=-np.ones(m),
    h_u=np.ones(m),
    x_l=np.full(n, -np.inf),
    x_u=np.full(n, np.inf),
  )


def banded_qp(n: int, bandwidth: int, density: float, n_global: int = 0, seed: int = 0) -> QP:
  """An unconstrained QP whose ``P`` has a random pattern within ``bandwidth`` of the diagonal, and ``n_global`` last variables coupled to a random half of the others.

  Its blocks come out irregular, unlike ``mpc_qp``'s, which reaches more of the decisions of
  ``extract_arrow_structure``.
  """
  rng = np.random.default_rng(seed)
  M = sparse.random_array((n, n), density=density, rng=rng, data_sampler=rng.standard_normal).tocoo()
  keep = np.abs(M.row - M.col) <= bandwidth
  coupled = np.zeros((n, n))
  for g in range(n - n_global, n):
    coupled[g, rng.choice(n - n_global, size=(n - n_global) // 2, replace=False)] = 1.0
  B = sparse.coo_array((M.data[keep], (M.row[keep], M.col[keep])), shape=(n, n)) + sparse.coo_array(coupled)
  P = B @ B.T + n * sparse.eye_array(n)
  return QP(
    P=sparse.csc_array(P),
    c=rng.standard_normal(n),
    A=sparse.csc_array((0, n)),
    b=np.zeros(0),
    G=sparse.csc_array((0, n)),
    h_l=np.zeros(0),
    h_u=np.zeros(0),
    x_l=np.full(n, -np.inf),
    x_u=np.full(n, np.inf),
  )
