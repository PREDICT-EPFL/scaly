"""The TinyMPC problem description, its cached Riccati quantities, and a NumPy reference solver.

TinyMPC (Nguyen et al., ICRA 2024; Schoedel et al., Conic-TinyMPC, 2024) solves the linear MPC problem

    minimize    sum_k 1/2 (x_k - xref_k)' Q (x_k - xref_k) + 1/2 (u_k - uref_k)' R (u_k - uref_k)
    subject to  x_{k+1} = A x_k + B u_k + f,   x_0 given,
                x_k in [x_min, x_max],  u_k in [u_min, u_max],  second-order cones on parts of x_k, u_k

by ADMM. Every constraint set gets a slack copy of the trajectory, so the primal step is an
unconstrained LQR problem with the penalty ``rho`` folded into ``Q`` and ``R``. Its Riccati gain is
the infinite-horizon one, computed once: each iteration is then only a backward pass for the
affine terms, a forward rollout, projections and dual updates.

``Problem`` holds one specialization; ``cache`` computes the Riccati quantities exactly as the
library's ``tiny_precompute_and_set_cache`` does. ``reference_solve`` is a line-by-line NumPy port
of the library's ``solve`` (TinyMPC ``src/tinympc/admm.cpp``, adaptive rho off), used as the test
oracle for the generated solver. It keeps the library's conventions where they differ from the
textbook ADMM, so that iterates and iteration counts can be compared:

- the linear cost uses ``Q + rho I`` (and ``R + rho I``) times the reference, as the library's
  ``work->Q`` stores the penalized diagonal;
- the Riccati cache folds in one ``rho`` however many constraint sets act on a variable;
- a disabled bound still has its slack and dual: ``vnew = x + g`` without the clip;
- termination tests the box slacks only, and does not update ``v``/``z`` in the iteration it stops.
- the cone projection is the library's ``project_soc``, a projection in the metric that scales the
  cone's axis by ``mu`` (the Euclidean one only for ``mu = 1``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from scaly.ocp.tinyadmm import STATE_FIELDS, Cone, Settings, tinympc_cache
from scaly.ocp.tinyadmm import LQRCache as Cache


@dataclass(frozen=True, eq=False)
class Problem:
  """One TinyMPC specialization: dynamics, diagonal weights, horizon, penalty, constraint structure."""

  A: np.ndarray
  B: np.ndarray
  Q: np.ndarray  # diagonal, (nx,)
  R: np.ndarray  # diagonal, (nu,)
  N: int
  rho: float
  f: np.ndarray | None = None
  settings: Settings = field(default_factory=Settings)
  state_cones: tuple[Cone, ...] = ()
  input_cones: tuple[Cone, ...] = ()

  @property
  def nx(self) -> int:
    return self.A.shape[0]

  @property
  def nu(self) -> int:
    return self.B.shape[1]

  @property
  def fdyn(self) -> np.ndarray:
    return np.zeros(self.nx) if self.f is None else np.asarray(self.f, dtype=float)


def cache(p: Problem) -> Cache:
  """The library's recursion (``scaly.ocp.tinyadmm.tinympc_cache``): ``P`` starts at ``rho I`` and
  stops when ``K`` moves by less than 1e-5."""
  return tinympc_cache(p.A, p.B, p.Q, p.R, p.rho, p.fdyn)


def zero_state(p: Problem) -> dict[str, np.ndarray]:
  sx, su = (p.N, p.nx), (p.N - 1, p.nu)
  return {name: np.zeros(sx if name in ("x", "v", "vnew", "g", "gc") else su) for name in STATE_FIELDS}


def project_soc(s: np.ndarray, mu: float) -> np.ndarray:
  """The library's ``project_soc``: zero below the polar cone, ``s`` inside, else onto the boundary."""
  u0 = s[-1] * mu
  u1 = s[:-1]
  a = np.linalg.norm(u1)
  if a <= -u0:
    return np.zeros_like(s)
  if a <= u0:
    return s
  return 0.5 * (1.0 + u0 / a) * np.append(u1, a / mu)


def _project_cones(w: np.ndarray, cones: tuple[Cone, ...]) -> np.ndarray:
  w = w.copy()
  for k in range(w.shape[0]):
    for c in cones:
      w[k, c.start : c.start + c.dim] = project_soc(w[k, c.start : c.start + c.dim], c.mu)
  return w


@dataclass
class Bounds:
  x_min: np.ndarray  # (N, nx)
  x_max: np.ndarray
  u_min: np.ndarray  # (N - 1, nu)
  u_max: np.ndarray


def reference_solve(
  p: Problem, c: Cache, state: dict[str, np.ndarray], x0: np.ndarray, xref: np.ndarray, uref: np.ndarray, bounds: Bounds
) -> tuple[dict[str, np.ndarray], int, bool]:
  """One call of the library's ``tiny_solve`` after ``tiny_set_x0``/``tiny_set_x_ref``/``tiny_set_u_ref``.
  Returns the new state, the iteration count and whether it converged."""
  s = {k: v.copy() for k, v in state.items()}
  st = p.settings
  rho, n = p.rho, p.N
  qw, rw = p.Q + rho, p.R + rho
  a, b, f = p.A, p.B, p.fdyn
  s["x"][0] = x0
  vcnew = s["x"].copy()
  zcnew = s["u"].copy()
  for it in range(st.max_iter):
    # update_linear_cost
    q = -(xref * qw)
    q = q - rho * (s["vnew"] - s["g"])
    if p.state_cones:
      q = q - rho * (vcnew - s["gc"])
    r = -(uref * rw)
    r = r - rho * (s["znew"] - s["y"])
    if p.input_cones:
      r = r - rho * (zcnew - s["yc"])
    pp = np.zeros((n, p.nx))
    pp[n - 1] = -(xref[n - 1] @ c.P)
    pp[n - 1] = pp[n - 1] - rho * (s["vnew"][n - 1] - s["g"][n - 1])
    if p.state_cones:
      pp[n - 1] = pp[n - 1] - rho * (vcnew[n - 1] - s["gc"][n - 1])
    # backward_pass_grad
    d = np.zeros((n - 1, p.nu))
    for i in range(n - 2, -1, -1):
      d[i] = c.Quu_inv @ (b.T @ pp[i + 1] + r[i] + c.BPf)
      pp[i] = q[i] + c.AmBKt @ pp[i + 1] - c.K.T @ r[i] + c.APf
    # forward_pass
    x, u = s["x"], s["u"]
    for i in range(n - 1):
      u[i] = -(c.K @ x[i]) - d[i]
      x[i + 1] = a @ x[i] + b @ u[i] + f
    # update_slack
    vnew = x + s["g"]
    znew = u + s["y"]
    if st.en_state_bound:
      vnew = np.minimum(bounds.x_max, np.maximum(bounds.x_min, vnew))
    if st.en_input_bound:
      znew = np.minimum(bounds.u_max, np.maximum(bounds.u_min, znew))
    if p.state_cones:
      vcnew = _project_cones(x + s["gc"], p.state_cones)
    if p.input_cones:
      zcnew = _project_cones(u + s["yc"], p.input_cones)
    s["vnew"], s["znew"] = vnew, znew
    # update_dual
    s["g"] = s["g"] + x - vnew
    s["y"] = s["y"] + u - znew
    if p.state_cones:
      s["gc"] = s["gc"] + x - vcnew
    if p.input_cones:
      s["yc"] = s["yc"] + u - zcnew
    # termination_condition
    prs = np.abs(x - vnew).max()
    drs = np.abs(s["v"] - vnew).max() * rho
    pri = np.abs(u - znew).max()
    dri = np.abs(s["z"] - znew).max() * rho
    if prs < st.abs_pri_tol and pri < st.abs_pri_tol and drs < st.abs_dua_tol and dri < st.abs_dua_tol:
      return s, it + 1, True
    s["v"], s["z"] = vnew.copy(), znew.copy()
  return s, st.max_iter, False


class ReferenceSolver:
  """``reference_solve`` with its warm-start state kept between calls."""

  def __init__(self, p: Problem, bounds: Bounds, c: Cache | None = None) -> None:
    self.problem, self.bounds = p, bounds
    self.cache = cache(p) if c is None else c
    self.state = zero_state(p)

  def __call__(self, x0: np.ndarray, xref: np.ndarray, uref: np.ndarray) -> tuple[np.ndarray, int, bool]:
    self.state, iterations, solved = reference_solve(self.problem, self.cache, self.state, x0, xref, uref, self.bounds)
    return self.state["u"][0].copy(), iterations, solved
