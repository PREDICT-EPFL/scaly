"""Test problems for the interior-point work: the stored Maros–Mészáros subset in PIQP's form."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
from scipy import sparse

MM_DIR = Path(__file__).resolve().parents[1] / "data" / "maros_meszaros"
INFINITE = 1e19  # at or beyond this, a bound is absent: the set writes 1e20, sometimes rounded to 9.999999999999998e19


@dataclass(frozen=True)
class QP:
  """``minimize 1/2 x^T P x + c^T x + r`` subject to ``A x = b``, ``h_l <= G x <= h_u`` and
  ``x_l <= x <= x_u``; an absent bound is ``-inf`` or ``inf``. ``P`` holds both triangles."""

  name: str
  P: sparse.csc_array
  c: np.ndarray
  r: float
  A: sparse.csc_array
  b: np.ndarray
  G: sparse.csc_array
  h_l: np.ndarray
  h_u: np.ndarray
  x_l: np.ndarray
  x_u: np.ndarray

  @property
  def n(self) -> int:
    return self.P.shape[0]

  def objective(self, x: np.ndarray) -> float:
    return float(0.5 * x @ (self.P @ x) + self.c @ x + self.r)


def maros_meszaros_names() -> list[str]:
  return sorted(p.stem for p in MM_DIR.glob("*.npz"))


@dataclass(frozen=True)
class RawProblem:
  """A stored problem exactly as distributed: ``minimize 1/2 x^T P x + q^T x + r`` subject to
  ``l <= A x <= u``, with ``P`` in full and +-1e20 (or nearly) for an absent bound."""

  P: sparse.csc_array
  q: np.ndarray
  r: float
  A: sparse.csc_array
  l: np.ndarray  # noqa: E741
  u: np.ndarray


@cache
def raw(name: str) -> RawProblem:
  z = np.load(MM_DIR / f"{name}.npz")

  def csc(m: str) -> sparse.csc_array:
    return sparse.csc_array((z[f"{m}_data"], z[f"{m}_indices"], z[f"{m}_indptr"]), shape=tuple(int(d) for d in z[f"{m}_shape"]))

  return RawProblem(P=csc("P"), q=z["q"], r=float(z["r"]), A=csc("A"), l=z["l"], u=z["u"])


def maros_meszaros(name: str) -> QP:
  """One stored problem in PIQP's form. A row of ``A`` with a single coefficient of 1 is a bound on
  its variable (bounds on one variable intersect), a row with ``l == u`` is an equality, and every
  other row is a two-sided inequality, kept even when both sides are infinite."""
  p = raw(name)
  a = sparse.csr_array(p.A)
  lo = np.where(p.l <= -INFINITE, -np.inf, p.l)
  hi = np.where(p.u >= INFINITE, np.inf, p.u)
  n = a.shape[1]
  x_l, x_u = np.full(n, -np.inf), np.full(n, np.inf)
  counts = np.diff(a.indptr)
  eq, ineq = [], []
  for i in range(a.shape[0]):
    start = a.indptr[i]
    if counts[i] == 1 and a.data[start] == 1.0 and lo[i] != hi[i]:
      j = a.indices[start]
      x_l[j], x_u[j] = max(x_l[j], lo[i]), min(x_u[j], hi[i])
    elif lo[i] == hi[i]:
      eq.append(i)
    else:
      ineq.append(i)
  return QP(
    name=name,
    P=sparse.csc_array(p.P),
    c=np.asarray(p.q, dtype=np.float64),
    r=p.r,
    A=sparse.csc_array(a[eq]),
    b=hi[eq],
    G=sparse.csc_array(a[ineq]),
    h_l=lo[ineq],
    h_u=hi[ineq],
    x_l=x_l,
    x_u=x_u,
  )


def _qp(name: str, P, c, *, A=None, b=None, G=None, h_l=None, h_u=None, x_l=None, x_u=None) -> QP:
  n = len(c)
  empty = sparse.csc_array((0, n))
  G = sparse.csc_array(G) if G is not None else empty
  m = G.shape[0]
  return QP(
    name=name,
    P=sparse.csc_array(P, shape=(n, n)),
    c=np.asarray(c, dtype=np.float64),
    r=0.0,
    A=sparse.csc_array(A) if A is not None else empty,
    b=np.asarray(b if b is not None else [], dtype=np.float64),
    G=G,
    h_l=np.asarray(h_l, dtype=np.float64) if h_l is not None else np.full(m, -np.inf),
    h_u=np.asarray(h_u, dtype=np.float64) if h_u is not None else np.full(m, np.inf),
    x_l=np.asarray(x_l, dtype=np.float64) if x_l is not None else np.full(n, -np.inf),
    x_u=np.asarray(x_u, dtype=np.float64) if x_u is not None else np.full(n, np.inf),
  )


def infeasible_problems() -> dict[str, tuple[QP, str]]:
  """Small problems with no solution, and why: ``"primal"`` (no feasible point) or ``"dual"``
  (the objective is unbounded below on the feasible set)."""
  z2 = sparse.csc_array((2, 2))
  return {
    "box_contradiction": (_qp("box_contradiction", z2, [1.0, 1.0], G=np.array([[1.0, 1.0]]), h_l=[3.0], x_u=[1.0, 1.0]), "primal"),
    "parallel_equalities": (_qp("parallel_equalities", np.eye(2), [0.0, 0.0], A=np.array([[1.0, 1.0], [1.0, 1.0]]), b=[1.0, 2.0]), "primal"),
    "empty_slab": (_qp("empty_slab", np.eye(3), [0.0, 0.0, 0.0], G=np.array([[1.0, -1.0, 0.0], [-1.0, 1.0, 0.0]]), h_u=[-1.0, -1.0]), "primal"),
    "unbounded_lp": (_qp("unbounded_lp", z2, [-1.0, -1.0], G=np.array([[1.0, -1.0]]), h_u=[1.0], x_l=[0.0, 0.0]), "dual"),
    "unbounded_ray": (_qp("unbounded_ray", np.diag([1.0, 0.0]), [0.0, -1.0], x_l=[-1.0, 0.0]), "dual"),
  }


def mpc_qp(nx: int, nu: int, horizon: int, *, seed: int = 0, name: str | None = None) -> QP:
  """A linear MPC problem in the sparse (simultaneous) form: states ``x_0 .. x_N`` then inputs
  ``u_0 .. u_{N-1}``, dynamics as equalities, ``x_0`` fixed, input and state boxes, quadratic cost."""
  rng = np.random.default_rng(seed)
  a = np.eye(nx) + 0.1 * rng.standard_normal((nx, nx))
  a /= max(1.0, 1.05 * np.abs(np.linalg.eigvals(a)).max())
  b = 0.3 * rng.standard_normal((nx, nu))
  nz, nv = (horizon + 1) * nx, horizon * nu
  q, r = rng.uniform(0.5, 2.0, nx), rng.uniform(0.05, 0.2, nu)
  P = sparse.diags_array(np.r_[np.tile(q, horizon + 1), np.tile(r, horizon)]).tocsc()
  dyn = sparse.hstack(
    [
      sparse.kron(sparse.eye(horizon, horizon + 1, k=1), np.eye(nx)) - sparse.kron(sparse.eye(horizon, horizon + 1), a),
      -sparse.kron(sparse.eye(horizon), b),
    ]
  )
  init = sparse.hstack([sparse.eye(nx, nz), sparse.csc_array((nx, nv))])
  x0 = rng.uniform(-1.0, 1.0, nx)
  return _qp(
    name or f"mpc_{nx}x{nu}_N{horizon}",
    P,
    np.zeros(nz + nv),
    A=sparse.vstack([init, dyn], format="csc"),
    b=np.r_[x0, np.zeros(horizon * nx)],
    x_l=np.r_[np.full(nz, -2.0), np.full(nv, -0.5)],
    x_u=np.r_[np.full(nz, 2.0), np.full(nv, 0.5)],
  )
