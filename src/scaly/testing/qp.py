"""Reference QPs: the stored Maros–Mészáros subset in PIQP's form, random QPs and LPs, MPC-shaped QPs, and QPs with no solution."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
from scipy import sparse

MM_DIR = Path(__file__).resolve().parent / "data" / "maros_meszaros"
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
  """The names of the stored Maros–Meszaros problems, sorted."""
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
  """The stored problem ``name`` as distributed, read once."""
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


def make_qp(name: str, P, c, *, A=None, b=None, G=None, h_l=None, h_u=None, x_l=None, x_u=None) -> QP:
  """A ``QP`` from dense or sparse data: ``A x = b``, ``h_l <= G x <= h_u`` and ``x_l <= x <= x_u``,
  each absent part empty or unbounded."""
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
    "box_contradiction": (make_qp("box_contradiction", z2, [1.0, 1.0], G=np.array([[1.0, 1.0]]), h_l=[3.0], x_u=[1.0, 1.0]), "primal"),
    "parallel_equalities": (make_qp("parallel_equalities", np.eye(2), [0.0, 0.0], A=np.array([[1.0, 1.0], [1.0, 1.0]]), b=[1.0, 2.0]), "primal"),
    "empty_slab": (make_qp("empty_slab", np.eye(3), [0.0, 0.0, 0.0], G=np.array([[1.0, -1.0, 0.0], [-1.0, 1.0, 0.0]]), h_u=[-1.0, -1.0]), "primal"),
    "unbounded_lp": (make_qp("unbounded_lp", z2, [-1.0, -1.0], G=np.array([[1.0, -1.0]]), h_u=[1.0], x_l=[0.0, 0.0]), "dual"),
    "unbounded_ray": (make_qp("unbounded_ray", np.diag([1.0, 0.0]), [0.0, -1.0], x_l=[-1.0, 0.0]), "dual"),
  }


def mpc_qp(nx: int, nu: int, horizon: int, *, seed: int = 0, path_rows: int = 0, name: str | None = None) -> QP:
  """A linear MPC problem in the sparse (simultaneous) form: states ``x_0 .. x_N`` then inputs
  ``u_0 .. u_{N-1}``, dynamics as equalities, ``x_0`` fixed, input and state boxes, quadratic cost.
  ``path_rows`` adds that many two-sided inequality rows a stage over ``x_k`` and ``u_k``, with
  random coefficients and bounds at half of what the boxes allow, so that some of them bind."""
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
  x_l, x_u = np.r_[np.full(nz, -2.0), np.full(nv, -0.5)], np.r_[np.full(nz, 2.0), np.full(nv, 0.5)]
  G, reach = None, None
  if path_rows:
    # Drawn from a generator of their own, so the rest of the problem is the one without them.
    coef = np.random.default_rng([seed, path_rows]).standard_normal((horizon * path_rows, nx + nu))
    stage = np.repeat(np.arange(horizon), path_rows)
    cols = np.concatenate([stage[:, None] * nx + np.arange(nx), nz + stage[:, None] * nu + np.arange(nu)], axis=1)
    G = sparse.csc_array((coef.ravel(), (np.repeat(np.arange(horizon * path_rows), nx + nu), cols.ravel())), shape=(horizon * path_rows, nz + nv))
    reach = 0.5 * (abs(G) @ x_u)
  return make_qp(
    name or f"mpc_{nx}x{nu}_N{horizon}" + (f"_r{path_rows}" if path_rows else ""),
    P,
    np.zeros(nz + nv),
    A=sparse.vstack([init, dyn], format="csc"),
    b=np.r_[x0, np.zeros(horizon * nx)],
    G=G,
    h_l=None if reach is None else -reach,
    h_u=reach,
    x_l=x_l,
    x_u=x_u,
  )


def kkt_residuals(qp: QP, *, x, y, z_l, z_u, z_bl, z_bu, **_) -> tuple[float, float]:
  """Primal and dual residuals (infinity norms) of a solution in PIQP's layout, from the problem data:
  ``P x + c + A^T y + G^T (z_u - z_l) + z_bu - z_bl = 0`` and the constraints."""
  primal = [np.abs(qp.A @ x - qp.b).max(initial=0.0)]
  gx = qp.G @ x
  primal += [np.maximum(qp.h_l - gx, 0.0).max(initial=0.0), np.maximum(gx - qp.h_u, 0.0).max(initial=0.0)]
  primal += [np.maximum(qp.x_l - x, 0.0).max(initial=0.0), np.maximum(x - qp.x_u, 0.0).max(initial=0.0)]
  grad = qp.P @ x + qp.c + qp.A.T @ y + qp.G.T @ (z_u - z_l) + z_bu - z_bl
  return max(primal), float(np.abs(grad).max(initial=0.0))


def random_qp(n: int, m: int, p: int, *, density: float = 0.15, seed: int = 0, lp: bool = False) -> QP:
  """A random feasible QP (an LP with ``lp=True``): ``P = M^T M + 1e-2 I`` sparse, ``m`` two-sided or
  one-sided inequalities and ``p`` equalities around a random point, and boxes on half the variables."""
  rng = np.random.default_rng(seed)
  if lp:
    P = sparse.csc_array((n, n))
  else:
    M = sparse.random_array((n, n), density=density, rng=rng)
    P = sparse.csc_array(M.T @ M + 1e-2 * sparse.eye_array(n))
  A = sparse.random_array((p, n), density=max(density, 2.0 / n), rng=rng, data_sampler=rng.standard_normal)
  G = sparse.random_array((m, n), density=max(density, 2.0 / n), rng=rng, data_sampler=rng.standard_normal)
  x0 = rng.standard_normal(n)
  gx = G @ x0
  kind = rng.integers(0, 3, m)  # 0: lower only, 1: upper only, 2: both
  h_l = np.where(kind != 1, gx - rng.uniform(0.0, 1.0, m), -np.inf)
  h_u = np.where(kind != 0, gx + rng.uniform(0.0, 1.0, m), np.inf)
  boxed = rng.random(n) < 0.5
  x_l = np.where(boxed, x0 - rng.uniform(0.0, 2.0, n), -np.inf)
  x_u = np.where(boxed, x0 + rng.uniform(0.0, 2.0, n), np.inf)
  c = rng.standard_normal(n)
  return make_qp(f"random_{'lp' if lp else 'qp'}_{n}_{m}_{p}_{seed}", P, c, A=A, b=A @ x0, G=G, h_l=h_l, h_u=h_u, x_l=x_l, x_u=x_u)


__all__ = [
  "INFINITE",
  "QP",
  "RawProblem",
  "infeasible_problems",
  "kkt_residuals",
  "make_qp",
  "maros_meszaros",
  "maros_meszaros_names",
  "mpc_qp",
  "random_qp",
  "raw",
]
