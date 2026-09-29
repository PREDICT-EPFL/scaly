"""The QP suite: stored Maros–Mészáros problems, random QPs and QPs with no solution, and the contract a QP method meets on them.

Through the signature every opt solver has, a conforming method returns

- on a problem with a solution: ``Status.OK``, a point feasible to the tolerance, signed multipliers
  (positive on an active upper side, negative on an active lower one, zero on an absent side) that
  make it stationary and complementary, and an ``Info`` whose objective is the problem's own there,
  constant included, and whose primal residual is the point's; again after the parameter changes,
  on the same compiled solver;
- on a problem with none: a status that is not a solution.
"""

from __future__ import annotations

from functools import cache
from typing import Any

import numpy as np

import scaly as sc
from scaly.testing.qp import QP, infeasible_problems, maros_meszaros, random_qp

SOLVABLE = [
  *("HS21", "TAME", "HS35", "QPTEST", "ZECEVIC2", "HS35MOD", "HS76", "HS51", "HS52", "HS53", "HS268", "GENHS28", "LOTSCHD"),
  *("HS118", "QAFIRO", "CVXQP1_S", "QPCBLEND", "QADLITTL", "DUAL4", "DPKLO1", "DUALC1", "QSHARE2B"),
  "random_qp_0",
  "random_qp_1",
]
TOL = 1e-6  # relative to the data's scale; the default tolerances are 1e-8 absolute and 1e-9 relative


@cache
def stored(name: str) -> QP:
  """A problem of the suite by name: a stored Maros–Mészáros problem, or ``random_qp_<seed>``."""
  if name.startswith("random_qp_"):
    return random_qp(30, 20, 5, seed=int(name.removeprefix("random_qp_")))
  return maros_meszaros(name)


def as_problem(q: QP) -> sc.opt.NLP:
  """``q`` as an ``sc.opt.problem``: its matrices and bounds constants, its linear term the parameter."""
  P, A, G = (sc.const(m.toarray()) for m in (q.P, q.A, q.G))
  p, m = q.A.shape[0], q.G.shape[0]

  @sc.opt.problem(vars=sc.L("x", q.n), params=sc.L("c", q.n), name=f"conformance_{q.name}")
  def problem(x: sc.Expr, c: sc.Expr) -> sc.opt.ProblemSpec:
    return sc.opt.ProblemSpec(
      minimize=0.5 * (x @ (P @ x)) + c @ x + q.r,
      eq=(A @ x - sc.const(q.b),) if p else (),
      ineq=(sc.opt.bounded(G @ x, lo=sc.const(q.h_l), hi=sc.const(q.h_u), name="g"),) if m else (),
      lb=sc.const(q.x_l),
      ub=sc.const(q.x_u),
    )

  return problem


def solve(q: QP, method: Any) -> tuple[sc.Function, list]:
  """``q``'s solver by ``method`` (a name or an instance with its options) and a zero warm start."""
  label = method.removeprefix("opt.") if isinstance(method, str) else method.name.removeprefix("opt.")
  fun = sc.opt.solver(as_problem(q), method, name=f"conformance_{q.name}_{label}")
  zeros = np.zeros(q.n)
  return fun, [zeros, zeros, np.zeros(q.A.shape[0]), np.zeros(q.G.shape[0])]


def violation(q: QP, x: np.ndarray) -> float:
  """The largest violation of ``q``'s constraints and bounds at ``x``."""
  gx = q.G @ x
  parts = [q.A @ x - q.b, q.h_l - gx, gx - q.h_u, q.x_l - x, x - q.x_u]
  return max(float(np.maximum(v, 0.0).max(initial=0.0)) if k else float(np.abs(v).max(initial=0.0)) for k, v in enumerate(parts))


def complementarity(lam: np.ndarray, value: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
  """The largest ``|lam|`` times its side's slack; on an absent side, ``|lam|`` itself."""
  up, down = np.maximum(lam, 0.0), np.maximum(-lam, 0.0)
  up_gap = np.where(np.isfinite(hi), up * np.abs(np.where(np.isfinite(hi), hi, 0.0) - value), up)
  down_gap = np.where(np.isfinite(lo), down * np.abs(value - np.where(np.isfinite(lo), lo, 0.0)), down)
  return float(max(up_gap.max(initial=0.0), down_gap.max(initial=0.0)))


def check_solution(q: QP, c: np.ndarray, out: tuple) -> None:
  """Assert the contract on one solve's outputs ``(x, lam_box, lam_eq, lam_ineq, info)``."""
  x, lam_box, lam_eq, lam_ineq, info = out
  x, lam_box, lam_eq, lam_ineq = (np.asarray(v, dtype=float) for v in (x, lam_box, lam_eq, lam_ineq))
  assert sc.Status(int(info.status)) == sc.Status.OK
  assert 1 <= int(info.iter) <= 250
  gx = q.G @ x
  terms = [q.P @ x, c, q.A.T @ lam_eq, q.G.T @ lam_ineq, lam_box]
  scale = 1.0 + max(float(np.abs(t).max(initial=0.0)) for t in terms)
  size = 1.0 + max(float(np.abs(v[np.isfinite(v)]).max(initial=0.0)) for v in (q.b, q.h_l, q.h_u, q.x_l, q.x_u, gx, x))
  assert violation(q, x) <= TOL * size
  assert float(np.abs(sum(terms)).max(initial=0.0)) <= TOL * scale, "stationarity"
  assert complementarity(lam_ineq, gx, q.h_l, q.h_u) <= TOL * scale * size, "inequality complementarity"
  assert complementarity(lam_box, x, q.x_l, q.x_u) <= TOL * scale * size, "box complementarity"
  objective = float(0.5 * x @ (q.P @ x) + c @ x + q.r)
  assert abs(float(info.objective) - objective) <= 1e-9 * (1.0 + abs(objective))
  assert 0.0 <= float(info.primal_residual) <= TOL * size


UNSOLVABLE = sorted(infeasible_problems())
"""The problems with no solution, by name."""


def check_solves(method: Any, name: str) -> None:
  """``method`` meets the contract on the solvable problem ``name``, twice: at its linear term, and
  at twice that on the same compiled solver."""
  q = stored(name)
  fun, warm = solve(q, method)
  for c in (q.c, 2.0 * q.c):
    check_solution(q, c, fun(*warm, c))


def check_refuses(method: Any, name: str) -> None:
  """``method`` does not report the problem ``name``, which has no solution, solved."""
  q, _ = infeasible_problems()[name]
  fun, warm = solve(q, method)
  info = fun(*warm, q.c)[-1]
  assert not sc.Status(int(info.status)).ok


__all__ = ["SOLVABLE", "TOL", "UNSOLVABLE", "as_problem", "check_refuses", "check_solution", "check_solves", "solve", "stored", "violation"]
