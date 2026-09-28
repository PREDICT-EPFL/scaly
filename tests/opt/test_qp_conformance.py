"""QP conformance: every QP method, at its default options, answers the same problems to one contract.

The problems are small and mid-size members of the stored Maros–Mészáros subset (``tests/data``),
written as ``sc.opt.problem``s with their linear term as the parameter and their constant in the
objective, two random QPs with every kind of bound, and the problems with no solution. A method
conforms when, through the signature every opt solver has, it returns

- on a problem with a solution: ``Status.OK``, a point feasible to the tolerance, signed multipliers
  (positive on an active upper side, negative on an active lower one, zero on an absent side) that
  make it stationary and complementary, and an ``Info`` whose objective is the problem's own there,
  constant included, and whose primal residual is the point's; again after the parameter changes,
  on the same compiled solver;
- on a problem with none: a status that is not a solution.

It runs over ``opt.ipm`` and ``opt.piqp``; step 7.1 of the restructure moves it into
``scaly.testing.conformance``, over every installed QP method.
"""

from __future__ import annotations

from functools import cache

import numpy as np
import pytest

import scaly as sc
from tests.opt.ipm.problems import QP, infeasible_problems, maros_meszaros, random_qp

METHODS = [pytest.param("opt.ipm", id="opt.ipm"), pytest.param("opt.piqp", id="opt.piqp", marks=pytest.mark.solver("piqp"))]
SOLVABLE = [
  *("HS21", "TAME", "HS35", "QPTEST", "ZECEVIC2", "HS35MOD", "HS76", "HS51", "HS52", "HS53", "HS268", "GENHS28", "LOTSCHD"),
  *("HS118", "QAFIRO", "CVXQP1_S", "QPCBLEND", "QADLITTL", "DUAL4", "DPKLO1", "DUALC1", "QSHARE2B"),
  "random_qp_0",
  "random_qp_1",
]
TOL = 1e-6  # relative to the data's scale; the default tolerances are 1e-8 absolute and 1e-9 relative


@cache
def stored(name: str) -> QP:
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


def solve(q: QP, method: str) -> tuple[sc.Function, list]:
  fun = sc.opt.solver(as_problem(q), method, name=f"conformance_{q.name}_{method.removeprefix('opt.')}")
  zeros = np.zeros(q.n)
  return fun, [zeros, zeros, np.zeros(q.A.shape[0]), np.zeros(q.G.shape[0])]


def violation(q: QP, x: np.ndarray) -> float:
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


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("name", SOLVABLE)
def test_a_solvable_qp(name: str, method: str) -> None:
  q = stored(name)
  fun, warm = solve(q, method)
  for c in (q.c, 2.0 * q.c):  # the second solve reuses the compiled solver, and PIQP its workspace
    check_solution(q, c, fun(*warm, c))


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("name", sorted(infeasible_problems()))
def test_a_qp_with_no_solution_is_not_reported_solved(name: str, method: str) -> None:
  q, _ = infeasible_problems()[name]
  fun, warm = solve(q, method)
  info = fun(*warm, q.c)[-1]
  assert not sc.Status(int(info.status)).ok
