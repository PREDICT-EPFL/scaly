# /// script
# requires-python = ">=3.12"
# dependencies = ["scaly", "scaly-piqp"]
# ///
"""Portfolio optimization with a factor risk model: a sparse QP for PIQP and an efficient frontier.

With ``n`` assets whose returns follow ``k`` factors, the covariance is ``Sigma = F F^T + diag(d)``
(``F`` is ``n x k``), dense as a matrix but cheap in the factored form. Introducing the factor
exposures ``y = F^T x`` turns the Markowitz problem

    maximize mu^T x - gamma x^T Sigma x   subject to   1^T x = 1,  0 <= x <= x_max

into a QP whose Hessian is diagonal and whose constraint matrix has ``n k + n`` nonzeros:

    minimize gamma (|y|^2 + x^T diag(d) x) - mu^T x   subject to   y = F^T x,  1^T x = 1,  0 <= x <= x_max.

``F``, ``d``, ``mu`` and the risk aversion ``gamma`` are parameters, so one generated solver serves
every rebalancing. ``options={"sparse": True}`` makes Scaly derive the sparsity patterns of the
extracted ``P`` and ``A`` when the solver is built and bake them into the generated PIQP wrapper,
which then refills values only.

Sweeping ``gamma`` traces the efficient frontier. The factored and the dense formulation (the full
``Sigma``, through ``sc.opt.QP``) agree, and so does SciPy's SLSQP on a small instance.

The generated C lands in ``examples/generated/portfolio_qp/``.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from scipy import optimize

import scaly as sc
from scaly.codegen import write_module

GENERATED = Path(__file__).resolve().parents[1] / "generated" / "portfolio_qp"
N_ASSETS, N_FACTORS, X_MAX = 120, 6, 0.05
GAMMAS = np.logspace(0, 2.5, 11)
TIGHT = {"eps_abs": 1e-10, "eps_rel": 1e-10, "eps_duality_gap_abs": 1e-10, "eps_duality_gap_rel": 1e-10}


@sc.opt.problem(
  vars=sc.G(sc.L("x", N_ASSETS), sc.L("y", N_FACTORS)),
  params=sc.G(sc.L("mu", N_ASSETS), sc.L("F", (N_ASSETS, N_FACTORS)), sc.L("d", N_ASSETS), sc.L("gamma", ())),
)
def markowitz(variables: tuple[sc.Expr, sc.Expr], params: tuple[sc.Expr, ...]) -> sc.opt.ProblemSpec:
  x, y = variables
  mu, f, d, gamma = params
  return sc.opt.ProblemSpec(
    minimize=gamma * (sc.sumsqr(y) + (d * x * x).sum()) - mu @ x,
    eq=(y - f.T @ x, x.sum() - 1.0),
    lb=(sc.const(np.zeros(N_ASSETS)), sc.opt.NO_LB),
    ub=(sc.const(np.full(N_ASSETS, X_MAX)), sc.opt.NO_UB),
  )


solve_factored = sc.opt.solver(markowitz, sc.opt.PIQP(sparse=True, options={**TIGHT}), name="markowitz_sparse")


def market(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  rng = np.random.default_rng(seed)
  f = rng.standard_normal((N_ASSETS, N_FACTORS)) * np.sqrt(rng.uniform(0.001, 0.01, N_FACTORS))
  d = rng.uniform(0.002, 0.03, N_ASSETS)
  mu = 0.03 + f @ rng.uniform(0.2, 0.6, N_FACTORS) + 0.01 * rng.standard_normal(N_ASSETS)
  return mu, f, d


def solve(mu: np.ndarray, f: np.ndarray, d: np.ndarray, gamma: float) -> np.ndarray:
  n_eq = N_FACTORS + 1
  (x, _), *_ = solve_factored(
    (np.zeros(N_ASSETS), np.zeros(N_FACTORS)), (np.zeros(N_ASSETS), np.zeros(N_FACTORS)), np.zeros(n_eq), np.zeros(0), (mu, f, d, np.array(gamma))
  )
  return x


def dense_reference(mu: np.ndarray, f: np.ndarray, d: np.ndarray, gamma: float) -> np.ndarray:
  """The same portfolio from the dense ``Sigma``, as a matrix-data QP (box bounds as inequalities)."""
  qp = sc.opt.QP(N_ASSETS, 1, N_ASSETS)
  solve_dense = sc.opt.solver(qp, sc.opt.PIQP(options=TIGHT), name="markowitz_dense")
  sigma = f @ f.T + np.diag(d)
  data = ((2 * gamma * sigma, -mu), (np.ones((1, N_ASSETS)), np.ones(1)), (np.eye(N_ASSETS), np.zeros(N_ASSETS), np.full(N_ASSETS, X_MAX)))
  x, *_ = solve_dense(np.zeros(N_ASSETS), np.zeros(N_ASSETS), np.zeros(1), np.zeros(N_ASSETS), data)
  return x


def slsqp_reference(mu: np.ndarray, f: np.ndarray, d: np.ndarray, gamma: float) -> np.ndarray:
  sigma = f @ f.T + np.diag(d)
  res = optimize.minimize(
    lambda x: gamma * x @ sigma @ x - mu @ x,
    np.full(N_ASSETS, 1.0 / N_ASSETS),
    jac=lambda x: 2 * gamma * sigma @ x - mu,
    constraints=[{"type": "eq", "fun": lambda x: x.sum() - 1.0, "jac": lambda x: np.ones((1, N_ASSETS))}],
    bounds=[(0.0, X_MAX)] * N_ASSETS,
    method="SLSQP",
    options={"ftol": 1e-15, "maxiter": 1000},
  )
  return res.x


def main() -> dict:
  mu, f, d = market()
  sigma = f @ f.T + np.diag(d)
  solve(mu, f, d, 1.0)  # compile
  frontier, start = [], time.perf_counter()
  for gamma in GAMMAS:
    x = solve(mu, f, d, gamma)
    frontier.append((gamma, float(mu @ x), float(np.sqrt(x @ sigma @ x)), int(np.sum(x > 1e-4)), float(x.sum()), float(x.min()), float(x.max())))
  per_solve = (time.perf_counter() - start) / len(GAMMAS)
  x = solve(mu, f, d, 5.0)
  return {
    "frontier": frontier,
    "per_solve": per_solve,
    "dense_error": np.abs(x - dense_reference(mu, f, d, 5.0)).max(),
    "slsqp_error": np.abs(x - slsqp_reference(mu, f, d, 5.0)).max(),
    "nnz_note": (N_ASSETS * N_FACTORS + N_ASSETS, N_ASSETS * N_ASSETS),
  }


if __name__ == "__main__":
  out = main()
  print(f"efficient frontier, {N_ASSETS} assets and {N_FACTORS} factors, {1e3 * out['per_solve']:.2f} ms per sparse PIQP solve:")
  print("   gamma   return  risk (std)  holdings")
  for gamma, ret, risk, held, total, lo, hi in out["frontier"]:
    assert abs(total - 1) < 1e-6 and lo > -1e-8 and hi < X_MAX + 1e-8
    print(f"  {gamma:6.2f}  {ret:7.4f}  {risk:10.4f}  {held:8d}")
  print(f"gamma = 5: factored sparse QP vs the dense-Sigma QP {out['dense_error']:.1e}, vs SciPy SLSQP {out['slsqp_error']:.1e}")
  write_module(solve_factored, GENERATED)
  print(f"generated C for the sparse PIQP solver in {GENERATED}")
