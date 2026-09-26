"""Four QP families, each written once as a parametric ``sc.problem`` and handed unchanged to every solver.

| Case | Structure | What varies at run time |
| --- | --- | --- |
| ``mpc`` | Oscillating masses, sparse multiple shooting: banded equalities, boxes on inputs and positions | the initial state (only ``b``) |
| ``portfolio`` | Markowitz with a factor risk model: diagonal ``P``, the factor loadings in ``A``, a box | returns, loadings, idiosyncratic risk, risk aversion (``P``, ``c``, ``A``) |
| ``svm`` | Soft-margin support vector machine: ``|w|^2`` plus hinge slacks, one inequality row per sample | the samples and labels (``G``) |
| ``dense`` | ``sc.qp_problem``: every matrix a dense parameter, two-sided inequalities | everything |

``Case.params()`` draws the parameter values (a fixed seed), in the problem's parameter tree.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import linalg as sla

import scaly as sc


@dataclass(frozen=True)
class Case:
  """A problem, the parameter values to solve it at, and a line for the tables."""

  name: str
  problem: sc.Problem
  params: Callable[[], Any]
  about: str


# --- linear MPC -------------------------------------------------------------------------------------


def masses(n_masses: int = 4, dt: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
  """The oscillating-masses benchmark: unit masses and springs between two walls, ``n_masses - 1``
  actuators each pushing a neighbouring pair apart; exact zero-order-hold discretisation."""
  nx, nu = 2 * n_masses, n_masses - 1
  springs = 2 * np.eye(n_masses) - np.eye(n_masses, k=1) - np.eye(n_masses, k=-1)
  push = np.eye(n_masses, nu) - np.eye(n_masses, nu, k=-1)
  a_c = np.block([[np.zeros((n_masses, n_masses)), np.eye(n_masses)], [-springs, np.zeros((n_masses, n_masses))]])
  b_c = np.r_[np.zeros((n_masses, nu)), push]
  e = sla.expm(np.block([[a_c, b_c], [np.zeros((nu, nx + nu))]]) * dt)
  return e[:nx, :nx], e[:nx, nx:]


def mpc(horizon: int = 20, n_masses: int = 4, u_max: float = 0.5, p_max: float = 1.0) -> Case:
  """Regulate the masses to rest from a displaced start: ``sum_k x_k^T Q x_k + u_k^T R u_k`` with a
  terminal weight, ``x_{k+1} = A x_k + B u_k``, ``|u| <= u_max`` and ``|positions| <= p_max``. The
  variables are the whole trajectory (sparse formulation), the parameter is ``x_0``."""
  a, b = masses(n_masses)
  nx, nu = a.shape[0], b.shape[1]
  q, r, q_n = np.ones(nx), 0.1 * np.ones(nu), 10.0 * np.ones(nx)
  x_lb = np.r_[-p_max * np.ones(n_masses), -np.inf * np.ones(n_masses)]
  x_ub = -x_lb

  @sc.problem(vars=sc.G(sc.L("u", (horizon, nu)), sc.L("x", (horizon, nx))), params=sc.L("x0", nx), name=f"mpc_N{horizon}")
  def masses_mpc(variables: tuple[sc.Expr, sc.Expr], x0: sc.Expr) -> sc.ProblemSpec:
    u, x = variables
    previous = sc.concat([x0.reshape((1, nx)), x[:-1]])
    weights = np.vstack([np.tile(q, (horizon - 1, 1)), q_n])
    return sc.ProblemSpec(
      minimize=0.5 * ((sc.const(weights) * x * x).sum() + (sc.const(np.tile(r, (horizon, 1))) * u * u).sum()),
      eq=((x - previous @ sc.const(a.T) - u @ sc.const(b.T)).vec(),),
      lb=(sc.const(np.full((horizon, nu), -u_max)), sc.const(np.tile(x_lb, (horizon, 1)))),
      ub=(sc.const(np.full((horizon, nu), u_max)), sc.const(np.tile(x_ub, (horizon, 1)))),
    )

  def params() -> np.ndarray:
    return np.r_[0.9, -0.5, 0.7, -0.9, np.zeros(n_masses - 4), np.zeros(n_masses)][:nx]

  return Case(f"mpc_N{horizon}", masses_mpc, params, f"oscillating masses, {n_masses} masses, horizon {horizon}")


# --- portfolio --------------------------------------------------------------------------------------


def portfolio(n_assets: int = 60, n_factors: int = 5, x_max: float = 0.1) -> Case:
  """``minimize gamma (|y|^2 + x^T diag(d) x) - mu^T x`` subject to ``y = F^T x``, ``1^T x = 1`` and
  ``0 <= x <= x_max``: the covariance ``F F^T + diag(d)`` in factored form."""

  @sc.problem(
    vars=sc.G(sc.L("x", n_assets), sc.L("y", n_factors)),
    params=sc.G(sc.L("mu", n_assets), sc.L("F", (n_assets, n_factors)), sc.L("d", n_assets), sc.L("gamma", ())),
    name="portfolio",
  )
  def markowitz(variables: tuple[sc.Expr, sc.Expr], params: tuple[sc.Expr, ...]) -> sc.ProblemSpec:
    x, y = variables
    mu, f, d, gamma = params
    return sc.ProblemSpec(
      minimize=gamma * (sc.sumsqr(y) + (d * x * x).sum()) - mu @ x,
      eq=(y - f.T @ x, x.sum() - 1.0),
      lb=(sc.const(np.zeros(n_assets)), sc.NO_LB),
      ub=(sc.const(np.full(n_assets, x_max)), sc.NO_UB),
    )

  def params() -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(1)
    f = rng.standard_normal((n_assets, n_factors)) * np.sqrt(rng.uniform(0.001, 0.01, n_factors))
    d = rng.uniform(0.002, 0.03, n_assets)
    mu = 0.03 + f @ rng.uniform(0.2, 0.6, n_factors) + 0.01 * rng.standard_normal(n_assets)
    return mu, f, d, np.array(5.0)

  return Case("portfolio", markowitz, params, f"factor-model Markowitz, {n_assets} assets, {n_factors} factors")


# --- support vector machine -------------------------------------------------------------------------


def svm(n_samples: int = 100, n_features: int = 10, c: float = 1.0) -> Case:
  """``minimize 1/2 |w|^2 + c 1^T t`` subject to ``y_i (a_i^T w + b) + t_i >= 1`` and ``t >= 0``: the
  soft-margin classifier of two overlapping Gaussian clouds, the data as parameters."""

  @sc.problem(
    vars=sc.G(sc.L("w", n_features), sc.L("b", ()), sc.L("t", n_samples)),
    params=sc.G(sc.L("data", (n_samples, n_features)), sc.L("labels", n_samples)),
    name="svm",
  )
  def soft_margin(variables: tuple[sc.Expr, sc.Expr, sc.Expr], params: tuple[sc.Expr, sc.Expr]) -> sc.ProblemSpec:
    w, b, t = variables
    data, labels = params
    return sc.ProblemSpec(
      minimize=0.5 * sc.sumsqr(w) + c * t.sum(),
      ineq=(sc.bounded(labels * (data @ w + b) + t, lo=1.0, name="margin"),),
      lb=(sc.NO_LB, sc.NO_LB, sc.const(np.zeros(n_samples))),
    )

  def params() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(2)
    labels = np.where(np.arange(n_samples) < n_samples // 2, 1.0, -1.0)
    centre = np.r_[1.0, -0.5, np.zeros(n_features - 2)]
    data = rng.standard_normal((n_samples, n_features)) + labels[:, None] * centre
    return data, labels

  return Case("svm", soft_margin, params, f"soft-margin SVM, {n_samples} samples, {n_features} features")


# --- dense ------------------------------------------------------------------------------------------


def dense(n: int = 30, n_eq: int = 5, n_ineq: int = 40) -> Case:
  """A random strictly convex QP in matrix form (``sc.qp_problem``): ``P``, ``c``, ``A``, ``b``, ``G``
  and two-sided bounds all parameters, all dense, the matrices full."""
  problem = sc.qp_problem(n, n_eq, n_ineq)

  def params() -> Any:
    rng = np.random.default_rng(3)
    m = rng.standard_normal((n, n))
    p = m @ m.T / n + 0.1 * np.eye(n)
    a, g = rng.standard_normal((n_eq, n)), rng.standard_normal((n_ineq, n))
    x_feasible = rng.standard_normal(n)
    slack = rng.uniform(0.1, 1.0, n_ineq)
    return (p, rng.standard_normal(n)), (a, a @ x_feasible), (g, g @ x_feasible - slack, g @ x_feasible + slack)

  return Case("dense", problem, params, f"random dense QP, n = {n}, {n_eq} equalities, {n_ineq} two-sided inequalities")


def cases() -> dict[str, Case]:
  """The four families at their default sizes."""
  return {c.name: c for c in (mpc(), portfolio(), svm(), dense())}
