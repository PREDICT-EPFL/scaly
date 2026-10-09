from __future__ import annotations

import numpy as np
import pytest

import scaly as sc
from scaly.solvers.qp import _qp_matrix_sparsity


def test_sparse_qp_dependency_mask_keeps_entries_that_probe_to_zero() -> None:
  """A parameter-dependent entry whose value happens to be zero at the probe
  draw must stay in the pattern (the dependency mask, not the probe, keeps it)."""
  t = sc.sym("t", 1)
  zero = sc.const(0.0)
  P = sc.stack([sc.stack([sc.const(2.0), t[0] - t[0]]), sc.stack([zero, sc.const(2.0)])], axis=0)
  sparsity = _qp_matrix_sparsity(P, (t,), np.diag([2.0, 2.0]), triu=True)
  assert set(zip(sparsity.rows, sparsity.cols)) == {(0, 0), (0, 1), (1, 1)}


@pytest.mark.parametrize("location", ["cost", "eq", "ineq", "lb", "ub", "ineq-lo", "ineq-hi", "parameter"])
@pytest.mark.parametrize("form", ["direct", "call", "map"])
def test_qp_proof_rejects_stopped_derivatives(location: str, form: str) -> None:
  @sc.function(sc.arg("value"), outputs=sc.arg("frozen"))
  def freeze(value: sc.Expr) -> sc.Expr:
    return sc.stop_gradient(value)

  def terms(x: sc.Expr, p: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
    source = p if location == "parameter" else x
    frozen = sc.stop_gradient(source) if form == "direct" else freeze(source) if form == "call" else sc.vmap(freeze, 2)(source)
    cost = (
      (x * x * frozen * frozen).sum() - 4 * x[0] + p.sum()
      if location == "cost"
      else (x * x * frozen).sum()
      if location == "parameter"
      else sc.sumsqr(x) + p.sum()
    )
    quantity = x * frozen - 1 if location in {"eq", "ineq"} else frozen
    return cost, quantity

  @sc.function(sc.arg("x", 2), sc.arg("p", 2), outputs=sc.group(sc.arg("cost"), sc.arg("quantity")))
  def values(x: sc.Expr, p: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
    return terms(x, p)

  x, p = np.array([0.3, 1.4]), np.array([1.2, 0.8])
  cost, quantity = values(x, p)
  expected_cost = (
    np.sum(x**4) - 4 * x[0] + np.sum(p) if location == "cost" else np.sum(x**2 * p) if location == "parameter" else np.sum(x**2) + np.sum(p)
  )
  np.testing.assert_allclose(cost, expected_cost)
  np.testing.assert_allclose(quantity, x * x - 1 if location in {"eq", "ineq"} else p if location == "parameter" else x)

  @sc.problem(vars=sc.arg("x", 2), params=sc.arg("p", 2))
  def stopped(x: sc.Expr, p: sc.Expr) -> sc.ProblemSpec[sc.Expr]:
    cost, quantity = terms(x, p)
    inequality = (
      sc.bounded(quantity, hi=1) if location == "ineq" else sc.bounded(x, lo=quantity) if location == "ineq-lo" else sc.bounded(x, hi=quantity)
    )
    return sc.ProblemSpec(
      minimize=cost,
      eq=(quantity,) if location == "eq" else (),
      ineq=(inequality,) if location in {"ineq", "ineq-lo", "ineq-hi"} else (),
      lb=quantity if location == "lb" else None,
      ub=quantity if location == "ub" else None,
    )

  with pytest.raises(sc.NotQuadratic, match="stop_gradient"):
    sc.solver(stopped, "piqp", options={"sparse": True})
