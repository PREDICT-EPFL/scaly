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

  @sc.function(sc.arg("x", 2), sc.arg("p", 2), outputs=sc.group(sc.arg("cost", ()), sc.arg("quantity", 2)))
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


@pytest.mark.parametrize("location", ["cost", "eq", "ineq", "lb", "ineq-hi"])
@pytest.mark.parametrize("form", ["direct", "call", "map"])
@pytest.mark.parametrize("operation", ["select", "maximum", "minimum", "abs", "floor", "ceil", "reduce_max", "reduce_min", "norm_1", "norm_inf"])
def test_qp_proof_treats_a_condition_on_the_variables_as_nonlinear(location: str, form: str, operation: str) -> None:
  """``where(x > 0, x, -x)`` has a zero Hessian but is not quadratic, and a bound of ``x > 0`` has a
  zero Jacobian but depends on the variables."""

  def bare(value: sc.Expr) -> sc.Expr:
    if operation != "select":
      return {
        "maximum": lambda: sc.maximum(value, 0),
        "minimum": lambda: sc.minimum(value, 0),
        "abs": value.abs,
        "floor": value.floor,
        "ceil": value.ceil,
        "reduce_max": lambda: sc.reduce_max(value),
        "reduce_min": lambda: sc.reduce_min(value),
        "norm_1": lambda: sc.norm_1(value),
        "norm_inf": lambda: sc.norm_inf(value),
      }[operation]()
    return sc.cast(value > 0.0, "float64") if location in {"lb", "ineq-hi"} else sc.where(value > 0.0, value, -value)

  absolute = sc.function(sc.arg("value"), outputs=sc.arg("absolute"))(bare)

  def kink(x: sc.Expr) -> sc.Expr:
    return absolute(x) if form == "call" else sc.vmap(absolute, 2)(x) if form == "map" else bare(x)

  @sc.problem(vars=sc.arg("x", 2), params=sc.arg("p", 2))
  def kinked(x: sc.Expr, p: sc.Expr) -> sc.ProblemSpec[sc.Expr]:
    term = kink(x)
    return sc.ProblemSpec(
      minimize=sc.sumsqr(x) + p.sum() + (term.sum() if location == "cost" else 0.0),
      eq=(term - 1.0,) if location == "eq" else (),
      ineq=(sc.bounded(term, hi=1.0),) if location == "ineq" else (sc.bounded(x, hi=term),) if location == "ineq-hi" else (),
      lb=term if location == "lb" else None,
    )

  with pytest.raises(sc.NotQuadratic, match="depends on the variables"):
    sc.solver(kinked, "piqp")


@pytest.mark.solver("piqp")
def test_qp_proof_accepts_a_condition_on_the_parameters() -> None:
  @sc.problem(vars=sc.arg("x", 2), params=sc.arg("p", 2))
  def switched(x: sc.Expr, p: sc.Expr) -> sc.ProblemSpec[sc.Expr]:
    return sc.ProblemSpec(minimize=sc.sumsqr(x - sc.where(p > 0.0, p, 2.0 * p)))

  x = sc.solver(switched, "piqp")(np.array([1.5, -1.0]))[0]
  np.testing.assert_allclose(x, [1.5, -2.0], atol=1e-6)


@pytest.mark.solver("piqp")
def test_sparse_qp_keeps_matrix_entries_a_parameter_condition_can_switch_on() -> None:
  """The pattern probe draws one parameter value, at which the gated entry may be zero."""

  @sc.problem(vars=sc.arg("x", 2), params=sc.arg("p", 1))
  def gated(x: sc.Expr, p: sc.Expr) -> sc.ProblemSpec[sc.Expr]:
    return sc.ProblemSpec(minimize=sc.sumsqr(x), eq=(x[0] + sc.where(p[0] < 0.0, 1.0, 0.0) * x[1] - 1.0,))

  for sparse in (False, True):
    solve = sc.solver(gated, "piqp", options={"sparse": sparse}, name=f"gated_qp_{sparse}")
    np.testing.assert_allclose(solve(np.array([-1.0]))[0], [0.5, 0.5], atol=1e-6)
    np.testing.assert_allclose(solve(np.array([1.0]))[0], [1.0, 0.0], atol=1e-6)
