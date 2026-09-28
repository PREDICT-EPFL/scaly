"""A solver's solution has no derivative rule of its own: a derivative that reaches one raises, its
structural pattern is dense in every argument, and ``custom_derivative`` supplies the rule.

Before, both directions returned zero and the pattern was empty, so a Jacobian through a solve was
silently zero and a custom rule's values were dropped by the empty pattern."""

from __future__ import annotations

import numpy as np
import pytest

import scaly as sc
from scaly.ad import jacobian_sparsity, jvp, jvp_many, vjp
from scaly.passes.expr import simplify_cse_fixpoint


@sc.problem(vars=sc.L("x", 2), params=sc.L("p", ()), name="solver_derivatives_qp")
def _qp(x, p):
  """``x0 = clip((p - 1) / 2, 0.2, 0.8)``, ``x1 = 1 - x0``: interior for ``1.4 < p < 2.6``."""
  return sc.ProblemSpec(
    minimize=x[0] ** 2 + x[1] ** 2 + x[0] * x[1] - (2.0 + p) * x[0] - 4.0 * x[1],
    eq=(x.sum() - 1.0,),
    lb=sc.const(np.zeros(2)),
    ub=sc.const(np.full(2, 0.8)),
  )


SOLVE = sc.solver(_qp, "piqp", name="solver_derivatives_solve", options={"eps_abs": 1e-10, "eps_rel": 1e-10})
ZEROS = (sc.const(np.zeros(2)), sc.const(np.zeros(2)), sc.const(np.zeros(1)), sc.const(np.zeros(0)))


def _solution(solve: sc.Function, p: sc.Expr) -> sc.Expr:
  return solve._flat_symbolic_call([*ZEROS, p])[0]


def test_a_derivative_through_a_solver_raises() -> None:
  t = sc.sym("t", 2)
  x = _solution(SOLVE, 2.0 * t[0] + t[1])
  with pytest.raises(NotImplementedError, match="solver_derivatives_solve.*custom_derivative"):
    jvp(x, t, sc.const(np.ones(2)))
  with pytest.raises(NotImplementedError, match="custom_derivative"):
    jvp_many(x, t, sc.const(np.eye(2)))
  with pytest.raises(NotImplementedError, match="custom_derivative"):
    vjp([x], [t], [sc.const(np.ones(2))])
  with pytest.raises(NotImplementedError, match="custom_derivative"):
    sc.jacobian(x, t)


def test_a_solve_independent_of_the_variable_is_a_constant() -> None:
  t, q = sc.sym("t", 2), sc.sym("q", ())
  derivative = simplify_cse_fixpoint(sc.jacobian(_solution(SOLVE, q) + t, t))
  assert derivative.op == sc.ExprOp.CONST
  np.testing.assert_array_equal(derivative.value, np.eye(2))


def test_the_pattern_of_a_solution_is_dense_in_the_columns_its_arguments_read() -> None:
  t = sc.sym("t", 3)
  x = _solution(SOLVE, 2.0 * t[0] + t[2])
  pattern = jacobian_sparsity(sc.concat([x, t[1:2]]), t)
  assert set(zip(pattern.rows, pattern.cols, strict=True)) == {(0, 0), (0, 2), (1, 0), (1, 2), (2, 1)}


def _ruled() -> sc.Function:
  """The solve with the interior sensitivity ``dx/dp = (1/2, -1/2)`` as its forward rule."""
  names = [*SOLVE.input_names, *(f"d{name}" for name in SOLVE.input_names)]
  inputs = [sc.sym(name, e.shape) for name, e in zip(SOLVE.input_names, SOLVE.inputs, strict=True)]
  tangents = [sc.sym(f"d{name}", e.shape) for name, e in zip(SOLVE.input_names, SOLVE.inputs, strict=True)]
  dp = tangents[-1]
  outs = [sc.const(np.array([0.5, -0.5])) * dp, sc.const(np.zeros(2)) * dp, sc.const(np.zeros(1)) * dp, sc.const(np.zeros(0))]
  rule = sc.Function._from_exprs("solver_derivatives_rule", [*inputs, *tangents], outs, names, [f"d{name}" for name in SOLVE.output_names])
  return sc.custom_derivative(SOLVE, jvp=rule)


def test_a_custom_rule_keeps_the_dense_pattern() -> None:
  ruled = _ruled()
  t = sc.sym("t", 2)
  x = _solution(ruled, 2.0 * t[0] + t[1])
  jac = sc.sparse_jacobian(x, t)  # an empty pattern here would drop the rule's values
  assert set(zip(jac.sparsity.rows, jac.sparsity.cols, strict=True)) == {(0, 0), (0, 1), (1, 0), (1, 1)}


@pytest.mark.solver("piqp")
def test_a_custom_rule_through_a_solver_reaches_the_sparse_jacobian() -> None:
  ruled = _ruled()
  t = sc.sym("t", 2)
  x = _solution(ruled, 2.0 * t[0] + t[1])
  fn = sc.Function._from_exprs("solver_derivatives_jac", [t], [x, sc.sparse_jacobian(x, t).values, sc.jacobian(x, t)], ["t"], ["x", "v", "j"])
  value, compact, dense = fn(np.array([0.9, 0.2]))  # p = 2: the interior, x = (0.5, 0.5)
  np.testing.assert_allclose(value, [0.5, 0.5], atol=1e-8)
  expected = np.array([[1.0, 0.5], [-1.0, -0.5]])
  np.testing.assert_allclose(dense, expected, atol=1e-12)
  np.testing.assert_allclose(np.sort(compact), np.sort(expected.reshape(-1)), atol=1e-12)
