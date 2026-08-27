"""Tests for typed backend-free problems and NLP solver construction."""

from __future__ import annotations

from typing import Any, cast
import numpy as np
import pytest

import alloy as al
from alloy.codegen import render_c_source
from alloy.ir.expr import topo
from alloy.solvers.registry import SolverPluginError


@al.problem(vars=al.L("x", 3), params=al.L("scale", ()))
def quadratic(x: al.Expr, scale: al.Expr) -> al.ProblemSpec[al.Expr]:
  return al.ProblemSpec(
    minimize=(x * x).sum() * scale,
    lb=al.const(np.full(3, -1.0)),
    ub=al.const(np.full(3, 1.0)),
  )


@al.problem(vars=al.G(al.L("u", 2), al.L("s", 1)), params=al.G(al.L("x", 4), al.L("u_ref", 2)))
def filter_problem(
  variables: tuple[al.Expr, al.Expr],
  params: tuple[al.Expr, al.Expr],
) -> al.ProblemSpec[tuple[al.Expr, al.Expr]]:
  u, s = variables
  x, u_ref = params
  barrier = x[:2] @ u + x[2:].sum()
  return al.ProblemSpec(
    minimize=0.5 * ((u - u_ref) * (u - u_ref)).sum() + 10.0 * s.sum(),
    eq=(u[0:1] - u[1:2],),
    ineq=(
      al.bounded(barrier + s, lo=0.0, name="cbf"),
      al.bounded(u, lo=-1.0, hi=1.0, name="u_box"),
    ),
    lb=(al.const(np.full(2, -1.0)), al.const(np.zeros(1))),
  )


@al.problem(vars=al.G(al.L("u", 2), al.L("s", 1)), params=al.G(al.L("target", 2), al.L("bias", 1)))
def two_block_quadratic(
  variables: tuple[al.Expr, al.Expr],
  params: tuple[al.Expr, al.Expr],
) -> al.ProblemSpec[tuple[al.Expr, al.Expr]]:
  u, s = variables
  target, bias = params
  return al.ProblemSpec(minimize=((u - target) * (u - target)).sum() + ((s - bias) * (s - bias)).sum())


def _two_block_inputs() -> Any:
  return (
    (np.zeros(2), np.zeros(1)),
    (np.zeros(2), np.zeros(1)),
    np.zeros(0),
    np.zeros(0),
    (np.array([0.25, -0.75]), np.array([0.4])),
  )


def test_problem_carries_spec_trees_and_declared_names() -> None:
  assert quadratic.name == "quadratic"
  assert quadratic.vars.names == ("x",)
  assert quadratic.params.names == ("scale",)
  assert filter_problem.vars.names == ("u", "s")
  assert filter_problem.params.names == ("x", "u_ref")
  assert filter_problem.n_eq == 1
  assert filter_problem.n_ineq == 3
  assert tuple(group.name for group in filter_problem.spec.ineq) == ("cbf", "u_box")


def test_problem_validates_body_and_declared_inputs_at_construction() -> None:
  with pytest.raises(TypeError, match="cost must be scalar"):
    al.problem(vars=al.L("x", 2), params=al.L("p", ()))(lambda x, p: al.ProblemSpec(minimize=x))

  with pytest.raises(TypeError, match="variables' structure|declared structure|expected an Expr"):
    al.problem(vars=al.G(al.L("u", 2), al.L("s", 1)), params=al.L("p", ()))(
      lambda variables, p: al.ProblemSpec(minimize=variables[0].sum(), lb=cast(Any, variables[0]))
    )

  undeclared = al.sym("undeclared", ())

  with pytest.raises(ValueError, match="undeclared symbolic inputs"):
    al.problem(vars=al.L("x", 2), params=al.L("p", ()))(lambda x, p: al.ProblemSpec(minimize=x.sum() + undeclared))


def test_problem_infers_closed_over_parameters_and_retypes_them_nondifferentiable() -> None:
  state = al.sym("state", 2)

  @al.problem(vars=al.L("u", 2))
  def inferred(u: al.Expr) -> al.ProblemSpec[al.Expr]:
    return al.ProblemSpec(minimize=((u - state) * (u - state)).sum())

  assert inferred.params.names == ("state",)
  assert not inferred._param_symbols[0].type.diff
  assert state not in topo((inferred.spec.minimize,))


def test_nlp_solver_is_plain_typed_function_and_reuses_problem_oracles() -> None:
  ipopt = al.solver(filter_problem, "ipopt", name="filter_ipopt")
  sqp = al.solver(filter_problem, "sqp", name="filter_sqp")
  another_sqp = al.solver(filter_problem, "sqp", name="filter_sqp_again")

  assert isinstance(ipopt, al.Function)
  assert not hasattr(al, "SolverFunction")
  assert ipopt.input_names == ("u", "s", "lam:u", "lam:s", "lam_eq", "lam_ineq", "x", "u_ref")
  assert ipopt.output_names == ("u", "s", "lam:u", "lam:s", "lam_eq", "lam_ineq")
  assert ipopt.input_shapes == ((2,), (1,), (2,), (1,), (1,), (3,), (4,), (2,))
  assert ipopt.output_shapes == ((2,), (1,), (2,), (1,), (1,), (3,))
  assert ipopt.descriptor.base is sqp.descriptor.base
  assert ipopt.descriptor.grad is sqp.descriptor.grad
  assert ipopt.descriptor.jac is sqp.descriptor.jac
  assert ipopt.descriptor.hess is not sqp.descriptor.hess
  assert sqp.descriptor.hess is another_sqp.descriptor.hess
  assert sqp.descriptor.hess.output_names == ("sphess_gamma_u_s_u_s",)
  assert al.Function.__doc__ is not None
  assert ipopt.descriptor.hess_sparsity is not None
  assert sqp.descriptor.hess_sparsity is not None
  assert all(row >= col for row, col in zip(ipopt.descriptor.hess_sparsity.rows, ipopt.descriptor.hess_sparsity.cols, strict=True))
  assert all(row <= col for row, col in zip(sqp.descriptor.hess_sparsity.rows, sqp.descriptor.hess_sparsity.cols, strict=True))


def test_two_solvers_from_one_problem_render_one_translation_unit() -> None:
  left = al.solver(filter_problem, "sqp", name="filter_left")
  right = al.solver(filter_problem, "sqp", name="filter_right")
  ipopt = al.solver(filter_problem, "ipopt", name="filter_third")
  warm = (
    (al.const(np.zeros(2)), al.const(np.zeros(1))),
    (al.const(np.zeros(2)), al.const(np.zeros(1))),
    al.const(np.zeros(1)),
    al.const(np.zeros(3)),
    (al.const(np.zeros(4)), al.const(np.zeros(2))),
  )
  outputs = (left.symbolic_call(warm)[0][0], right.symbolic_call(warm)[0][0], ipopt.symbolic_call(warm)[0][0])
  host = al.Function._from_exprs("shared_problem_host", (), outputs, (), ("left", "right", "third"))
  source = render_c_source(host)
  assert source.count("static inline void filter_problem_hess_upper_raw(") == 1
  assert source.count("static inline void filter_problem_hess_lower_raw(") == 1


@pytest.mark.solver("sqp")
def test_sqp_numerical_call_preserves_multiple_variable_blocks() -> None:
  solve = al.solver(two_block_quadratic, "sqp", name="two_block_sqp")
  result = solve.numerical_call(_two_block_inputs())
  np.testing.assert_allclose(result[0][0], [0.25, -0.75], atol=2e-6)
  np.testing.assert_allclose(result[0][1], [0.4], atol=2e-6)
  assert result[1][0].shape == (2,)
  assert result[1][1].shape == (1,)


@pytest.mark.solver("ipopt")
def test_ipopt_numerical_call_preserves_multiple_variable_blocks() -> None:
  solve = al.solver(two_block_quadratic, "ipopt", name="two_block_ipopt")
  result = solve.numerical_call(_two_block_inputs())
  np.testing.assert_allclose(result[0][0], [0.25, -0.75], atol=2e-6)
  np.testing.assert_allclose(result[0][1], [0.4], atol=2e-6)
  assert result[1][0].shape == (2,)
  assert result[1][1].shape == (1,)


def test_solver_rejects_an_unknown_backend() -> None:
  with pytest.raises(SolverPluginError, match="no solver plugin"):
    al.solver(quadratic, "missing")


def test_bounded_requires_at_least_one_bound() -> None:
  with pytest.raises(ValueError, match="at least one"):
    al.bounded(al.sym("g", 1))


def test_single_block_solver_signature_is_not_nested() -> None:
  solve = al.solver(quadratic, "sqp", name="quadratic_sqp")
  assert solve.input_names == ("x", "lam:x", "lam_eq", "lam_ineq", "scale")
  assert solve.output_names == ("x", "lam:x", "lam_eq", "lam_ineq")


def test_nlp_solver_symbolic_call_preserves_variable_blocks() -> None:
  solve = al.solver(filter_problem, "sqp", name="filter_nested")
  u0, s0 = al.sym("u0", 2), al.sym("s0", 1)
  result = solve.symbolic_call(
    (
      (u0, s0),
      (al.const(np.zeros(2)), al.const(np.zeros(1))),
      al.const(np.zeros(1)),
      al.const(np.zeros(3)),
      (al.const(np.zeros(4)), al.const(np.zeros(2))),
    )
  )
  assert isinstance(result[0], tuple)
  assert result[0][0].shape == (2,)
  assert result[0][1].shape == (1,)
  assert result[1][0].shape == (2,)
  assert result[2].shape == (1,)
  assert result[3].shape == (3,)
