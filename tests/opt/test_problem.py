"""Tests for typed backend-free problems and NLP solver construction."""

from __future__ import annotations

from typing import Any, cast
import numpy as np
import pytest

import scaly as sc
from scaly.codegen import render_c_source
from scaly.ir.expr import topo
from scaly.function.method import MethodError
from scaly.opt.external.graph import solver_descriptor


@sc.opt.problem(vars=sc.L("x", 3), params=sc.L("scale", ()))
def quadratic(x: sc.Expr, scale: sc.Expr) -> sc.opt.ProblemSpec[sc.Expr]:
  return sc.opt.ProblemSpec(
    minimize=(x * x).sum() * scale,
    lb=sc.const(-1.0),
    ub=sc.const(1.0),
  )


@sc.opt.problem(vars=sc.G(sc.L("u", 2), sc.L("s", 1)), params=sc.G(sc.L("x", 4), sc.L("u_ref", 2)))
def filter_problem(
  variables: tuple[sc.Expr, sc.Expr],
  params: tuple[sc.Expr, sc.Expr],
) -> sc.opt.ProblemSpec[tuple[sc.Expr, sc.Expr]]:
  u, s = variables
  x, u_ref = params
  barrier = x[:2] @ u + x[2:].sum()
  return sc.opt.ProblemSpec(
    minimize=0.5 * ((u - u_ref) * (u - u_ref)).sum() + 10.0 * s.sum(),
    eq=(u[0:1] - u[1:2],),
    ineq=(
      sc.opt.bounded(barrier + s, lo=0.0, name="cbf"),
      sc.opt.bounded(u, lo=-1.0, hi=1.0, name="u_box"),
    ),
    lb=(sc.opt.NO_LB, sc.const(0.0)),
    ub=(sc.opt.NO_UB, sc.opt.NO_UB),
  )


@sc.opt.problem(vars=sc.G(sc.L("u", 2), sc.L("s", 1)), params=sc.G(sc.L("target", 2), sc.L("bias", 1)))
def two_block_quadratic(
  variables: tuple[sc.Expr, sc.Expr],
  params: tuple[sc.Expr, sc.Expr],
) -> sc.opt.ProblemSpec[tuple[sc.Expr, sc.Expr]]:
  u, s = variables
  target, bias = params
  return sc.opt.ProblemSpec(minimize=((u - target) * (u - target)).sum() + ((s - bias) * (s - bias)).sum())


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
  assert quadratic.spec.minimize.shape == ()
  assert filter_problem.vars.names == ("u", "s")
  assert filter_problem.params.names == ("x", "u_ref")
  assert filter_problem.n_eq == 1
  assert filter_problem.n_ineq == 3
  assert tuple(group.name for group in filter_problem.spec.ineq) == ("cbf", "u_box")


def test_box_bound_leaves_broadcast_and_keep_ieee_infinity_in_core_oracle() -> None:
  assert isinstance(quadratic.spec.lb, sc.Expr) and quadratic.spec.lb.shape == (3,)
  assert isinstance(quadratic.spec.ub, sc.Expr) and quadratic.spec.ub.shape == (3,)

  solve = sc.opt.solver(filter_problem, "sqp", name="filter_bound_oracle")
  bounds = solver_descriptor(solve).bounds
  assert isinstance(bounds, sc.Function)
  x_lb, x_ub, l_ineq, u_ineq = bounds((np.zeros(4), np.zeros(2)))
  np.testing.assert_array_equal(x_lb, [-np.inf, -np.inf, 0.0])
  np.testing.assert_array_equal(x_ub, [np.inf, np.inf, np.inf])
  np.testing.assert_array_equal(l_ineq, [0.0, -1.0, -1.0])
  np.testing.assert_array_equal(u_ineq, [np.inf, 1.0, 1.0])


def test_qp_is_a_typed_problem() -> None:
  qp = sc.opt.QP(3, 1, 2)
  assert qp.vars.names == ("x",)
  assert qp.params.names == ("P", "c", "A", "b", "G", "g_lb", "g_ub")
  assert qp.n_eq == 1
  assert qp.n_ineq == 2

  empty = sc.opt.QP(3, 0, 0)
  assert empty.params.shapes[2:4] == ((0, 3), (0,))
  assert empty.n_eq == 0
  assert empty.n_ineq == 0


def test_problem_validates_body_and_declared_inputs_at_construction() -> None:
  with pytest.raises(TypeError, match="cost must be scalar"):
    sc.opt.problem(vars=sc.L("x", 2), params=sc.L("p", ()))(lambda x, p: sc.opt.ProblemSpec(minimize=x))

  with pytest.raises(TypeError, match="variables' structure|declared structure|expected an Expr"):
    sc.opt.problem(vars=sc.G(sc.L("u", 2), sc.L("s", 1)), params=sc.L("p", ()))(
      lambda variables, p: sc.opt.ProblemSpec(minimize=variables[0].sum(), lb=cast(Any, variables[0]))
    )

  undeclared = sc.sym("undeclared", ())

  with pytest.raises(ValueError, match="undeclared symbolic inputs"):
    sc.opt.problem(vars=sc.L("x", 2), params=sc.L("p", ()))(lambda x, p: sc.opt.ProblemSpec(minimize=x.sum() + undeclared))


def test_problem_infers_closed_over_parameters_and_retypes_them_nondifferentiable() -> None:
  state = sc.sym("state", 2)

  @sc.opt.problem(vars=sc.L("u", 2))
  def inferred(u: sc.Expr) -> sc.opt.ProblemSpec[sc.Expr]:
    return sc.opt.ProblemSpec(minimize=((u - state) * (u - state)).sum())

  assert inferred.params.names == ("state",)
  assert not inferred._param_symbols[0].type.diff
  assert state not in topo((inferred.spec.minimize,))


def test_nlp_solver_is_plain_typed_function_and_reuses_problem_oracles() -> None:
  ipopt = sc.opt.solver(filter_problem, "ipopt", name="filter_ipopt")
  sqp = sc.opt.solver(filter_problem, "sqp", name="filter_sqp")
  another_sqp = sc.opt.solver(filter_problem, "sqp", name="filter_sqp_again")

  assert isinstance(ipopt, sc.Function)
  assert ipopt.input_names == ("u", "s", "lam:u", "lam:s", "lam_eq", "lam_ineq", "x", "u_ref")
  assert ipopt.output_names == (
    "u",
    "s",
    "lam:u",
    "lam:s",
    "lam_eq",
    "lam_ineq",
    "info:status",
    "info:iter",
    "info:objective",
    "info:primal_residual",
  )
  assert ipopt.input_shapes == ((2,), (1,), (2,), (1,), (1,), (3,), (4,), (2,))
  assert ipopt.output_shapes == ((2,), (1,), (2,), (1,), (1,), (3,), (), (), (), ())
  ipopt_desc, sqp_desc = solver_descriptor(ipopt), solver_descriptor(sqp)
  sqp_hess = sqp_desc.hess
  assert isinstance(sqp_hess, sc.Function)
  assert sqp_hess.output_names == ("sphess_gamma_u_s_u_s",)
  assert sqp_hess.input_names == ("u_s", "x", "u_ref", "lam:f", "lam:g")
  assert ipopt_desc.base is sqp_desc.base
  assert ipopt_desc.grad is sqp_desc.grad
  assert ipopt_desc.jac is sqp_desc.jac
  assert ipopt_desc.hess is not sqp_desc.hess
  assert sqp_desc.hess is solver_descriptor(another_sqp).hess
  assert sc.Function.__doc__ is not None
  assert ipopt_desc.hess_sparsity is not None
  assert sqp_desc.hess_sparsity is not None
  assert all(row >= col for row, col in zip(ipopt_desc.hess_sparsity.rows, ipopt_desc.hess_sparsity.cols, strict=True))
  assert all(row <= col for row, col in zip(sqp_desc.hess_sparsity.rows, sqp_desc.hess_sparsity.cols, strict=True))
  assert not hasattr(sc, "SolverFunction")


@sc.function(sc.L("stage", 2), output=sc.L("row", ...), name="single_block_stage")
def single_block_stage(stage: sc.Expr) -> sc.Expr:
  return stage.sin()


def test_descriptor_lagrangian_hessian_matches_dense_reference() -> None:
  @sc.opt.problem(vars=sc.G(sc.L("u", 2), sc.L("s", 1)), params=sc.L("weight", ()), name="descriptor_hessian")
  def nonlinear_hessian(variables: tuple[sc.Expr, sc.Expr], weight: sc.Expr) -> sc.opt.ProblemSpec[tuple[sc.Expr, sc.Expr]]:
    u, s = variables
    objective = 0.5 * weight * u[0] ** 2 + u[0] * u[1] * s[0]
    equality = u[0] * s[0] + u[1] ** 2
    inequality = u[0] ** 2 + s[0] ** 2
    return sc.opt.ProblemSpec(minimize=objective, eq=(equality,), ineq=(sc.opt.bounded(inequality, hi=3.0),))

  solve = sc.opt.solver(nonlinear_hessian, "sqp", name="descriptor_hessian_sqp")
  hess = solver_descriptor(solve).hess
  assert isinstance(hess, sc.Function)
  sparsity = hess.output_sparsities[0]
  assert sparsity is not None

  x = np.array([0.4, -0.7, 0.2])
  weight = np.array(1.3)
  lam_f = np.array(1.7)
  lam_g = np.array([-0.6, 0.8])
  values = hess(((x, weight), (lam_f, lam_g)))
  actual = np.zeros((3, 3))
  actual[np.asarray(sparsity.rows), np.asarray(sparsity.cols)] = values
  actual += np.triu(actual, 1).T

  hess_f = np.array([[weight, x[2], x[1]], [x[2], 0.0, x[0]], [x[1], x[0], 0.0]])
  hess_eq = np.array([[0.0, 0.0, 1.0], [0.0, 2.0, 0.0], [1.0, 0.0, 0.0]])
  hess_ineq = np.diag([2.0, 0.0, 2.0])
  expected = lam_f * hess_f + lam_g[0] * hess_eq + lam_g[1] * hess_ineq
  np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


def test_single_block_problem_preserves_vmap_decision_input() -> None:
  @sc.opt.problem(vars=sc.L("z", 6), params=sc.L("p", ()), name="single_block_vmap")
  def mapped_problem(z: sc.Expr, p: sc.Expr) -> sc.opt.ProblemSpec[sc.Expr]:
    rows = sc.vmap(single_block_stage, 3, {"stage": (z, 0, 2)})
    return sc.opt.ProblemSpec(minimize=(z * z).sum() + p, eq=(rows,))

  solve = sc.opt.solver(mapped_problem, "sqp", name="single_block_vmap_sqp")
  desc = solver_descriptor(solve)
  assert isinstance(desc.base, sc.Function)
  mapped = next(node for node in topo(desc.base.outputs) if node.op == sc.ExprOp.VMAP)
  assert mapped.args[0] is desc.base.inputs[0]

  hess = desc.hess
  assert isinstance(hess, sc.Function)
  sparsity = hess.output_sparsities[0]
  assert sparsity is not None
  z, p = np.linspace(-0.7, 0.9, 6), np.array(0.3)
  lam_f, lam_g = np.array(1.7), np.linspace(-1.1, 0.8, 6)
  actual = np.zeros(sparsity.shape)
  actual[np.asarray(sparsity.rows), np.asarray(sparsity.cols)] = hess(((z, p), (lam_f, lam_g)))
  np.testing.assert_allclose(actual, np.diag(2.0 * lam_f - lam_g * np.sin(z)), rtol=1e-13, atol=1e-13)


def test_two_solvers_from_one_problem_render_one_translation_unit() -> None:
  left = sc.opt.solver(filter_problem, "sqp", name="filter_left")
  right = sc.opt.solver(filter_problem, "sqp", name="filter_right")
  ipopt = sc.opt.solver(filter_problem, "ipopt", name="filter_third")
  warm = (
    (sc.const(np.zeros(2)), sc.const(np.zeros(1))),
    (sc.const(np.zeros(2)), sc.const(np.zeros(1))),
    sc.const(np.zeros(1)),
    sc.const(np.zeros(3)),
    (sc.const(np.zeros(4)), sc.const(np.zeros(2))),
  )
  outputs = (left.symbolic_call(*warm)[0][0], right.symbolic_call(*warm)[0][0], ipopt.symbolic_call(*warm)[0][0])
  host = sc.Function.from_exprs("shared_problem_host", (), outputs, (), ("left", "right", "third"))
  source = render_c_source(host)
  assert source.count("static inline void filter_problem_hess_upper_raw(") == 1
  assert source.count("static inline void filter_problem_hess_lower_raw(") == 1


@pytest.mark.method("opt.sqp")
def test_sqp_numerical_call_preserves_multiple_variable_blocks() -> None:
  solve = sc.opt.solver(two_block_quadratic, "sqp", name="two_block_sqp")
  result = solve.numerical_call(*_two_block_inputs())
  np.testing.assert_allclose(result[0][0], [0.25, -0.75], atol=2e-6)
  np.testing.assert_allclose(result[0][1], [0.4], atol=2e-6)
  assert result[1][0].shape == (2,)
  assert result[1][1].shape == (1,)


@pytest.mark.method("opt.ipopt")
def test_ipopt_numerical_call_preserves_multiple_variable_blocks() -> None:
  solve = sc.opt.solver(two_block_quadratic, "ipopt", name="two_block_ipopt")
  result = solve.numerical_call(*_two_block_inputs())
  np.testing.assert_allclose(result[0][0], [0.25, -0.75], atol=2e-6)
  np.testing.assert_allclose(result[0][1], [0.4], atol=2e-6)
  assert result[1][0].shape == (2,)
  assert result[1][1].shape == (1,)


@pytest.mark.method("opt.piqp")
def test_piqp_numerical_call_preserves_multiple_variable_blocks() -> None:
  solve = sc.opt.solver(two_block_quadratic, "piqp", name="two_block_piqp")
  result = solve.numerical_call(*_two_block_inputs())
  np.testing.assert_allclose(result[0][0], [0.25, -0.75], atol=2e-6)
  np.testing.assert_allclose(result[0][1], [0.4], atol=2e-6)
  assert result[1][0].shape == (2,)
  assert result[1][1].shape == (1,)


def test_solver_rejects_an_unknown_backend() -> None:
  with pytest.raises(MethodError, match="no method opt."):
    sc.opt.solver(quadratic, "missing")


def test_solver_name_and_backend_are_selected_at_construction() -> None:
  assert sc.opt.solver(quadratic, "ipopt").name == "quadratic_ipopt"
  assert sc.opt.solver(quadratic, "piqp", name="quadratic_fast").name == "quadratic_fast"


def test_qp_backend_proves_quadratic_cost_and_affine_constraints() -> None:
  assert sc.opt.solver(filter_problem, "piqp").name == "filter_problem_piqp"

  @sc.opt.problem(vars=sc.L("x", 2), params=sc.L("center", 2))
  def power_cost(x: sc.Expr, center: sc.Expr) -> sc.opt.ProblemSpec[sc.Expr]:
    return sc.opt.ProblemSpec(minimize=((x - center) ** 2).sum())

  assert sc.opt.solver(power_cost, "piqp").name == "power_cost_piqp"
  qp = sc.opt.QP(3, 1, 2)
  assert sc.opt.solver(qp, "piqp").input_names[-7:] == qp.params.names

  @sc.opt.problem(vars=sc.L("x", 2), params=sc.L("p", ()))
  def quartic_cost(x: sc.Expr, p: sc.Expr) -> sc.opt.ProblemSpec[sc.Expr]:
    return sc.opt.ProblemSpec(minimize=(x * x * x * x).sum() + p)

  with pytest.raises(sc.opt.NotQuadratic, match="cost is not quadratic"):
    sc.opt.solver(quartic_cost, "piqp")

  @sc.opt.problem(vars=sc.L("x", 2), params=sc.L("p", ()))
  def cubic_equality(x: sc.Expr, p: sc.Expr) -> sc.opt.ProblemSpec[sc.Expr]:
    return sc.opt.ProblemSpec(minimize=(x * x).sum() + p, eq=(x * x * x,))

  with pytest.raises(sc.opt.NotQuadratic, match=r"eq\[0\] is not affine"):
    sc.opt.solver(cubic_equality, "piqp")

  @sc.opt.problem(vars=sc.L("x", 2), params=sc.L("p", ()))
  def wavy_inequality(x: sc.Expr, p: sc.Expr) -> sc.opt.ProblemSpec[sc.Expr]:
    return sc.opt.ProblemSpec(minimize=(x * x).sum() + p, ineq=(sc.opt.bounded(x.sin(), hi=1.0, name="w"),))

  with pytest.raises(sc.opt.NotQuadratic, match="ineq w is not affine"):
    sc.opt.solver(wavy_inequality, "piqp")

  @sc.opt.problem(vars=sc.L("x", 2), params=sc.L("p", ()))
  def variable_box_bound(x: sc.Expr, p: sc.Expr) -> sc.opt.ProblemSpec[sc.Expr]:
    return sc.opt.ProblemSpec(minimize=(x * x).sum() + p, lb=x)

  with pytest.raises(sc.opt.NotQuadratic, match="lb for 'x' depends on the variables"):
    sc.opt.solver(variable_box_bound, "piqp")

  @sc.opt.problem(vars=sc.L("x", 2), params=sc.L("p", ()))
  def variable_box_upper_bound(x: sc.Expr, p: sc.Expr) -> sc.opt.ProblemSpec[sc.Expr]:
    return sc.opt.ProblemSpec(minimize=(x * x).sum() + p, ub=x)

  with pytest.raises(sc.opt.NotQuadratic, match="ub for 'x' depends on the variables"):
    sc.opt.solver(variable_box_upper_bound, "piqp")

  @sc.opt.problem(vars=sc.L("x", 2), params=sc.L("p", ()))
  def variable_group_bound(x: sc.Expr, p: sc.Expr) -> sc.opt.ProblemSpec[sc.Expr]:
    return sc.opt.ProblemSpec(minimize=(x * x).sum() + p, ineq=(sc.opt.bounded(x, lo=x, name="moving"),))

  with pytest.raises(sc.opt.NotQuadratic, match="ineq moving lower bound depends on the variables"):
    sc.opt.solver(variable_group_bound, "piqp")

  @sc.opt.problem(vars=sc.L("x", 2), params=sc.L("p", ()))
  def variable_group_upper_bound(x: sc.Expr, p: sc.Expr) -> sc.opt.ProblemSpec[sc.Expr]:
    return sc.opt.ProblemSpec(minimize=(x * x).sum() + p, ineq=(sc.opt.bounded(x, hi=x, name="moving"),))

  with pytest.raises(sc.opt.NotQuadratic, match="ineq moving upper bound depends on the variables"):
    sc.opt.solver(variable_group_upper_bound, "piqp")

  inner = sc.opt.solver(quadratic, "sqp", name="nested_qp_gate_inner")

  @sc.opt.problem(vars=sc.L("outer", 3), params=sc.L("p", ()))
  def nested_solver_cost(outer: sc.Expr, p: sc.Expr) -> sc.opt.ProblemSpec[sc.Expr]:
    nested = inner.symbolic_call(
      sc.const(np.zeros(3)),
      sc.const(np.zeros(3)),
      sc.const(np.zeros(0)),
      sc.const(np.zeros(0)),
      outer[0],
    )[0]
    return sc.opt.ProblemSpec(minimize=((outer - nested) * (outer - nested)).sum() + p)

  with pytest.raises(sc.opt.NotQuadratic, match="cannot prove QP structure through a nested solver"):
    sc.opt.solver(nested_solver_cost, "piqp")


def test_nlp_backend_accepts_a_nonlinear_problem() -> None:
  @sc.opt.problem(vars=sc.L("x", 2), params=sc.L("p", ()))
  def nonlinear(x: sc.Expr, p: sc.Expr) -> sc.opt.ProblemSpec[sc.Expr]:
    return sc.opt.ProblemSpec(minimize=((1.0 - x[0]) ** 2 + p * (x[1] - x[0] ** 2) ** 2), ineq=(sc.opt.bounded(x[0].sin(), hi=0.5),))

  assert sc.opt.solver(nonlinear, "ipopt").input_shapes == ((2,), (2,), (0,), (1,), ())

  qp = sc.opt.QP(3, 1, 2)
  assert sc.opt.solver(qp, "ipopt").input_names == sc.opt.solver(qp, "piqp").input_names


def _zero_group_inputs(n_eq: int, n_ineq: int) -> tuple[Any, ...]:
  return (
    np.zeros(2),
    np.zeros(2),
    np.zeros(n_eq),
    np.zeros(n_ineq),
    (
      (np.eye(2), np.zeros(2)),
      (np.zeros((n_eq, 2)), np.zeros(n_eq)),
      (np.zeros((n_ineq, 2)), -np.ones(n_ineq), np.ones(n_ineq)),
    ),
  )


@pytest.mark.method("opt.ipopt")
@pytest.mark.parametrize(("n_eq", "n_ineq"), ((0, 0), (1, 0), (0, 1)))
def test_ipopt_compiles_present_zero_length_constraint_groups(n_eq: int, n_ineq: int) -> None:
  solve = sc.opt.solver(sc.opt.QP(2, n_eq, n_ineq), "ipopt", name=f"empty_ipopt_{n_eq}_{n_ineq}")
  result = solve.numerical_call(*_zero_group_inputs(n_eq, n_ineq))
  np.testing.assert_allclose(result[0], np.zeros(2), atol=1e-7)


@pytest.mark.method("opt.sqp")
@pytest.mark.parametrize(("n_eq", "n_ineq"), ((0, 0), (1, 0), (0, 1)))
def test_sqp_compiles_present_zero_length_constraint_groups(n_eq: int, n_ineq: int) -> None:
  solve = sc.opt.solver(sc.opt.QP(2, n_eq, n_ineq), "sqp", name=f"empty_sqp_{n_eq}_{n_ineq}")
  result = solve.numerical_call(*_zero_group_inputs(n_eq, n_ineq))
  np.testing.assert_allclose(result[0], np.zeros(2), atol=1e-7)


def test_bounded_requires_at_least_one_bound() -> None:
  with pytest.raises(ValueError, match="at least one"):
    sc.opt.bounded(sc.sym("g", 1))


def test_single_block_solver_signature_is_not_nested() -> None:
  solve = sc.opt.solver(quadratic, "sqp", name="quadratic_sqp")
  assert solve.input_names == ("x", "lam:x", "lam_eq", "lam_ineq", "scale")
  assert solve.input_shapes == ((3,), (3,), (0,), (0,), ())
  assert solve.output_names == ("x", "lam:x", "lam_eq", "lam_ineq", "info:status", "info:iter", "info:objective", "info:primal_residual")


def test_nlp_solver_symbolic_call_preserves_variable_blocks() -> None:
  solve = sc.opt.solver(filter_problem, "sqp", name="filter_nested")
  u0, s0 = sc.sym("u0", 2), sc.sym("s0", 1)
  result = solve.symbolic_call(
    (u0, s0),
    (sc.const(np.zeros(2)), sc.const(np.zeros(1))),
    sc.const(np.zeros(1)),
    sc.const(np.zeros(3)),
    (sc.const(np.zeros(4)), sc.const(np.zeros(2))),
  )
  assert isinstance(result[0], tuple)
  assert result[0][0].shape == (2,)
  assert result[0][1].shape == (1,)
  assert result[1][0].shape == (2,)
  assert result[2].shape == (1,)
  assert result[3].shape == (3,)


def test_an_unnamed_leaf_outside_a_decorator_is_refused() -> None:
  with pytest.raises(ValueError, match="unnamed leaf"):
    sc.opt.problem(vars=sc.L(3), params=sc.L("p", ()))(lambda x, p: sc.opt.ProblemSpec(minimize=x.sum()))
