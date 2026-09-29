from __future__ import annotations

from typing import Any, cast

import pytest

from typing_playground.expr import Buffer, Expr, const
from typing_playground.opti import NotQuadratic, ProblemSpec, bounded, problem, qp_problem, solver
from typing_playground.tests.definitions import (
  filter_piqp,
  filter_problem,
  filter_sqp,
  qp3,
  qp3_ipopt,
  qp3_piqp,
  quadratic,
  quadratic_ipopt,
  quadratic_piqp,
  rosenbrock,
  rosenbrock_ipopt,
)
from typing_playground.trees import L


def test_problem_carries_spec_and_trees() -> None:
  assert quadratic.name == "quadratic" and quadratic.vars.names == ("x",) and quadratic.params.names == ("scale",)
  assert quadratic.spec.minimize.shape == () and quadratic.n_eq == 0 and quadratic.n_ineq == 0
  assert filter_problem.n_eq == 1 and filter_problem.n_ineq == 3
  assert [b.name for b in filter_problem.spec.ineq] == ["cbf", "u_box"]


def test_problem_checks_the_spec_at_the_decorator() -> None:
  with pytest.raises(TypeError, match="scalar"):
    problem(vars=L("x", 2), params=L("p", ()))(lambda x, p: ProblemSpec(minimize=x))
  with pytest.raises(TypeError, match="structure"):
    problem(vars=L("x", 2), params=L("p", ()))(lambda x, p: ProblemSpec(minimize=x.sum(), lb=cast(Any, (x, x))))
  with pytest.raises(ValueError, match="at least one"):
    bounded(Expr((2,)))


def test_qp_problem_is_a_problem() -> None:
  assert qp3.vars.names == ("x",) and qp3.params.names == ("P", "c", "A", "b", "G", "g_lb", "g_ub")
  assert qp3.n_eq == 1 and qp3.n_ineq == 2
  empty = qp_problem(3, 0, 0)
  assert empty.params.shapes[2:4] == ((0, 3), (0,)) and empty.n_eq == 0 and empty.n_ineq == 0


def test_solver_signature_is_fixed() -> None:
  assert quadratic_ipopt.instantiate().input_names == ("x", "lam:x", "lam_eq", "lam_ineq", "scale")
  assert quadratic_ipopt.instantiate().input_shapes == ((3,), (3,), (0,), (0,), ())
  assert quadratic_ipopt.instantiate().output_names == ("x", "lam:x", "lam_eq", "lam_ineq")
  assert filter_sqp.instantiate().input_names == ("u", "s", "lam:u", "lam:s", "lam_eq", "lam_ineq", "x", "u_ref")
  assert filter_sqp.instantiate().input_shapes[4:6] == ((1,), (3,))


def test_solver_name_and_backend() -> None:
  assert quadratic_ipopt.name == "quadratic_ipopt" and quadratic_piqp.name == "quadratic_fast"
  with pytest.raises(ValueError, match="unknown backend"):
    solver(quadratic, "osqp")


def test_solution_and_warm_start_have_the_variables_structure() -> None:
  x0 = (Buffer((2,)), Buffer((1,)))
  out = filter_sqp.numerical_call(x0, x0, Buffer((1,)), Buffer((3,)), (Buffer((4,)), Buffer((2,))))
  (u, s), (lam_u, lam_s), lam_eq, lam_ineq = out
  assert u.shape == (2,) and s.shape == (1,) and lam_u.shape == (2,) and lam_s.shape == (1,) and lam_eq.shape == (1,) and lam_ineq.shape == (3,)


def test_qp_backend_is_gated_by_the_quadratic_proof() -> None:
  assert filter_piqp.name == "filter_problem_piqp"  # quadratic cost, affine constraints: accepted
  assert qp3_piqp.instantiate().input_names[-7:] == qp3.params.names  # the data form is updatable per call
  with pytest.raises(NotQuadratic, match="cost has degree 4"):
    solver(rosenbrock, "piqp")
  quartic_constraint = problem(vars=L("x", 2), params=L("p", ()))(lambda x, p: ProblemSpec(minimize=(x * x).sum(), eq=(x * x * x,)))
  with pytest.raises(NotQuadratic, match=r"eq\[0\] has degree 3"):
    solver(quartic_constraint, "piqp")
  wavy = problem(vars=L("x", 2), params=L("p", ()))(lambda x, p: ProblemSpec(minimize=(x * x).sum(), ineq=(bounded(x.sin(), hi=1.0, name="w"),)))
  with pytest.raises(NotQuadratic, match="ineq w has degree None"):
    solver(wavy, "piqp")


def test_nlp_backend_takes_any_spec() -> None:
  assert rosenbrock_ipopt.instantiate().input_shapes == ((2,), (2,), (0,), (1,), ())
  assert qp3_ipopt.instantiate().input_names == qp3_piqp.instantiate().input_names


def test_degree_tracking_used_by_the_proof() -> None:
  x = L("x", 3).symbols(degree=1)
  p = L("p", (3, 3)).symbols(degree=0)
  assert (x @ p @ x).degree == 2 and (p @ x).degree == 1 and (x / p[0]).degree == 1
  assert (x / x[0]).degree is None and x.sin().degree is None and (x**3).degree == 3 and const(1.0).degree == 0
