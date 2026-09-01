"""Static acceptance tests for typed functions, derivatives, problems, and solvers."""

from __future__ import annotations

from typing import TYPE_CHECKING, assert_type

import numpy as np

import alloy as al
from alloy.function import Tree


@al.function(al.L("x", 3), al.G(al.L("first", ...), al.L("second", 3)))
def duplicate(x: al.Expr) -> tuple[al.Expr, al.Expr]:
  return x, x


@al.function(al.G(al.L("x", 3), al.L("y", 3)), al.L("prod", ...))
def multiply(inputs: tuple[al.Expr, al.Expr]) -> al.Expr:
  x, y = inputs
  return x * y


@al.function(al.L("x", 3), al.L("square", ...))
def square(x: al.Expr) -> al.Expr:
  return multiply.symbolic_call(duplicate.symbolic_call(x))


@al.function(al.G(al.L("x", 3), al.L("p", ())), al.L("f", ...))
def cost(inputs: tuple[al.Expr, al.Expr]) -> al.Expr:
  x, p = inputs
  return (x * x).sum() * p


@al.function(
  al.G(al.G(al.L("state", 4), al.L("u", 2)), al.G(al.L("pw", 10), al.L("physics", 3), al.L("dt", ()))),
  al.L("next", ...),
)
def step(inputs: tuple[tuple[al.Expr, al.Expr], tuple[al.Expr, al.Expr, al.Expr]]) -> al.Expr:
  (state, _u), (_pw, _physics, _dt) = inputs
  return state


@al.function(al.G(al.L("state", 4), al.L("u", 2), al.L("pw", 10), al.L("physics", 3), al.L("dt", ())), al.L("next", ...))
def step_flat(inputs: tuple[al.Expr, al.Expr, al.Expr, al.Expr, al.Expr]) -> al.Expr:
  state, _u, _pw, _physics, _dt = inputs
  return state


@al.problem(vars=al.L("x", 3), params=al.L("scale", ()))
def quadratic(x: al.Expr, scale: al.Expr) -> al.ProblemSpec[al.Expr]:
  return al.ProblemSpec(minimize=(x * x).sum() * scale, lb=al.const(-1.0), ub=al.const(1.0))


@al.problem(vars=al.G(al.L("u", 2), al.L("s", 1)), params=al.G(al.L("x", 4), al.L("u_ref", 2)))
def filter_problem(variables: tuple[al.Expr, al.Expr], params: tuple[al.Expr, al.Expr]) -> al.ProblemSpec[tuple[al.Expr, al.Expr]]:
  u, s = variables
  x, u_ref = params
  barrier = x[:2] @ u + x[2:].sum()
  return al.ProblemSpec(
    minimize=0.5 * ((u - u_ref) * (u - u_ref)).sum() + 10.0 * s.sum(),
    eq=(u[0:1] - u[1:2],),
    ineq=(al.bounded(barrier + s, lo=0.0, name="cbf"), al.bounded(u, lo=-1.0, hi=1.0, name="u_box")),
    lb=(al.NO_LB, al.const(0.0)),
    ub=(al.NO_UB, al.NO_UB),
  )


quadratic_ipopt = al.solver(quadratic, "ipopt")
filter_sqp = al.solver(filter_problem, "sqp")
qp3 = al.qp_problem(3, 1, 2)
qp3_piqp = al.solver(qp3, "piqp")

grad_f_x = al.gradient(cost, "f", "x")
hess_f_x = al.hessian(cost, "f", "x")
jac_square_x = al.jacobian(square, "square", "x")
fwd_f_x = al.forward(cost, "f", "x")
adj_square_x = al.adjoint(square, "square", "x")
hess_l = al.lagrangian_hessian(duplicate, "x")


if TYPE_CHECKING:
  assert_type(al.NO_LB, al.Expr)
  assert_type(al.NO_UB, al.Expr)
  al.L("x", "3")  # ty: ignore[invalid-argument-type]
  al.G(al.L("x", 3))  # ty: ignore[no-matching-overload]
  al.G(al.L("x", 3), al.L("y", 3), ("z", 3))  # ty: ignore[invalid-argument-type]
  assert_type(al.G(al.L("x", 3), al.L("p", ())), Tree[tuple[al.Expr, al.Expr], tuple[np.ndarray, np.ndarray]])
  assert_type(
    al.G(al.G(al.L("a", 1), al.L("b", 1)), al.L("c", 1)),
    Tree[tuple[tuple[al.Expr, al.Expr], al.Expr], tuple[tuple[np.ndarray, np.ndarray], np.ndarray]],
  )

  assert_type(duplicate, al.Function[al.Expr, np.ndarray, tuple[al.Expr, al.Expr], tuple[np.ndarray, np.ndarray]])
  assert_type(duplicate.symbolic_call(al.sym("x", 3)), tuple[al.Expr, al.Expr])
  assert_type(duplicate.numerical_call(np.zeros(3)), tuple[np.ndarray, np.ndarray])
  assert_type(multiply.numerical_call((np.zeros(3), np.zeros(3))), np.ndarray)
  assert_type(
    step.numerical_call(((np.zeros(4), np.zeros(2)), (np.zeros(10), np.zeros(3), np.zeros(())))),
    np.ndarray,
  )
  assert_type(step_flat.numerical_call((np.zeros(4), np.zeros(2), np.zeros(10), np.zeros(3), np.zeros(()))), np.ndarray)
  multiply.numerical_call((np.zeros(3),))  # ty: ignore[invalid-argument-type]
  multiply.numerical_call(np.zeros(3))  # ty: ignore[invalid-argument-type]
  multiply.numerical_call((al.sym("x", 3), al.sym("y", 3)))  # ty: ignore[invalid-argument-type]
  multiply.symbolic_call((np.zeros(3), np.zeros(3)))  # ty: ignore[invalid-argument-type]
  duplicate.symbolic_call((al.sym("x", 3),))  # ty: ignore[invalid-argument-type]
  step.numerical_call((np.zeros(4), np.zeros(2), np.zeros(10), np.zeros(3), np.zeros(())))  # ty: ignore[invalid-argument-type]
  step_flat.numerical_call(((np.zeros(4), np.zeros(2)), (np.zeros(10), np.zeros(3), np.zeros(()))))  # ty: ignore[invalid-argument-type]

  al.function(al.G(al.L("x", 3), al.L("y", 3)), al.L("z", ...))(lambda x: x)  # ty: ignore[invalid-argument-type]
  al.function(al.L("x", 3), al.G(al.L("a", ...), al.L("b", ...)))(lambda x: x)  # ty: ignore[invalid-argument-type]

  assert_type(multiply.symbolic_call(duplicate.symbolic_call(al.sym("x", 3))), al.Expr)
  multiply.symbolic_call(square.symbolic_call(al.sym("x", 3)))  # ty: ignore[invalid-argument-type]

  assert_type(grad_f_x, al.Function[tuple[al.Expr, al.Expr], tuple[np.ndarray, np.ndarray], al.Expr, np.ndarray])
  assert_type(hess_f_x, al.Function[tuple[al.Expr, al.Expr], tuple[np.ndarray, np.ndarray], al.Expr, np.ndarray])
  assert_type(jac_square_x, al.Function[al.Expr, np.ndarray, al.Expr, np.ndarray])
  assert_type(grad_f_x.numerical_call((np.zeros(3), np.zeros(()))), np.ndarray)
  assert_type(
    fwd_f_x,
    al.Function[tuple[tuple[al.Expr, al.Expr], al.Expr], tuple[tuple[np.ndarray, np.ndarray], np.ndarray], al.Expr, np.ndarray],
  )
  assert_type(adj_square_x, al.Function[tuple[al.Expr, al.Expr], tuple[np.ndarray, np.ndarray], al.Expr, np.ndarray])
  assert_type(
    hess_l,
    al.Function[tuple[al.Expr, tuple[al.Expr, al.Expr]], tuple[np.ndarray, tuple[np.ndarray, np.ndarray]], al.Expr, np.ndarray],
  )
  assert_type(fwd_f_x.numerical_call(((np.zeros(3), np.zeros(())), np.zeros(3))), np.ndarray)
  assert_type(hess_l.numerical_call((np.zeros(3), (np.zeros(3), np.zeros(3)))), np.ndarray)

  assert_type(quadratic, al.Problem[al.Expr, np.ndarray, al.Expr, np.ndarray])
  assert_type(
    filter_problem,
    al.Problem[
      tuple[al.Expr, al.Expr],
      tuple[np.ndarray, np.ndarray],
      tuple[al.Expr, al.Expr],
      tuple[np.ndarray, np.ndarray],
    ],
  )
  assert_type(qp3, al.Problem[al.Expr, np.ndarray, al.QPData[al.Expr], al.QPData[np.ndarray]])
  al.problem(vars=al.G(al.L("u", 2), al.L("s", 1)), params=al.L("p", ()))(lambda variables, p: al.ProblemSpec(minimize=variables.sum()))  # ty: ignore[unresolved-attribute]
  al.problem(vars=al.L("x", 2), params=al.L("p", ()))(lambda x, p: al.ProblemSpec(minimize=x.sum(), lb=(x, x)))  # ty: ignore[invalid-argument-type]

  assert_type(
    quadratic_ipopt,
    al.Function[
      tuple[al.Expr, al.Expr, al.Expr, al.Expr, al.Expr],
      tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
      tuple[al.Expr, al.Expr, al.Expr, al.Expr],
      tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ],
  )
  assert_type(
    filter_sqp,
    al.Function[
      tuple[tuple[al.Expr, al.Expr], tuple[al.Expr, al.Expr], al.Expr, al.Expr, tuple[al.Expr, al.Expr]],
      tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]],
      tuple[tuple[al.Expr, al.Expr], tuple[al.Expr, al.Expr], al.Expr, al.Expr],
      tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], np.ndarray, np.ndarray],
    ],
  )
  assert_type(
    filter_sqp.numerical_call(
      (
        (np.zeros(2), np.zeros(1)),
        (np.zeros(2), np.zeros(1)),
        np.zeros(1),
        np.zeros(3),
        (np.zeros(4), np.zeros(2)),
      )
    ),
    tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], np.ndarray, np.ndarray],
  )
  assert_type(
    filter_sqp.symbolic_call(
      (
        (al.sym("u0", 2), al.sym("s0", 1)),
        (al.sym("lam_u0", 2), al.sym("lam_s0", 1)),
        al.sym("lam_eq0", 1),
        al.sym("lam_ineq0", 3),
        (al.sym("x0", 4), al.sym("u_ref0", 2)),
      )
    ),
    tuple[tuple[al.Expr, al.Expr], tuple[al.Expr, al.Expr], al.Expr, al.Expr],
  )
  assert_type(
    qp3_piqp.numerical_call(
      (
        np.zeros(3),
        np.zeros(3),
        np.zeros(1),
        np.zeros(2),
        (
          (np.zeros((3, 3)), np.zeros(3)),
          (np.zeros((1, 3)), np.zeros(1)),
          (np.zeros((2, 3)), np.zeros(2), np.zeros(2)),
        ),
      )
    ),
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
  )
  quadratic_ipopt.numerical_call((np.zeros(3), np.zeros(3), np.zeros(0), np.zeros(0)))  # ty: ignore[invalid-argument-type]
  filter_sqp.numerical_call(((np.zeros(2),), (np.zeros(2), np.zeros(1)), np.zeros(1), np.zeros(3), (np.zeros(4), np.zeros(2))))  # ty: ignore[invalid-argument-type]
  filter_sqp.numerical_call(((al.sym("u", 2), al.sym("s", 1)), (np.zeros(2), np.zeros(1)), np.zeros(1), np.zeros(3), (np.zeros(4), np.zeros(2))))  # ty: ignore[invalid-argument-type]

  grad_f_x.numerical_call(np.zeros(3))  # ty: ignore[invalid-argument-type]
  fwd_f_x.numerical_call((np.zeros(3), np.zeros(()), np.zeros(3)))  # ty: ignore[invalid-argument-type]
  hess_l.numerical_call((np.zeros(3), np.zeros(6)))  # ty: ignore[invalid-argument-type]
