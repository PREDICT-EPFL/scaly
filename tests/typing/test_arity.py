"""Static acceptance tests for typed functions, derivatives, problems, and solvers."""

from __future__ import annotations

from typing import TYPE_CHECKING, assert_type

import numpy as np

import scaly as sc
from scaly.function import Tree


@sc.function(sc.L("x", 3), output=sc.G(sc.L("first", ...), sc.L("second", 3)))
def duplicate(x: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
  return x, x


@sc.function(sc.G(sc.L("x", 3), sc.L("y", 3)), output=sc.L("prod", ...))
def multiply(inputs: tuple[sc.Expr, sc.Expr]) -> sc.Expr:
  x, y = inputs
  return x * y


@sc.function(sc.L("x", 3), output=sc.L("square", ...))
def square(x: sc.Expr) -> sc.Expr:
  return multiply.symbolic_call(duplicate.symbolic_call(x))


@sc.function(sc.G(sc.L("x", 3), sc.L("p", ())), output=sc.L("f", ...))
def cost(inputs: tuple[sc.Expr, sc.Expr]) -> sc.Expr:
  x, p = inputs
  return (x * x).sum() * p


@sc.function(
  sc.G(sc.G(sc.L("state", 4), sc.L("u", 2)), sc.G(sc.L("pw", 10), sc.L("physics", 3), sc.L("dt", ()))),
  output=sc.L("next", ...),
)
def step(inputs: tuple[tuple[sc.Expr, sc.Expr], tuple[sc.Expr, sc.Expr, sc.Expr]]) -> sc.Expr:
  (state, _u), (_pw, _physics, _dt) = inputs
  return state


@sc.function(sc.G(sc.L("state", 4), sc.L("u", 2), sc.L("pw", 10), sc.L("physics", 3), sc.L("dt", ())), output=sc.L("next", ...))
def step_flat(inputs: tuple[sc.Expr, sc.Expr, sc.Expr, sc.Expr, sc.Expr]) -> sc.Expr:
  state, _u, _pw, _physics, _dt = inputs
  return state


@sc.problem(vars=sc.L("x", 3), params=sc.L("scale", ()))
def quadratic(x: sc.Expr, scale: sc.Expr) -> sc.ProblemSpec[sc.Expr]:
  return sc.ProblemSpec(minimize=(x * x).sum() * scale, lb=sc.const(-1.0), ub=sc.const(1.0))


@sc.problem(vars=sc.G(sc.L("u", 2), sc.L("s", 1)), params=sc.G(sc.L("x", 4), sc.L("u_ref", 2)))
def filter_problem(variables: tuple[sc.Expr, sc.Expr], params: tuple[sc.Expr, sc.Expr]) -> sc.ProblemSpec[tuple[sc.Expr, sc.Expr]]:
  u, s = variables
  x, u_ref = params
  barrier = x[:2] @ u + x[2:].sum()
  return sc.ProblemSpec(
    minimize=0.5 * ((u - u_ref) * (u - u_ref)).sum() + 10.0 * s.sum(),
    eq=(u[0:1] - u[1:2],),
    ineq=(sc.bounded(barrier + s, lo=0.0, name="cbf"), sc.bounded(u, lo=-1.0, hi=1.0, name="u_box")),
    lb=(sc.NO_LB, sc.const(0.0)),
    ub=(sc.NO_UB, sc.NO_UB),
  )


quadratic_ipopt = sc.solver(quadratic, "ipopt")
filter_sqp = sc.solver(filter_problem, "sqp")
qp3 = sc.qp_problem(3, 1, 2)
qp3_piqp = sc.solver(qp3, "piqp")

grad_f_x = sc.gradient(cost, "f", "x")
hess_f_x = sc.hessian(cost, "f", "x")
jac_square_x = sc.jacobian(square, "square", "x")
fwd_f_x = sc.forward(cost, "f", "x")
adj_square_x = sc.adjoint(square, "square", "x")
hess_l = sc.lagrangian_hessian(duplicate, "x")


if TYPE_CHECKING:
  assert_type(sc.NO_LB, sc.Expr)
  assert_type(sc.NO_UB, sc.Expr)
  sc.L("x", "3")  # ty: ignore[invalid-argument-type]
  sc.G(sc.L("x", 3))  # ty: ignore[no-matching-overload]
  sc.G(sc.L("x", 3), sc.L("y", 3), ("z", 3))  # ty: ignore[invalid-argument-type]
  assert_type(sc.G(sc.L("x", 3), sc.L("p", ())), Tree[tuple[sc.Expr, sc.Expr], tuple[np.ndarray, np.ndarray]])
  assert_type(
    sc.G(sc.G(sc.L("a", 1), sc.L("b", 1)), sc.L("c", 1)),
    Tree[tuple[tuple[sc.Expr, sc.Expr], sc.Expr], tuple[tuple[np.ndarray, np.ndarray], np.ndarray]],
  )

  assert_type(duplicate, sc.Function[sc.Expr, np.ndarray, tuple[sc.Expr, sc.Expr], tuple[np.ndarray, np.ndarray]])
  assert_type(duplicate.symbolic_call(sc.sym("x", 3)), tuple[sc.Expr, sc.Expr])
  assert_type(duplicate.numerical_call(np.zeros(3)), tuple[np.ndarray, np.ndarray])
  assert_type(multiply.numerical_call((np.zeros(3), np.zeros(3))), np.ndarray)
  assert_type(
    step.numerical_call(((np.zeros(4), np.zeros(2)), (np.zeros(10), np.zeros(3), np.zeros(())))),
    np.ndarray,
  )
  assert_type(step_flat.numerical_call((np.zeros(4), np.zeros(2), np.zeros(10), np.zeros(3), np.zeros(()))), np.ndarray)
  multiply.numerical_call((np.zeros(3),))  # ty: ignore[invalid-argument-type]
  multiply.numerical_call(np.zeros(3))  # ty: ignore[invalid-argument-type]
  multiply.numerical_call((sc.sym("x", 3), sc.sym("y", 3)))  # ty: ignore[invalid-argument-type]
  multiply.symbolic_call((np.zeros(3), np.zeros(3)))  # ty: ignore[invalid-argument-type]
  duplicate.symbolic_call((sc.sym("x", 3),))  # ty: ignore[invalid-argument-type]
  step.numerical_call((np.zeros(4), np.zeros(2), np.zeros(10), np.zeros(3), np.zeros(())))  # ty: ignore[invalid-argument-type]
  step_flat.numerical_call(((np.zeros(4), np.zeros(2)), (np.zeros(10), np.zeros(3), np.zeros(()))))  # ty: ignore[invalid-argument-type]

  sc.function(sc.G(sc.L("x", 3), sc.L("y", 3)), output=sc.L("z", ...))(lambda x: x)  # ty: ignore[invalid-argument-type]
  sc.function(sc.L("x", 3), output=sc.G(sc.L("a", ...), sc.L("b", ...)))(lambda x: x)  # ty: ignore[invalid-argument-type]

  assert_type(multiply.symbolic_call(duplicate.symbolic_call(sc.sym("x", 3))), sc.Expr)
  multiply.symbolic_call(square.symbolic_call(sc.sym("x", 3)))  # ty: ignore[invalid-argument-type]

  # __call__ dispatches on the leaf kind and is typed as precisely as the two named methods.
  assert_type(duplicate(sc.sym("x", 3)), tuple[sc.Expr, sc.Expr])
  assert_type(duplicate(np.zeros(3)), tuple[np.ndarray, np.ndarray])
  assert_type(multiply((sc.sym("x", 3), sc.sym("y", 3))), sc.Expr)
  assert_type(multiply((np.zeros(3), np.zeros(3))), np.ndarray)
  assert_type(step(((np.zeros(4), np.zeros(2)), (np.zeros(10), np.zeros(3), np.zeros(())))), np.ndarray)
  assert_type(multiply(duplicate(sc.sym("x", 3))), sc.Expr)
  assert_type(hess_l((np.zeros(3), (np.zeros(3), np.zeros(3)))), np.ndarray)
  multiply((np.zeros(3),))  # ty: ignore[no-matching-overload]
  multiply((sc.sym("x", 3), np.zeros(3)))  # ty: ignore[no-matching-overload]
  step((np.zeros(4), np.zeros(2), np.zeros(10), np.zeros(3), np.zeros(())))  # ty: ignore[no-matching-overload]
  multiply(square(sc.sym("x", 3)))  # ty: ignore[no-matching-overload]

  assert_type(grad_f_x, sc.Function[tuple[sc.Expr, sc.Expr], tuple[np.ndarray, np.ndarray], sc.Expr, np.ndarray])
  assert_type(hess_f_x, sc.Function[tuple[sc.Expr, sc.Expr], tuple[np.ndarray, np.ndarray], sc.Expr, np.ndarray])
  assert_type(jac_square_x, sc.Function[sc.Expr, np.ndarray, sc.Expr, np.ndarray])
  assert_type(grad_f_x.numerical_call((np.zeros(3), np.zeros(()))), np.ndarray)
  assert_type(
    fwd_f_x,
    sc.Function[tuple[tuple[sc.Expr, sc.Expr], sc.Expr], tuple[tuple[np.ndarray, np.ndarray], np.ndarray], sc.Expr, np.ndarray],
  )
  assert_type(adj_square_x, sc.Function[tuple[sc.Expr, sc.Expr], tuple[np.ndarray, np.ndarray], sc.Expr, np.ndarray])
  assert_type(
    hess_l,
    sc.Function[tuple[sc.Expr, tuple[sc.Expr, sc.Expr]], tuple[np.ndarray, tuple[np.ndarray, np.ndarray]], sc.Expr, np.ndarray],
  )
  assert_type(fwd_f_x.numerical_call(((np.zeros(3), np.zeros(())), np.zeros(3))), np.ndarray)
  assert_type(hess_l.numerical_call((np.zeros(3), (np.zeros(3), np.zeros(3)))), np.ndarray)

  assert_type(quadratic, sc.Problem[sc.Expr, np.ndarray, sc.Expr, np.ndarray])
  assert_type(
    filter_problem,
    sc.Problem[
      tuple[sc.Expr, sc.Expr],
      tuple[np.ndarray, np.ndarray],
      tuple[sc.Expr, sc.Expr],
      tuple[np.ndarray, np.ndarray],
    ],
  )
  assert_type(qp3, sc.Problem[sc.Expr, np.ndarray, sc.QPData[sc.Expr], sc.QPData[np.ndarray]])
  sc.problem(vars=sc.G(sc.L("u", 2), sc.L("s", 1)), params=sc.L("p", ()))(lambda variables, p: sc.ProblemSpec(minimize=variables.sum()))  # ty: ignore[unresolved-attribute]
  sc.problem(vars=sc.L("x", 2), params=sc.L("p", ()))(lambda x, p: sc.ProblemSpec(minimize=x.sum(), lb=(x, x)))  # ty: ignore[invalid-argument-type]

  assert_type(
    quadratic_ipopt,
    sc.Function[
      tuple[sc.Expr, sc.Expr, sc.Expr, sc.Expr, sc.Expr],
      tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
      tuple[sc.Expr, sc.Expr, sc.Expr, sc.Expr],
      tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ],
  )
  assert_type(
    filter_sqp,
    sc.Function[
      tuple[tuple[sc.Expr, sc.Expr], tuple[sc.Expr, sc.Expr], sc.Expr, sc.Expr, tuple[sc.Expr, sc.Expr]],
      tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]],
      tuple[tuple[sc.Expr, sc.Expr], tuple[sc.Expr, sc.Expr], sc.Expr, sc.Expr],
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
        (sc.sym("u0", 2), sc.sym("s0", 1)),
        (sc.sym("lam_u0", 2), sc.sym("lam_s0", 1)),
        sc.sym("lam_eq0", 1),
        sc.sym("lam_ineq0", 3),
        (sc.sym("x0", 4), sc.sym("u_ref0", 2)),
      )
    ),
    tuple[tuple[sc.Expr, sc.Expr], tuple[sc.Expr, sc.Expr], sc.Expr, sc.Expr],
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
  filter_sqp.numerical_call(((sc.sym("u", 2), sc.sym("s", 1)), (np.zeros(2), np.zeros(1)), np.zeros(1), np.zeros(3), (np.zeros(4), np.zeros(2))))  # ty: ignore[invalid-argument-type]

  grad_f_x.numerical_call(np.zeros(3))  # ty: ignore[invalid-argument-type]
  fwd_f_x.numerical_call((np.zeros(3), np.zeros(()), np.zeros(3)))  # ty: ignore[invalid-argument-type]
  hess_l.numerical_call((np.zeros(3), np.zeros(6)))  # ty: ignore[invalid-argument-type]
