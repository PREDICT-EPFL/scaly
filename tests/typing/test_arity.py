"""Static acceptance tests for typed functions, derivatives, problems, and solvers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, assert_type

import numpy as np

import scaly as sc
from scaly.function import Tree
from scaly.function.concrete import ConcreteFunction


@sc.function(sc.arg("x", 3), outputs=sc.group(sc.arg("first"), sc.arg("second", 3)))
def duplicate(x: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
  return x, x


@sc.function(sc.group(sc.arg("x", 3), sc.arg("y", 3)), outputs=sc.arg("prod"))
def multiply(inputs: tuple[sc.Expr, sc.Expr]) -> sc.Expr:
  x, y = inputs
  return x * y


@sc.function(sc.arg("x", 3), outputs=sc.arg("square"))
def square(x: sc.Expr) -> sc.Expr:
  return multiply.symbolic_call(duplicate.symbolic_call(x))


@sc.function(sc.group(sc.arg("x", 3), sc.arg("p", ())), outputs=sc.arg("f"))
def cost(inputs: tuple[sc.Expr, sc.Expr]) -> sc.Expr:
  x, p = inputs
  return (x * x).sum() * p


@sc.function(
  sc.group(sc.group(sc.arg("state", 4), sc.arg("u", 2)), sc.group(sc.arg("pw", 10), sc.arg("physics", 3), sc.arg("dt", ()))),
  outputs=sc.arg("next"),
)
def step(inputs: tuple[tuple[sc.Expr, sc.Expr], tuple[sc.Expr, sc.Expr, sc.Expr]]) -> sc.Expr:
  (state, _u), (_pw, _physics, _dt) = inputs
  return state


@sc.function(sc.group(sc.arg("state", 4), sc.arg("u", 2), sc.arg("pw", 10), sc.arg("physics", 3), sc.arg("dt", ())), outputs=sc.arg("next"))
def step_flat(inputs: tuple[sc.Expr, sc.Expr, sc.Expr, sc.Expr, sc.Expr]) -> sc.Expr:
  state, _u, _pw, _physics, _dt = inputs
  return state


@sc.problem(vars=sc.arg("x", 3), params=sc.arg("scale", ()))
def quadratic(x: sc.Expr, scale: sc.Expr) -> sc.ProblemSpec[sc.Expr]:
  return sc.ProblemSpec(minimize=(x * x).sum() * scale, lb=sc.const(-1.0), ub=sc.const(1.0))


@sc.problem(vars=sc.group(sc.arg("u", 2), sc.arg("s", 1)), params=sc.group(sc.arg("x", 4), sc.arg("u_ref", 2)))
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
  sc.arg("x", "3")  # ty: ignore[invalid-argument-type]
  sc.group(sc.arg("x", 3))
  sc.group(sc.arg("x", 3), sc.arg("y", 3), ("z", 3))  # ty: ignore[invalid-argument-type]
  assert_type(sc.group(sc.arg("x", 3), sc.arg("p", ())), Tree[tuple[sc.Expr, sc.Expr], tuple[np.ndarray, np.ndarray]])
  assert_type(
    sc.group(sc.group(sc.arg("a", 1), sc.arg("b", 1)), sc.arg("c", 1)),
    Tree[tuple[tuple[sc.Expr, sc.Expr], sc.Expr], tuple[tuple[np.ndarray, np.ndarray], np.ndarray]],
  )

  assert_type(duplicate, sc.Function[tuple[sc.Expr], tuple[np.ndarray], tuple[sc.Expr, sc.Expr], tuple[np.ndarray, np.ndarray]])
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

  sc.function(sc.group(sc.arg("x", 3), sc.arg("y", 3)), outputs=sc.arg("z"))(lambda x: x)  # ty: ignore[invalid-argument-type]
  sc.function(sc.arg("x", 3), outputs=sc.group(sc.arg("a"), sc.arg("b")))(lambda x: x)  # ty: ignore[invalid-argument-type]

  assert_type(multiply.symbolic_call(duplicate.symbolic_call(sc.sym("x", 3))), sc.Expr)
  multiply.symbolic_call(square.symbolic_call(sc.sym("x", 3)))  # ty: ignore[invalid-argument-type]

  # __call__ dispatches on the leaf kind and is typed as precisely as the two named methods.
  assert_type(duplicate(sc.sym("x", 3)), tuple[sc.Expr, sc.Expr])
  assert_type(duplicate(np.zeros(3)), tuple[np.ndarray, np.ndarray])
  assert_type(multiply((sc.sym("x", 3), sc.sym("y", 3))), sc.Expr)
  assert_type(multiply((np.zeros(3), np.zeros(3))), np.ndarray)
  assert_type(step(((np.zeros(4), np.zeros(2)), (np.zeros(10), np.zeros(3), np.zeros(())))), np.ndarray)
  assert_type(multiply(duplicate(sc.sym("x", 3))), sc.Expr)
  assert_type(hess_l(*(np.zeros(3), (np.zeros(3), np.zeros(3)))), np.ndarray)
  multiply((np.zeros(3),))  # ty: ignore[no-matching-overload]
  multiply((sc.sym("x", 3), np.zeros(3)))  # ty: ignore[no-matching-overload]
  step((np.zeros(4), np.zeros(2), np.zeros(10), np.zeros(3), np.zeros(())))  # ty: ignore[no-matching-overload]
  multiply(square(sc.sym("x", 3)))  # ty: ignore[no-matching-overload]

  assert_type(grad_f_x, sc.Function[tuple[tuple[sc.Expr, sc.Expr]], tuple[tuple[np.ndarray, np.ndarray]], sc.Expr, np.ndarray])
  assert_type(hess_f_x, sc.Function[tuple[tuple[sc.Expr, sc.Expr]], tuple[tuple[np.ndarray, np.ndarray]], sc.Expr, np.ndarray])
  assert_type(jac_square_x, sc.Function[tuple[sc.Expr], tuple[np.ndarray], sc.Expr, np.ndarray])
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
  assert_type(fwd_f_x.numerical_call(*((np.zeros(3), np.zeros(())), np.zeros(3))), np.ndarray)
  assert_type(hess_l.numerical_call(*(np.zeros(3), (np.zeros(3), np.zeros(3)))), np.ndarray)

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
  sc.problem(vars=sc.group(sc.arg("u", 2), sc.arg("s", 1)), params=sc.arg("p", ()))(lambda variables, p: sc.ProblemSpec(minimize=variables.sum()))  # ty: ignore[unresolved-attribute]
  sc.problem(vars=sc.arg("x", 2), params=sc.arg("p", ()))(lambda x, p: sc.ProblemSpec(minimize=x.sum(), lb=(x, x)))  # ty: ignore[invalid-argument-type]

  assert_type(quadratic_ipopt, sc.Solver[sc.Expr, np.ndarray, sc.Expr, np.ndarray])
  assert_type(
    quadratic_ipopt.function,
    sc.Function[
      tuple[sc.Expr, sc.Expr, sc.Expr, sc.Expr, sc.Expr],
      tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
      tuple[sc.Expr, sc.Expr, sc.Expr, sc.Expr],
      tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ],
  )
  assert_type(quadratic_ipopt(np.zeros(())), tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray])
  assert_type(quadratic_ipopt(np.zeros(()), x0=np.zeros(3)), tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray])
  assert_type(quadratic_ipopt(sc.sym("scale", ())), tuple[sc.Expr, sc.Expr, sc.Expr, sc.Expr])
  assert_type(
    filter_sqp,
    sc.Solver[tuple[sc.Expr, sc.Expr], tuple[np.ndarray, np.ndarray], tuple[sc.Expr, sc.Expr], tuple[np.ndarray, np.ndarray]],
  )
  assert_type(
    filter_sqp((np.zeros(4), np.zeros(2))),
    tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], np.ndarray, np.ndarray],
  )
  assert_type(
    filter_sqp((sc.sym("x0", 4), sc.sym("u_ref0", 2)), x0=(sc.sym("u0", 2), sc.sym("s0", 1))),
    tuple[tuple[sc.Expr, sc.Expr], tuple[sc.Expr, sc.Expr], sc.Expr, sc.Expr],
  )
  assert_type(
    filter_sqp.function,
    sc.Function[
      tuple[tuple[sc.Expr, sc.Expr], tuple[sc.Expr, sc.Expr], sc.Expr, sc.Expr, tuple[sc.Expr, sc.Expr]],
      tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]],
      tuple[tuple[sc.Expr, sc.Expr], tuple[sc.Expr, sc.Expr], sc.Expr, sc.Expr],
      tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], np.ndarray, np.ndarray],
    ],
  )
  assert_type(
    filter_sqp.function.numerical_call(
      *(
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
    filter_sqp.function.symbolic_call(
      *(
        (sc.sym("u0", 2), sc.sym("s0", 1)),
        (sc.sym("lam_u0", 2), sc.sym("lam_s0", 1)),
        sc.sym("lam_eq0", 1),
        sc.sym("lam_ineq0", 3),
        (sc.sym("x0", 4), sc.sym("u_ref0", 2)),
      )
    ),
    tuple[tuple[sc.Expr, sc.Expr], tuple[sc.Expr, sc.Expr], sc.Expr, sc.Expr],
  )
  assert_type(qp3_piqp, sc.Solver[sc.Expr, np.ndarray, sc.QPData[sc.Expr], sc.QPData[np.ndarray]])
  assert_type(
    qp3_piqp.function.numerical_call(
      *(
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
  quadratic_ipopt.function.numerical_call(*(np.zeros(3), np.zeros(3), np.zeros(0), np.zeros(0)))  # ty: ignore[invalid-argument-type]
  filter_sqp((np.zeros(4),))  # ty: ignore[no-matching-overload]
  filter_sqp((np.zeros(4), np.zeros(2)), x0=np.zeros(3))  # ty: ignore[no-matching-overload]
  filter_sqp.function.numerical_call(*((np.zeros(2),), (np.zeros(2), np.zeros(1)), np.zeros(1), np.zeros(3), (np.zeros(4), np.zeros(2))))  # ty: ignore[invalid-argument-type]
  filter_sqp.function.numerical_call(  # ty: ignore[invalid-argument-type]
    *((sc.sym("u", 2), sc.sym("s", 1)), (np.zeros(2), np.zeros(1)), np.zeros(1), np.zeros(3), (np.zeros(4), np.zeros(2)))
  )

  grad_f_x.numerical_call(np.zeros(3))  # ty: ignore[invalid-argument-type]
  fwd_f_x.numerical_call(*(np.zeros(3), np.zeros(()), np.zeros(3)))  # ty: ignore[invalid-argument-type]
  hess_l.numerical_call(*(np.zeros(3), np.zeros(6)))  # ty: ignore[invalid-argument-type]


if TYPE_CHECKING:

  @sc.function(sc.arg("x", 3), sc.arg("p", ()))
  def energy(x: sc.Expr, p: sc.Expr) -> sc.Expr:
    return (x * x).sum() * p

  assert_type(energy, sc.Function[tuple[sc.Expr, sc.Expr], tuple[np.ndarray, np.ndarray], sc.Expr, np.ndarray])
  assert_type(energy(np.zeros(3), np.array(2.0)), np.ndarray)
  assert_type(energy(sc.sym("x", 3), sc.sym("p", ())), sc.Expr)
  energy(sc.sym("x", 3), np.array(2.0))  # ty: ignore[no-matching-overload]
  energy(np.zeros(3))  # ty: ignore[no-matching-overload]
  sc.function(sc.arg("x", 3), sc.arg("y", 3), outputs=sc.arg("z"))(lambda x: x)  # ty: ignore[invalid-argument-type]
  sc.function(sc.group(sc.arg("x", 3), sc.arg("y", 3)), outputs=sc.arg("z"))(lambda x, y: x)  # ty: ignore[invalid-argument-type]
  assert_type(sc.group(sc.arg("x", 3)), Tree[tuple[sc.Expr], tuple[np.ndarray]])

  @sc.function(outputs=sc.arg("value", 2))
  def constant() -> sc.Expr:
    return sc.const([2.0, 3.0])

  assert_type(constant, sc.Function[tuple[()], tuple[()], sc.Expr, np.ndarray])
  assert_type(constant(), np.ndarray)
  assert_type(constant.symbolic_call(), sc.Expr)
  constant(np.zeros(2))  # ty: ignore[no-matching-overload]


if TYPE_CHECKING:

  @sc.function(sc.arg("x"), sc.arg("p", ()), outputs=sc.arg("f", ()))
  def template(x: sc.Expr, p: sc.Expr) -> sc.Expr:
    return (x * x).sum() * p

  assert_type(template, sc.Function[tuple[sc.Expr, sc.Expr], tuple[np.ndarray, np.ndarray], sc.Expr, np.ndarray])
  assert_type(template.instantiate((3, ())), ConcreteFunction[tuple[sc.Expr, sc.Expr], tuple[np.ndarray, np.ndarray], sc.Expr, np.ndarray])
  assert_type(template.instantiate((3, ())).numerical_call(np.zeros(3), np.array(2.0)), np.ndarray)
  assert_type(template.instantiate((3, ())).input_shapes, tuple[tuple[int, ...], ...])
  assert_type(sc.forward(template), sc.Function[tuple[sc.Expr, sc.Expr, sc.Expr], tuple[np.ndarray, np.ndarray, np.ndarray], sc.Expr, np.ndarray])

  @sc.function()
  def bare(x: sc.Expr, p: sc.Expr) -> sc.Expr:
    return x * p

  assert_type(bare, sc.Function[tuple[sc.Expr, sc.Expr], Any, sc.Expr, Any])
  assert_type(bare.symbolic_call(sc.sym("x", 3), sc.sym("p", ())), sc.Expr)
  bare.symbolic_call(np.zeros(3), sc.sym("p", ()))  # ty: ignore[invalid-argument-type]

  @sc.function(sc.arg("x", 3))
  def inferred(x: sc.Expr) -> tuple[sc.Expr, tuple[sc.Expr, sc.Expr]]:
    return x, (x, x.sum())

  assert_type(inferred, sc.Function[tuple[sc.Expr], tuple[np.ndarray], tuple[sc.Expr, tuple[sc.Expr, sc.Expr]], Any])
  assert_type(inferred.symbolic_call(sc.sym("x", 3)), tuple[sc.Expr, tuple[sc.Expr, sc.Expr]])

  @sc.function(
    sc.arg("a", 1),
    sc.arg("b", 1),
    sc.arg("c", 1),
    sc.arg("d", 1),
    sc.arg("e", 1),
    sc.arg("f", 1),
    sc.arg("g", 1),
    sc.arg("h", 1),
    outputs=sc.arg("y", 1),
  )
  def eight(a: sc.Expr, b: sc.Expr, c: sc.Expr, d: sc.Expr, e: sc.Expr, f: sc.Expr, g: sc.Expr, h: sc.Expr) -> sc.Expr:
    return a + b + c + d + e + f + g + h

  assert_type(eight(np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1)), np.ndarray)
  sc.function(sc.arg("a"), sc.arg("b"), sc.arg("c"), sc.arg("d"), sc.arg("e"), sc.arg("f"), sc.arg("g"), sc.arg("h"), sc.arg("i"))  # ty: ignore[no-matching-overload]


if TYPE_CHECKING:
  assert_type(sc.vmap(template, 4), sc.Function[tuple[sc.Expr, sc.Expr], tuple[np.ndarray, np.ndarray], sc.Expr, np.ndarray])
  assert_type(sc.vmap(template, 4)(sc.sym("batch", (4, 3)), sc.broadcast(sc.sym("p", ()))), sc.Expr)
  assert_type(sc.vmap(template, 4)(np.zeros((4, 3)), sc.broadcast(np.array(2.0))), np.ndarray)
  assert_type(sc.broadcast(sc.sym("p", ())), sc.Expr)
  array: np.ndarray = np.zeros(5)
  assert_type(sc.window(array, 0, 1), np.ndarray)
  assert_type(sc.vmap(duplicate, 4)(np.zeros((4, 3))), tuple[np.ndarray, np.ndarray])
  sc.vmap(template, 4)(np.zeros((4, 3)))  # ty: ignore[no-matching-overload]
  sc.vmap(template, 4)(sc.sym("x", (4, 3)), np.zeros(4))  # ty: ignore[no-matching-overload]
