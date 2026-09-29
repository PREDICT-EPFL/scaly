"""Concrete definitions shared by the behavioral and the static tests."""

from __future__ import annotations

import numpy as np

from typing_playground.expr import Buffer, Expr, const
from typing_playground.function import Function, adjoint, forward, function, gradient, hessian, jacobian, lagrangian_hessian, vmap
from typing_playground.opti import ProblemSpec, bounded, problem, qp_problem, solver
from typing_playground.trees import G, L

# --- functions ---


@function(L("x", 3), outputs=G(L("first", ...), L("second", 3)))
def duplicate(x: Expr) -> tuple[Expr, Expr]:
  return x, x


@function(L("x", 3), L("y", 3), outputs=L("prod", ...))
def multiply(x: Expr, y: Expr) -> Expr:
  return x * y


@function(L("x", 3), outputs=L("square", ...))
def square(x: Expr) -> Expr:
  return multiply.symbolic_call(*duplicate.symbolic_call(x))


@function(L("x", 3), L("p", ()), outputs=L("f", ...))
def cost(x: Expr, p: Expr) -> Expr:
  return (x * x).sum() * p


# Five inputs grouped the way the problem thinks about them ...
@function(G(L("state", 4), L("u", 2)), G(L("pw", 10), L("physics", 3), L("dt", ())), outputs=L("next", ...))
def step(xu: tuple[Expr, Expr], rest: tuple[Expr, Expr, Expr]) -> Expr:
  state, _u = xu
  return state


# ... or flat, when there is no natural grouping. Same leaves, same C signature.
@function(L("state", 4), L("u", 2), L("pw", 10), L("physics", 3), L("dt", ()), outputs=L("next", ...))
def step_flat(state: Expr, u: Expr, pw: Expr, physics: Expr, dt: Expr) -> Expr:
  return state


grad_f_x = gradient(cost, "f", "x")
hess_f_x = hessian(cost, "f", "x")
jac_square_x = jacobian(square, "square", "x")
fwd_f_x = forward(cost, "f", "x")
adj_square_x = adjoint(square, "square", "x")
hess_l = lagrangian_hessian(duplicate, "x")
cost_batch = vmap(cost, 7)


# A group is one parameter: the body receives the tuple.
@function(G(L("x", 3), L("p", ())), outputs=L("f", ...))
def cost_packed(inputs: tuple[Expr, Expr]) -> Expr:
  x, p = inputs
  return (x * x).sum() * p


# No parameters at all: a constant, evaluated by `constant()`.
@function(outputs=L("c", ...))
def constant() -> Expr:
  return Expr((2,))


# --- holes ---


@function(L("x"), L("p"), outputs=L("f"), name="cost")
def cost_t(x: Expr, p: Expr) -> Expr:
  return (x * x).sum() * p


@function()
def scale(a, b):  # the "average user" spelling: no declaration, no annotations
  return a * b


@function()
def scale_annotated(a: Expr, b: Expr) -> Expr:  # bare, but the annotations type the symbolic side
  return a * b


def fresh_cost(name: str = "cost") -> Function[tuple[Expr, Expr], tuple[Buffer, Buffer], Expr, Buffer]:
  """A function with holes that nothing else has instantiated, for cache tests."""

  @function(L("x"), L("p"), outputs=L("f"), name=name)
  def fresh(x: Expr, p: Expr) -> Expr:
    return (x * x).sum() * p

  return fresh


# Trace-time resolution: decorating `caller` instantiates cost_t for shapes ((2,), (2,)).
@function(L("x", 2), outputs=L("y"))
def caller(x: Expr) -> Expr:
  return cost_t.symbolic_call(x, x)


# --- problems ---


@problem(vars=L("x", 3), params=L("scale", ()))
def quadratic(x: Expr, scale: Expr) -> ProblemSpec[Expr]:
  return ProblemSpec(minimize=(x * x).sum() * scale, lb=const(-1.0), ub=const(1.0))


# Multi-block variables, two constraint groups, box bounds in the variables' structure. The body
# binds `vs`/`ps`, not `u`/`s`/`x`: declared names are external, body names are local.
@problem(vars=G(L("u", 2), L("s", 1)), params=G(L("x", 4), L("u_ref", 2)))
def filter_problem(vs: tuple[Expr, Expr], ps: tuple[Expr, Expr]) -> ProblemSpec[tuple[Expr, Expr]]:
  u, s = vs
  x, u_ref = ps
  barrier = x[:2] @ u + x[2:].sum()  # affine in u, parametrised by x
  return ProblemSpec(
    minimize=0.5 * ((u - u_ref) * (u - u_ref)).sum() + 10.0 * s.sum(),
    eq=(u[0:1] - u[1:2],),
    ineq=(bounded(barrier + s, lo=0.0, name="cbf"), bounded(u, lo=-1.0, hi=1.0, name="u_box")),
    lb=(const(np.full(2, -1.0)), const(np.zeros(1))),
  )


@problem(vars=L("x", 2), params=L("w", ()))
def rosenbrock(x: Expr, w: Expr) -> ProblemSpec[Expr]:
  return ProblemSpec(minimize=(1 - x[0]) ** 2 + w * (x[1] - x[0] ** 2) ** 2, ineq=(bounded(x[0].sin(), hi=0.5, name="wave"),))


qp3 = qp_problem(3, 1, 2)

quadratic_ipopt = solver(quadratic, "ipopt")
quadratic_piqp = solver(quadratic, "piqp", name="quadratic_fast")
filter_sqp = solver(filter_problem, "sqp")
filter_piqp = solver(filter_problem, "piqp")
rosenbrock_ipopt = solver(rosenbrock, "ipopt")
qp3_piqp = solver(qp3, "piqp")
qp3_ipopt = solver(qp3, "ipopt")
