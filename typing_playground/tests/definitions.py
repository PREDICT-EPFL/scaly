"""Concrete definitions shared by the behavioral and the static tests."""

from __future__ import annotations

import numpy as np

from typing_playground.expr import Buffer, Expr, const
from typing_playground.function import Function, adjoint, forward, function, gradient, hessian, jacobian, lagrangian_hessian, vmap
from typing_playground.opti import ProblemSpec, bounded, problem, qp_problem, solver
from typing_playground.trees import arg, group

# --- functions ---


@function(arg("x", 3), outputs=group(arg("first", ...), arg("second", 3)))
def duplicate(x: Expr) -> tuple[Expr, Expr]:
  return x, x


@function(arg("x", 3), arg("p", ()))  # no outputs=: one traced leaf named "energy"
def energy(x: Expr, p: Expr) -> Expr:
  return (x * x).sum() * p


@function(arg("x", 3), arg("y", 3), outputs=arg("prod", ...))
def multiply(x: Expr, y: Expr) -> Expr:
  return x * y


@function(arg("x", 3), outputs=arg("square", ...))
def square(x: Expr) -> Expr:
  return multiply.symbolic_call(*duplicate.symbolic_call(x))


@function(arg("x", 3), arg("p", ()), outputs=arg("f", ...))
def cost(x: Expr, p: Expr) -> Expr:
  return (x * x).sum() * p


# Five inputs grouped the way the problem thinks about them ...
@function(group(arg("state", 4), arg("u", 2)), group(arg("pw", 10), arg("physics", 3), arg("dt", ())), outputs=arg("next", ...))
def step(xu: tuple[Expr, Expr], rest: tuple[Expr, Expr, Expr]) -> Expr:
  state, _u = xu
  return state


# ... or flat, when there is no natural grouping. Same leaves, same C signature.
@function(arg("state", 4), arg("u", 2), arg("pw", 10), arg("physics", 3), arg("dt", ()), outputs=arg("next", ...))
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
@function(group(arg("x", 3), arg("p", ())), outputs=arg("f", ...))
def cost_packed(inputs: tuple[Expr, Expr]) -> Expr:
  x, p = inputs
  return (x * x).sum() * p


# No parameters at all: a constant, evaluated by `constant()`.
@function(outputs=arg("c", ...))
def constant() -> Expr:
  return Expr((2,))


# --- holes ---


@function(arg("x"), arg("p"), outputs=arg("f"))
def cost2(x: Expr, p: Expr) -> Expr:
  return (x * x).sum() * p


@function(group(arg("x"), arg("y")), arg("p"), outputs=arg("f"))
def cost3(xy: tuple[Expr, Expr], p: Expr):
  x, y = xy
  return (x * y).sum() * p


@function(arg("x"), arg("p"), outputs=arg("f"), name="cost")
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

  @function(arg("x"), arg("p"), outputs=arg("f"), name=name)
  def fresh(x: Expr, p: Expr) -> Expr:
    return (x * x).sum() * p

  return fresh


# Trace-time resolution: decorating `caller` instantiates cost_t for shapes ((2,), (2,)).
@function(arg("x", 2), outputs=arg("y"))
def caller(x: Expr) -> Expr:
  return cost_t.symbolic_call(x, x)


# --- problems ---


@problem(vars=arg("x", 3), params=arg("scale", ()))
def quadratic(x: Expr, scale: Expr) -> ProblemSpec[Expr]:
  return ProblemSpec(minimize=(x * x).sum() * scale, lb=const(-1.0), ub=const(1.0))


# Multi-block variables, two constraint groups, box bounds in the variables' structure. The body
# binds `vs`/`ps`, not `u`/`s`/`x`: declared names are external, body names are local.
@problem(vars=group(arg("u", 2), arg("s", 1)), params=group(arg("x", 4), arg("u_ref", 2)))
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


@problem(vars=arg("x", 2), params=arg("w", ()))
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
