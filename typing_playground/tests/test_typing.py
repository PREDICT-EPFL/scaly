"""Static acceptance tests: ``uv run ty check --error-on-warning typing_playground``.

Every ``ty: ignore`` marks an expected error; an unused one fails the check. pytest collects
nothing from this file: the block below never runs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, assert_type

from typing_playground.expr import Buffer, Expr
from typing_playground.function import Function, function
from typing_playground.opti import Problem, ProblemSpec, QPData, problem
from typing_playground.templates import FunctionTemplate, forward, gradient, lagrangian_hessian, template
from typing_playground.tests.definitions import (
  adj_square_x,
  caller,
  cost,
  cost_batch,
  cost_t,
  duplicate,
  filter_problem,
  filter_sqp,
  fwd_f_x,
  grad_f_x,
  hess_f_x,
  hess_l,
  jac_square_x,
  multiply,
  qp3,
  qp3_piqp,
  quadratic,
  quadratic_ipopt,
  scale,
  square,
  step,
  step_flat,
)
from typing_playground.trees import G, L, Tree

if TYPE_CHECKING:
  # declarations: the count of a group is static, and only trees may be grouped
  L("x", "3")  # ty: ignore[invalid-argument-type]
  G(L("x", 3))  # ty: ignore[no-matching-overload]
  G(L("x", 3), L("y", 3), ("z", 3))  # ty: ignore[invalid-argument-type]
  assert_type(G(L("x", 3), L("p", ())), Tree[tuple[Expr, Expr], tuple[Buffer, Buffer]])
  assert_type(G(G(L("a", 1), L("b", 1)), L("c", 1)), Tree[tuple[tuple[Expr, Expr], Expr], tuple[tuple[Buffer, Buffer], Buffer]])

  # functions: structure, count and leaf kind are all checked, on both sides
  assert_type(duplicate, Function[Expr, Buffer, tuple[Expr, Expr], tuple[Buffer, Buffer]])
  assert_type(duplicate.symbolic_call(Expr((3,))), tuple[Expr, Expr])
  assert_type(duplicate.numerical_call(Buffer((3,))), tuple[Buffer, Buffer])
  assert_type(multiply.numerical_call((Buffer((3,)), Buffer((3,)))), Buffer)
  assert_type(step.numerical_call(((Buffer((4,)), Buffer((2,))), (Buffer((10,)), Buffer((3,)), Buffer(())))), Buffer)
  assert_type(step_flat.numerical_call((Buffer((4,)), Buffer((2,)), Buffer((10,)), Buffer((3,)), Buffer(()))), Buffer)
  multiply.numerical_call((Buffer((3,)),))  # ty: ignore[invalid-argument-type]
  multiply.numerical_call(Buffer((3,)))  # ty: ignore[invalid-argument-type]
  multiply.numerical_call((Expr((3,)), Expr((3,))))  # ty: ignore[invalid-argument-type]
  multiply.symbolic_call((Buffer((3,)), Buffer((3,))))  # ty: ignore[invalid-argument-type]
  duplicate.symbolic_call((Expr((3,)),))  # ty: ignore[invalid-argument-type]
  step.numerical_call((Buffer((4,)), Buffer((2,)), Buffer((10,)), Buffer((3,)), Buffer(())))  # ty: ignore[invalid-argument-type]
  step_flat.numerical_call(((Buffer((4,)), Buffer((2,))), (Buffer((10,)), Buffer((3,)), Buffer(()))))  # ty: ignore[invalid-argument-type]

  # decorator and body must agree
  function(G(L("x", 3), L("y", 3)), L("z", ...))(lambda x: x)  # ty: ignore[invalid-argument-type]
  function(L("x", 3), G(L("a", ...), L("b", ...)))(lambda x: x)  # ty: ignore[invalid-argument-type]

  # composition preserves types
  assert_type(multiply.symbolic_call(duplicate.symbolic_call(Expr((3,)))), Expr)
  multiply.symbolic_call(square.symbolic_call(Expr((3,))))  # ty: ignore[invalid-argument-type]

  # derivatives: the source's input tree is preserved, seeded modes pair it with the new group
  assert_type(grad_f_x, Function[tuple[Expr, Expr], tuple[Buffer, Buffer], Expr, Buffer])
  assert_type(hess_f_x, Function[tuple[Expr, Expr], tuple[Buffer, Buffer], Expr, Buffer])
  assert_type(jac_square_x, Function[Expr, Buffer, Expr, Buffer])
  assert_type(grad_f_x.numerical_call((Buffer((3,)), Buffer(()))), Buffer)
  assert_type(fwd_f_x, Function[tuple[tuple[Expr, Expr], Expr], tuple[tuple[Buffer, Buffer], Buffer], Expr, Buffer])
  assert_type(adj_square_x, Function[tuple[Expr, Expr], tuple[Buffer, Buffer], Expr, Buffer])
  assert_type(hess_l, Function[tuple[Expr, tuple[Expr, Expr]], tuple[Buffer, tuple[Buffer, Buffer]], Expr, Buffer])
  assert_type(fwd_f_x.numerical_call(((Buffer((3,)), Buffer(())), Buffer((3,)))), Buffer)
  assert_type(hess_l.numerical_call((Buffer((3,)), (Buffer((3,)), Buffer((3,))))), Buffer)
  grad_f_x.numerical_call(Buffer((3,)))  # ty: ignore[invalid-argument-type]
  fwd_f_x.numerical_call((Buffer((3,)), Buffer(()), Buffer((3,))))  # ty: ignore[invalid-argument-type]
  hess_l.numerical_call((Buffer((3,)), Buffer((6,))))  # ty: ignore[invalid-argument-type]

  # vmap is typed by the callee's trees
  assert_type(cost_batch, Function[tuple[Expr, Expr], tuple[Buffer, Buffer], Expr, Buffer])

  # templates: exactly Function's four variables, so calls typecheck identically
  assert_type(cost_t, FunctionTemplate[tuple[Expr, Expr], tuple[Buffer, Buffer], Expr, Buffer])
  assert_type(cost_t.numerical_call((Buffer((30, 40)), Buffer((5,)))), Buffer)
  assert_type(cost_t.symbolic_call((Expr((3,)), Expr(()))), Expr)
  assert_type(cost_t.instantiate(((30, 40), (5,))), Function[tuple[Expr, Expr], tuple[Buffer, Buffer], Expr, Buffer])
  assert_type(caller, Function[Expr, Buffer, Expr, Buffer])
  cost_t.numerical_call(Buffer((3,)))  # ty: ignore[invalid-argument-type]
  cost_t.numerical_call((Expr((3,)), Expr((3,))))  # ty: ignore[invalid-argument-type]
  cost_t.symbolic_call((Buffer((3,)), Buffer((3,))))  # ty: ignore[invalid-argument-type]
  template(G(L("x"), L("p")), L("f"))(lambda x, p: x)  # ty: ignore[invalid-argument-type]

  # one wrapper, two overloads: Functions stay Functions, templates stay templates
  assert_type(gradient(cost, "f", "x"), Function[tuple[Expr, Expr], tuple[Buffer, Buffer], Expr, Buffer])
  assert_type(gradient(cost_t, "f", "x"), FunctionTemplate[tuple[Expr, Expr], tuple[Buffer, Buffer], Expr, Buffer])
  assert_type(forward(cost_t, "f", "x"), FunctionTemplate[tuple[tuple[Expr, Expr], Expr], tuple[tuple[Buffer, Buffer], Buffer], Expr, Buffer])
  assert_type(lagrangian_hessian(cost_t, "x"), FunctionTemplate[tuple[tuple[Expr, Expr], Expr], tuple[tuple[Buffer, Buffer], Buffer], Expr, Buffer])
  gradient(cost_t, "f", "x").numerical_call(Buffer((3,)))  # ty: ignore[invalid-argument-type]

  # the bare mode is Any end to end: nothing below is an error. That is the stated trade.
  assert_type(scale, FunctionTemplate[Any, Any, Any, Any])
  scale.numerical_call((Expr((3,)), Buffer((3,)), "nonsense"))

  # problems: the body's parameter types are the declared trees, the spec's bounds have the vars' structure
  assert_type(quadratic, Problem[Expr, Buffer, Expr, Buffer])
  assert_type(filter_problem, Problem[tuple[Expr, Expr], tuple[Buffer, Buffer], tuple[Expr, Expr], tuple[Buffer, Buffer]])
  assert_type(qp3, Problem[Expr, Buffer, QPData[Expr], QPData[Buffer]])
  problem(vars=G(L("u", 2), L("s", 1)), params=L("p", ()))(lambda x, p: ProblemSpec(minimize=x.sum()))  # ty: ignore[unresolved-attribute]
  problem(vars=L("x", 2), params=L("p", ()))(lambda x, p: ProblemSpec(minimize=x.sum(), lb=(x, x)))  # ty: ignore[invalid-argument-type]

  # solvers are Functions with a fixed five-group input and four-group output
  assert_type(
    quadratic_ipopt,
    Function[
      tuple[Expr, Expr, Expr, Expr, Expr],
      tuple[Buffer, Buffer, Buffer, Buffer, Buffer],
      tuple[Expr, Expr, Expr, Expr],
      tuple[Buffer, Buffer, Buffer, Buffer],
    ],
  )
  assert_type(
    filter_sqp,
    Function[
      tuple[tuple[Expr, Expr], tuple[Expr, Expr], Expr, Expr, tuple[Expr, Expr]],
      tuple[tuple[Buffer, Buffer], tuple[Buffer, Buffer], Buffer, Buffer, tuple[Buffer, Buffer]],
      tuple[tuple[Expr, Expr], tuple[Expr, Expr], Expr, Expr],
      tuple[tuple[Buffer, Buffer], tuple[Buffer, Buffer], Buffer, Buffer],
    ],
  )
  assert_type(
    qp3_piqp.numerical_call(
      (
        Buffer((3,)),
        Buffer((3,)),
        Buffer((1,)),
        Buffer((2,)),
        ((Buffer((3, 3)), Buffer((3,))), (Buffer((1, 3)), Buffer((1,))), (Buffer((2, 3)), Buffer((2,)), Buffer((2,)))),
      )
    ),
    tuple[Buffer, Buffer, Buffer, Buffer],
  )
  quadratic_ipopt.numerical_call((Buffer((3,)), Buffer((3,)), Buffer((0,)), Buffer((0,))))  # ty: ignore[invalid-argument-type]
  filter_sqp.numerical_call(((Buffer((2,)),), (Buffer((2,)), Buffer((1,))), Buffer((1,)), Buffer((3,)), (Buffer((4,)), Buffer((2,)))))  # ty: ignore[invalid-argument-type]
  filter_sqp.numerical_call(((Expr((2,)), Expr((1,))), (Buffer((2,)), Buffer((1,))), Buffer((1,)), Buffer((3,)), (Buffer((4,)), Buffer((2,)))))  # ty: ignore[invalid-argument-type]
  # a solver nests in a larger graph like any Function
  assert_type(
    filter_sqp.symbolic_call(((Expr((2,)), Expr((1,))), (Expr((2,)), Expr((1,))), Expr((1,)), Expr((3,)), (Expr((4,)), Expr((2,))))),
    tuple[tuple[Expr, Expr], tuple[Expr, Expr], Expr, Expr],
  )
