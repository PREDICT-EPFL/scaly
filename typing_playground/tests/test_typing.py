"""Static acceptance tests: ``uv run ty check --error-on-warning typing_playground``.

Every ``ty: ignore`` marks an expected error; an unused one fails the check. pytest collects
nothing from this file: the block below never runs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, assert_type

from typing_playground.expr import Buffer, Expr
from typing_playground.concrete import ConcreteFunction
from typing_playground.function import Function, forward, function, gradient, lagrangian_hessian, vmap
from typing_playground.opti import Problem, ProblemSpec, QPData, problem
from typing_playground.tests.definitions import (
  adj_square_x,
  caller,
  constant,
  energy,
  cost,
  cost_batch,
  cost_packed,
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
  scale_annotated,
  square,
  step,
  step_flat,
)
from typing_playground.trees import Tree, arg, group

if TYPE_CHECKING:
  # declarations: the count of a group is static, and only trees may be grouped
  arg("x", "3")  # ty: ignore[invalid-argument-type]
  group()  # ty: ignore[no-matching-overload]
  group(arg("x", 3), arg("y", 3), ("z", 3))  # ty: ignore[invalid-argument-type]
  assert_type(group(arg("x", 3)), Tree[tuple[Expr], tuple[Buffer]])  # never normalized to the leaf
  assert_type(group(arg("x", 3), arg("p", ())), Tree[tuple[Expr, Expr], tuple[Buffer, Buffer]])
  assert_type(group(group(arg("a", 1), arg("b", 1)), arg("c", 1)), Tree[tuple[tuple[Expr, Expr], Expr], tuple[tuple[Buffer, Buffer], Buffer]])

  # functions: the input types are the parameter lists; count, structure and leaf kind are checked on both sides
  assert_type(duplicate, Function[tuple[Expr], tuple[Buffer], tuple[Expr, Expr], tuple[Buffer, Buffer]])
  assert_type(multiply, Function[tuple[Expr, Expr], tuple[Buffer, Buffer], Expr, Buffer])
  assert_type(duplicate.symbolic_call(Expr((3,))), tuple[Expr, Expr])
  assert_type(duplicate.numerical_call(Buffer((3,))), tuple[Buffer, Buffer])
  assert_type(multiply.numerical_call(Buffer((3,)), Buffer((3,))), Buffer)
  assert_type(step.numerical_call((Buffer((4,)), Buffer((2,))), (Buffer((10,)), Buffer((3,)), Buffer(()))), Buffer)
  assert_type(step_flat.numerical_call(Buffer((4,)), Buffer((2,)), Buffer((10,)), Buffer((3,)), Buffer(())), Buffer)
  multiply.numerical_call(Buffer((3,)))  # ty: ignore[invalid-argument-type]
  multiply.numerical_call((Buffer((3,)), Buffer((3,))))  # ty: ignore[invalid-argument-type]
  multiply.numerical_call(Expr((3,)), Expr((3,)))  # ty: ignore[invalid-argument-type]
  multiply.symbolic_call(Buffer((3,)), Buffer((3,)))  # ty: ignore[invalid-argument-type]
  duplicate.symbolic_call(Expr((3,)), Expr((3,)))  # ty: ignore[invalid-argument-type]
  step.numerical_call(Buffer((4,)), Buffer((2,)), Buffer((10,)), Buffer((3,)), Buffer(()))  # ty: ignore[invalid-argument-type]
  step_flat.numerical_call((Buffer((4,)), Buffer((2,))), (Buffer((10,)), Buffer((3,)), Buffer(())))  # ty: ignore[invalid-argument-type]

  # decorator and body must agree: one tree per parameter, a group being one parameter; 0 to 8 of them
  function(arg("x", 3), arg("y", 3), outputs=arg("z", ...))(lambda x, y: x)
  function(group(arg("x", 3), arg("y", 3)), outputs=arg("z", ...))(lambda xy: xy[0])
  function(group(arg("x", 3), arg("y", 3)), outputs=arg("z", ...))(lambda x, y: x)  # ty: ignore[invalid-argument-type]
  assert_type(cost_packed, Function[tuple[tuple[Expr, Expr]], tuple[tuple[Buffer, Buffer]], Expr, Buffer])
  assert_type(cost_packed.numerical_call((Buffer((3,)), Buffer(()))), Buffer)
  # without outputs= the output type is the body's, numerically typed for an Expr and Any otherwise;
  # a forgotten `outputs=` is one parameter too many for the body
  assert_type(energy, Function[tuple[Expr, Expr], tuple[Buffer, Buffer], Expr, Buffer])
  assert_type(function(arg("x", 3))(lambda x: (x, x)), Function[tuple[Expr], tuple[Buffer], tuple[Expr, Expr], Any])
  function(group(arg("x", 3), arg("y", 3)), arg("z", ...))(lambda xy: xy[0])  # ty: ignore[no-matching-overload]
  assert_type(constant, Function[tuple[()], tuple[()], Expr, Buffer])
  assert_type(constant(), Buffer)
  assert_type(constant.symbolic_call(), Expr)
  constant(Buffer(()))  # ty: ignore[no-matching-overload]
  eight = function(arg("a"), arg("b"), arg("c"), arg("d"), arg("e"), arg("f"), arg("g"), arg("h"), outputs=arg("y"))(lambda a, b, c, d, e, f, g, h: a)
  assert_type(
    eight,
    Function[
      tuple[Expr, Expr, Expr, Expr, Expr, Expr, Expr, Expr], tuple[Buffer, Buffer, Buffer, Buffer, Buffer, Buffer, Buffer, Buffer], Expr, Buffer
    ],
  )
  function(arg("a"), arg("b"), arg("c"), arg("d"), arg("e"), arg("f"), arg("g"), arg("h"), arg("i"), outputs=arg("y"))  # ty: ignore[no-matching-overload]
  function(arg("x", 3), arg("y", 3), outputs=arg("z", ...))(lambda x: x)  # ty: ignore[invalid-argument-type]
  function(arg("x", 3), arg("y", 3), outputs=arg("z", ...))(lambda inputs: inputs[0])  # ty: ignore[invalid-argument-type]
  function(arg("x", 3), outputs=arg("z", ...))(lambda x, y: x)  # ty: ignore[invalid-argument-type]
  function(arg("x", 3), outputs=group(arg("a", ...), arg("b", ...)))(lambda x: x)  # ty: ignore[invalid-argument-type]

  @function(arg("x", 3), arg("y", 3), outputs=arg("z", ...))  # ty: ignore[invalid-argument-type]
  def wrong_kind(x: Buffer, y: Expr) -> Expr:
    return y

  # `__call__` dispatches on the leaf kind, both sides typed; mixing them matches neither overload
  assert_type(cost(Buffer((3,)), Buffer(())), Buffer)
  assert_type(cost(Expr((3,)), Expr(())), Expr)
  assert_type(duplicate(Expr((3,))), tuple[Expr, Expr])
  assert_type(cost_t(Buffer((3,)), Buffer(())), Buffer)
  assert_type(cost.instantiate()(Expr((3,)), Expr(())), Expr)
  assert_type(scale_annotated(Expr((3,)), Expr(())), Any)  # the numerical overload wins on an `Any` side
  cost(Expr((3,)), Buffer(()))  # ty: ignore[no-matching-overload]
  cost(Buffer((3,)))  # ty: ignore[no-matching-overload]

  # composition preserves types
  assert_type(multiply.symbolic_call(*duplicate.symbolic_call(Expr((3,)))), Expr)
  multiply.symbolic_call(square.symbolic_call(Expr((3,))))  # ty: ignore[invalid-argument-type]

  # derivatives keep the source's parameters; seeded modes append one, flat, in a single overload
  assert_type(grad_f_x, Function[tuple[Expr, Expr], tuple[Buffer, Buffer], Expr, Buffer])
  assert_type(hess_f_x, Function[tuple[Expr, Expr], tuple[Buffer, Buffer], Expr, Buffer])
  assert_type(jac_square_x, Function[tuple[Expr], tuple[Buffer], Expr, Buffer])
  assert_type(grad_f_x.numerical_call(Buffer((3,)), Buffer(())), Buffer)
  assert_type(fwd_f_x, Function[tuple[Expr, Expr, Expr], tuple[Buffer, Buffer, Buffer], Expr, Buffer])
  assert_type(adj_square_x, Function[tuple[Expr, Expr], tuple[Buffer, Buffer], Expr, Buffer])
  assert_type(adj_square_x.numerical_call(Buffer((3,)), Buffer((3,))), Buffer)
  assert_type(hess_l, Function[tuple[Expr, tuple[Expr, Expr]], tuple[Buffer, tuple[Buffer, Buffer]], Expr, Buffer])
  assert_type(
    forward(step, "next", "state"),
    Function[
      tuple[tuple[Expr, Expr], tuple[Expr, Expr, Expr], Expr], tuple[tuple[Buffer, Buffer], tuple[Buffer, Buffer, Buffer], Buffer], Expr, Buffer
    ],
  )
  assert_type(fwd_f_x.numerical_call(Buffer((3,)), Buffer(()), Buffer((3,))), Buffer)
  assert_type(hess_l.numerical_call(Buffer((3,)), (Buffer((3,)), Buffer((3,)))), Buffer)
  grad_f_x.numerical_call(Buffer((3,)))  # ty: ignore[invalid-argument-type]
  fwd_f_x.numerical_call((Buffer((3,)), Buffer(())), Buffer((3,)))  # ty: ignore[invalid-argument-type]
  hess_l.numerical_call(Buffer((3,)), Buffer((3,)), Buffer((3,)))  # ty: ignore[invalid-argument-type]

  # vmap is typed by the callee's trees, with or without holes
  assert_type(cost_batch, Function[tuple[Expr, Expr], tuple[Buffer, Buffer], Expr, Buffer])
  assert_type(cost_batch.numerical_call(Buffer((7, 3)), Buffer((7,))), Buffer)
  assert_type(vmap(cost_t, 4), Function[tuple[Expr, Expr], tuple[Buffer, Buffer], Expr, Buffer])

  # holes are invisible to the types; shapes live on the instance, which types its calls the same way
  assert_type(cost_t, Function[tuple[Expr, Expr], tuple[Buffer, Buffer], Expr, Buffer])
  assert_type(cost_t.numerical_call(Buffer((30, 40)), Buffer((5,))), Buffer)
  assert_type(cost_t.symbolic_call(Expr((3,)), Expr(())), Expr)
  assert_type(cost_t.instantiate(((30, 40), (5,))), ConcreteFunction[tuple[Expr, Expr], tuple[Buffer, Buffer], Expr, Buffer])
  assert_type(cost.instantiate().numerical_call(Buffer((3,)), Buffer(())), Buffer)
  assert_type(cost.instantiate().input_shapes, tuple[tuple[int, ...], ...])
  cost.input_shapes  # ty: ignore[unresolved-attribute]
  assert_type(caller, Function[tuple[Expr], tuple[Buffer], Expr, Buffer])
  cost_t.numerical_call(Buffer((3,)))  # ty: ignore[invalid-argument-type]
  cost_t.numerical_call(Expr((3,)), Expr((3,)))  # ty: ignore[invalid-argument-type]
  cost_t.symbolic_call(Buffer((3,)), Buffer((3,)))  # ty: ignore[invalid-argument-type]
  function(arg("x"), arg("p"), outputs=arg("f"))(lambda x, p: x)
  function(arg("x"), arg("p"), outputs=arg("f"))(lambda inputs: inputs[0])  # ty: ignore[invalid-argument-type]

  # one signature per wrapper, holes or not
  assert_type(gradient(cost_t, "f", "x"), Function[tuple[Expr, Expr], tuple[Buffer, Buffer], Expr, Buffer])
  assert_type(forward(cost_t, "f", "x"), Function[tuple[Expr, Expr, Expr], tuple[Buffer, Buffer, Buffer], Expr, Buffer])
  assert_type(lagrangian_hessian(cost_t, "x"), Function[tuple[Expr, Expr, Expr], tuple[Buffer, Buffer, Buffer], Expr, Buffer])
  gradient(cost_t, "f", "x").numerical_call(Buffer((3,)))  # ty: ignore[invalid-argument-type]

  # the bare mode checks arity; the symbolic side is the body's annotations, the numerical side Any
  assert_type(scale_annotated, Function[tuple[Expr, Expr], Any, Expr, Any])
  assert_type(scale_annotated.symbolic_call(Expr((3,)), Expr(())), Expr)
  scale_annotated.symbolic_call(Buffer((3,)), Expr(()))  # ty: ignore[invalid-argument-type]
  scale.symbolic_call(Expr((3,)))  # ty: ignore[invalid-argument-type]
  scale.numerical_call(Expr((3,)), Buffer((3,)), "nonsense")

  # problems: the body's parameter types are the declared trees, the spec's bounds have the vars' structure
  assert_type(quadratic, Problem[Expr, Buffer, Expr, Buffer])
  assert_type(filter_problem, Problem[tuple[Expr, Expr], tuple[Buffer, Buffer], tuple[Expr, Expr], tuple[Buffer, Buffer]])
  assert_type(qp3, Problem[Expr, Buffer, QPData[Expr], QPData[Buffer]])
  problem(vars=group(arg("u", 2), arg("s", 1)), params=arg("p", ()))(lambda x, p: ProblemSpec(minimize=x.sum()))  # ty: ignore[unresolved-attribute]
  problem(vars=arg("x", 2), params=arg("p", ()))(lambda x, p: ProblemSpec(minimize=x.sum(), lb=(x, x)))  # ty: ignore[invalid-argument-type]

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
      Buffer((3,)),
      Buffer((3,)),
      Buffer((1,)),
      Buffer((2,)),
      ((Buffer((3, 3)), Buffer((3,))), (Buffer((1, 3)), Buffer((1,))), (Buffer((2, 3)), Buffer((2,)), Buffer((2,)))),
    ),
    tuple[Buffer, Buffer, Buffer, Buffer],
  )
  quadratic_ipopt.numerical_call(Buffer((3,)), Buffer((3,)), Buffer((0,)), Buffer((0,)))  # ty: ignore[invalid-argument-type]
  filter_sqp.numerical_call((Buffer((2,)),), (Buffer((2,)), Buffer((1,))), Buffer((1,)), Buffer((3,)), (Buffer((4,)), Buffer((2,))))  # ty: ignore[invalid-argument-type]
  filter_sqp.numerical_call((Expr((2,)), Expr((1,))), (Buffer((2,)), Buffer((1,))), Buffer((1,)), Buffer((3,)), (Buffer((4,)), Buffer((2,))))  # ty: ignore[invalid-argument-type]
  # a solver nests in a larger graph like any Function
  assert_type(
    filter_sqp.symbolic_call((Expr((2,)), Expr((1,))), (Expr((2,)), Expr((1,))), Expr((1,)), Expr((3,)), (Expr((4,)), Expr((2,)))),
    tuple[tuple[Expr, Expr], tuple[Expr, Expr], Expr, Expr],
  )
