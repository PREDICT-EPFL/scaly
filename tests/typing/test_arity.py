"""Static acceptance tests for typed function trees and derivatives."""

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


grad_f_x = al.gradient(cost, "f", "x")
hess_f_x = al.hessian(cost, "f", "x")
jac_square_x = al.jacobian(square, "square", "x")
fwd_f_x = al.forward(cost, "f", "x")
adj_square_x = al.adjoint(square, "square", "x")
hess_l = al.lagrangian_hessian(duplicate, "x")


if TYPE_CHECKING:
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
  grad_f_x.numerical_call(np.zeros(3))  # ty: ignore[invalid-argument-type]
  fwd_f_x.numerical_call((np.zeros(3), np.zeros(()), np.zeros(3)))  # ty: ignore[invalid-argument-type]
  hess_l.numerical_call((np.zeros(3), np.zeros(6)))  # ty: ignore[invalid-argument-type]
