"""The ergonomic frontend: the ``@function`` decorator and the named derivative wrappers.

These are the canonical convenience layer over ``Function.factory`` and the typed specs in
``factory.py`` — ``al.jacobian(fn, "x", "y")`` rather than a hand-assembled request.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from ..ir.expr import Expr, ExprOp, as_expr
from .factory import adj, fwd, grad, hess, jac, sphess, spjac
from .model import Function
from ..ir.types import TensorType, as_shape


def function(name: str, inputs: Mapping[str, int | tuple[int, ...] | TensorType]) -> Callable[[Callable[..., Any]], Function]:
  """Build a Python-scoped symbolic function into an Alloy ``Function``.

  It creates fresh input placeholders scoped to the decorated function,
  and returns the same IR-level ``Function`` used by explicit ``Expr.sym``
  construction.
  """

  def decorate(fn: Callable[..., Any]) -> Function:
    input_names = tuple(inputs)
    input_exprs = tuple(_sym_from_spec(n, inputs[n]) for n in input_names)
    outputs, output_names = _normalize_outputs(fn(*input_exprs))
    return Function(name, input_exprs, outputs, input_names, output_names)

  return decorate


def _sym_from_spec(name: str, spec: int | tuple[int, ...] | TensorType) -> Expr:
  if isinstance(spec, TensorType):
    return Expr(ExprOp.INPUT, type=spec, name=name)
  return Expr.sym(name, as_shape(spec))


def _normalize_outputs(ret: Any) -> tuple[tuple[Expr, ...], tuple[str, ...]]:
  if isinstance(ret, Mapping):
    return tuple(as_expr(v) for v in ret.values()), tuple(str(k) for k in ret)
  if isinstance(ret, tuple | list):
    outs = tuple(as_expr(x) for x in ret)
  else:
    outs = (as_expr(ret),)
  return outs, tuple(f"out{i}" for i in range(len(outs)))


def jacobian(fn: Function, input_name: str, output_name: str, *, name: str | None = None, extra_inputs: Sequence[str] = ()) -> Function:
  """Create a one-output function computing ``d output_name / d input_name``.

  ``extra_inputs`` are passed through to the derived function unchanged, after the differentiated
  input — what a caller needs when the source function also takes parameters.
  """

  return fn.factory(name or f"{fn.name}_jac_{output_name}_{input_name}", [input_name, *extra_inputs], [jac(output_name, input_name)])


def gradient(fn: Function, input_name: str, output_name: str, *, name: str | None = None, extra_inputs: Sequence[str] = ()) -> Function:
  """Create a one-output function computing the gradient of a scalar output."""

  return fn.factory(name or f"{fn.name}_grad_{output_name}_{input_name}", [input_name, *extra_inputs], [grad(output_name, input_name)])


def hessian(fn: Function, input_name: str, output_name: str, *, name: str | None = None, extra_inputs: Sequence[str] = ()) -> Function:
  """Create a one-output function computing the Hessian of a scalar output."""

  return fn.factory(name or f"{fn.name}_hess_{output_name}_{input_name}_{input_name}", [input_name, *extra_inputs], [hess(output_name, input_name)])


def forward(fn: Function, input_name: str, output_name: str, *, name: str | None = None, extra_inputs: Sequence[str] = ()) -> Function:
  """Create a seeded forward-mode function computing ``J(output, input) @ fwd``."""

  return fn.factory(
    name or f"{fn.name}_fwd_{output_name}_{input_name}", [input_name, f"fwd:{input_name}", *extra_inputs], [fwd(output_name, input_name)]
  )


def adjoint(fn: Function, input_name: str, output_name: str, *, name: str | None = None, extra_inputs: Sequence[str] = ()) -> Function:
  """Create a seeded reverse-mode function computing ``J(output, input).T @ lam``."""

  return fn.factory(
    name or f"{fn.name}_adj_{output_name}_{input_name}", [input_name, f"lam:{output_name}", *extra_inputs], [adj(output_name, input_name)]
  )


def spjacobian(fn: Function, input_name: str, output_name: str, *, name: str | None = None, extra_inputs: Sequence[str] = ()) -> Function:
  """Create a one-output function computing compact nonzero Jacobian values."""

  return fn.factory(name or f"{fn.name}_spjac_{output_name}_{input_name}", [input_name, *extra_inputs], [spjac(output_name, input_name)])


def sphessian(fn: Function, input_name: str, output_name: str, *, name: str | None = None, extra_inputs: Sequence[str] = ()) -> Function:
  """Create a one-output function computing compact nonzero Hessian values."""

  return fn.factory(
    name or f"{fn.name}_sphess_{output_name}_{input_name}_{input_name}", [input_name, *extra_inputs], [sphess(output_name, input_name)]
  )


def lagrangian_hessian(
  fn: Function, input_name: str, output_names: Sequence[str], *, name: str | None = None, aux_name: str = "gamma", extra_inputs: Sequence[str] = ()
) -> Function:
  """Create a Hessian of ``sum(dot(lam:out, out) for out in output_names)``.

  The generated function inputs are ``input_name``, one ``lam:<output>`` input for every output in
  ``output_names``, then ``extra_inputs``.
  """

  inputs = [input_name, *(f"lam:{out}" for out in output_names), *extra_inputs]
  return fn.factory(
    name or f"{fn.name}_hess_{aux_name}_{input_name}_{input_name}",
    inputs,
    [hess(aux_name, input_name)],
    aux={aux_name: output_names},
  )


def sparse_lagrangian_hessian(
  fn: Function, input_name: str, output_names: Sequence[str], *, name: str | None = None, aux_name: str = "gamma", extra_inputs: Sequence[str] = ()
) -> Function:
  """Create compact nonzero values for a Hessian of a Lagrangian-style auxiliary output."""

  inputs = [input_name, *(f"lam:{out}" for out in output_names), *extra_inputs]
  return fn.factory(
    name or f"{fn.name}_sphess_{aux_name}_{input_name}_{input_name}",
    inputs,
    [sphess(aux_name, input_name)],
    aux={aux_name: output_names},
  )
