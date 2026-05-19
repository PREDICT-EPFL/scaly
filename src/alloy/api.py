from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .expr import Expr, as_expr
from .function import Function
from .ops import Ops
from .types import TensorType, as_shape


def function(name: str, inputs: Mapping[str, int | tuple[int, ...] | TensorType]) -> Callable[[Callable[..., Any]], Function]:
  """Trace a Python-scoped symbolic function into an Alloy ``Function``.

  This is the anvil-style construction layer: input placeholders are fresh and scoped
  to the decorated function, while the returned object is still the same IR-level
  ``Function`` used by explicit ``Expr.sym`` construction.
  """

  def decorate(fn: Callable[..., Any]) -> Function:
    input_names = tuple(inputs)
    input_exprs = tuple(_sym_from_spec(n, inputs[n]) for n in input_names)
    outputs, output_names = _normalize_outputs(fn(*input_exprs))
    return Function(name, input_exprs, outputs, input_names, output_names)

  return decorate


def _sym_from_spec(name: str, spec: int | tuple[int, ...] | TensorType) -> Expr:
  if isinstance(spec, TensorType):
    return Expr(Ops.INPUT, type=spec, name=name)
  return Expr.sym(name, as_shape(spec))


def _normalize_outputs(ret: Any) -> tuple[tuple[Expr, ...], tuple[str, ...]]:
  if isinstance(ret, Mapping):
    return tuple(as_expr(v) for v in ret.values()), tuple(str(k) for k in ret)
  if isinstance(ret, tuple | list):
    outs = tuple(as_expr(x) for x in ret)
  else:
    outs = (as_expr(ret),)
  return outs, tuple(f"out{i}" for i in range(len(outs)))


def jacobian(fn: Function, input_name: str, output_name: str, *, name: str | None = None) -> Function:
  """Create a one-output function computing ``d output_name / d input_name``.

  This is convenience sugar over ``Function.factory`` for humans; the factory request string
  remains the canonical representation.
  """

  return fn.factory(name or f"{fn.name}_jac_{output_name}_{input_name}", [input_name], [f"jac:{output_name}:{input_name}"])


def gradient(fn: Function, input_name: str, output_name: str, *, name: str | None = None) -> Function:
  """Create a one-output function computing the gradient of a scalar output."""

  return fn.factory(name or f"{fn.name}_grad_{output_name}_{input_name}", [input_name], [f"grad:{output_name}:{input_name}"])


def hessian(fn: Function, input_name: str, output_name: str, *, name: str | None = None) -> Function:
  """Create a one-output function computing the Hessian of a scalar output."""

  return fn.factory(
    name or f"{fn.name}_hess_{output_name}_{input_name}_{input_name}", [input_name], [f"hess:{output_name}:{input_name}:{input_name}"]
  )


def forward(fn: Function, input_name: str, output_name: str, *, name: str | None = None) -> Function:
  """Create a seeded forward-mode function computing ``J(output, input) @ fwd``."""

  return fn.factory(name or f"{fn.name}_fwd_{output_name}_{input_name}", [input_name, f"fwd:{input_name}"], [f"fwd:{output_name}:{input_name}"])


def adjoint(fn: Function, input_name: str, output_name: str, *, name: str | None = None) -> Function:
  """Create a seeded reverse-mode function computing ``J(output, input).T @ lam``."""

  return fn.factory(name or f"{fn.name}_adj_{output_name}_{input_name}", [input_name, f"lam:{output_name}"], [f"adj:{output_name}:{input_name}"])


def spjacobian(fn: Function, input_name: str, output_name: str, *, name: str | None = None) -> Function:
  """Create a one-output function computing compact nonzero Jacobian values."""

  return fn.factory(name or f"{fn.name}_spjac_{output_name}_{input_name}", [input_name], [f"spjac:{output_name}:{input_name}"])


def sphessian(fn: Function, input_name: str, output_name: str, *, name: str | None = None) -> Function:
  """Create a one-output function computing compact nonzero Hessian values."""

  return fn.factory(
    name or f"{fn.name}_sphess_{output_name}_{input_name}_{input_name}", [input_name], [f"sphess:{output_name}:{input_name}:{input_name}"]
  )


def lagrangian_hessian(fn: Function, input_name: str, output_names: Sequence[str], *, name: str | None = None, aux_name: str = "gamma") -> Function:
  """Create a Hessian of ``sum(dot(lam:out, out) for out in output_names)``.

  The generated function inputs are ``input_name`` followed by one ``lam:<output>`` input for
  every output in ``output_names``.
  """

  inputs = [input_name, *(f"lam:{out}" for out in output_names)]
  return fn.factory(
    name or f"{fn.name}_hess_{aux_name}_{input_name}_{input_name}",
    inputs,
    [f"hess:{aux_name}:{input_name}:{input_name}"],
    aux={aux_name: output_names},
  )


def sparse_lagrangian_hessian(
  fn: Function, input_name: str, output_names: Sequence[str], *, name: str | None = None, aux_name: str = "gamma"
) -> Function:
  """Create compact nonzero values for a Hessian of a Lagrangian-style auxiliary output."""

  inputs = [input_name, *(f"lam:{out}" for out in output_names)]
  return fn.factory(
    name or f"{fn.name}_sphess_{aux_name}_{input_name}_{input_name}",
    inputs,
    [f"sphess:{aux_name}:{input_name}:{input_name}"],
    aux={aux_name: output_names},
  )
