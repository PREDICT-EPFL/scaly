"""The ergonomic frontend: function construction and named derivative wrappers."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, overload

from ..ad.derivatives import gradient as _gradient_expr
from ..ad.derivatives import hessian as _hessian_expr
from ..ad.derivatives import jacobian as _jacobian_expr
from ..ad.sparse import SparseJacobian, Triangle, sparse_hessian as _expr_sparse_hessian
from ..ad.sparse import sparse_jacobian as _expr_sparse_jacobian
from ..ir.expr import Expr, ExprOp, as_expr
from ..ir.types import TensorType, as_shape
from .factory import Adj, Fwd, Grad, Hess, Jac, SpHess, SpJac
from .model import Function


def function(name: str, inputs: Mapping[str, int | tuple[int, ...] | TensorType]) -> Callable[[Callable[..., Any]], Function]:
  """Build a Python-scoped symbolic function into an Alloy Function.

  It creates input placeholders scoped to the decorated function and returns the same
  IR-level Function used by explicit Expr.sym construction.
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


def _expr_wrt(
  operation: str,
  args: tuple[Expr | str, ...],
  wrt: Expr | str | None,
  of: str | None,
  name: str | None,
  extra_inputs: Sequence[str],
) -> Expr:
  if name is not None or extra_inputs:
    raise TypeError(f"{operation} Expr form does not accept name or extra_inputs")
  if of is not None or len(args) > 1 or (args and wrt is not None):
    raise TypeError(f"the Expr form is {operation}(expr, wrt)")
  candidate = wrt if wrt is not None else (args[0] if args else None)
  if not isinstance(candidate, Expr):
    raise TypeError(f"the Expr form is {operation}(expr, wrt)")
  return candidate


def _function_names(operation: str, args: tuple[Expr | str, ...], wrt: Expr | str | None, of: str | None) -> tuple[str, str]:
  if len(args) == 2 and wrt is None and of is None:
    function_of, function_wrt = args
  elif len(args) == 1 and wrt is not None and of is None:
    function_of, function_wrt = args[0], wrt
  elif not args:
    function_of, function_wrt = of, wrt
  else:
    raise TypeError(f"the Function form is {operation}(fn, of, wrt)")
  if not isinstance(function_of, str) or not isinstance(function_wrt, str):
    raise TypeError(f"the Function form is {operation}(fn, of, wrt)")
  return function_of, function_wrt


@overload
def jacobian(source: Expr, wrt: Expr) -> Expr: ...


@overload
def jacobian(source: Function, of: str, wrt: str, *, name: str | None = None, extra_inputs: Sequence[str] = ()) -> Function: ...


def jacobian(
  source: Expr | Function,
  *args: Expr | str,
  wrt: Expr | str | None = None,
  of: str | None = None,
  name: str | None = None,
  extra_inputs: Sequence[str] = (),
) -> Expr | Function:
  """Build a dense Jacobian for an Expr or a named Function output."""
  if isinstance(source, Expr):
    return _jacobian_expr(source, _expr_wrt("jacobian", args, wrt, of, name, extra_inputs))
  if not isinstance(source, Function):
    raise TypeError("jacobian source must be an Expr or Function")
  function_of, function_wrt = _function_names("jacobian", args, wrt, of)
  return source.factory(name or f"{source.name}_jac_{function_of}_{function_wrt}", [function_wrt, *extra_inputs], [Jac(function_of, function_wrt)])


@overload
def gradient(source: Expr, wrt: Expr) -> Expr: ...


@overload
def gradient(source: Function, of: str, wrt: str, *, name: str | None = None, extra_inputs: Sequence[str] = ()) -> Function: ...


def gradient(
  source: Expr | Function,
  *args: Expr | str,
  wrt: Expr | str | None = None,
  of: str | None = None,
  name: str | None = None,
  extra_inputs: Sequence[str] = (),
) -> Expr | Function:
  """Build a gradient for an Expr or a named Function output."""
  if isinstance(source, Expr):
    return _gradient_expr(source, _expr_wrt("gradient", args, wrt, of, name, extra_inputs))
  if not isinstance(source, Function):
    raise TypeError("gradient source must be an Expr or Function")
  function_of, function_wrt = _function_names("gradient", args, wrt, of)
  return source.factory(name or f"{source.name}_grad_{function_of}_{function_wrt}", [function_wrt, *extra_inputs], [Grad(function_of, function_wrt)])


@overload
def hessian(source: Expr, wrt: Expr) -> Expr: ...


@overload
def hessian(source: Function, of: str, wrt: str, *, name: str | None = None, extra_inputs: Sequence[str] = ()) -> Function: ...


def hessian(
  source: Expr | Function,
  *args: Expr | str,
  wrt: Expr | str | None = None,
  of: str | None = None,
  name: str | None = None,
  extra_inputs: Sequence[str] = (),
) -> Expr | Function:
  """Build a Hessian for an Expr or a named Function output."""
  if isinstance(source, Expr):
    return _hessian_expr(source, _expr_wrt("hessian", args, wrt, of, name, extra_inputs))
  if not isinstance(source, Function):
    raise TypeError("hessian source must be an Expr or Function")
  function_of, function_wrt = _function_names("hessian", args, wrt, of)
  return source.factory(
    name or f"{source.name}_hess_{function_of}_{function_wrt}_{function_wrt}", [function_wrt, *extra_inputs], [Hess(function_of, function_wrt)]
  )


@overload
def sparse_jacobian(source: Expr, wrt: Expr) -> SparseJacobian: ...


@overload
def sparse_jacobian(source: Function, of: str, wrt: str, *, name: str | None = None, extra_inputs: Sequence[str] = ()) -> Function: ...


def sparse_jacobian(
  source: Expr | Function,
  *args: Expr | str,
  wrt: Expr | str | None = None,
  of: str | None = None,
  name: str | None = None,
  extra_inputs: Sequence[str] = (),
) -> SparseJacobian | Function:
  """Build compact nonzero Jacobian values for an Expr or a named Function output."""
  if isinstance(source, Expr):
    return _expr_sparse_jacobian(source, _expr_wrt("sparse_jacobian", args, wrt, of, name, extra_inputs))
  if not isinstance(source, Function):
    raise TypeError("sparse_jacobian source must be an Expr or Function")
  function_of, function_wrt = _function_names("sparse_jacobian", args, wrt, of)
  return source.factory(
    name or f"{source.name}_spjac_{function_of}_{function_wrt}", [function_wrt, *extra_inputs], [SpJac(function_of, function_wrt)]
  )


@overload
def sparse_hessian(source: Expr, wrt: Expr, *, triangle: Triangle = "full") -> SparseJacobian: ...


@overload
def sparse_hessian(
  source: Function,
  of: str,
  wrt: str,
  *,
  name: str | None = None,
  extra_inputs: Sequence[str] = (),
  triangle: Triangle = "full",
) -> Function: ...


def sparse_hessian(
  source: Expr | Function,
  *args: Expr | str,
  wrt: Expr | str | None = None,
  of: str | None = None,
  name: str | None = None,
  extra_inputs: Sequence[str] = (),
  triangle: Triangle = "full",
) -> SparseJacobian | Function:
  """Build compact nonzero Hessian values for an Expr or a named Function output.

  ``triangle`` selects the symmetric pattern returned by the sparse Hessian: ``"full"`` keeps
  every entry, while ``"lower"`` and ``"upper"`` keep one triangle in the full pattern's order.
  """
  if isinstance(source, Expr):
    return _expr_sparse_hessian(source, _expr_wrt("sparse_hessian", args, wrt, of, name, extra_inputs), triangle=triangle)
  if not isinstance(source, Function):
    raise TypeError("sparse_hessian source must be an Expr or Function")
  function_of, function_wrt = _function_names("sparse_hessian", args, wrt, of)
  return source.factory(
    name or f"{source.name}_sphess_{function_of}_{function_wrt}_{function_wrt}",
    [function_wrt, *extra_inputs],
    [SpHess(function_of, function_wrt, triangle=triangle)],
  )


def forward(
  fn: Function,
  of: str,
  wrt: str,
  *,
  name: str | None = None,
  extra_inputs: Sequence[str] = (),
) -> Function:
  """Create a seeded forward-mode Function computing J(of, wrt) @ fwd:wrt."""
  return fn.factory(name or f"{fn.name}_fwd_{of}_{wrt}", [wrt, f"fwd:{wrt}", *extra_inputs], [Fwd(of, wrt)])


def adjoint(
  fn: Function,
  of: str,
  wrt: str,
  *,
  name: str | None = None,
  extra_inputs: Sequence[str] = (),
) -> Function:
  """Create a seeded reverse-mode Function computing J(of, wrt).T @ lam:of."""
  return fn.factory(name or f"{fn.name}_adj_{of}_{wrt}", [wrt, f"lam:{of}", *extra_inputs], [Adj(of, wrt)])


def lagrangian_hessian(
  fn: Function,
  of: Sequence[str],
  wrt: str,
  *,
  name: str | None = None,
  aux_name: str = "gamma",
  extra_inputs: Sequence[str] = (),
) -> Function:
  """Create a dense Hessian of a weighted combination of named outputs."""
  inputs = [wrt, *(f"lam:{out}" for out in of), *extra_inputs]
  return fn.factory(
    name or f"{fn.name}_hess_{aux_name}_{wrt}_{wrt}",
    inputs,
    [Hess(aux_name, wrt)],
    aux={aux_name: of},
  )


def sparse_lagrangian_hessian(
  fn: Function,
  of: Sequence[str],
  wrt: str,
  *,
  name: str | None = None,
  aux_name: str = "gamma",
  extra_inputs: Sequence[str] = (),
  triangle: Triangle = "full",
) -> Function:
  """Create compact nonzero values for a Hessian of a weighted combination of outputs.

  ``triangle`` selects the symmetric pattern returned by the sparse Hessian: ``"full"`` keeps
  every entry, while ``"lower"`` and ``"upper"`` keep one triangle in the full pattern's order.
  """
  inputs = [wrt, *(f"lam:{out}" for out in of), *extra_inputs]
  return fn.factory(
    name or f"{fn.name}_sphess_{aux_name}_{wrt}_{wrt}",
    inputs,
    [SpHess(aux_name, wrt, triangle=triangle)],
    aux={aux_name: of},
  )
