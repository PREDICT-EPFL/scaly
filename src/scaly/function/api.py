"""The ergonomic frontend: function construction and named derivative wrappers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast, overload

import numpy as np

from ..ad.derivatives import gradient as _gradient_expr
from ..ad.derivatives import hessian as _hessian_expr
from ..ad.derivatives import jacobian as _jacobian_expr
from ..ad.sparse import SparseJacobian, Triangle, sparse_hessian as _expr_sparse_hessian
from ..ad.sparse import sparse_jacobian as _expr_sparse_jacobian
from ..ir.expr import Expr
from .tree import G, L, Tree
from .factory import Adj, Fwd, Grad, Hess, Jac, SpHess, SpJac
from .model import Function


def function[SI, NI, SO, NO](
  inputs: Tree[SI, NI], outputs: Tree[SO, NO], /, *, name: str | None = None
) -> Callable[[Callable[[SI], SO]], Function[SI, NI, SO, NO]]:
  """Trace a callable over declared input and output pytrees."""

  def decorate(fn: Callable[[SI], SO]) -> Function[SI, NI, SO, NO]:
    return Function(name or getattr(fn, "__name__", "fn"), fn, inputs, outputs)

  return decorate


def _expr_wrt(
  operation: str,
  args: tuple[Expr | str, ...],
  wrt: Expr | str | None,
  of: str | None,
  name: str | None,
) -> Expr:
  if name is not None:
    raise TypeError(f"{operation} Expr form does not accept name")
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


def _checked_names(source: Function[Any, Any, Any, Any], of: str, wrt: str) -> None:
  source.output_tree.index(of)
  source.input_tree.index(wrt)


def _typed_result[SI, NI](result: Function[Any, Any, Any, Any], input_tree: Tree[SI, NI]) -> Function[SI, NI, Expr, np.ndarray]:
  output_tree = L(result.output_names[0], result.outputs[0].type)
  return cast(Function[SI, NI, Expr, np.ndarray], result._with_trees(input_tree, output_tree))


def _factory_input_tree[SI, NI](result: Function[Any, Any, Any, Any], tree: Tree[SI, NI]) -> Tree[SI, NI]:
  input_map = result.input_map()
  return tree.with_types(tuple(input_map[name].type for name in tree.names))


def _unseeded[SI, NI](
  source: Function[SI, NI, Any, Any], name: str, of: str, wrt: str, spec: Jac | Grad | Hess | SpJac | SpHess
) -> Function[SI, NI, Expr, np.ndarray]:
  _checked_names(source, of, wrt)
  return _typed_result(source.factory(name, list(source.input_names), [spec]), source.input_tree)


@overload
def jacobian(source: Expr, wrt: Expr) -> Expr: ...


@overload
def jacobian[SI, NI, SO, NO](
  source: Function[SI, NI, SO, NO], of: str, wrt: str, *, name: str | None = None
) -> Function[SI, NI, Expr, np.ndarray]: ...


def jacobian(
  source: Expr | Function,
  *args: Expr | str,
  wrt: Expr | str | None = None,
  of: str | None = None,
  name: str | None = None,
) -> Expr | Function:
  """Build a dense Jacobian for an Expr or a named Function output."""
  if isinstance(source, Expr):
    return _jacobian_expr(source, _expr_wrt("jacobian", args, wrt, of, name))
  if not isinstance(source, Function):
    raise TypeError("jacobian source must be an Expr or Function")
  function_of, function_wrt = _function_names("jacobian", args, wrt, of)
  return _unseeded(source, name or f"{source.name}_jac_{function_of}_{function_wrt}", function_of, function_wrt, Jac(function_of, function_wrt))


@overload
def gradient(source: Expr, wrt: Expr) -> Expr: ...


@overload
def gradient[SI, NI, SO, NO](
  source: Function[SI, NI, SO, NO], of: str, wrt: str, *, name: str | None = None
) -> Function[SI, NI, Expr, np.ndarray]: ...


def gradient(
  source: Expr | Function,
  *args: Expr | str,
  wrt: Expr | str | None = None,
  of: str | None = None,
  name: str | None = None,
) -> Expr | Function:
  """Build a gradient for an Expr or a named Function output."""
  if isinstance(source, Expr):
    return _gradient_expr(source, _expr_wrt("gradient", args, wrt, of, name))
  if not isinstance(source, Function):
    raise TypeError("gradient source must be an Expr or Function")
  function_of, function_wrt = _function_names("gradient", args, wrt, of)
  return _unseeded(source, name or f"{source.name}_grad_{function_of}_{function_wrt}", function_of, function_wrt, Grad(function_of, function_wrt))


@overload
def hessian(source: Expr, wrt: Expr) -> Expr: ...


@overload
def hessian[SI, NI, SO, NO](
  source: Function[SI, NI, SO, NO], of: str, wrt: str, *, name: str | None = None
) -> Function[SI, NI, Expr, np.ndarray]: ...


def hessian(
  source: Expr | Function,
  *args: Expr | str,
  wrt: Expr | str | None = None,
  of: str | None = None,
  name: str | None = None,
) -> Expr | Function:
  """Build a Hessian for an Expr or a named Function output."""
  if isinstance(source, Expr):
    return _hessian_expr(source, _expr_wrt("hessian", args, wrt, of, name))
  if not isinstance(source, Function):
    raise TypeError("hessian source must be an Expr or Function")
  function_of, function_wrt = _function_names("hessian", args, wrt, of)
  return _unseeded(
    source, name or f"{source.name}_hess_{function_of}_{function_wrt}_{function_wrt}", function_of, function_wrt, Hess(function_of, function_wrt)
  )


@overload
def sparse_jacobian(source: Expr, wrt: Expr) -> SparseJacobian: ...


@overload
def sparse_jacobian[SI, NI, SO, NO](
  source: Function[SI, NI, SO, NO], of: str, wrt: str, *, name: str | None = None
) -> Function[SI, NI, Expr, np.ndarray]: ...


def sparse_jacobian(
  source: Expr | Function,
  *args: Expr | str,
  wrt: Expr | str | None = None,
  of: str | None = None,
  name: str | None = None,
) -> SparseJacobian | Function:
  """Build compact nonzero Jacobian values for an Expr or a named Function output."""
  if isinstance(source, Expr):
    return _expr_sparse_jacobian(source, _expr_wrt("sparse_jacobian", args, wrt, of, name))
  if not isinstance(source, Function):
    raise TypeError("sparse_jacobian source must be an Expr or Function")
  function_of, function_wrt = _function_names("sparse_jacobian", args, wrt, of)
  return _unseeded(source, name or f"{source.name}_spjac_{function_of}_{function_wrt}", function_of, function_wrt, SpJac(function_of, function_wrt))


@overload
def sparse_hessian(source: Expr, wrt: Expr, *, triangle: Triangle = "full") -> SparseJacobian: ...


@overload
def sparse_hessian[SI, NI, SO, NO](
  source: Function[SI, NI, SO, NO],
  of: str,
  wrt: str,
  *,
  name: str | None = None,
  triangle: Triangle = "full",
) -> Function[SI, NI, Expr, np.ndarray]: ...


def sparse_hessian(
  source: Expr | Function,
  *args: Expr | str,
  wrt: Expr | str | None = None,
  of: str | None = None,
  name: str | None = None,
  triangle: Triangle = "full",
) -> SparseJacobian | Function:
  """Build compact nonzero Hessian values for an Expr or a named Function output.

  ``triangle`` selects the symmetric pattern returned by the sparse Hessian: ``"full"`` keeps
  every entry, while ``"lower"`` and ``"upper"`` keep one triangle in the full pattern's order.
  """
  if isinstance(source, Expr):
    return _expr_sparse_hessian(source, _expr_wrt("sparse_hessian", args, wrt, of, name), triangle=triangle)
  if not isinstance(source, Function):
    raise TypeError("sparse_hessian source must be an Expr or Function")
  function_of, function_wrt = _function_names("sparse_hessian", args, wrt, of)
  return _unseeded(
    source,
    name or f"{source.name}_sphess_{function_of}_{function_wrt}_{function_wrt}",
    function_of,
    function_wrt,
    SpHess(function_of, function_wrt, triangle=triangle),
  )


def forward[SI, NI, SO, NO](
  fn: Function[SI, NI, SO, NO],
  of: str,
  wrt: str,
  *,
  name: str | None = None,
) -> Function[tuple[SI, Expr], tuple[NI, np.ndarray], Expr, np.ndarray]:
  """Create a seeded forward-mode Function computing J(of, wrt) @ fwd:wrt."""
  _checked_names(fn, of, wrt)
  seed = L(f"fwd:{wrt}", fn.inputs[fn.input_tree.index(wrt)].type)
  result = fn.factory(name or f"{fn.name}_fwd_{of}_{wrt}", [*fn.input_names, f"fwd:{wrt}"], [Fwd(of, wrt)])
  return _typed_result(result, G(fn.input_tree, _factory_input_tree(result, seed)))


def adjoint[SI, NI, SO, NO](
  fn: Function[SI, NI, SO, NO],
  of: str,
  wrt: str,
  *,
  name: str | None = None,
) -> Function[tuple[SI, Expr], tuple[NI, np.ndarray], Expr, np.ndarray]:
  """Create a seeded reverse-mode Function computing J(of, wrt).T @ lam:of."""
  _checked_names(fn, of, wrt)
  seed = L(f"lam:{of}", fn.outputs[fn.output_tree.index(of)].type)
  result = fn.factory(name or f"{fn.name}_adj_{of}_{wrt}", [*fn.input_names, f"lam:{of}"], [Adj(of, wrt)])
  return _typed_result(result, G(fn.input_tree, _factory_input_tree(result, seed)))


def lagrangian_hessian[SI, NI, SO, NO](
  fn: Function[SI, NI, SO, NO],
  wrt: str,
  *,
  name: str | None = None,
  aux_name: str = "gamma",
) -> Function[tuple[SI, SO], tuple[NI, NO], Expr, np.ndarray]:
  """Create the dense Hessian of all outputs weighted by the declared output tree."""
  fn.input_tree.index(wrt)
  output_names = list(fn.output_names)
  inputs = [*fn.input_names, *(f"lam:{out}" for out in output_names)]
  result = fn.factory(
    name or f"{fn.name}_hess_{aux_name}_{wrt}_{wrt}",
    inputs,
    [Hess(aux_name, wrt)],
    aux={aux_name: output_names},
  )
  seed_tree = _factory_input_tree(result, fn.output_tree.relabel("lam:"))
  return _typed_result(result, G(fn.input_tree, seed_tree))


def sparse_lagrangian_hessian[SI, NI, SO, NO](
  fn: Function[SI, NI, SO, NO],
  wrt: str,
  *,
  name: str | None = None,
  aux_name: str = "gamma",
  triangle: Triangle = "full",
) -> Function[tuple[SI, SO], tuple[NI, NO], Expr, np.ndarray]:
  """Create compact values for the weighted Hessian of all declared outputs."""
  fn.input_tree.index(wrt)
  output_names = list(fn.output_names)
  inputs = [*fn.input_names, *(f"lam:{out}" for out in output_names)]
  result = fn.factory(
    name or f"{fn.name}_sphess_{aux_name}_{wrt}_{wrt}",
    inputs,
    [SpHess(aux_name, wrt, triangle=triangle)],
    aux={aux_name: output_names},
  )
  seed_tree = _factory_input_tree(result, fn.output_tree.relabel("lam:"))
  return _typed_result(result, G(fn.input_tree, seed_tree))
