"""The ergonomic frontend: function construction and named derivative wrappers."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, overload

import numpy as np

from ..ad.derivatives import gradient as _gradient_expr
from ..ad.derivatives import hessian as _hessian_expr
from ..ad.derivatives import jacobian as _jacobian_expr
from ..ad.sparse import SparseJacobian, Triangle, sparse_hessian as _expr_sparse_hessian
from ..ad.sparse import sparse_jacobian as _expr_sparse_jacobian
from ..ir.expr import Expr
from .tree import L, Spec, Tree, _G, as_tree, param_list

if TYPE_CHECKING:
  from .tree import NA, NB, NC, ND, NE, NF, NG, NH, NO, SA, SB, SC, SD, SE, SF, SG, SH, SO
from .factory import Adj, Fwd, Grad, Hess, Jac, SpHess, SpJac
from .model import ConcreteFunction, Function


# fmt: off
@overload
def function(*, output: Tree[SO, NO] | Spec, name: str | None = None) -> Callable[[Callable[[], SO]], Function[[], [], SO, NO]]: ...
@overload
def function(a: Tree[SA, NA] | Spec, /, *, output: Tree[SO, NO] | Spec, name: str | None = None) -> Callable[[Callable[[SA], SO]], Function[[SA], [NA], SO, NO]]: ...
@overload
def function(a: Tree[SA, NA] | Spec, b: Tree[SB, NB] | Spec, /, *, output: Tree[SO, NO] | Spec, name: str | None = None) -> Callable[[Callable[[SA, SB], SO]], Function[[SA, SB], [NA, NB], SO, NO]]: ...
@overload
def function(a: Tree[SA, NA] | Spec, b: Tree[SB, NB] | Spec, c: Tree[SC, NC] | Spec, /, *, output: Tree[SO, NO] | Spec, name: str | None = None) -> Callable[[Callable[[SA, SB, SC], SO]], Function[[SA, SB, SC], [NA, NB, NC], SO, NO]]: ...
@overload
def function(a: Tree[SA, NA] | Spec, b: Tree[SB, NB] | Spec, c: Tree[SC, NC] | Spec, d: Tree[SD, ND] | Spec, /, *, output: Tree[SO, NO] | Spec, name: str | None = None) -> Callable[[Callable[[SA, SB, SC, SD], SO]], Function[[SA, SB, SC, SD], [NA, NB, NC, ND], SO, NO]]: ...
@overload
def function(a: Tree[SA, NA] | Spec, b: Tree[SB, NB] | Spec, c: Tree[SC, NC] | Spec, d: Tree[SD, ND] | Spec, e: Tree[SE, NE] | Spec, /, *, output: Tree[SO, NO] | Spec, name: str | None = None) -> Callable[[Callable[[SA, SB, SC, SD, SE], SO]], Function[[SA, SB, SC, SD, SE], [NA, NB, NC, ND, NE], SO, NO]]: ...
@overload
def function(a: Tree[SA, NA] | Spec, b: Tree[SB, NB] | Spec, c: Tree[SC, NC] | Spec, d: Tree[SD, ND] | Spec, e: Tree[SE, NE] | Spec, f: Tree[SF, NF] | Spec, /, *, output: Tree[SO, NO] | Spec, name: str | None = None) -> Callable[[Callable[[SA, SB, SC, SD, SE, SF], SO]], Function[[SA, SB, SC, SD, SE, SF], [NA, NB, NC, ND, NE, NF], SO, NO]]: ...
@overload
def function(a: Tree[SA, NA] | Spec, b: Tree[SB, NB] | Spec, c: Tree[SC, NC] | Spec, d: Tree[SD, ND] | Spec, e: Tree[SE, NE] | Spec, f: Tree[SF, NF] | Spec, g: Tree[SG, NG] | Spec, /, *, output: Tree[SO, NO] | Spec, name: str | None = None) -> Callable[[Callable[[SA, SB, SC, SD, SE, SF, SG], SO]], Function[[SA, SB, SC, SD, SE, SF, SG], [NA, NB, NC, ND, NE, NF, NG], SO, NO]]: ...
@overload
def function(a: Tree[SA, NA] | Spec, b: Tree[SB, NB] | Spec, c: Tree[SC, NC] | Spec, d: Tree[SD, ND] | Spec, e: Tree[SE, NE] | Spec, f: Tree[SF, NF] | Spec, g: Tree[SG, NG] | Spec, h: Tree[SH, NH] | Spec, /, *, output: Tree[SO, NO] | Spec, name: str | None = None) -> Callable[[Callable[[SA, SB, SC, SD, SE, SF, SG, SH], SO]], Function[[SA, SB, SC, SD, SE, SF, SG, SH], [NA, NB, NC, ND, NE, NF, NG, NH], SO, NO]]: ...
@overload
def function(*slots: Tree[Any, Any] | Spec, output: Tree[Any, Any] | Spec, name: str | None = None) -> Callable[[Callable[..., Any]], Function[..., ..., Any, Any]]: ...
# fmt: on
def function(*slots: Tree[Any, Any] | Spec, output: Tree[Any, Any] | Spec | None = None, name: str | None = None) -> Any:
  """Trace a Python body into a named ``Function``: one declaration per parameter, and the ``output``.

  Each slot is a tree spec: ``sc.L``, ``sc.G`` or ``sc.S``, or shorthand for one leaf, a shape
  (``3``, ``(n, m)``, a ``TensorType``) or a name (a leaf whose shape the trace decides). Unnamed
  leaves take the parameter's name (``p``, or ``p_0``, ``p_1`` inside a group); unnamed outputs take
  the function's. The names become the generated C signature and the ``of``/``wrt`` of derivatives.

  With every input shape declared, the body is traced now and the result is a ``ConcreteFunction``.
  An input shape left open (``sc.L()``, ``(n, None)``) makes a template: each call binds the holes
  and traces one instance per distinct binding; see ``Function``.
  """

  def decorate(fn: Callable[..., Any]) -> Function[..., ..., Any, Any]:
    fn_name = name or getattr(fn, "__name__", "fn")
    names = _parameters(fn, fn_name)
    if output is None:
      if len(slots) == len(names) + 1:
        raise TypeError(
          f"{fn_name}: sc.function takes one declaration per parameter and the output as output=; "
          "did you mean sc.function(<inputs>, output=<outputs>)?"
        )
      raise TypeError(f"{fn_name}: declare the output tree with output=")
    if len(slots) != len(names):
      raise TypeError(f"{fn_name}: declared {len(slots)} parameters, the body takes {len(names)} ({', '.join(names)})")
    inputs = param_list(*(as_tree(slot).named(param) for slot, param in zip(slots, names, strict=True)))
    outputs = as_tree(output).named(fn_name)
    if inputs.has_holes:
      return Function(fn_name, fn, inputs, outputs)
    return ConcreteFunction(fn_name, fn, inputs, outputs)

  return decorate


def _parameters(fn: Callable[..., Any], name: str) -> tuple[str, ...]:
  """The body's parameter names; each is one declaration slot, so only plain positional parameters are allowed."""
  try:
    signature = inspect.signature(fn)
  except (TypeError, ValueError):
    raise TypeError(f"{name}: sc.function needs a Python callable with a signature, got {type(fn).__name__}") from None
  plain = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
  refused = [p.name for p in signature.parameters.values() if p.kind not in plain or p.default is not p.empty]
  if refused:
    raise TypeError(
      f"{name}: parameters {refused} have defaults or are variadic or keyword-only; a body takes plain positional parameters, one per declaration"
    )
  return tuple(signature.parameters)


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


def _checked_names(source: ConcreteFunction[Any, Any, Any, Any], of: str, wrt: str) -> None:
  source.output_tree.index(of)
  source.input_tree.index(wrt)


def _typed_result(result: ConcreteFunction[Any, Any, Any, Any], input_tree: _G) -> Any:
  output_tree = L(result.output_names[0], result.outputs[0].type)
  return result._with_trees(input_tree, output_tree)


def _factory_input_tree(result: ConcreteFunction[Any, Any, Any, Any], tree: Tree[Any, Any]) -> Tree[Any, Any]:
  input_map = result.input_map()
  return tree.with_types(tuple(input_map[name].type for name in tree.names))


def _unseeded(source: ConcreteFunction[Any, Any, Any, Any], name: str, of: str, wrt: str, spec: Jac | Grad | Hess | SpJac | SpHess) -> Any:
  _checked_names(source, of, wrt)
  return _typed_result(source.factory(name, list(source.input_names), [spec]), source.input_tree)


@overload
def jacobian(source: Expr, wrt: Expr) -> Expr: ...


@overload
def jacobian[**PS, **PN](source: Function[PS, PN, Any, Any], of: str, wrt: str, *, name: str | None = None) -> Function[PS, PN, Expr, np.ndarray]: ...


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
  return _unseeded(
    source.concrete, name or f"{source.name}_jac_{function_of}_{function_wrt}", function_of, function_wrt, Jac(function_of, function_wrt)
  )


@overload
def gradient(source: Expr, wrt: Expr) -> Expr: ...


@overload
def gradient[**PS, **PN](source: Function[PS, PN, Any, Any], of: str, wrt: str, *, name: str | None = None) -> Function[PS, PN, Expr, np.ndarray]: ...


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
  return _unseeded(
    source.concrete, name or f"{source.name}_grad_{function_of}_{function_wrt}", function_of, function_wrt, Grad(function_of, function_wrt)
  )


@overload
def hessian(source: Expr, wrt: Expr) -> Expr: ...


@overload
def hessian[**PS, **PN](source: Function[PS, PN, Any, Any], of: str, wrt: str, *, name: str | None = None) -> Function[PS, PN, Expr, np.ndarray]: ...


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
    source.concrete,
    name or f"{source.name}_hess_{function_of}_{function_wrt}_{function_wrt}",
    function_of,
    function_wrt,
    Hess(function_of, function_wrt),
  )


@overload
def sparse_jacobian(source: Expr, wrt: Expr) -> SparseJacobian: ...


@overload
def sparse_jacobian[**PS, **PN](
  source: Function[PS, PN, Any, Any], of: str, wrt: str, *, name: str | None = None
) -> Function[PS, PN, Expr, np.ndarray]: ...


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
  return _unseeded(
    source.concrete, name or f"{source.name}_spjac_{function_of}_{function_wrt}", function_of, function_wrt, SpJac(function_of, function_wrt)
  )


@overload
def sparse_hessian(source: Expr, wrt: Expr, *, triangle: Triangle = "full") -> SparseJacobian: ...


@overload
def sparse_hessian[**PS, **PN](
  source: Function[PS, PN, Any, Any],
  of: str,
  wrt: str,
  *,
  name: str | None = None,
  triangle: Triangle = "full",
) -> Function[PS, PN, Expr, np.ndarray]: ...


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
    source.concrete,
    name or f"{source.name}_sphess_{function_of}_{function_wrt}_{function_wrt}",
    function_of,
    function_wrt,
    SpHess(function_of, function_wrt, triangle=triangle),
  )


# fmt: off
@overload
def forward(fn: Function[[], [], SO, NO], of: str, wrt: str, *, name: str | None = None) -> Function[[Expr], [np.ndarray], Expr, np.ndarray]: ...
@overload
def forward(fn: Function[[SA], [NA], SO, NO], of: str, wrt: str, *, name: str | None = None) -> Function[[SA, Expr], [NA, np.ndarray], Expr, np.ndarray]: ...
@overload
def forward(fn: Function[[SA, SB], [NA, NB], SO, NO], of: str, wrt: str, *, name: str | None = None) -> Function[[SA, SB, Expr], [NA, NB, np.ndarray], Expr, np.ndarray]: ...
@overload
def forward(fn: Function[[SA, SB, SC], [NA, NB, NC], SO, NO], of: str, wrt: str, *, name: str | None = None) -> Function[[SA, SB, SC, Expr], [NA, NB, NC, np.ndarray], Expr, np.ndarray]: ...
@overload
def forward(fn: Function[[SA, SB, SC, SD], [NA, NB, NC, ND], SO, NO], of: str, wrt: str, *, name: str | None = None) -> Function[[SA, SB, SC, SD, Expr], [NA, NB, NC, ND, np.ndarray], Expr, np.ndarray]: ...
@overload
def forward(fn: Function[[SA, SB, SC, SD, SE], [NA, NB, NC, ND, NE], SO, NO], of: str, wrt: str, *, name: str | None = None) -> Function[[SA, SB, SC, SD, SE, Expr], [NA, NB, NC, ND, NE, np.ndarray], Expr, np.ndarray]: ...
@overload
def forward(fn: Function[[SA, SB, SC, SD, SE, SF], [NA, NB, NC, ND, NE, NF], SO, NO], of: str, wrt: str, *, name: str | None = None) -> Function[[SA, SB, SC, SD, SE, SF, Expr], [NA, NB, NC, ND, NE, NF, np.ndarray], Expr, np.ndarray]: ...
@overload
def forward(fn: Function[[SA, SB, SC, SD, SE, SF, SG], [NA, NB, NC, ND, NE, NF, NG], SO, NO], of: str, wrt: str, *, name: str | None = None) -> Function[[SA, SB, SC, SD, SE, SF, SG, Expr], [NA, NB, NC, ND, NE, NF, NG, np.ndarray], Expr, np.ndarray]: ...
@overload
def forward(fn: Function[[SA, SB, SC, SD, SE, SF, SG, SH], [NA, NB, NC, ND, NE, NF, NG, NH], SO, NO], of: str, wrt: str, *, name: str | None = None) -> Function[[SA, SB, SC, SD, SE, SF, SG, SH, Expr], [NA, NB, NC, ND, NE, NF, NG, NH, np.ndarray], Expr, np.ndarray]: ...
@overload
def forward(fn: Function[..., ..., Any, Any], of: str, wrt: str, *, name: str | None = None) -> Function[..., ..., Expr, np.ndarray]: ...
# fmt: on
def forward(fn: Function[Any, Any, Any, Any], of: str, wrt: str, *, name: str | None = None) -> Any:
  """Create a seeded forward-mode Function computing ``J(of, wrt) @ seed``, called as ``f(*inputs, seed)``."""
  fn = fn.concrete
  _checked_names(fn, of, wrt)
  seed = L(f"fwd:{wrt}", fn.inputs[fn.input_tree.index(wrt)].type)
  result = fn.factory(name or f"{fn.name}_fwd_{of}_{wrt}", [*fn.input_names, f"fwd:{wrt}"], [Fwd(of, wrt)])
  return _typed_result(result, param_list(*fn.input_tree.parts, _factory_input_tree(result, seed)))


# fmt: off
@overload
def adjoint(fn: Function[[], [], SO, NO], of: str, wrt: str, *, name: str | None = None) -> Function[[Expr], [np.ndarray], Expr, np.ndarray]: ...
@overload
def adjoint(fn: Function[[SA], [NA], SO, NO], of: str, wrt: str, *, name: str | None = None) -> Function[[SA, Expr], [NA, np.ndarray], Expr, np.ndarray]: ...
@overload
def adjoint(fn: Function[[SA, SB], [NA, NB], SO, NO], of: str, wrt: str, *, name: str | None = None) -> Function[[SA, SB, Expr], [NA, NB, np.ndarray], Expr, np.ndarray]: ...
@overload
def adjoint(fn: Function[[SA, SB, SC], [NA, NB, NC], SO, NO], of: str, wrt: str, *, name: str | None = None) -> Function[[SA, SB, SC, Expr], [NA, NB, NC, np.ndarray], Expr, np.ndarray]: ...
@overload
def adjoint(fn: Function[[SA, SB, SC, SD], [NA, NB, NC, ND], SO, NO], of: str, wrt: str, *, name: str | None = None) -> Function[[SA, SB, SC, SD, Expr], [NA, NB, NC, ND, np.ndarray], Expr, np.ndarray]: ...
@overload
def adjoint(fn: Function[[SA, SB, SC, SD, SE], [NA, NB, NC, ND, NE], SO, NO], of: str, wrt: str, *, name: str | None = None) -> Function[[SA, SB, SC, SD, SE, Expr], [NA, NB, NC, ND, NE, np.ndarray], Expr, np.ndarray]: ...
@overload
def adjoint(fn: Function[[SA, SB, SC, SD, SE, SF], [NA, NB, NC, ND, NE, NF], SO, NO], of: str, wrt: str, *, name: str | None = None) -> Function[[SA, SB, SC, SD, SE, SF, Expr], [NA, NB, NC, ND, NE, NF, np.ndarray], Expr, np.ndarray]: ...
@overload
def adjoint(fn: Function[[SA, SB, SC, SD, SE, SF, SG], [NA, NB, NC, ND, NE, NF, NG], SO, NO], of: str, wrt: str, *, name: str | None = None) -> Function[[SA, SB, SC, SD, SE, SF, SG, Expr], [NA, NB, NC, ND, NE, NF, NG, np.ndarray], Expr, np.ndarray]: ...
@overload
def adjoint(fn: Function[[SA, SB, SC, SD, SE, SF, SG, SH], [NA, NB, NC, ND, NE, NF, NG, NH], SO, NO], of: str, wrt: str, *, name: str | None = None) -> Function[[SA, SB, SC, SD, SE, SF, SG, SH, Expr], [NA, NB, NC, ND, NE, NF, NG, NH, np.ndarray], Expr, np.ndarray]: ...
@overload
def adjoint(fn: Function[..., ..., Any, Any], of: str, wrt: str, *, name: str | None = None) -> Function[..., ..., Expr, np.ndarray]: ...
# fmt: on
def adjoint(fn: Function[Any, Any, Any, Any], of: str, wrt: str, *, name: str | None = None) -> Any:
  """Create a seeded reverse-mode Function computing ``J(of, wrt).T @ lam``, called as ``f(*inputs, lam)``."""
  fn = fn.concrete
  _checked_names(fn, of, wrt)
  seed = L(f"lam:{of}", fn.outputs[fn.output_tree.index(of)].type)
  result = fn.factory(name or f"{fn.name}_adj_{of}_{wrt}", [*fn.input_names, f"lam:{of}"], [Adj(of, wrt)])
  return _typed_result(result, param_list(*fn.input_tree.parts, _factory_input_tree(result, seed)))


# fmt: off
@overload
def lagrangian_hessian(fn: Function[[], [], SO, NO], wrt: str, *, name: str | None = None, aux_name: str = "gamma") -> Function[[SO], [NO], Expr, np.ndarray]: ...
@overload
def lagrangian_hessian(fn: Function[[SA], [NA], SO, NO], wrt: str, *, name: str | None = None, aux_name: str = "gamma") -> Function[[SA, SO], [NA, NO], Expr, np.ndarray]: ...
@overload
def lagrangian_hessian(fn: Function[[SA, SB], [NA, NB], SO, NO], wrt: str, *, name: str | None = None, aux_name: str = "gamma") -> Function[[SA, SB, SO], [NA, NB, NO], Expr, np.ndarray]: ...
@overload
def lagrangian_hessian(fn: Function[[SA, SB, SC], [NA, NB, NC], SO, NO], wrt: str, *, name: str | None = None, aux_name: str = "gamma") -> Function[[SA, SB, SC, SO], [NA, NB, NC, NO], Expr, np.ndarray]: ...
@overload
def lagrangian_hessian(fn: Function[[SA, SB, SC, SD], [NA, NB, NC, ND], SO, NO], wrt: str, *, name: str | None = None, aux_name: str = "gamma") -> Function[[SA, SB, SC, SD, SO], [NA, NB, NC, ND, NO], Expr, np.ndarray]: ...
@overload
def lagrangian_hessian(fn: Function[[SA, SB, SC, SD, SE], [NA, NB, NC, ND, NE], SO, NO], wrt: str, *, name: str | None = None, aux_name: str = "gamma") -> Function[[SA, SB, SC, SD, SE, SO], [NA, NB, NC, ND, NE, NO], Expr, np.ndarray]: ...
@overload
def lagrangian_hessian(fn: Function[[SA, SB, SC, SD, SE, SF], [NA, NB, NC, ND, NE, NF], SO, NO], wrt: str, *, name: str | None = None, aux_name: str = "gamma") -> Function[[SA, SB, SC, SD, SE, SF, SO], [NA, NB, NC, ND, NE, NF, NO], Expr, np.ndarray]: ...
@overload
def lagrangian_hessian(fn: Function[[SA, SB, SC, SD, SE, SF, SG], [NA, NB, NC, ND, NE, NF, NG], SO, NO], wrt: str, *, name: str | None = None, aux_name: str = "gamma") -> Function[[SA, SB, SC, SD, SE, SF, SG, SO], [NA, NB, NC, ND, NE, NF, NG, NO], Expr, np.ndarray]: ...
@overload
def lagrangian_hessian(fn: Function[[SA, SB, SC, SD, SE, SF, SG, SH], [NA, NB, NC, ND, NE, NF, NG, NH], SO, NO], wrt: str, *, name: str | None = None, aux_name: str = "gamma") -> Function[[SA, SB, SC, SD, SE, SF, SG, SH, SO], [NA, NB, NC, ND, NE, NF, NG, NH, NO], Expr, np.ndarray]: ...
@overload
def lagrangian_hessian(fn: Function[..., ..., Any, Any], wrt: str, *, name: str | None = None, aux_name: str = "gamma") -> Function[..., ..., Expr, np.ndarray]: ...
# fmt: on
def lagrangian_hessian(fn: Function[Any, Any, Any, Any], wrt: str, *, name: str | None = None, aux_name: str = "gamma") -> Any:
  """Create the dense Hessian of all outputs weighted by multipliers shaped as the output tree, called as ``f(*inputs, lam)``."""
  fn = fn.concrete
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
  return _typed_result(result, param_list(*fn.input_tree.parts, seed_tree))


# fmt: off
@overload
def sparse_lagrangian_hessian(fn: Function[[], [], SO, NO], wrt: str, *, name: str | None = None, aux_name: str = "gamma", triangle: Triangle = "full") -> Function[[SO], [NO], Expr, np.ndarray]: ...
@overload
def sparse_lagrangian_hessian(fn: Function[[SA], [NA], SO, NO], wrt: str, *, name: str | None = None, aux_name: str = "gamma", triangle: Triangle = "full") -> Function[[SA, SO], [NA, NO], Expr, np.ndarray]: ...
@overload
def sparse_lagrangian_hessian(fn: Function[[SA, SB], [NA, NB], SO, NO], wrt: str, *, name: str | None = None, aux_name: str = "gamma", triangle: Triangle = "full") -> Function[[SA, SB, SO], [NA, NB, NO], Expr, np.ndarray]: ...
@overload
def sparse_lagrangian_hessian(fn: Function[[SA, SB, SC], [NA, NB, NC], SO, NO], wrt: str, *, name: str | None = None, aux_name: str = "gamma", triangle: Triangle = "full") -> Function[[SA, SB, SC, SO], [NA, NB, NC, NO], Expr, np.ndarray]: ...
@overload
def sparse_lagrangian_hessian(fn: Function[[SA, SB, SC, SD], [NA, NB, NC, ND], SO, NO], wrt: str, *, name: str | None = None, aux_name: str = "gamma", triangle: Triangle = "full") -> Function[[SA, SB, SC, SD, SO], [NA, NB, NC, ND, NO], Expr, np.ndarray]: ...
@overload
def sparse_lagrangian_hessian(fn: Function[[SA, SB, SC, SD, SE], [NA, NB, NC, ND, NE], SO, NO], wrt: str, *, name: str | None = None, aux_name: str = "gamma", triangle: Triangle = "full") -> Function[[SA, SB, SC, SD, SE, SO], [NA, NB, NC, ND, NE, NO], Expr, np.ndarray]: ...
@overload
def sparse_lagrangian_hessian(fn: Function[[SA, SB, SC, SD, SE, SF], [NA, NB, NC, ND, NE, NF], SO, NO], wrt: str, *, name: str | None = None, aux_name: str = "gamma", triangle: Triangle = "full") -> Function[[SA, SB, SC, SD, SE, SF, SO], [NA, NB, NC, ND, NE, NF, NO], Expr, np.ndarray]: ...
@overload
def sparse_lagrangian_hessian(fn: Function[[SA, SB, SC, SD, SE, SF, SG], [NA, NB, NC, ND, NE, NF, NG], SO, NO], wrt: str, *, name: str | None = None, aux_name: str = "gamma", triangle: Triangle = "full") -> Function[[SA, SB, SC, SD, SE, SF, SG, SO], [NA, NB, NC, ND, NE, NF, NG, NO], Expr, np.ndarray]: ...
@overload
def sparse_lagrangian_hessian(fn: Function[[SA, SB, SC, SD, SE, SF, SG, SH], [NA, NB, NC, ND, NE, NF, NG, NH], SO, NO], wrt: str, *, name: str | None = None, aux_name: str = "gamma", triangle: Triangle = "full") -> Function[[SA, SB, SC, SD, SE, SF, SG, SH, SO], [NA, NB, NC, ND, NE, NF, NG, NH, NO], Expr, np.ndarray]: ...
@overload
def sparse_lagrangian_hessian(fn: Function[..., ..., Any, Any], wrt: str, *, name: str | None = None, aux_name: str = "gamma", triangle: Triangle = "full") -> Function[..., ..., Expr, np.ndarray]: ...
# fmt: on
def sparse_lagrangian_hessian(
  fn: Function[Any, Any, Any, Any], wrt: str, *, name: str | None = None, aux_name: str = "gamma", triangle: Triangle = "full"
) -> Any:
  """Create compact values for the weighted Hessian of all declared outputs, called as ``f(*inputs, lam)``."""
  fn = fn.concrete
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
  return _typed_result(result, param_list(*fn.input_tree.parts, seed_tree))
