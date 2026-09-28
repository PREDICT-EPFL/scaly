"""The ergonomic frontend: function construction and named derivative wrappers."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol, cast, overload

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


class _Bare(Protocol):
  """``@sc.function`` with nothing declared: typed by the body's own signature on the symbolic side."""

  def __call__[**P, R](self, fn: Callable[P, R], /) -> Function[P, ..., R, Any]: ...


class _OutputOnly[SO, NO](Protocol):
  """``@sc.function(output=...)``: no inputs, or inputs bound at each call."""

  @overload
  def __call__(self, fn: Callable[[], SO], /) -> Function[[], [], SO, NO]: ...

  @overload
  def __call__[**P](self, fn: Callable[P, SO], /) -> Function[P, ..., SO, NO]: ...


class _Inferred[**PS, **PN](Protocol):
  """Declared inputs and no ``output=``: the body's return type is the symbolic output."""

  def __call__[R](self, fn: Callable[PS, R], /) -> Function[PS, PN, R, Any]: ...


# fmt: off
@overload
def function[**P, R](fn: Callable[P, R], /) -> Function[P, ..., R, Any]: ...
@overload
def function(*, name: str | None = None) -> _Bare: ...
@overload
def function(*, output: Tree[SO, NO] | Spec, name: str | None = None) -> _OutputOnly[SO, NO]: ...
@overload
def function(a: Tree[SA, NA] | Spec, /, *, output: Tree[SO, NO] | Spec, name: str | None = None) -> Callable[[Callable[[SA], SO]], Function[[SA], [NA], SO, NO]]: ...
@overload
def function(a: Tree[SA, NA] | Spec, /, *, name: str | None = None) -> _Inferred[[SA], [NA]]: ...
@overload
def function(a: Tree[SA, NA] | Spec, b: Tree[SB, NB] | Spec, /, *, output: Tree[SO, NO] | Spec, name: str | None = None) -> Callable[[Callable[[SA, SB], SO]], Function[[SA, SB], [NA, NB], SO, NO]]: ...
@overload
def function(a: Tree[SA, NA] | Spec, b: Tree[SB, NB] | Spec, /, *, name: str | None = None) -> _Inferred[[SA, SB], [NA, NB]]: ...
@overload
def function(a: Tree[SA, NA] | Spec, b: Tree[SB, NB] | Spec, c: Tree[SC, NC] | Spec, /, *, output: Tree[SO, NO] | Spec, name: str | None = None) -> Callable[[Callable[[SA, SB, SC], SO]], Function[[SA, SB, SC], [NA, NB, NC], SO, NO]]: ...
@overload
def function(a: Tree[SA, NA] | Spec, b: Tree[SB, NB] | Spec, c: Tree[SC, NC] | Spec, /, *, name: str | None = None) -> _Inferred[[SA, SB, SC], [NA, NB, NC]]: ...
@overload
def function(a: Tree[SA, NA] | Spec, b: Tree[SB, NB] | Spec, c: Tree[SC, NC] | Spec, d: Tree[SD, ND] | Spec, /, *, output: Tree[SO, NO] | Spec, name: str | None = None) -> Callable[[Callable[[SA, SB, SC, SD], SO]], Function[[SA, SB, SC, SD], [NA, NB, NC, ND], SO, NO]]: ...
@overload
def function(a: Tree[SA, NA] | Spec, b: Tree[SB, NB] | Spec, c: Tree[SC, NC] | Spec, d: Tree[SD, ND] | Spec, /, *, name: str | None = None) -> _Inferred[[SA, SB, SC, SD], [NA, NB, NC, ND]]: ...
@overload
def function(a: Tree[SA, NA] | Spec, b: Tree[SB, NB] | Spec, c: Tree[SC, NC] | Spec, d: Tree[SD, ND] | Spec, e: Tree[SE, NE] | Spec, /, *, output: Tree[SO, NO] | Spec, name: str | None = None) -> Callable[[Callable[[SA, SB, SC, SD, SE], SO]], Function[[SA, SB, SC, SD, SE], [NA, NB, NC, ND, NE], SO, NO]]: ...
@overload
def function(a: Tree[SA, NA] | Spec, b: Tree[SB, NB] | Spec, c: Tree[SC, NC] | Spec, d: Tree[SD, ND] | Spec, e: Tree[SE, NE] | Spec, /, *, name: str | None = None) -> _Inferred[[SA, SB, SC, SD, SE], [NA, NB, NC, ND, NE]]: ...
@overload
def function(a: Tree[SA, NA] | Spec, b: Tree[SB, NB] | Spec, c: Tree[SC, NC] | Spec, d: Tree[SD, ND] | Spec, e: Tree[SE, NE] | Spec, f: Tree[SF, NF] | Spec, /, *, output: Tree[SO, NO] | Spec, name: str | None = None) -> Callable[[Callable[[SA, SB, SC, SD, SE, SF], SO]], Function[[SA, SB, SC, SD, SE, SF], [NA, NB, NC, ND, NE, NF], SO, NO]]: ...
@overload
def function(a: Tree[SA, NA] | Spec, b: Tree[SB, NB] | Spec, c: Tree[SC, NC] | Spec, d: Tree[SD, ND] | Spec, e: Tree[SE, NE] | Spec, f: Tree[SF, NF] | Spec, /, *, name: str | None = None) -> _Inferred[[SA, SB, SC, SD, SE, SF], [NA, NB, NC, ND, NE, NF]]: ...
@overload
def function(a: Tree[SA, NA] | Spec, b: Tree[SB, NB] | Spec, c: Tree[SC, NC] | Spec, d: Tree[SD, ND] | Spec, e: Tree[SE, NE] | Spec, f: Tree[SF, NF] | Spec, g: Tree[SG, NG] | Spec, /, *, output: Tree[SO, NO] | Spec, name: str | None = None) -> Callable[[Callable[[SA, SB, SC, SD, SE, SF, SG], SO]], Function[[SA, SB, SC, SD, SE, SF, SG], [NA, NB, NC, ND, NE, NF, NG], SO, NO]]: ...
@overload
def function(a: Tree[SA, NA] | Spec, b: Tree[SB, NB] | Spec, c: Tree[SC, NC] | Spec, d: Tree[SD, ND] | Spec, e: Tree[SE, NE] | Spec, f: Tree[SF, NF] | Spec, g: Tree[SG, NG] | Spec, /, *, name: str | None = None) -> _Inferred[[SA, SB, SC, SD, SE, SF, SG], [NA, NB, NC, ND, NE, NF, NG]]: ...
@overload
def function(a: Tree[SA, NA] | Spec, b: Tree[SB, NB] | Spec, c: Tree[SC, NC] | Spec, d: Tree[SD, ND] | Spec, e: Tree[SE, NE] | Spec, f: Tree[SF, NF] | Spec, g: Tree[SG, NG] | Spec, h: Tree[SH, NH] | Spec, /, *, output: Tree[SO, NO] | Spec, name: str | None = None) -> Callable[[Callable[[SA, SB, SC, SD, SE, SF, SG, SH], SO]], Function[[SA, SB, SC, SD, SE, SF, SG, SH], [NA, NB, NC, ND, NE, NF, NG, NH], SO, NO]]: ...
@overload
def function(a: Tree[SA, NA] | Spec, b: Tree[SB, NB] | Spec, c: Tree[SC, NC] | Spec, d: Tree[SD, ND] | Spec, e: Tree[SE, NE] | Spec, f: Tree[SF, NF] | Spec, g: Tree[SG, NG] | Spec, h: Tree[SH, NH] | Spec, /, *, name: str | None = None) -> _Inferred[[SA, SB, SC, SD, SE, SF, SG, SH], [NA, NB, NC, ND, NE, NF, NG, NH]]: ...
@overload
def function(*slots: Tree[Any, Any] | Spec, output: Tree[Any, Any] | Spec | None = None, name: str | None = None) -> Callable[[Callable[..., Any]], Function[..., ..., Any, Any]]: ...
# fmt: on
def function(*slots: Any, output: Tree[Any, Any] | Spec | None = None, name: str | None = None) -> Any:
  """Trace a Python body into a named ``Function``: one declaration per parameter, and the ``output``.

  Each slot is a tree spec: ``sc.L``, ``sc.G`` or ``sc.S``, or shorthand for one leaf, a shape
  (``3``, ``(n, m)``, a ``TensorType``) or a name (a leaf whose shape the trace decides). Unnamed
  leaves take the parameter's name (``p``, or ``p_0``, ``p_1`` inside a group); unnamed outputs take
  the function's. The names become the generated C signature and the ``of``/``wrt`` of derivatives.
  ``output`` may be omitted: the output tree is then read off the trace, a single leaf named after the
  function and a tuple's leaves ``f_0``, ``f_1``, ...

  With every input shape declared, the body is traced now and the result is a ``ConcreteFunction``.
  An input shape left open (``sc.L()``, ``(n, None)``) makes a template: each call binds the holes and
  traces one instance per distinct binding; see ``Function``. With no slots at all (``@sc.function``,
  with or without parentheses), a body with parameters is a template whose calls bind everything: a
  tuple argument is structure, an ``Expr`` a leaf of its shape and dtype, an array-like a ``float64``
  leaf of its shape.
  """
  if len(slots) == 1 and callable(slots[0]) and not isinstance(slots[0], Tree) and output is None and name is None:
    return function()(slots[0])  # @sc.function, without parentheses

  def decorate(fn: Callable[..., Any]) -> Function[..., ..., Any, Any]:
    fn_name = name or getattr(fn, "__name__", "fn")
    names = _parameters(fn, fn_name)
    outputs = None if output is None else as_tree(output).named(fn_name)
    if not slots:
      return Function(fn_name, fn, None, outputs) if names else ConcreteFunction(fn_name, fn, param_list(), outputs)
    if len(slots) != len(names):
      if output is None and len(slots) == len(names) + 1:
        raise TypeError(
          f"{fn_name}: sc.function takes one declaration per parameter and the output as output=; "
          "did you mean sc.function(<inputs>, output=<outputs>)?"
        )
      raise TypeError(f"{fn_name}: declared {len(slots)} parameters, the body takes {len(names)} ({', '.join(names)})")
    inputs = param_list(*(as_tree(slot).named(param) for slot, param in zip(slots, names, strict=True)))
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


def _function_names(operation: str, args: tuple[Expr | str, ...], wrt: Expr | str | None, of: str | None) -> tuple[str | None, str | None]:
  """``(of, wrt)`` from a Function-form call: two strings are both, one string is ``wrt`` (or ``of``
  when ``wrt=`` is given), and a name left out is the Function's only output or only input."""
  if len(args) > 2 or not all(isinstance(arg, str) for arg in args) or not isinstance(wrt, (str, type(None))):
    raise TypeError(f"the Function form is {operation}(fn, of, wrt), or {operation}(fn, wrt) for a Function with one output")
  if len(args) == 2 and of is None and wrt is None:
    return cast(str, args[0]), cast(str, args[1])
  if len(args) == 1 and wrt is None:
    return of, cast(str, args[0])
  if len(args) == 1 and of is None:
    return cast(str, args[0]), wrt
  if not args:
    return of, wrt
  raise TypeError(f"{operation}: of and wrt are given twice")


def _resolved_names(
  operation: str, owner: str, inputs: tuple[str, ...] | None, outputs: tuple[str, ...] | None, of: str | None, wrt: str | None, weighted: bool
) -> tuple[str | None, str | None]:
  """``of`` and ``wrt`` checked against the names ``owner`` declares (``None`` when not known yet), a name
  left out defaulting to the only output or input. A ``weighted`` derivative (the Lagrangian Hessians)
  takes every output and no ``of``."""
  if weighted:
    of = ""
  elif of is None and outputs is not None:
    if len(outputs) != 1:
      raise TypeError(f"{operation}: {owner} has outputs {outputs}; say which with of=")
    of = outputs[0]
  if wrt is None and inputs is not None:
    if len(inputs) != 1:
      raise TypeError(f"{operation}: {owner} has inputs {inputs}; say which with wrt=")
    wrt = inputs[0]
  if wrt is not None and inputs is not None and wrt not in inputs and outputs is not None and wrt in outputs:
    raise ValueError(
      f"{operation}: {wrt!r} is an output of {owner}, not an input; one name is wrt, as in "
      f"sc.{operation}(f, 'x'), and two are of and wrt, as in sc.{operation}(f, {wrt!r}, 'x')"
    )
  if wrt is not None and inputs is not None and wrt not in inputs:
    raise ValueError(f"unknown name {wrt!r}; declared {inputs}")
  if of and outputs is not None and of not in outputs:
    raise ValueError(f"unknown name {of!r}; declared {outputs}")
  return of, wrt


def _derived(
  operation: str,
  source: Function[Any, Any, Any, Any],
  of: str | None,
  wrt: str | None,
  name: str | None,
  default: Callable[[str, str, str], str],
  build: Callable[[ConcreteFunction[Any, Any, Any, Any], str, str, str], Any],
  extra: tuple[str, ...] = (),
  *,
  weighted: bool = False,
) -> Any:
  """A derivative of ``source``: built now from a concrete one, per instance from a template.

  ``default(source name, of, wrt)`` names a derivative; with ``name`` given, a template's instances
  are ``{name}__{tokens}``, the tokens of the source instance they derive from. Names the template
  declares are checked now; anything that needs shapes waits for the instance."""
  if not isinstance(source, Function):
    raise TypeError(f"{operation} source must be an Expr or Function")

  def derive(instance: ConcreteFunction[Any, Any, Any, Any]) -> Any:
    of_, wrt_ = cast(tuple[str, str], _resolved_names(operation, instance.name, instance.input_names, instance.output_names, of, wrt, weighted))
    derived_name = default(instance.name, of_, wrt_) if name is None else (name if instance is source else f"{name}__{instance.tokens}")
    return build(instance, derived_name, of_, wrt_)

  if source.is_concrete:
    return derive(source.concrete)
  # A template's declared names are checked now, and settle defaults; the rest waits for an instance.
  declared_inputs = source._slots.names if source._slots is not None and source._source is None else None
  declared_outputs = source._output.names if source._output is not None else None
  known_of, known_wrt = _resolved_names(operation, source.name, declared_inputs, declared_outputs, of, wrt, weighted)
  label = default(source.name, known_of, known_wrt) if known_of is not None and known_wrt is not None else f"{source.name}_{operation}"
  return source.lift(derive, name or label, extra)


def _typed_result(result: ConcreteFunction[Any, Any, Any, Any], input_tree: _G) -> Any:
  output_tree = L(result.output_names[0], result.outputs[0].type)
  return result._with_trees(input_tree, output_tree)


def _factory_input_tree(result: ConcreteFunction[Any, Any, Any, Any], tree: Tree[Any, Any]) -> Tree[Any, Any]:
  input_map = result.input_map()
  return tree.with_types(tuple(input_map[name].type for name in tree.names))


def _unseeded(source: ConcreteFunction[Any, Any, Any, Any], name: str, of: str, wrt: str, spec: Jac | Grad | Hess | SpJac | SpHess) -> Any:
  return _typed_result(source.factory(name, list(source.input_names), [spec]), source.input_tree)


@overload
def jacobian(source: Expr, wrt: Expr) -> Expr: ...


@overload
def jacobian[**PS, **PN](
  source: Function[PS, PN, Any, Any], /, *names: str, of: str | None = None, wrt: str | None = None, name: str | None = None
) -> Function[PS, PN, Expr, np.ndarray]: ...


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
  function_of, function_wrt = _function_names("jacobian", args, wrt, of)
  return _derived(
    "jacobian",
    source,
    function_of,
    function_wrt,
    name,
    lambda s, o, w: f"{s}_jac_{o}_{w}",
    lambda src, derived_name, of, wrt: _unseeded(src, derived_name, of, wrt, Jac(of, wrt)),
  )


@overload
def gradient(source: Expr, wrt: Expr) -> Expr: ...


@overload
def gradient[**PS, **PN](
  source: Function[PS, PN, Any, Any], /, *names: str, of: str | None = None, wrt: str | None = None, name: str | None = None
) -> Function[PS, PN, Expr, np.ndarray]: ...


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
  function_of, function_wrt = _function_names("gradient", args, wrt, of)
  return _derived(
    "gradient",
    source,
    function_of,
    function_wrt,
    name,
    lambda s, o, w: f"{s}_grad_{o}_{w}",
    lambda src, derived_name, of, wrt: _unseeded(src, derived_name, of, wrt, Grad(of, wrt)),
  )


@overload
def hessian(source: Expr, wrt: Expr) -> Expr: ...


@overload
def hessian[**PS, **PN](
  source: Function[PS, PN, Any, Any], /, *names: str, of: str | None = None, wrt: str | None = None, name: str | None = None
) -> Function[PS, PN, Expr, np.ndarray]: ...


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
  function_of, function_wrt = _function_names("hessian", args, wrt, of)
  return _derived(
    "hessian",
    source,
    function_of,
    function_wrt,
    name,
    lambda s, o, w: f"{s}_hess_{o}_{w}_{w}",
    lambda src, derived_name, of, wrt: _unseeded(src, derived_name, of, wrt, Hess(of, wrt)),
  )


@overload
def sparse_jacobian(source: Expr, wrt: Expr) -> SparseJacobian: ...


@overload
def sparse_jacobian[**PS, **PN](
  source: Function[PS, PN, Any, Any], /, *names: str, of: str | None = None, wrt: str | None = None, name: str | None = None
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
  function_of, function_wrt = _function_names("sparse_jacobian", args, wrt, of)
  return _derived(
    "sparse_jacobian",
    source,
    function_of,
    function_wrt,
    name,
    lambda s, o, w: f"{s}_spjac_{o}_{w}",
    lambda src, derived_name, of, wrt: _unseeded(src, derived_name, of, wrt, SpJac(of, wrt)),
  )


@overload
def sparse_hessian(source: Expr, wrt: Expr, *, triangle: Triangle = "full") -> SparseJacobian: ...


@overload
def sparse_hessian[**PS, **PN](
  source: Function[PS, PN, Any, Any],
  /,
  *names: str,
  of: str | None = None,
  wrt: str | None = None,
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
  function_of, function_wrt = _function_names("sparse_hessian", args, wrt, of)
  return _derived(
    "sparse_hessian",
    source,
    function_of,
    function_wrt,
    name,
    lambda s, o, w: f"{s}_sphess_{o}_{w}_{w}",
    lambda src, derived_name, of, wrt: _unseeded(src, derived_name, of, wrt, SpHess(of, wrt, triangle=triangle)),
  )


# fmt: off
@overload
def forward(fn: Function[[], [], SO, NO], /, *names: str, of: str | None = None, wrt: str | None = None, name: str | None = None) -> Function[[Expr], [np.ndarray], Expr, np.ndarray]: ...
@overload
def forward(fn: Function[[SA], [NA], SO, NO], /, *names: str, of: str | None = None, wrt: str | None = None, name: str | None = None) -> Function[[SA, Expr], [NA, np.ndarray], Expr, np.ndarray]: ...
@overload
def forward(fn: Function[[SA, SB], [NA, NB], SO, NO], /, *names: str, of: str | None = None, wrt: str | None = None, name: str | None = None) -> Function[[SA, SB, Expr], [NA, NB, np.ndarray], Expr, np.ndarray]: ...
@overload
def forward(fn: Function[[SA, SB, SC], [NA, NB, NC], SO, NO], /, *names: str, of: str | None = None, wrt: str | None = None, name: str | None = None) -> Function[[SA, SB, SC, Expr], [NA, NB, NC, np.ndarray], Expr, np.ndarray]: ...
@overload
def forward(fn: Function[[SA, SB, SC, SD], [NA, NB, NC, ND], SO, NO], /, *names: str, of: str | None = None, wrt: str | None = None, name: str | None = None) -> Function[[SA, SB, SC, SD, Expr], [NA, NB, NC, ND, np.ndarray], Expr, np.ndarray]: ...
@overload
def forward(fn: Function[[SA, SB, SC, SD, SE], [NA, NB, NC, ND, NE], SO, NO], /, *names: str, of: str | None = None, wrt: str | None = None, name: str | None = None) -> Function[[SA, SB, SC, SD, SE, Expr], [NA, NB, NC, ND, NE, np.ndarray], Expr, np.ndarray]: ...
@overload
def forward(fn: Function[[SA, SB, SC, SD, SE, SF], [NA, NB, NC, ND, NE, NF], SO, NO], /, *names: str, of: str | None = None, wrt: str | None = None, name: str | None = None) -> Function[[SA, SB, SC, SD, SE, SF, Expr], [NA, NB, NC, ND, NE, NF, np.ndarray], Expr, np.ndarray]: ...
@overload
def forward(fn: Function[[SA, SB, SC, SD, SE, SF, SG], [NA, NB, NC, ND, NE, NF, NG], SO, NO], /, *names: str, of: str | None = None, wrt: str | None = None, name: str | None = None) -> Function[[SA, SB, SC, SD, SE, SF, SG, Expr], [NA, NB, NC, ND, NE, NF, NG, np.ndarray], Expr, np.ndarray]: ...
@overload
def forward(fn: Function[[SA, SB, SC, SD, SE, SF, SG, SH], [NA, NB, NC, ND, NE, NF, NG, NH], SO, NO], /, *names: str, of: str | None = None, wrt: str | None = None, name: str | None = None) -> Function[[SA, SB, SC, SD, SE, SF, SG, SH, Expr], [NA, NB, NC, ND, NE, NF, NG, NH, np.ndarray], Expr, np.ndarray]: ...
@overload
def forward(fn: Function[..., ..., Any, Any], /, *names: str, of: str | None = None, wrt: str | None = None, name: str | None = None) -> Function[..., ..., Expr, np.ndarray]: ...
# fmt: on
def forward(fn: Function[Any, Any, Any, Any], /, *names: str, of: str | None = None, wrt: str | None = None, name: str | None = None) -> Any:
  """Create a seeded forward-mode Function computing ``J(of, wrt) @ seed``, called as ``f(*inputs, seed)``."""
  of, wrt = _function_names("forward", names, wrt, of)
  return _derived("forward", fn, of, wrt, name, lambda s, o, w: f"{s}_fwd_{o}_{w}", _forward, ("seed",))


def _forward(fn: ConcreteFunction[Any, Any, Any, Any], name: str, of: str, wrt: str) -> Any:
  seed = L(f"fwd:{wrt}", fn.inputs[fn.input_tree.index(wrt)].type)
  result = fn.factory(name, [*fn.input_names, f"fwd:{wrt}"], [Fwd(of, wrt)])
  return _typed_result(result, param_list(*fn.input_tree.parts, _factory_input_tree(result, seed)))


# fmt: off
@overload
def adjoint(fn: Function[[], [], SO, NO], /, *names: str, of: str | None = None, wrt: str | None = None, name: str | None = None) -> Function[[Expr], [np.ndarray], Expr, np.ndarray]: ...
@overload
def adjoint(fn: Function[[SA], [NA], SO, NO], /, *names: str, of: str | None = None, wrt: str | None = None, name: str | None = None) -> Function[[SA, Expr], [NA, np.ndarray], Expr, np.ndarray]: ...
@overload
def adjoint(fn: Function[[SA, SB], [NA, NB], SO, NO], /, *names: str, of: str | None = None, wrt: str | None = None, name: str | None = None) -> Function[[SA, SB, Expr], [NA, NB, np.ndarray], Expr, np.ndarray]: ...
@overload
def adjoint(fn: Function[[SA, SB, SC], [NA, NB, NC], SO, NO], /, *names: str, of: str | None = None, wrt: str | None = None, name: str | None = None) -> Function[[SA, SB, SC, Expr], [NA, NB, NC, np.ndarray], Expr, np.ndarray]: ...
@overload
def adjoint(fn: Function[[SA, SB, SC, SD], [NA, NB, NC, ND], SO, NO], /, *names: str, of: str | None = None, wrt: str | None = None, name: str | None = None) -> Function[[SA, SB, SC, SD, Expr], [NA, NB, NC, ND, np.ndarray], Expr, np.ndarray]: ...
@overload
def adjoint(fn: Function[[SA, SB, SC, SD, SE], [NA, NB, NC, ND, NE], SO, NO], /, *names: str, of: str | None = None, wrt: str | None = None, name: str | None = None) -> Function[[SA, SB, SC, SD, SE, Expr], [NA, NB, NC, ND, NE, np.ndarray], Expr, np.ndarray]: ...
@overload
def adjoint(fn: Function[[SA, SB, SC, SD, SE, SF], [NA, NB, NC, ND, NE, NF], SO, NO], /, *names: str, of: str | None = None, wrt: str | None = None, name: str | None = None) -> Function[[SA, SB, SC, SD, SE, SF, Expr], [NA, NB, NC, ND, NE, NF, np.ndarray], Expr, np.ndarray]: ...
@overload
def adjoint(fn: Function[[SA, SB, SC, SD, SE, SF, SG], [NA, NB, NC, ND, NE, NF, NG], SO, NO], /, *names: str, of: str | None = None, wrt: str | None = None, name: str | None = None) -> Function[[SA, SB, SC, SD, SE, SF, SG, Expr], [NA, NB, NC, ND, NE, NF, NG, np.ndarray], Expr, np.ndarray]: ...
@overload
def adjoint(fn: Function[[SA, SB, SC, SD, SE, SF, SG, SH], [NA, NB, NC, ND, NE, NF, NG, NH], SO, NO], /, *names: str, of: str | None = None, wrt: str | None = None, name: str | None = None) -> Function[[SA, SB, SC, SD, SE, SF, SG, SH, Expr], [NA, NB, NC, ND, NE, NF, NG, NH, np.ndarray], Expr, np.ndarray]: ...
@overload
def adjoint(fn: Function[..., ..., Any, Any], /, *names: str, of: str | None = None, wrt: str | None = None, name: str | None = None) -> Function[..., ..., Expr, np.ndarray]: ...
# fmt: on
def adjoint(fn: Function[Any, Any, Any, Any], /, *names: str, of: str | None = None, wrt: str | None = None, name: str | None = None) -> Any:
  """Create a seeded reverse-mode Function computing ``J(of, wrt).T @ lam``, called as ``f(*inputs, lam)``."""
  of, wrt = _function_names("adjoint", names, wrt, of)
  return _derived("adjoint", fn, of, wrt, name, lambda s, o, w: f"{s}_adj_{o}_{w}", _adjoint, ("lam",))


def _adjoint(fn: ConcreteFunction[Any, Any, Any, Any], name: str, of: str, wrt: str) -> Any:
  seed = L(f"lam:{of}", fn.outputs[fn.output_tree.index(of)].type)
  result = fn.factory(name, [*fn.input_names, f"lam:{of}"], [Adj(of, wrt)])
  return _typed_result(result, param_list(*fn.input_tree.parts, _factory_input_tree(result, seed)))


# fmt: off
@overload
def lagrangian_hessian(fn: Function[[], [], SO, NO], /, wrt: str | None = None, *, name: str | None = None, aux_name: str = "gamma") -> Function[[SO], [NO], Expr, np.ndarray]: ...
@overload
def lagrangian_hessian(fn: Function[[SA], [NA], SO, NO], /, wrt: str | None = None, *, name: str | None = None, aux_name: str = "gamma") -> Function[[SA, SO], [NA, NO], Expr, np.ndarray]: ...
@overload
def lagrangian_hessian(fn: Function[[SA, SB], [NA, NB], SO, NO], /, wrt: str | None = None, *, name: str | None = None, aux_name: str = "gamma") -> Function[[SA, SB, SO], [NA, NB, NO], Expr, np.ndarray]: ...
@overload
def lagrangian_hessian(fn: Function[[SA, SB, SC], [NA, NB, NC], SO, NO], /, wrt: str | None = None, *, name: str | None = None, aux_name: str = "gamma") -> Function[[SA, SB, SC, SO], [NA, NB, NC, NO], Expr, np.ndarray]: ...
@overload
def lagrangian_hessian(fn: Function[[SA, SB, SC, SD], [NA, NB, NC, ND], SO, NO], /, wrt: str | None = None, *, name: str | None = None, aux_name: str = "gamma") -> Function[[SA, SB, SC, SD, SO], [NA, NB, NC, ND, NO], Expr, np.ndarray]: ...
@overload
def lagrangian_hessian(fn: Function[[SA, SB, SC, SD, SE], [NA, NB, NC, ND, NE], SO, NO], /, wrt: str | None = None, *, name: str | None = None, aux_name: str = "gamma") -> Function[[SA, SB, SC, SD, SE, SO], [NA, NB, NC, ND, NE, NO], Expr, np.ndarray]: ...
@overload
def lagrangian_hessian(fn: Function[[SA, SB, SC, SD, SE, SF], [NA, NB, NC, ND, NE, NF], SO, NO], /, wrt: str | None = None, *, name: str | None = None, aux_name: str = "gamma") -> Function[[SA, SB, SC, SD, SE, SF, SO], [NA, NB, NC, ND, NE, NF, NO], Expr, np.ndarray]: ...
@overload
def lagrangian_hessian(fn: Function[[SA, SB, SC, SD, SE, SF, SG], [NA, NB, NC, ND, NE, NF, NG], SO, NO], /, wrt: str | None = None, *, name: str | None = None, aux_name: str = "gamma") -> Function[[SA, SB, SC, SD, SE, SF, SG, SO], [NA, NB, NC, ND, NE, NF, NG, NO], Expr, np.ndarray]: ...
@overload
def lagrangian_hessian(fn: Function[[SA, SB, SC, SD, SE, SF, SG, SH], [NA, NB, NC, ND, NE, NF, NG, NH], SO, NO], /, wrt: str | None = None, *, name: str | None = None, aux_name: str = "gamma") -> Function[[SA, SB, SC, SD, SE, SF, SG, SH, SO], [NA, NB, NC, ND, NE, NF, NG, NH, NO], Expr, np.ndarray]: ...
@overload
def lagrangian_hessian(fn: Function[..., ..., Any, Any], /, wrt: str | None = None, *, name: str | None = None, aux_name: str = "gamma") -> Function[..., ..., Expr, np.ndarray]: ...
# fmt: on
def lagrangian_hessian(fn: Function[Any, Any, Any, Any], /, wrt: str | None = None, *, name: str | None = None, aux_name: str = "gamma") -> Any:
  """Create the dense Hessian of all outputs weighted by multipliers shaped as the output tree, called as ``f(*inputs, lam)``."""
  return _derived(
    "lagrangian_hessian",
    fn,
    None,
    wrt,
    name,
    lambda s, o, w: f"{s}_hess_{aux_name}_{w}_{w}",
    lambda src, derived_name, of, wrt: _weighted(src, derived_name, wrt, Hess(aux_name, wrt), aux_name),
    ("lam",),
    weighted=True,
  )


def _weighted(fn: ConcreteFunction[Any, Any, Any, Any], name: str, wrt: str, spec: Hess | SpHess, aux_name: str) -> Any:
  output_names = list(fn.output_names)
  result = fn.factory(name, [*fn.input_names, *(f"lam:{out}" for out in output_names)], [spec], aux={aux_name: output_names})
  seed_tree = _factory_input_tree(result, fn.output_tree.relabel("lam:"))
  return _typed_result(result, param_list(*fn.input_tree.parts, seed_tree))


# fmt: off
@overload
def sparse_lagrangian_hessian(fn: Function[[], [], SO, NO], /, wrt: str | None = None, *, name: str | None = None, aux_name: str = "gamma", triangle: Triangle = "full") -> Function[[SO], [NO], Expr, np.ndarray]: ...
@overload
def sparse_lagrangian_hessian(fn: Function[[SA], [NA], SO, NO], /, wrt: str | None = None, *, name: str | None = None, aux_name: str = "gamma", triangle: Triangle = "full") -> Function[[SA, SO], [NA, NO], Expr, np.ndarray]: ...
@overload
def sparse_lagrangian_hessian(fn: Function[[SA, SB], [NA, NB], SO, NO], /, wrt: str | None = None, *, name: str | None = None, aux_name: str = "gamma", triangle: Triangle = "full") -> Function[[SA, SB, SO], [NA, NB, NO], Expr, np.ndarray]: ...
@overload
def sparse_lagrangian_hessian(fn: Function[[SA, SB, SC], [NA, NB, NC], SO, NO], /, wrt: str | None = None, *, name: str | None = None, aux_name: str = "gamma", triangle: Triangle = "full") -> Function[[SA, SB, SC, SO], [NA, NB, NC, NO], Expr, np.ndarray]: ...
@overload
def sparse_lagrangian_hessian(fn: Function[[SA, SB, SC, SD], [NA, NB, NC, ND], SO, NO], /, wrt: str | None = None, *, name: str | None = None, aux_name: str = "gamma", triangle: Triangle = "full") -> Function[[SA, SB, SC, SD, SO], [NA, NB, NC, ND, NO], Expr, np.ndarray]: ...
@overload
def sparse_lagrangian_hessian(fn: Function[[SA, SB, SC, SD, SE], [NA, NB, NC, ND, NE], SO, NO], /, wrt: str | None = None, *, name: str | None = None, aux_name: str = "gamma", triangle: Triangle = "full") -> Function[[SA, SB, SC, SD, SE, SO], [NA, NB, NC, ND, NE, NO], Expr, np.ndarray]: ...
@overload
def sparse_lagrangian_hessian(fn: Function[[SA, SB, SC, SD, SE, SF], [NA, NB, NC, ND, NE, NF], SO, NO], /, wrt: str | None = None, *, name: str | None = None, aux_name: str = "gamma", triangle: Triangle = "full") -> Function[[SA, SB, SC, SD, SE, SF, SO], [NA, NB, NC, ND, NE, NF, NO], Expr, np.ndarray]: ...
@overload
def sparse_lagrangian_hessian(fn: Function[[SA, SB, SC, SD, SE, SF, SG], [NA, NB, NC, ND, NE, NF, NG], SO, NO], /, wrt: str | None = None, *, name: str | None = None, aux_name: str = "gamma", triangle: Triangle = "full") -> Function[[SA, SB, SC, SD, SE, SF, SG, SO], [NA, NB, NC, ND, NE, NF, NG, NO], Expr, np.ndarray]: ...
@overload
def sparse_lagrangian_hessian(fn: Function[[SA, SB, SC, SD, SE, SF, SG, SH], [NA, NB, NC, ND, NE, NF, NG, NH], SO, NO], /, wrt: str | None = None, *, name: str | None = None, aux_name: str = "gamma", triangle: Triangle = "full") -> Function[[SA, SB, SC, SD, SE, SF, SG, SH, SO], [NA, NB, NC, ND, NE, NF, NG, NH, NO], Expr, np.ndarray]: ...
@overload
def sparse_lagrangian_hessian(fn: Function[..., ..., Any, Any], /, wrt: str | None = None, *, name: str | None = None, aux_name: str = "gamma", triangle: Triangle = "full") -> Function[..., ..., Expr, np.ndarray]: ...
# fmt: on
def sparse_lagrangian_hessian(
  fn: Function[Any, Any, Any, Any], /, wrt: str | None = None, *, name: str | None = None, aux_name: str = "gamma", triangle: Triangle = "full"
) -> Any:
  """Create compact values for the weighted Hessian of all declared outputs, called as ``f(*inputs, lam)``."""
  return _derived(
    "sparse_lagrangian_hessian",
    fn,
    None,
    wrt,
    name,
    lambda s, o, w: f"{s}_sphess_{aux_name}_{w}_{w}",
    lambda src, derived_name, of, wrt: _weighted(src, derived_name, wrt, SpHess(aux_name, wrt, triangle=triangle), aux_name),
    ("lam",),
    weighted=True,
  )
