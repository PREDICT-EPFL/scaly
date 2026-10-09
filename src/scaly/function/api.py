"""The ergonomic frontend: function construction and named derivative wrappers."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, Protocol, cast, overload

from ..ad.derivatives import gradient as _gradient_expr
from ..ad.derivatives import hessian as _hessian_expr
from ..ad.derivatives import jacobian as _jacobian_expr
from ..ad.sparse import SparseJacobian, Triangle
from ..ad.sparse import sparse_hessian as _expr_sparse_hessian
from ..ad.sparse import sparse_jacobian as _expr_sparse_jacobian
from ..ir.expr import Expr
from .concrete import ConcreteFunction
from .factory import Adj, Fwd, Grad, Hess, Jac, SpHess, SpJac
from .model import Function, derived_name, lift
from .tree import Array, Tree, append_parameter, arg, parameter_list


class _InferredOutputs[NI, *Ss](Protocol):
  """What ``@function(...)`` without ``outputs=`` returns. The symbolic output type is the body's;
  the numerical one is ``Array`` for an ``Expr`` body and ``Any`` otherwise, since no map from one
  to the other exists (README, "Why a wrapper class exists")."""

  @overload
  def __call__(self, fn: Callable[[*Ss], Expr], /) -> Function[tuple[*Ss], NI, Expr, Array]: ...
  @overload
  def __call__[SO](self, fn: Callable[[*Ss], SO], /) -> Function[tuple[*Ss], NI, SO, Any]: ...


class _Bare(Protocol):
  """What ``@function()`` returns. A body without parameters counts as fully declared."""

  @overload
  def __call__(self, fn: Callable[[], Expr], /) -> Function[tuple[()], tuple[()], Expr, Array]: ...
  @overload
  def __call__[SO](self, fn: Callable[[], SO], /) -> Function[tuple[()], tuple[()], SO, Any]: ...
  @overload
  def __call__[*Ss, SO](self, fn: Callable[[*Ss], SO], /) -> Function[tuple[*Ss], Any, SO, Any]: ...


# One overload per width: turning the slots' `Tree[S, N]`s into the two parameter lists is the type-level
# map Python lacks (README, "Why a wrapper class exists"). The second ladder omits `outputs`, which the
# trace then supplies. Bare comes last: no slots and no `outputs`.
# fmt: off
@overload
def function[SO, NO](*, outputs: Tree[SO, NO], name: str | None = None) -> Callable[[Callable[[], SO]], Function[tuple[()], tuple[()], SO, NO]]: ...
@overload
def function[SA, NA, SO, NO](a: Tree[SA, NA], /, *, outputs: Tree[SO, NO], name: str | None = None) -> Callable[[Callable[[SA], SO]], Function[tuple[SA], tuple[NA], SO, NO]]: ...
@overload
def function[SA, NA, SB, NB, SO, NO](a: Tree[SA, NA], b: Tree[SB, NB], /, *, outputs: Tree[SO, NO], name: str | None = None) -> Callable[[Callable[[SA, SB], SO]], Function[tuple[SA, SB], tuple[NA, NB], SO, NO]]: ...
@overload
def function[SA, NA, SB, NB, SC, NC, SO, NO](a: Tree[SA, NA], b: Tree[SB, NB], c: Tree[SC, NC], /, *, outputs: Tree[SO, NO], name: str | None = None) -> Callable[[Callable[[SA, SB, SC], SO]], Function[tuple[SA, SB, SC], tuple[NA, NB, NC], SO, NO]]: ...
@overload
def function[SA, NA, SB, NB, SC, NC, SD, ND, SO, NO](a: Tree[SA, NA], b: Tree[SB, NB], c: Tree[SC, NC], d: Tree[SD, ND], /, *, outputs: Tree[SO, NO], name: str | None = None) -> Callable[[Callable[[SA, SB, SC, SD], SO]], Function[tuple[SA, SB, SC, SD], tuple[NA, NB, NC, ND], SO, NO]]: ...
@overload
def function[SA, NA, SB, NB, SC, NC, SD, ND, SE, NE, SO, NO](a: Tree[SA, NA], b: Tree[SB, NB], c: Tree[SC, NC], d: Tree[SD, ND], e: Tree[SE, NE], /, *, outputs: Tree[SO, NO], name: str | None = None) -> Callable[[Callable[[SA, SB, SC, SD, SE], SO]], Function[tuple[SA, SB, SC, SD, SE], tuple[NA, NB, NC, ND, NE], SO, NO]]: ...
@overload
def function[SA, NA, SB, NB, SC, NC, SD, ND, SE, NE, SF, NF, SO, NO](a: Tree[SA, NA], b: Tree[SB, NB], c: Tree[SC, NC], d: Tree[SD, ND], e: Tree[SE, NE], f: Tree[SF, NF], /, *, outputs: Tree[SO, NO], name: str | None = None) -> Callable[[Callable[[SA, SB, SC, SD, SE, SF], SO]], Function[tuple[SA, SB, SC, SD, SE, SF], tuple[NA, NB, NC, ND, NE, NF], SO, NO]]: ...
@overload
def function[SA, NA, SB, NB, SC, NC, SD, ND, SE, NE, SF, NF, SG, NG, SO, NO](a: Tree[SA, NA], b: Tree[SB, NB], c: Tree[SC, NC], d: Tree[SD, ND], e: Tree[SE, NE], f: Tree[SF, NF], g: Tree[SG, NG], /, *, outputs: Tree[SO, NO], name: str | None = None) -> Callable[[Callable[[SA, SB, SC, SD, SE, SF, SG], SO]], Function[tuple[SA, SB, SC, SD, SE, SF, SG], tuple[NA, NB, NC, ND, NE, NF, NG], SO, NO]]: ...
@overload
def function[SA, NA, SB, NB, SC, NC, SD, ND, SE, NE, SF, NF, SG, NG, SH, NH, SO, NO](a: Tree[SA, NA], b: Tree[SB, NB], c: Tree[SC, NC], d: Tree[SD, ND], e: Tree[SE, NE], f: Tree[SF, NF], g: Tree[SG, NG], h: Tree[SH, NH], /, *, outputs: Tree[SO, NO], name: str | None = None) -> Callable[[Callable[[SA, SB, SC, SD, SE, SF, SG, SH], SO]], Function[tuple[SA, SB, SC, SD, SE, SF, SG, SH], tuple[NA, NB, NC, ND, NE, NF, NG, NH], SO, NO]]: ...
@overload
def function[SA, NA](a: Tree[SA, NA], /, *, name: str | None = None) -> _InferredOutputs[tuple[NA], SA]: ...
@overload
def function[SA, NA, SB, NB](a: Tree[SA, NA], b: Tree[SB, NB], /, *, name: str | None = None) -> _InferredOutputs[tuple[NA, NB], SA, SB]: ...
@overload
def function[SA, NA, SB, NB, SC, NC](a: Tree[SA, NA], b: Tree[SB, NB], c: Tree[SC, NC], /, *, name: str | None = None) -> _InferredOutputs[tuple[NA, NB, NC], SA, SB, SC]: ...
@overload
def function[SA, NA, SB, NB, SC, NC, SD, ND](a: Tree[SA, NA], b: Tree[SB, NB], c: Tree[SC, NC], d: Tree[SD, ND], /, *, name: str | None = None) -> _InferredOutputs[tuple[NA, NB, NC, ND], SA, SB, SC, SD]: ...
@overload
def function[SA, NA, SB, NB, SC, NC, SD, ND, SE, NE](a: Tree[SA, NA], b: Tree[SB, NB], c: Tree[SC, NC], d: Tree[SD, ND], e: Tree[SE, NE], /, *, name: str | None = None) -> _InferredOutputs[tuple[NA, NB, NC, ND, NE], SA, SB, SC, SD, SE]: ...
@overload
def function[SA, NA, SB, NB, SC, NC, SD, ND, SE, NE, SF, NF](a: Tree[SA, NA], b: Tree[SB, NB], c: Tree[SC, NC], d: Tree[SD, ND], e: Tree[SE, NE], f: Tree[SF, NF], /, *, name: str | None = None) -> _InferredOutputs[tuple[NA, NB, NC, ND, NE, NF], SA, SB, SC, SD, SE, SF]: ...
@overload
def function[SA, NA, SB, NB, SC, NC, SD, ND, SE, NE, SF, NF, SG, NG](a: Tree[SA, NA], b: Tree[SB, NB], c: Tree[SC, NC], d: Tree[SD, ND], e: Tree[SE, NE], f: Tree[SF, NF], g: Tree[SG, NG], /, *, name: str | None = None) -> _InferredOutputs[tuple[NA, NB, NC, ND, NE, NF, NG], SA, SB, SC, SD, SE, SF, SG]: ...
@overload
def function[SA, NA, SB, NB, SC, NC, SD, ND, SE, NE, SF, NF, SG, NG, SH, NH](a: Tree[SA, NA], b: Tree[SB, NB], c: Tree[SC, NC], d: Tree[SD, ND], e: Tree[SE, NE], f: Tree[SF, NF], g: Tree[SG, NG], h: Tree[SH, NH], /, *, name: str | None = None) -> _InferredOutputs[tuple[NA, NB, NC, ND, NE, NF, NG, NH], SA, SB, SC, SD, SE, SF, SG, SH]: ...
@overload
def function(*, name: str | None = None) -> _Bare: ...
# fmt: on
def function(
  *inputs: Tree[Any, Any], outputs: Tree[Any, Any] | None = None, name: str | None = None
) -> Callable[[Callable[..., Any]], Function[Any, Any, Any, Any]]:
  """Trace a callable with one declaration per parameter and an optional output tree."""

  def decorate(fn: Callable[..., Any]) -> Function[Any, Any, Any, Any]:
    function_name = name or getattr(fn, "__name__", "fn")
    parameters = tuple(inspect.signature(fn).parameters.values())
    if (inputs or outputs is not None) and (
      len(parameters) != len(inputs) or any(p.kind not in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) for p in parameters)
    ):
      raise TypeError(f"{function_name}: declare one tree per positional parameter, got {len(inputs)} declarations for {len(parameters)} parameters")
    return Function(function_name, fn, None if not inputs and outputs is None and parameters else parameter_list(inputs), outputs)

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


def _function_names(operation: str, args: tuple[Expr | str, ...], wrt: Expr | str | None, of: str | None) -> tuple[str | None, str | None]:
  if len(args) == 2 and wrt is None and of is None:
    function_of, function_wrt = args
  elif len(args) == 1 and wrt is not None and of is None:
    function_of, function_wrt = args[0], wrt
  elif not args:
    function_of, function_wrt = of, wrt
  else:
    raise TypeError(f"the Function form is {operation}(fn, of, wrt)")
  if (function_of is not None and not isinstance(function_of, str)) or (function_wrt is not None and not isinstance(function_wrt, str)):
    raise TypeError(f"the Function form is {operation}(fn, of, wrt)")
  return function_of, function_wrt


def _typed_result[SI, NI](result: ConcreteFunction[Any, Any, Any, Any], input_tree: Tree[SI, NI]) -> ConcreteFunction[SI, NI, Expr, Array]:
  output_tree = arg(result.output_names[0], result.outputs[0].type)
  return result._with_trees(input_tree, output_tree)


def _declared(source: Function[Any, Any, Any, Any], of: str | None, wrt: str | None) -> tuple[Tree[Any, Any] | None, Tree[Any, Any] | None]:
  inputs, outputs = source.inputs, source.outputs
  if inputs is not None and not inputs.has_holes:
    concrete = source.instantiate()
    inputs, outputs = concrete.input_tree, concrete.output_tree
  if inputs is not None and wrt is not None:
    inputs.index(wrt)
  if outputs is not None and of is not None:
    outputs.index(of)
  return inputs, outputs


def _unseeded[SI, NI](
  source: Function[SI, NI, Any, Any] | ConcreteFunction[SI, NI, Any, Any],
  name: str | None,
  of: str | None,
  wrt: str | None,
  kind: type[Jac] | type[Grad] | type[Hess] | type[SpJac] | type[SpHess],
  *,
  triangle: Triangle = "full",
) -> Function[SI, NI, Expr, Array]:
  source = Function._from_instance(source) if isinstance(source, ConcreteFunction) else source
  inputs, outputs = _declared(source, of, wrt)
  if inputs is not None:
    wrt = _unique_name(inputs.names, wrt, "wrt")
  if outputs is not None:
    of = _unique_name(outputs.names, of, "of")

  def transform(concrete: ConcreteFunction) -> ConcreteFunction:
    output = _unique_name(concrete.output_names, of, "of")
    variable = _unique_name(concrete.input_names, wrt, "wrt")
    spec = kind(output, variable, triangle=triangle) if kind is SpHess else kind(output, variable)
    instance_name = derived_name(source, concrete, name, f"{concrete.name}_{spec.output_name}")
    key = (instance_name, spec)
    if key in concrete._memo.derivatives:
      return concrete._memo.derivatives[key]
    result = concrete.factory(instance_name, list(concrete.input_names), [spec])
    concrete._memo.derivatives[key] = cast(Any, _typed_result(result, concrete.input_tree))
    return concrete._memo.derivatives[key]

  declaration = None if of is None or wrt is None else arg(kind(of, wrt).output_name)
  public_name = name or (f"{source.name}_{kind(of, wrt).output_name}" if of is not None and wrt is not None else f"{source.name}_{kind.kind}")
  return lift(source, inputs, declaration, transform, name=public_name)


def _unique_name(names: tuple[str, ...], requested: str | None, role: str) -> str:
  if requested is not None:
    if requested not in names:
      raise ValueError(f"unknown name {requested!r}; declared {names}")
    return requested
  if len(names) != 1:
    raise ValueError(f"{role} must be specified; declared {names}")
  return names[0]


@overload
def jacobian(source: Expr, wrt: Expr) -> Expr: ...


@overload
def jacobian[SI, NI, SO, NO](
  source: Function[SI, NI, SO, NO] | ConcreteFunction[SI, NI, SO, NO], of: str | None = None, wrt: str | None = None, *, name: str | None = None
) -> Function[SI, NI, Expr, Array]: ...


def jacobian(
  source: Expr | Function | ConcreteFunction,
  *args: Expr | str,
  wrt: Expr | str | None = None,
  of: str | None = None,
  name: str | None = None,
) -> Expr | Function:
  """Build a dense Jacobian for an Expr or a named Function output."""
  if isinstance(source, Expr):
    return _jacobian_expr(source, _expr_wrt("jacobian", args, wrt, of, name))
  if not isinstance(source, Function | ConcreteFunction):
    raise TypeError("jacobian source must be an Expr or Function")
  function_of, function_wrt = _function_names("jacobian", args, wrt, of)
  return _unseeded(source, name, function_of, function_wrt, Jac)


@overload
def gradient(source: Expr, wrt: Expr) -> Expr: ...


@overload
def gradient[SI, NI, SO, NO](
  source: Function[SI, NI, SO, NO] | ConcreteFunction[SI, NI, SO, NO], of: str | None = None, wrt: str | None = None, *, name: str | None = None
) -> Function[SI, NI, Expr, Array]: ...


def gradient(
  source: Expr | Function | ConcreteFunction,
  *args: Expr | str,
  wrt: Expr | str | None = None,
  of: str | None = None,
  name: str | None = None,
) -> Expr | Function:
  """Build a gradient for an Expr or a named Function output."""
  if isinstance(source, Expr):
    return _gradient_expr(source, _expr_wrt("gradient", args, wrt, of, name))
  if not isinstance(source, Function | ConcreteFunction):
    raise TypeError("gradient source must be an Expr or Function")
  function_of, function_wrt = _function_names("gradient", args, wrt, of)
  return _unseeded(source, name, function_of, function_wrt, Grad)


@overload
def hessian(source: Expr, wrt: Expr) -> Expr: ...


@overload
def hessian[SI, NI, SO, NO](
  source: Function[SI, NI, SO, NO] | ConcreteFunction[SI, NI, SO, NO], of: str | None = None, wrt: str | None = None, *, name: str | None = None
) -> Function[SI, NI, Expr, Array]: ...


def hessian(
  source: Expr | Function | ConcreteFunction,
  *args: Expr | str,
  wrt: Expr | str | None = None,
  of: str | None = None,
  name: str | None = None,
) -> Expr | Function:
  """Build a Hessian for an Expr or a named Function output."""
  if isinstance(source, Expr):
    return _hessian_expr(source, _expr_wrt("hessian", args, wrt, of, name))
  if not isinstance(source, Function | ConcreteFunction):
    raise TypeError("hessian source must be an Expr or Function")
  function_of, function_wrt = _function_names("hessian", args, wrt, of)
  return _unseeded(source, name, function_of, function_wrt, Hess)


@overload
def sparse_jacobian(source: Expr, wrt: Expr) -> SparseJacobian: ...


@overload
def sparse_jacobian[SI, NI, SO, NO](
  source: Function[SI, NI, SO, NO] | ConcreteFunction[SI, NI, SO, NO], of: str | None = None, wrt: str | None = None, *, name: str | None = None
) -> Function[SI, NI, Expr, Array]: ...


def sparse_jacobian(
  source: Expr | Function | ConcreteFunction,
  *args: Expr | str,
  wrt: Expr | str | None = None,
  of: str | None = None,
  name: str | None = None,
) -> SparseJacobian | Function:
  """Build compact nonzero Jacobian values for an Expr or a named Function output."""
  if isinstance(source, Expr):
    return _expr_sparse_jacobian(source, _expr_wrt("sparse_jacobian", args, wrt, of, name))
  if not isinstance(source, Function | ConcreteFunction):
    raise TypeError("sparse_jacobian source must be an Expr or Function")
  function_of, function_wrt = _function_names("sparse_jacobian", args, wrt, of)
  return _unseeded(source, name, function_of, function_wrt, SpJac)


@overload
def sparse_hessian(source: Expr, wrt: Expr, *, triangle: Triangle = "full") -> SparseJacobian: ...


@overload
def sparse_hessian[SI, NI, SO, NO](
  source: Function[SI, NI, SO, NO] | ConcreteFunction[SI, NI, SO, NO],
  of: str | None = None,
  wrt: str | None = None,
  *,
  name: str | None = None,
  triangle: Triangle = "full",
) -> Function[SI, NI, Expr, Array]: ...


def sparse_hessian(
  source: Expr | Function | ConcreteFunction,
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
  if not isinstance(source, Function | ConcreteFunction):
    raise TypeError("sparse_hessian source must be an Expr or Function")
  function_of, function_wrt = _function_names("sparse_hessian", args, wrt, of)
  return _unseeded(source, name, function_of, function_wrt, SpHess, triangle=triangle)


def _seeded(
  source: Function, of: str | None, wrt: str | None, name: str | None, kind: str, aux_name: str = "gamma", triangle: Triangle = "full"
) -> Function:
  source = Function._from_instance(source) if isinstance(source, ConcreteFunction) else source
  inputs, outputs = _declared(source, of, wrt)
  if inputs is not None:
    wrt = _unique_name(inputs.names, wrt, "wrt")
  if kind in ("fwd", "adj") and outputs is not None:
    of = _unique_name(outputs.names, of, "of")
  if kind == "fwd":
    seed = None if inputs is None or wrt is None else arg(f"fwd:{wrt}", inputs.decls[inputs.index(wrt)])
  elif kind == "adj":
    seed = None if outputs is None or of is None else arg(f"lam:{of}", outputs.decls[outputs.index(of)])
  else:
    seed = None if outputs is None else outputs.relabel("lam:")
  seeded = None if inputs is None or seed is None else append_parameter(inputs, seed)

  def transform(concrete: ConcreteFunction) -> ConcreteFunction:
    variable = _unique_name(concrete.input_names, wrt, "wrt")
    if kind in ("fwd", "adj"):
      output = _unique_name(concrete.output_names, of, "of")
      seed_name = f"fwd:{variable}" if kind == "fwd" else f"lam:{output}"
      seed_type = (
        concrete.inputs[concrete.input_tree.index(variable)].type if kind == "fwd" else concrete.outputs[concrete.output_tree.index(output)].type
      )
      seed_tree = arg(seed_name, seed_type)
      spec = Fwd(output, variable) if kind == "fwd" else Adj(output, variable)
      factory_inputs = [*concrete.input_names, seed_name]
      aux = None
    else:
      output_names = list(concrete.output_names)
      seed_tree = concrete.output_tree.relabel("lam:")
      spec = Hess(aux_name, variable) if kind == "hess" else SpHess(aux_name, variable, triangle=triangle)
      factory_inputs = [*concrete.input_names, *seed_tree.names]
      aux = {aux_name: output_names}
    instance_name = derived_name(source, concrete, name, f"{concrete.name}_{spec.output_name}")
    key = (instance_name, spec)
    if key not in concrete._memo.derivatives:
      result = concrete.factory(instance_name, factory_inputs, [spec], aux=aux)
      concrete._memo.derivatives[key] = _typed_result(result, append_parameter(concrete.input_tree, seed_tree))
    return concrete._memo.derivatives[key]

  spec_name = (
    None
    if wrt is None or (kind in ("fwd", "adj") and of is None)
    else (
      Fwd(cast(str, of), wrt).output_name
      if kind == "fwd"
      else Adj(cast(str, of), wrt).output_name
      if kind == "adj"
      else Hess(aux_name, wrt).output_name
      if kind == "hess"
      else SpHess(aux_name, wrt).output_name
    )
  )
  return lift(
    source,
    seeded,
    None if spec_name is None else arg(spec_name),
    transform,
    name=name or (f"{source.name}_{spec_name}" if spec_name else f"{source.name}_{kind}"),
    source_skeleton=lambda skeleton: skeleton[:-1],
  )


def forward[*Ss, *Ns, SO, NO](
  fn: Function[tuple[*Ss], tuple[*Ns], SO, NO], of: str | None = None, wrt: str | None = None, *, name: str | None = None
) -> Function[tuple[*Ss, Expr], tuple[*Ns, Array], Expr, Array]:
  """Create a forward derivative with one seed parameter shaped and typed as ``wrt``."""
  return _seeded(fn, of, wrt, name, "fwd")


def adjoint[*Ss, *Ns, SO, NO](
  fn: Function[tuple[*Ss], tuple[*Ns], SO, NO], of: str | None = None, wrt: str | None = None, *, name: str | None = None
) -> Function[tuple[*Ss, Expr], tuple[*Ns, Array], Expr, Array]:
  """Create an adjoint derivative with one seed parameter shaped and typed as ``of``."""
  return _seeded(fn, of, wrt, name, "adj")


def lagrangian_hessian[*Ss, *Ns, SO, NO](
  fn: Function[tuple[*Ss], tuple[*Ns], SO, NO], wrt: str | None = None, *, name: str | None = None, aux_name: str = "gamma"
) -> Function[tuple[*Ss, SO], tuple[*Ns, NO], Expr, Array]:
  """Create the dense weighted Hessian with one multiplier parameter matching the output tree."""
  return _seeded(fn, None, wrt, name, "hess", aux_name)


def sparse_lagrangian_hessian[*Ss, *Ns, SO, NO](
  fn: Function[tuple[*Ss], tuple[*Ns], SO, NO],
  wrt: str | None = None,
  *,
  name: str | None = None,
  aux_name: str = "gamma",
  triangle: Triangle = "full",
) -> Function[tuple[*Ss, SO], tuple[*Ns, NO], Expr, Array]:
  """Create compact weighted Hessian values with multipliers matching the output tree."""
  return _seeded(fn, None, wrt, name, "sphess", aux_name, triangle)
