"""Function templates: a declaration with shape holes, instantiated to a ``Function`` per shape combination.

Purely additive to ``Function``: a template owns a body, a declaration (or none, in the bare mode)
and a cache of concrete instances keyed by leaf shapes. Transforms over templates are ``lift``ed
transforms over their instances, so each wrapper in ``function.py`` gains one overload here.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast, overload

from .expr import Buffer, Expr, Shape, as_shape
from .function import Function
from .function import forward as _forward
from .function import gradient as _gradient
from .function import hessian as _hessian
from .function import lagrangian_hessian as _lagrangian_hessian
from .trees import G, L, Tree, inferred_tree, leaves, same_structure, shapes_of, skeleton


def _mangle(name: str, shapes: tuple[Shape, ...]) -> str:
  return name + "__" + "_".join("x".join(map(str, s)) if s else "s" for s in shapes)


class FunctionTemplate[SymbolicInputs, NumericalInputs, SymbolicOutputs, NumericalOutputs]:
  """A body plus a declaration that may have shape holes. ``instances`` is the AOT surface:
  everything in it reaches C. A fully shaped declaration instantiates at the decorator, so it keeps
  today's fail-early behavior; holes are bound at the first call that supplies shapes, numerical or
  symbolic inside another trace."""

  def __init__(
    self,
    name: str,
    fn: Callable[[SymbolicInputs], SymbolicOutputs],
    inputs: Tree[SymbolicInputs, NumericalInputs] | None,
    outputs: Tree[SymbolicOutputs, NumericalOutputs] | None,
  ) -> None:
    if (inputs is None) != (outputs is None):
      raise TypeError("declare both trees or neither")
    self.name = name
    self._fn = fn
    self.inputs = inputs
    self.outputs = outputs
    self.instances: dict[Any, Function[SymbolicInputs, NumericalInputs, SymbolicOutputs, NumericalOutputs]] = {}
    if inputs is not None and not inputs.has_holes:
      self.instantiate(inputs.shapes)

  def _build(self, resolved: tuple[Shape, ...]) -> Function[SymbolicInputs, NumericalInputs, SymbolicOutputs, NumericalOutputs]:
    assert self.inputs is not None and self.outputs is not None
    return Function(_mangle(self.name, resolved), self._fn, self.inputs.with_shapes(resolved), self.outputs)

  def instantiate(self, shapes: tuple[int | Shape, ...], /) -> Function[SymbolicInputs, NumericalInputs, SymbolicOutputs, NumericalOutputs]:
    """Bind the holes to ``shapes`` (flat, leaf order) and trace; cached. Also the explicit AOT entry."""
    if self.inputs is None:
      raise TypeError(f"{self.name}: a bare template has no declared leaves to bind; call it instead")
    resolved = self.inputs.resolved(tuple(as_shape(s) for s in shapes))
    got = self.instances.get(resolved)
    if got is None:
      got = self._build(resolved)
      self.instances[resolved] = got
    return got

  def _resolve(self, value: object, what: str) -> Function[SymbolicInputs, NumericalInputs, SymbolicOutputs, NumericalOutputs]:
    if self.inputs is not None:
      if not same_structure(self.inputs, value):
        raise ValueError(f"{what}: value does not have the declared structure of {self.inputs.names}")
      return self.instantiate(shapes_of(value))
    # Bare mode: the structure is read off the value, so leaves must be exactly Expr or ndarray and
    # every tuple is structure. Array-likes are refused here because `(a, b)` could be two leaves or
    # one vector; the declared form disambiguates them and may coerce.
    try:
      key = skeleton(value)
    except TypeError as e:
      raise TypeError(
        f"{what}: a bare template reads its structure from the call, so every leaf must be an Expr or an ndarray ({e}); wrap array-likes in np.asarray or declare the input tree"
      ) from None
    got = self.instances.get(key)
    if got is None:
      fn = cast(Callable[[Any], Any], self._fn)
      in_tree = inferred_tree(value, (f"in{i}" for i in range(len(leaves(value)))))
      traced = fn(in_tree.symbols())  # pre-trace to learn the output structure; the real thing traces once
      out_tree = inferred_tree(traced, (f"out{i}" for i in range(len(leaves(traced)))))
      got = cast(
        Function[SymbolicInputs, NumericalInputs, SymbolicOutputs, NumericalOutputs],
        Function(_mangle(self.name, shapes_of(value)), fn, in_tree, out_tree),
      )
      self.instances[key] = got
    return got

  def symbolic_call(self, inputs: SymbolicInputs, /) -> SymbolicOutputs:
    return self._resolve(inputs, f"{self.name}.symbolic_call").symbolic_call(inputs)

  def numerical_call(self, inputs: NumericalInputs, /) -> NumericalOutputs:
    return self._resolve(inputs, f"{self.name}.numerical_call").numerical_call(inputs)


@overload
def template[SI, NI, SO, NO](
  inputs: Tree[SI, NI], outputs: Tree[SO, NO], /, *, name: str | None = None
) -> Callable[[Callable[[SI], SO]], FunctionTemplate[SI, NI, SO, NO]]: ...


@overload
def template(*, name: str | None = None) -> Callable[[Callable[[Any], Any]], FunctionTemplate[Any, Any, Any, Any]]: ...


def template(
  inputs: Tree[Any, Any] | None = None, outputs: Tree[Any, Any] | None = None, /, *, name: str | None = None
) -> Callable[[Callable[[Any], Any]], FunctionTemplate[Any, Any, Any, Any]]:
  """Declare a template; leave shapes (or, bare, the whole declaration) to the call sites."""

  def decorate(fn: Callable[[Any], Any]) -> FunctionTemplate[Any, Any, Any, Any]:
    return FunctionTemplate(name or getattr(fn, "__name__", "fn"), fn, inputs, outputs)

  return decorate


# --- transforms over templates: lift a Function -> Function transform to instances ---


class _Derived(FunctionTemplate[Any, Any, Any, Any]):
  """A template whose instances are ``transform(source instance)``. Its input tree may extend the
  source's (seeded modes); the source's leaves come first, so its shapes are a prefix."""

  def __init__(
    self,
    source: FunctionTemplate[Any, Any, Any, Any],
    inputs: Tree[Any, Any],
    outputs: Tree[Any, Any],
    transform: Callable[[Function[Any, Any, Any, Any]], Function[Any, Any, Any, Any]],
    *,
    name: str,
  ) -> None:
    self._source = source
    self._transform = transform
    super().__init__(name, source._fn, inputs, outputs)

  def _build(self, resolved: tuple[Shape, ...]) -> Function[Any, Any, Any, Any]:
    assert self._source.inputs is not None
    fn = self._transform(self._source.instantiate(resolved[: self._source.inputs.size]))
    if fn.input_shapes != resolved:
      raise ValueError(f"{self.name}: instantiated with shapes {resolved}, but the transform's inputs are {fn.input_shapes}")
    return fn


def lift[SI, NI, SO, NO](
  source: FunctionTemplate[Any, Any, Any, Any],
  inputs: Tree[SI, NI],
  outputs: Tree[SO, NO],
  transform: Callable[[Function[Any, Any, Any, Any]], Function[Any, Any, Any, Any]],
  *,
  name: str,
) -> FunctionTemplate[SI, NI, SO, NO]:
  """The one mechanism every transform over templates uses: the declared trees give the static
  types (with holes where the source has them), ``transform`` runs per instantiation."""
  if source.inputs is None:
    raise TypeError(f"{source.name}: a bare template has no declared names to transform")
  return cast(FunctionTemplate[SI, NI, SO, NO], _Derived(source, inputs, outputs, transform, name=name))


def _declared(t: FunctionTemplate[Any, Any, Any, Any], of: str | None, wrt: str) -> tuple[Tree[Any, Any], Tree[Any, Any]]:
  """Name checks need no shapes, so they happen when the derivative is built, as for Functions."""
  if t.inputs is None or t.outputs is None:
    raise TypeError(f"{t.name}: a bare template has no declared names to differentiate")
  t.inputs.index(wrt)
  if of is not None:
    t.outputs.index(of)
  return t.inputs, t.outputs


# fmt: off
@overload
def gradient[SI, NI](fn: Function[SI, NI, Any, Any], of: str, wrt: str, /) -> Function[SI, NI, Expr, Buffer]: ...
@overload
def gradient[SI, NI](fn: FunctionTemplate[SI, NI, Any, Any], of: str, wrt: str, /) -> FunctionTemplate[SI, NI, Expr, Buffer]: ...
# fmt: on
def gradient(fn: Function[Any, Any, Any, Any] | FunctionTemplate[Any, Any, Any, Any], of: str, wrt: str, /) -> Any:
  if isinstance(fn, Function):
    return _gradient(fn, of, wrt)
  inputs, _ = _declared(fn, of, wrt)
  return lift(fn, inputs, L(f"grad_{of}_{wrt}"), lambda f: _gradient(f, of, wrt), name=f"{fn.name}_grad_{of}_{wrt}")


# fmt: off
@overload
def hessian[SI, NI](fn: Function[SI, NI, Any, Any], of: str, wrt: str, /) -> Function[SI, NI, Expr, Buffer]: ...
@overload
def hessian[SI, NI](fn: FunctionTemplate[SI, NI, Any, Any], of: str, wrt: str, /) -> FunctionTemplate[SI, NI, Expr, Buffer]: ...
# fmt: on
def hessian(fn: Function[Any, Any, Any, Any] | FunctionTemplate[Any, Any, Any, Any], of: str, wrt: str, /) -> Any:
  if isinstance(fn, Function):
    return _hessian(fn, of, wrt)
  inputs, _ = _declared(fn, of, wrt)
  return lift(fn, inputs, L(f"hess_{of}_{wrt}_{wrt}"), lambda f: _hessian(f, of, wrt), name=f"{fn.name}_hess_{of}_{wrt}_{wrt}")


# fmt: off
@overload
def forward[SI, NI](fn: Function[SI, NI, Any, Any], of: str, wrt: str, /) -> Function[tuple[SI, Expr], tuple[NI, Buffer], Expr, Buffer]: ...
@overload
def forward[SI, NI](fn: FunctionTemplate[SI, NI, Any, Any], of: str, wrt: str, /) -> FunctionTemplate[tuple[SI, Expr], tuple[NI, Buffer], Expr, Buffer]: ...
# fmt: on
def forward(fn: Function[Any, Any, Any, Any] | FunctionTemplate[Any, Any, Any, Any], of: str, wrt: str, /) -> Any:
  """Seeded: the seed leaf is a hole too, bound with the rest at instantiation."""
  if isinstance(fn, Function):
    return _forward(fn, of, wrt)
  inputs, _ = _declared(fn, of, wrt)
  return lift(fn, G(inputs, L(f"fwd:{wrt}")), L(f"fwd_{of}_{wrt}"), lambda f: _forward(f, of, wrt), name=f"{fn.name}_fwd_{of}_{wrt}")


# fmt: off
@overload
def lagrangian_hessian[SI, NI, SO, NO](fn: Function[SI, NI, SO, NO], wrt: str, /) -> Function[tuple[SI, SO], tuple[NI, NO], Expr, Buffer]: ...
@overload
def lagrangian_hessian[SI, NI, SO, NO](fn: FunctionTemplate[SI, NI, SO, NO], wrt: str, /) -> FunctionTemplate[tuple[SI, SO], tuple[NI, NO], Expr, Buffer]: ...
# fmt: on
def lagrangian_hessian(fn: Function[Any, Any, Any, Any] | FunctionTemplate[Any, Any, Any, Any], wrt: str, /) -> Any:
  """The multipliers are the source's output tree relabelled; its holes stay holes."""
  if isinstance(fn, Function):
    return _lagrangian_hessian(fn, wrt)
  inputs, outputs = _declared(fn, None, wrt)
  return lift(
    fn,
    G(inputs, outputs.relabel("lam:")),
    L(f"hess_lagrangian_{wrt}"),
    lambda f: _lagrangian_hessian(f, wrt),
    name=f"{fn.name}_hess_lagrangian_{wrt}",
  )
