"""``Function``: a body over a declared parameter list, realized as one ``ConcreteFunction`` per binding.

The one public function type. A declaration may leave leaf shapes as holes (``L("x")``); a call binds
them to its arguments' shapes and traces the body once per distinct binding. A fully shaped
declaration has one instance, built at the decorator and named after the function, so it fails
early and keeps its C symbol. The decorator takes one tree per parameter, and the input type
parameters are the parameter lists, always tuples:
``Function[tuple[Expr, Expr], ...]`` has a body ``def f(x, p)`` and is called ``f.numerical_call(x, p)``.
A ``TypeVarTuple`` unpacks them into the body's and the calls' parameters, and appends the seed of a
seeded mode.

Transforms that need an instance (the derivative wrappers, ``vmap``) are ``lift``ed: the result is a
``Function`` whose instances are the transform of the source's.
"""

from __future__ import annotations

from collections.abc import Callable
from types import EllipsisType
from typing import Any, cast, overload

from . import concrete
from .concrete import ConcreteFunction, Instance
from .expr import Buffer, Expr, Shape, ShapeDecl, as_shape
from .trees import L, Tree, append_parameter, inferred_tree, leaves, parameter_list, same_structure, shapes_of, skeleton


def _mangle(name: str, shapes: tuple[Shape, ...]) -> str:
  return name + "__" + "_".join("x".join(map(str, s)) if s else "s" for s in shapes)


class Function[SymbolicInputs, NumericalInputs, SymbolicOutputs, NumericalOutputs]:
  """A body plus a declaration that may have shape holes, or none at all in the bare mode. Nothing
  shape-dependent lives here: ``inputs`` and ``outputs`` are the declared trees, holes included, and
  ``instantiate`` returns the ``ConcreteFunction`` that has the shapes. ``instances`` is everything
  that reaches C."""

  def __init__(
    self,
    name: str,
    fn: Callable[..., SymbolicOutputs],
    inputs: Tree[SymbolicInputs, NumericalInputs] | None,
    outputs: Tree[SymbolicOutputs, NumericalOutputs] | None,
  ) -> None:
    if (inputs is None) != (outputs is None):
      raise TypeError(f"{name}: declare outputs= with the inputs, or neither for the bare mode")
    self.name = name
    self._fn = fn
    self.inputs = inputs
    self.outputs = outputs
    self.instances: dict[Any, ConcreteFunction[SymbolicInputs, NumericalInputs, SymbolicOutputs, NumericalOutputs]] = {}
    if self.inputs is not None and not self.inputs.has_holes:
      self.instantiate()

  def _build(self, resolved: tuple[Shape, ...]) -> ConcreteFunction[SymbolicInputs, NumericalInputs, SymbolicOutputs, NumericalOutputs]:
    assert self.inputs is not None and self.outputs is not None
    name = _mangle(self.name, resolved) if self.inputs.has_holes else self.name
    return ConcreteFunction(name, self._fn, self.inputs.with_shapes(resolved), self.outputs)

  def instantiate(
    self, shapes: tuple[int | Shape, ...] | None = None, /
  ) -> ConcreteFunction[SymbolicInputs, NumericalInputs, SymbolicOutputs, NumericalOutputs]:
    """Bind the holes to ``shapes`` (flat, leaf order) and trace; cached. Without holes no shapes are
    needed and this is the one instance. Also the explicit ahead-of-time entry."""
    if self.inputs is None:
      raise TypeError(f"{self.name}: a bare function has no declared leaves to bind; call it instead")
    if shapes is None:
      if self.inputs.has_holes:
        raise TypeError(f"{self.name} has holes in {self.inputs.names}; pass the shapes to bind them")
      shapes = self.inputs.shapes
    resolved = self.inputs.resolved(tuple(as_shape(s) for s in shapes))
    got = self.instances.get(resolved)
    if got is None:
      got = self._build(resolved)
      self.instances[resolved] = got
    return got

  def _resolve(self, args: tuple[Any, ...], what: str) -> Instance:
    if self.inputs is not None:
      if not same_structure(self.inputs, args):
        raise ValueError(f"{what}: arguments do not have the declared structure of {self.inputs.names}")
      shapes = shapes_of(args)
      if any(d is not Ellipsis and d != s for d, s in zip(self.inputs.decls, shapes, strict=True)):
        raise ValueError(f"{what}: expected shapes {self.inputs.decls} for {self.inputs.names}, got {shapes}")
      return self.instantiate(shapes)
    # Bare mode: the structure is read off the arguments, so leaves must be exactly Expr or ndarray
    # and every tuple is structure. Array-likes are refused here because `(a, b)` could be two leaves
    # or one vector; the declared form disambiguates them and may coerce.
    try:
      key = skeleton(args)
    except TypeError as e:
      raise TypeError(
        f"{what}: a bare function reads its structure from the call, so every leaf must be an Expr or an ndarray ({e}); wrap array-likes in np.asarray or declare the input tree"
      ) from None
    got = self.instances.get(key)
    if got is None:
      in_tree = inferred_tree(args, (f"in{i}" for i in range(len(leaves(args)))))
      traced = self._fn(*in_tree.symbols())  # pre-trace to learn the output structure; the real thing traces once
      out_tree = inferred_tree(traced, (f"out{i}" for i in range(len(leaves(traced)))))
      got = ConcreteFunction(_mangle(self.name, shapes_of(args)), self._fn, in_tree, out_tree)
      self.instances[key] = got
    return got

  # Numerical first, for the reason `ConcreteFunction.__call__` gives.
  @overload
  def __call__[*Ns](self: Function[SymbolicInputs, tuple[*Ns], SymbolicOutputs, NumericalOutputs], *args: *Ns) -> NumericalOutputs: ...
  @overload
  def __call__[*Ss](self: Function[tuple[*Ss], NumericalInputs, SymbolicOutputs, NumericalOutputs], *args: *Ss) -> SymbolicOutputs: ...
  def __call__(self, *args: Any) -> Any:
    """Bind the instance the arguments call for and dispatch as its ``__call__`` does."""
    return self._resolve(args, self.name)(*args)

  def symbolic_call[*Ss](self: Function[tuple[*Ss], NumericalInputs, SymbolicOutputs, NumericalOutputs], *args: *Ss) -> SymbolicOutputs:
    return self._resolve(args, f"{self.name}.symbolic_call").symbolic_call(*args)

  def numerical_call[*Ns](self: Function[SymbolicInputs, tuple[*Ns], SymbolicOutputs, NumericalOutputs], *args: *Ns) -> NumericalOutputs:
    return self._resolve(args, f"{self.name}.numerical_call").numerical_call(*args)


# One overload per width: turning the slots' `Tree[S, N]`s into the two parameter lists is the type-level
# map Python lacks (README, "Why a wrapper class exists"). Bare comes last: no slots and no `outputs`.
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
def function[*Ss, SO](*, name: str | None = None) -> Callable[[Callable[[*Ss], SO]], Function[tuple[*Ss], Any, SO, Any]]: ...
# fmt: on
def function(
  *inputs: Tree[Any, Any], outputs: Tree[Any, Any] | None = None, name: str | None = None
) -> Callable[[Callable[..., Any]], Function[Any, Any, Any, Any]]:
  """Declare a function with one tree per parameter and an output tree; leaves without a shape are
  bound at each call. With nothing declared, the bare mode reads everything from the first call."""
  params = None if not inputs and outputs is None else parameter_list(inputs)

  def decorate(fn: Callable[..., Any]) -> Function[Any, Any, Any, Any]:
    return Function(name or getattr(fn, "__name__", "fn"), fn, params, outputs)

  return decorate


# --- transforms: lift a per-instance transform to a Function ---


class _Derived(Function[Any, Any, Any, Any]):
  """A function whose instances are ``transform(source instance)``. ``source_shapes`` maps a binding
  of its own leaves to the source's: a prefix for the seeded modes, one axis less for ``vmap``."""

  def __init__(
    self,
    source: Function[Any, Any, Any, Any],
    inputs: Tree[Any, Any],
    outputs: Tree[Any, Any],
    transform: Callable[[Instance], Instance],
    source_shapes: Callable[[tuple[Shape, ...]], tuple[Shape, ...]],
    *,
    name: str,
  ) -> None:
    self._source = source
    self._transform = transform
    self._source_shapes = source_shapes
    super().__init__(name, source._fn, inputs, outputs)

  def _build(self, resolved: tuple[Shape, ...]) -> Instance:
    fn = self._transform(self._source.instantiate(self._source_shapes(resolved)))
    if fn.input_shapes != resolved:
      raise ValueError(f"{self.name}: instantiated with shapes {resolved}, but the transform's inputs are {fn.input_shapes}")
    return fn


def lift[SI, NI, SO, NO](
  source: Function[Any, Any, Any, Any],
  inputs: Tree[SI, NI],
  outputs: Tree[SO, NO],
  transform: Callable[[Instance], Instance],
  *,
  name: str,
  source_shapes: Callable[[tuple[Shape, ...]], tuple[Shape, ...]] | None = None,
) -> Function[SI, NI, SO, NO]:
  """The one mechanism every transform uses: the declared trees give the static types, with holes
  where the source has them, and ``transform`` runs per instance. By default the source's leaves
  come first in ``inputs``, so its shapes are a prefix."""
  if source.inputs is None:
    raise TypeError(f"{source.name}: a bare function has no declared names to transform")
  n = source.inputs.size
  return cast(Function[SI, NI, SO, NO], _Derived(source, inputs, outputs, transform, source_shapes or (lambda s: s[:n]), name=name))


def _declared(fn: Function[Any, Any, Any, Any], of: str | None, wrt: str | None) -> tuple[Tree[Any, Any], Tree[Any, Any]]:
  """The trees a derived declaration builds on, with ``of`` and ``wrt`` checked. Names need no shapes,
  so the check is immediate. Without holes these are the instance's trees, so the derived function
  has none either and is built now, as its source was."""
  if fn.inputs is None or fn.outputs is None:
    raise TypeError(f"{fn.name}: a bare function has no declared names to transform")
  if wrt is not None:
    fn.inputs.index(wrt)
  if of is not None:
    fn.outputs.index(of)
  if fn.inputs.has_holes:
    return fn.inputs, fn.outputs
  instance = fn.instantiate()
  return instance.inputs, instance.outputs


def _decl(tree: Tree[Any, Any], name: str) -> ShapeDecl:
  return tree.decls[tree.index(name)]


# --- derivatives: same parameters, one output; seeded modes append the seed as one more parameter ---


def gradient[SI, NI](fn: Function[SI, NI, Any, Any], of: str, wrt: str, /) -> Function[SI, NI, Expr, Buffer]:
  """``d of / d wrt`` as a function of every parameter of ``fn``. Needs a scalar ``of``."""
  inputs, _ = _declared(fn, of, wrt)
  return lift(fn, inputs, L(f"grad_{of}_{wrt}"), lambda f: concrete.gradient(f, of, wrt), name=f"{fn.name}_grad_{of}_{wrt}")


def jacobian[SI, NI](fn: Function[SI, NI, Any, Any], of: str, wrt: str, /) -> Function[SI, NI, Expr, Buffer]:
  inputs, _ = _declared(fn, of, wrt)
  return lift(fn, inputs, L(f"jac_{of}_{wrt}"), lambda f: concrete.jacobian(f, of, wrt), name=f"{fn.name}_jac_{of}_{wrt}")


def hessian[SI, NI](fn: Function[SI, NI, Any, Any], of: str, wrt: str, /) -> Function[SI, NI, Expr, Buffer]:
  inputs, _ = _declared(fn, of, wrt)
  return lift(fn, inputs, L(f"hess_{of}_{wrt}_{wrt}"), lambda f: concrete.hessian(f, of, wrt), name=f"{fn.name}_hess_{of}_{wrt}_{wrt}")


def forward[*Ss, *Ns](
  fn: Function[tuple[*Ss], tuple[*Ns], Any, Any], of: str, wrt: str, /
) -> Function[tuple[*Ss, Expr], tuple[*Ns, Buffer], Expr, Buffer]:
  """``J(of, wrt) @ seed``, called as ``fwd.numerical_call(*inputs, seed)``. The seed is shaped as ``wrt``."""
  inputs, _ = _declared(fn, of, wrt)
  seeded = append_parameter(inputs, L(f"fwd:{wrt}", _decl(inputs, wrt)))
  return lift(fn, seeded, L(f"fwd_{of}_{wrt}"), lambda f: concrete.forward(f, of, wrt), name=f"{fn.name}_fwd_{of}_{wrt}")


def adjoint[*Ss, *Ns](
  fn: Function[tuple[*Ss], tuple[*Ns], Any, Any], of: str, wrt: str, /
) -> Function[tuple[*Ss, Expr], tuple[*Ns, Buffer], Expr, Buffer]:
  """``J(of, wrt).T @ lam``, called as ``adj.numerical_call(*inputs, lam)``. The seed is shaped as ``of``."""
  inputs, outputs = _declared(fn, of, wrt)
  seeded = append_parameter(inputs, L(f"lam:{of}", _decl(outputs, of)))
  return lift(fn, seeded, L(f"adj_{of}_{wrt}"), lambda f: concrete.adjoint(f, of, wrt), name=f"{fn.name}_adj_{of}_{wrt}")


def lagrangian_hessian[*Ss, *Ns, SO, NO](
  fn: Function[tuple[*Ss], tuple[*Ns], SO, NO], wrt: str, /
) -> Function[tuple[*Ss, SO], tuple[*Ns, NO], Expr, Buffer]:
  """Hessian of ``sum_i lam_i . out_i``: the multipliers are one more parameter, the output tree relabelled."""
  inputs, outputs = _declared(fn, None, wrt)
  seeded = append_parameter(inputs, outputs.relabel("lam:"))
  return lift(fn, seeded, L(f"hess_lagrangian_{wrt}"), lambda f: concrete.lagrangian_hessian(f, wrt), name=f"{fn.name}_hess_lagrangian_{wrt}")


# --- vmap: a candidate, not a settled design (README, "Open items") ---


def _batched[S, N](tree: Tree[S, N], length: int) -> Tree[S, N]:
  return tree.with_shapes(tuple(d if isinstance(d, EllipsisType) else (length, *d) for d in tree.decls))


def vmap[SI, NI, SO, NO](fn: Function[SI, NI, SO, NO], length: int, /) -> Function[SI, NI, SO, NO]:
  """Map ``fn`` over a leading axis of ``length``: every leaf of both trees gains that axis and the
  structure is preserved, so the result is typed by ``fn``'s trees. Holes stay holes; a binding
  strips the axis to find the per-iteration shapes. The real ``sc.vmap`` slices flat outer tensors
  and returns a bare ``Expr``; this is the shape of what could replace it."""
  inputs, outputs = _declared(fn, None, None)
  name = f"{fn.name}_vmap{length}"

  def per_iteration(shapes: tuple[Shape, ...]) -> tuple[Shape, ...]:
    if any(not s or s[0] != length for s in shapes):
      raise ValueError(f"{name}: every argument needs a leading axis of {length}, got shapes {shapes}")
    return tuple(s[1:] for s in shapes)

  return lift(fn, _batched(inputs, length), _batched(outputs, length), lambda f: concrete.vmap(f, length), name=name, source_shapes=per_iteration)
