"""``Function``: a body over a declared parameter list, realized as one ``ConcreteFunction`` per binding.

The one public function type. A declaration may leave leaf shapes as holes (``arg("x")``); a call binds
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
from typing import Any, Protocol, cast, overload

from . import concrete
from .concrete import ConcreteFunction, Instance
from .expr import Buffer, Expr, Shape, ShapeDecl, as_shape
from .trees import Tree, append_parameter, arg, inferred_tree, map_skeleton, parameter_list, same_structure, skeleton, skeleton_of, skeleton_shapes


def _mangle(name: str, shapes: tuple[Shape, ...]) -> str:
  return name + "__" + "_".join("x".join(map(str, s)) if s else "s" for s in shapes)


class Function[SymbolicInputs, NumericalInputs, SymbolicOutputs, NumericalOutputs]:
  """A body plus a declaration that may have shape holes. Nothing shape-dependent lives here:
  ``inputs`` and ``outputs`` are the declared trees, holes included, and ``None`` where the structure
  is read instead, off the call for ``inputs`` (the bare mode) and off the trace for ``outputs``
  (no ``outputs=``). ``instantiate`` returns the ``ConcreteFunction`` that has the shapes.
  ``instances`` is everything that reaches C."""

  def __init__(
    self,
    name: str,
    fn: Callable[..., SymbolicOutputs],
    inputs: Tree[SymbolicInputs, NumericalInputs] | None,
    outputs: Tree[SymbolicOutputs, NumericalOutputs] | None,
  ) -> None:
    self.name = name
    self._fn = fn
    self.inputs = inputs
    self.outputs = outputs
    self.instances: dict[Any, ConcreteFunction[SymbolicInputs, NumericalInputs, SymbolicOutputs, NumericalOutputs]] = {}
    if self.inputs is not None and not self.inputs.has_holes:
      self.instantiate()

  def _build(self, resolved: tuple[Shape, ...]) -> Instance:
    assert self.inputs is not None
    name = _mangle(self.name, resolved) if self.inputs.has_holes else self.name
    return ConcreteFunction(name, self._fn, self.inputs.with_shapes(resolved), self.outputs, self.name)

  def _build_bare(self, skel: Any) -> Instance:
    shapes = skeleton_shapes(skel)
    inputs = inferred_tree(skel, (f"in{i}" for i in range(len(shapes))))
    return ConcreteFunction(_mangle(self.name, shapes), self._fn, inputs, None, self.name)

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

  def _bind(self, skel: Any, what: str) -> Instance:
    """The instance for a call skeleton: checked against the declared inputs, or, in the bare mode,
    cached under the skeleton and read off it."""
    if self.inputs is None:
      got = self.instances.get(skel)
      if got is None:
        got = self._build_bare(skel)
        self.instances[skel] = got
      return got
    if not same_structure(self.inputs, skel):
      raise ValueError(f"{what}: arguments do not have the declared structure of {self.inputs.names}")
    shapes = skeleton_shapes(skel)
    if any(d is not Ellipsis and d != s for d, s in zip(self.inputs.decls, shapes, strict=True)):
      raise ValueError(f"{what}: expected shapes {self.inputs.decls} for {self.inputs.names}, got {shapes}")
    return self.instantiate(shapes)

  def _resolve(self, args: tuple[Any, ...], what: str) -> Instance:
    # The bare mode reads its structure from the call, so leaves must be exactly Expr or ndarray and
    # every tuple is structure. Array-likes are refused because `(a, b)` could be two leaves or one
    # vector; the declared form disambiguates them and may coerce.
    try:
      skel = skeleton(args)
    except TypeError as e:
      if self.inputs is not None:
        raise ValueError(f"{what}: arguments do not have the declared structure of {self.inputs.names}") from None
      raise TypeError(
        f"{what}: a bare function reads its structure from the call, so every leaf must be an Expr or an ndarray ({e}); wrap array-likes in np.asarray or declare the input tree"
      ) from None
    return self._bind(skel, what)

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


class _InferredOutputs[NI, *Ss](Protocol):
  """What ``@function(...)`` without ``outputs=`` returns. The symbolic output type is the body's;
  the numerical one is ``Buffer`` for an ``Expr`` body and ``Any`` otherwise, since no map from one
  to the other exists (README, "Why a wrapper class exists")."""

  @overload
  def __call__(self, fn: Callable[[*Ss], Expr], /) -> Function[tuple[*Ss], NI, Expr, Buffer]: ...
  @overload
  def __call__[SO](self, fn: Callable[[*Ss], SO], /) -> Function[tuple[*Ss], NI, SO, Any]: ...


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
def function[*Ss, SO](*, name: str | None = None) -> Callable[[Callable[[*Ss], SO]], Function[tuple[*Ss], Any, SO, Any]]: ...
# fmt: on
def function(
  *inputs: Tree[Any, Any], outputs: Tree[Any, Any] | None = None, name: str | None = None
) -> Callable[[Callable[..., Any]], Function[Any, Any, Any, Any]]:
  """Declare a function with one tree per parameter and, optionally, an output tree; leaves without a
  shape are bound at each call, and a missing output tree is read off each trace. With nothing
  declared, the bare mode also reads the inputs off each call."""
  params = None if not inputs and outputs is None else parameter_list(inputs)

  def decorate(fn: Callable[..., Any]) -> Function[Any, Any, Any, Any]:
    return Function(name or getattr(fn, "__name__", "fn"), fn, params, outputs)

  return decorate


# --- transforms: lift a per-instance transform to a Function ---


class _Derived(Function[Any, Any, Any, Any]):
  """A function whose instances are ``transform(source instance)``. ``source_skeleton`` maps a call
  skeleton of it to the source's: the identity, the seed dropped for the seeded modes, one axis less
  for ``vmap``. Its trees are ``None`` where the source's structure is only known at binding, and it
  then binds from the call as the bare mode does."""

  def __init__(
    self,
    source: Function[Any, Any, Any, Any],
    inputs: Tree[Any, Any] | None,
    outputs: Tree[Any, Any] | None,
    transform: Callable[[Instance], Instance],
    source_skeleton: Callable[[Any], Any],
    *,
    name: str,
  ) -> None:
    self._source = source
    self._transform = transform
    self._source_skeleton = source_skeleton
    super().__init__(name, source._fn, inputs, outputs)

  def _derive(self, skel: Any) -> Instance:
    fn = self._transform(self._source._bind(self._source_skeleton(skel), self.name))
    if skeleton_of(fn.inputs, fn.input_shapes) != skel:
      raise ValueError(f"{self.name}: bound with shapes {skeleton_shapes(skel)}, but the transform's inputs are {fn.input_shapes}")
    return fn

  def _build(self, resolved: tuple[Shape, ...]) -> Instance:
    assert self.inputs is not None
    return self._derive(skeleton_of(self.inputs, resolved))

  def _build_bare(self, skel: Any) -> Instance:
    return self._derive(skel)


def lift[SI, NI, SO, NO](
  source: Function[Any, Any, Any, Any],
  inputs: Tree[SI, NI] | None,
  outputs: Tree[SO, NO] | None,
  transform: Callable[[Instance], Instance],
  *,
  name: str,
  source_skeleton: Callable[[Any], Any] = lambda skel: skel,
) -> Function[SI, NI, SO, NO]:
  """The one mechanism every transform uses: the declared trees give the static types, with holes
  where the source has them and ``None`` where its structure is only known at binding, and
  ``transform`` runs per instance."""
  return cast(Function[SI, NI, SO, NO], _Derived(source, inputs, outputs, transform, source_skeleton, name=name))


def _without_seed(skel: Any) -> Any:
  return skel[:-1]


def _declared(fn: Function[Any, Any, Any, Any], of: str | None, wrt: str | None) -> tuple[Tree[Any, Any] | None, Tree[Any, Any] | None]:
  """The trees a derived declaration builds on, ``None`` where the source reads them at binding.
  Declared names are checked now and the others by the transform at binding. Without holes these are
  the instance's trees, so the derived function has none either and is built now, as its source was."""
  if fn.inputs is not None and wrt is not None:
    fn.inputs.index(wrt)
  if fn.inputs is None or fn.inputs.has_holes:
    if fn.outputs is not None and of is not None:
      fn.outputs.index(of)
    return fn.inputs, fn.outputs
  instance = fn.instantiate()
  if of is not None:
    instance.outputs.index(of)
  return instance.inputs, instance.outputs


def _decl(tree: Tree[Any, Any], name: str) -> ShapeDecl:
  return tree.decls[tree.index(name)]


# --- derivatives: same parameters, one output; seeded modes append the seed as one more parameter ---


def gradient[SI, NI](fn: Function[SI, NI, Any, Any], of: str, wrt: str, /) -> Function[SI, NI, Expr, Buffer]:
  """``d of / d wrt`` as a function of every parameter of ``fn``. Needs a scalar ``of``."""
  inputs, _ = _declared(fn, of, wrt)
  return lift(fn, inputs, arg(f"grad_{of}_{wrt}"), lambda f: concrete.gradient(f, of, wrt), name=f"{fn.name}_grad_{of}_{wrt}")


def jacobian[SI, NI](fn: Function[SI, NI, Any, Any], of: str, wrt: str, /) -> Function[SI, NI, Expr, Buffer]:
  inputs, _ = _declared(fn, of, wrt)
  return lift(fn, inputs, arg(f"jac_{of}_{wrt}"), lambda f: concrete.jacobian(f, of, wrt), name=f"{fn.name}_jac_{of}_{wrt}")


def hessian[SI, NI](fn: Function[SI, NI, Any, Any], of: str, wrt: str, /) -> Function[SI, NI, Expr, Buffer]:
  inputs, _ = _declared(fn, of, wrt)
  return lift(fn, inputs, arg(f"hess_{of}_{wrt}_{wrt}"), lambda f: concrete.hessian(f, of, wrt), name=f"{fn.name}_hess_{of}_{wrt}_{wrt}")


def forward[*Ss, *Ns](
  fn: Function[tuple[*Ss], tuple[*Ns], Any, Any], of: str, wrt: str, /
) -> Function[tuple[*Ss, Expr], tuple[*Ns, Buffer], Expr, Buffer]:
  """``J(of, wrt) @ seed``, called as ``fwd.numerical_call(*inputs, seed)``. The seed is shaped as ``wrt``."""
  inputs, _ = _declared(fn, of, wrt)
  seeded = None if inputs is None else append_parameter(inputs, arg(f"fwd:{wrt}", _decl(inputs, wrt)))
  return lift(
    fn, seeded, arg(f"fwd_{of}_{wrt}"), lambda f: concrete.forward(f, of, wrt), name=f"{fn.name}_fwd_{of}_{wrt}", source_skeleton=_without_seed
  )


def adjoint[*Ss, *Ns](
  fn: Function[tuple[*Ss], tuple[*Ns], Any, Any], of: str, wrt: str, /
) -> Function[tuple[*Ss, Expr], tuple[*Ns, Buffer], Expr, Buffer]:
  """``J(of, wrt).T @ lam``, called as ``adj.numerical_call(*inputs, lam)``. The seed is shaped as ``of``."""
  inputs, outputs = _declared(fn, of, wrt)
  seed = arg(f"lam:{of}", ... if outputs is None else _decl(outputs, of))
  seeded = None if inputs is None else append_parameter(inputs, seed)
  return lift(
    fn, seeded, arg(f"adj_{of}_{wrt}"), lambda f: concrete.adjoint(f, of, wrt), name=f"{fn.name}_adj_{of}_{wrt}", source_skeleton=_without_seed
  )


def lagrangian_hessian[*Ss, *Ns, SO, NO](
  fn: Function[tuple[*Ss], tuple[*Ns], SO, NO], wrt: str, /
) -> Function[tuple[*Ss, SO], tuple[*Ns, NO], Expr, Buffer]:
  """Hessian of ``sum_i lam_i . out_i``: the multipliers are one more parameter, the output tree relabelled."""
  inputs, outputs = _declared(fn, None, wrt)
  seeded = None if inputs is None or outputs is None else append_parameter(inputs, outputs.relabel("lam:"))
  return lift(
    fn,
    seeded,
    arg(f"hess_lagrangian_{wrt}"),
    lambda f: concrete.lagrangian_hessian(f, wrt),
    name=f"{fn.name}_hess_lagrangian_{wrt}",
    source_skeleton=_without_seed,
  )


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

  def per_iteration(shape: Shape) -> Shape:
    if not shape or shape[0] != length:
      raise ValueError(f"{name}: every argument needs a leading axis of {length}, got shape {shape}")
    return shape[1:]

  return lift(
    fn,
    None if inputs is None else _batched(inputs, length),
    None if outputs is None else _batched(outputs, length),
    lambda f: concrete.vmap(f, length),
    name=name,
    source_skeleton=lambda skel: map_skeleton(skel, per_iteration),
  )
