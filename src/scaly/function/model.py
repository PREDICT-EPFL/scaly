"""Function declarations and their cached concrete instances."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import hashlib
from typing import Any, cast, overload

import numpy as np

from ..ir.expr import Expr
from ..ir.types import TensorType, as_shape
from .concrete import ConcreteFunction, DerivSpec
from .tree import Tree, _G, _Leaf, _leaves, parameter_list


def _structure(value: Any) -> Any:
  return tuple(_structure(part) for part in value) if isinstance(value, tuple) else "leaf"


def _skeleton(tree: Tree[Any, Any], types: tuple[TensorType, ...]) -> Any:
  return tree.unflatten(types)


def _types(skeleton: Any) -> tuple[TensorType, ...]:
  return tuple(_leaves(skeleton))


def _mangle(name: str, skeleton: Any, open_types: tuple[TensorType, ...]) -> str:
  shapes = "_".join("x".join(map(str, type_.shape)) if type_.shape else "s" for type_ in open_types)
  nesting = hashlib.sha256(repr(_structure(skeleton)).encode()).hexdigest()[:6]
  return "_".join(part for part in (name, shapes, f"t{nesting}") if part)


class Function[SymbolicInputs, NumericalInputs, SymbolicOutputs, NumericalOutputs]:
  """A Python body and declarations, traced once for each binding of its shape holes.

  Calling with symbolic leaves embeds the selected instance in a graph. Numerical leaves
  evaluate its compiled code. Each call takes one positional argument per declared parameter.
  ``instantiate`` returns the concrete graph for inspection or ahead-of-time compilation.
  """

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
    self._argument_cache: dict[Any, ConcreteFunction[Any, Any, Any, Any]] = {}
    self._instance_names: dict[str, Any] = {}
    if inputs is not None and not inputs.has_holes:
      self.instantiate()

  def _build(self, types: tuple[TensorType, ...]) -> ConcreteFunction[Any, Any, Any, Any]:
    assert self.inputs is not None
    inputs = self.inputs.with_types(types)
    open_types = tuple(type_ for decl, type_ in zip(self.inputs.decls, types, strict=True) if decl is Ellipsis)
    name = _mangle(self.name, _skeleton(inputs, types), open_types) if self.inputs.has_holes else self.name
    return ConcreteFunction(name, self._fn, inputs, self.outputs, output_name=self.name)

  def _build_bare(self, skeleton: Any) -> ConcreteFunction[Any, Any, Any, Any]:
    names = iter(f"in{i}" for i in range(len(_types(skeleton))))

    def inferred(item: Any) -> Tree[Any, Any]:
      if isinstance(item, tuple):
        return parameter_list(tuple(inferred(part) for part in item))
      return _Leaf(next(names), item)

    inputs = inferred(skeleton)
    name = _mangle(self.name, skeleton, _types(skeleton))
    return ConcreteFunction(name, self._fn, inputs, None, output_name=self.name)

  def _cache(self, key: Any, build: Callable[[], ConcreteFunction[Any, Any, Any, Any]]) -> ConcreteFunction[Any, Any, Any, Any]:
    if key not in self.instances:
      instance = build()
      if instance.name in self._instance_names and self._instance_names[instance.name] != key:
        raise ValueError(f"{self.name}: instance name collision for {instance.name!r}")
      self._instance_names[instance.name] = key
      self.instances[key] = instance
    return self.instances[key]

  def instantiate(
    self, shapes: tuple[int | tuple[int, ...] | TensorType, ...] | None = None, /
  ) -> ConcreteFunction[SymbolicInputs, NumericalInputs, SymbolicOutputs, NumericalOutputs]:
    """Bind input shape holes in flat leaf order and return the cached graph instance.

    A fully declared function needs no arguments. Bare functions must be bound by calling them.
    ``TensorType`` bindings must agree with the declaration's dtype and differentiability.
    """
    if self.inputs is None:
      raise TypeError(f"{self.name}: a bare function has no declared leaves to bind; call it instead")
    if shapes is None:
      if self.inputs.has_holes:
        raise TypeError(f"{self.name} has shape holes in {self.inputs.names}; pass shapes to bind them")
      types = self.inputs.types
    else:
      if len(shapes) != self.inputs.size:
        raise TypeError(f"{self.name}: declared {self.inputs.size} leaves, got {len(shapes)} bindings")
      types = tuple(
        shape
        if isinstance(shape, TensorType)
        else TensorType(as_shape(shape), decl.dtype, decl.diff)
        if decl is not Ellipsis
        else TensorType(as_shape(shape))
        for decl, shape in zip(self.inputs.decls, shapes, strict=True)
      )
      for decl, type_ in zip(self.inputs.decls, types, strict=True):
        expected = TensorType(type_.shape) if decl is Ellipsis else decl
        if expected != type_:
          raise TypeError(f"{self.name}: binding type {type_} contradicts declaration {expected}")
    return cast(ConcreteFunction[SymbolicInputs, NumericalInputs, SymbolicOutputs, NumericalOutputs], self._cache(types, lambda: self._build(types)))

  def _resolve(self, args: tuple[Any, ...], what: str) -> ConcreteFunction[Any, Any, Any, Any]:
    def bind(tree: Tree[Any, Any] | None, value: Any) -> Any:
      if tree is not None and isinstance(tree, _G):
        if not isinstance(value, tuple) or len(value) != len(tree.parts):
          raise ValueError(f"{what}: value does not have the declared structure of {tree.names}")
        return tuple(bind(part, item) for part, item in zip(tree.parts, value, strict=True))
      if tree is None and isinstance(value, tuple):
        return tuple(bind(None, item) for item in value)
      if tree is None:
        if not isinstance(value, (Expr, np.ndarray)):
          raise TypeError(f"{what}: bare function leaves must be Expr or ndarray; use np.asarray or declare the input tree")
        if isinstance(value, Expr) and value.type.dtype != TensorType(()).dtype:
          raise ValueError(f"{what}: bare functions require float64 symbolic leaves, got {value.type.dtype}")
        return TensorType(value.shape)
      decl = tree.decls[0]
      if isinstance(value, Expr):
        shape = value.shape
        expected_dtype = TensorType(()).dtype if decl is Ellipsis else decl.dtype
        if value.type.dtype != expected_dtype:
          raise ValueError(f"{what}: expected dtype {expected_dtype} for {tree.names[0]!r}, got {value.type.dtype}")
      else:
        shape = np.shape(value)
      if decl is not Ellipsis and shape != decl.shape:
        raise ValueError(f"{what}: expected shape {decl.shape} for {tree.names[0]!r}, got {shape}")
      return TensorType(shape) if decl is Ellipsis else decl

    skeleton = bind(self.inputs, args)
    return self._bind(skeleton, what)

  def _bind(self, skeleton: Any, what: str) -> ConcreteFunction[Any, Any, Any, Any]:
    if self.inputs is not None and _structure(self.inputs.unflatten(self.inputs.decls)) != _structure(skeleton):
      raise ValueError(f"{what}: value does not have the declared structure of {self.inputs.names}")
    if skeleton not in self._argument_cache:
      instance = (
        self._cache(skeleton, lambda: self._build_bare(skeleton))
        if self.inputs is None
        else self.instantiate(tuple(type_.shape for type_ in _types(skeleton)))
      )
      self._argument_cache[skeleton] = instance
    return self._argument_cache[skeleton]

  @overload
  def __call__[*Ns](self: Function[SymbolicInputs, tuple[*Ns], SymbolicOutputs, NumericalOutputs], *args: *Ns) -> NumericalOutputs: ...

  @overload
  def __call__[*Ss](self: Function[tuple[*Ss], NumericalInputs, SymbolicOutputs, NumericalOutputs], *args: *Ss) -> SymbolicOutputs: ...

  def __call__(self, *args: Any) -> Any:
    """Bind shapes and call the instance. An empty call evaluates numerically."""
    return self._resolve(args, self.name)(*args)

  def symbolic_call[*Ss](self: Function[tuple[*Ss], NumericalInputs, SymbolicOutputs, NumericalOutputs], *args: *Ss) -> SymbolicOutputs:
    """Embed a symbolic call, including the zero-input case."""
    return self._resolve(args, f"{self.name}.symbolic_call").symbolic_call(*args)

  def numerical_call[*Ns](self: Function[SymbolicInputs, tuple[*Ns], SymbolicOutputs, NumericalOutputs], *args: *Ns) -> NumericalOutputs:
    """Bind shapes, compile the instance, and evaluate its numerical outputs."""
    return self._resolve(args, f"{self.name}.numerical_call").numerical_call(*args)

  @classmethod
  def _from_instance(cls, instance: ConcreteFunction[Any, Any, Any, Any]) -> Function[Any, Any, Any, Any]:
    function = cls.__new__(cls)
    function.name = instance.name
    function._fn = instance.symbolic_call
    function.inputs = instance.input_tree
    function.outputs = instance.output_tree
    function.instances = {instance.input_tree.types: instance}
    function._argument_cache = {}
    function._instance_names = {instance.name: instance.input_tree.types}
    return function

  def factory(self, name: str, inputs: Sequence[str], outputs: Sequence[str | DerivSpec], aux: Any = None) -> Function:
    """Derive named graph outputs from a fully specified instance."""
    return Function._from_instance(self.instantiate().factory(name, inputs, outputs, aux))

  def compile(self) -> None:
    """Compile a fully specified function ahead of its first numerical call."""
    self.instantiate().compile()

  def recompile(self) -> None:
    """Invalidate a fully specified function's cache; its next numerical call recompiles."""
    self.instantiate().recompile()

  def with_device(self, device: Any) -> Function[Any, Any, Any, Any]:
    """Return a fully specified function with the requested device placement."""
    return Function._from_instance(self.instantiate().with_device(device))

  def solver_stats(self, name: str | None = None) -> Any:
    """Return the latest statistics for a solver in the fully specified graph."""
    return self.instantiate().solver_stats(name)


def as_concrete(function: Function[Any, Any, Any, Any] | ConcreteFunction[Any, Any, Any, Any]) -> ConcreteFunction[Any, Any, Any, Any]:
  """Resolve a fully specified declaration at a graph consumer's boundary."""
  return function.instantiate() if isinstance(function, Function) else function


class _Derived(Function[Any, Any, Any, Any]):
  def __init__(
    self,
    source: Function[Any, Any, Any, Any],
    inputs: Tree[Any, Any] | None,
    outputs: Tree[Any, Any] | None,
    transform: Callable[[ConcreteFunction[Any, Any, Any, Any]], ConcreteFunction[Any, Any, Any, Any]],
    source_skeleton: Callable[[Any], Any],
    name: str,
  ) -> None:
    self._source = source
    self._transform = transform
    self._source_skeleton = source_skeleton
    super().__init__(name, source._fn, inputs, outputs)

  def _derive(self, skeleton: Any) -> ConcreteFunction[Any, Any, Any, Any]:
    source = self._source._bind(self._source_skeleton(skeleton), self.name)
    instance = self._transform(source)
    expected = _skeleton(instance.input_tree, instance.input_tree.types)

    def compatible(left: Any, right: Any) -> bool:
      if isinstance(left, tuple):
        return isinstance(right, tuple) and len(left) == len(right) and all(compatible(a, b) for a, b in zip(left, right, strict=True))
      return isinstance(right, TensorType) and left.shape == right.shape and left.dtype == right.dtype

    if not compatible(expected, skeleton):
      raise ValueError(f"{self.name}: seed or argument types do not match the bound source instance")
    return instance

  def _build(self, types: tuple[TensorType, ...]) -> ConcreteFunction[Any, Any, Any, Any]:
    assert self.inputs is not None
    return self._derive(_skeleton(self.inputs, types))

  def _build_bare(self, skeleton: Any) -> ConcreteFunction[Any, Any, Any, Any]:
    return self._derive(skeleton)


def lift[SI, NI, SO, NO](
  source: Function[Any, Any, Any, Any],
  inputs: Tree[SI, NI] | None,
  outputs: Tree[SO, NO] | None,
  transform: Callable[[ConcreteFunction[Any, Any, Any, Any]], ConcreteFunction[Any, Any, Any, Any]],
  *,
  name: str,
  source_skeleton: Callable[[Any], Any] = lambda skeleton: skeleton,
) -> Function[SI, NI, SO, NO]:
  """Apply a concrete graph transform once for each source binding."""
  return cast(Function[SI, NI, SO, NO], _Derived(source, inputs, outputs, transform, source_skeleton, name))


def derived_name(source: Function, concrete: ConcreteFunction, requested: str | None, default: str) -> str:
  """Give explicit derivative names the source's deterministic specialization suffix."""
  if requested is None:
    return default
  if source.inputs is None:
    open_types = concrete.input_tree.types
  else:
    open_types = tuple(type_ for decl, type_ in zip(source.inputs.decls, concrete.input_tree.types, strict=True) if decl is Ellipsis)
  return (
    _mangle(requested, _skeleton(concrete.input_tree, concrete.input_tree.types), open_types) if open_types or source.inputs is None else requested
  )
