"""Typed mapped callables, argument markers, affine views, and the internal VMAP builder."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast, overload

import numpy as np

from ..ir.expr import Expr, ExprOp, as_expr, common_lowering
from ..ir.types import TensorType
from .model import Function, as_concrete, lift
from .tree import Tree, _G, _Leaf, _leaves
from .concrete import ConcreteFunction


def _mapped_call(callee: Any, length: int, inputs: Any, output: int = 0) -> Expr:
  """Create an ``ExprOp.VMAP`` node: ``length`` independent calls of ``callee`` whose i-th argument list is
  sliced out of outer tensors with per-input ``(start, stride)`` strides.

  ``inputs`` is a sequence ordered to match ``callee.inputs``, or a mapping from callee input name
  to the same entries. An entry is usually a bare outer tensor: one of size ``length * formal.size``
  is cut into ``length`` contiguous chunks, one of size ``formal.size`` is broadcast to every
  iteration. The explicit ``(outer_tensor, start, stride)`` tuple covers overlapping or offset
  windows: the i-th iteration reads ``outer[start + i*stride : start + i*stride + formal.size]``,
  and ``stride=0`` broadcasts the same slice every iteration.

  Only the outer tensors must be rank-1. Callee formals and outputs may be rank-2 (as well as scalar
  or rank-1). Each iteration reads a flat slice of ``formal.size`` values and the produced node has
  shape ``(length * callee.outputs[output].size,)``, with iteration outputs concatenated flat.
  """
  if not isinstance(callee, Function | ConcreteFunction):
    raise TypeError(f"vmap callee must be an scaly Function, got {type(callee).__name__}")
  callee = as_concrete(callee)
  length = int(length)
  if length < 0:
    raise ValueError(f"vmap length must be non-negative, got {length}")
  if not 0 <= output < len(callee.outputs):
    raise ValueError(f"vmap output index {output} out of range for callee with {len(callee.outputs)} outputs")
  out_expr = callee.outputs[output]

  if isinstance(inputs, Mapping):
    extra = [n for n in inputs if n not in callee.input_names]
    if extra:
      raise ValueError(f"vmap inputs reference unknown callee input names {extra}; callee accepts {list(callee.input_names)}")
    missing = [n for n in callee.input_names if n not in inputs]
    if missing:
      raise ValueError(f"vmap inputs missing entries for callee inputs {missing}")
    specs = tuple(inputs[n] for n in callee.input_names)
  else:
    specs = tuple(inputs)
    if len(specs) != len(callee.inputs):
      raise ValueError(f"vmap expects {len(callee.inputs)} input specs, got {len(specs)}")
  outers: list[Expr] = []
  starts: list[int] = []
  strides: list[int] = []
  for i, spec in enumerate(specs):
    formal = callee.inputs[i]
    if isinstance(spec, tuple):
      outer, start, stride = spec
      outer = as_expr(outer)
    else:
      outer = as_expr(spec)
      start = 0
      if outer.size == length * formal.size:
        stride = formal.size
      elif outer.size == formal.size:
        stride = 0
      else:
        raise ValueError(
          f"vmap input {callee.input_names[i]!r} has size {outer.size}; expected {length * formal.size} "
          f"({length} chunks of {formal.size}) or {formal.size} (broadcast)"
        )
    if len(outer.shape) != 1:
      raise NotImplementedError(f"vmap currently requires rank-1 outer tensors, got {outer.shape} for input {i}")
    start = int(start)
    stride = int(stride)
    if start < 0:
      raise ValueError(f"vmap input {i} start must be non-negative, got {start}")
    if stride < 0:
      raise ValueError(f"vmap input {i} stride must be non-negative, got {stride}")
    if length > 0:
      end = start + (length - 1) * stride + formal.size
      if end > outer.size:
        raise ValueError(
          f"vmap input {i} reads past outer tensor of size {outer.size}: start={start}, stride={stride}, length={length}, slice_size={formal.size}"
        )
    outers.append(outer)
    starts.append(start)
    strides.append(stride)

  diff = out_expr.type.diff and any(o.type.diff for o in outers)
  return Expr(
    ExprOp.VMAP,
    tuple(outers),
    TensorType((length * out_expr.size,), out_expr.type.dtype, diff=diff),
    attrs={
      "callee": callee,
      "output": int(output),
      "length": length,
      "starts": tuple(starts),
      "strides": tuple(strides),
      "slice_size": int(out_expr.size),
    },
    lowering=common_lowering(*outers) if outers else "auto",
  )


@dataclass(frozen=True)
class _Broadcast:
  value: Any


@dataclass(frozen=True)
class _Window:
  value: Any
  start: int
  stride: int


def broadcast[T](value: T, /) -> T:
  """Mark one tensor argument to be passed whole to every mapped iteration."""
  return cast(T, _Broadcast(value))


def window[T](value: T, start: int, stride: int, /) -> T:
  """Mark rank-1 windows whose width comes from the callee's declared shape.

  Iteration ``i`` reads at ``start + i * stride``. For a shape hole, use a leading-axis
  view or explicitly instantiate the callee to specify the window width.
  """
  if start < 0 or stride < 0:
    raise ValueError("window start and stride must be non-negative")
  return cast(T, _Window(value, start, stride))


def _batch_tree(tree: Tree, length: int) -> Tree:
  if isinstance(tree, _G):
    return _G(tuple(_batch_tree(part, length) for part in tree.parts), public=False)
  decl = tree.decls[0]
  return _Leaf(tree.names[0], Ellipsis if decl is Ellipsis else TensorType((length, *decl.shape), decl.dtype, decl.diff))


def _view_window(value: Expr, length: int, width: int, *, repeated: bool = False) -> tuple[Expr, int, int]:
  base = value
  movements = []
  while base.op in (ExprOp.SLICE, ExprOp.RESHAPE, ExprOp.TRANSPOSE):
    movements.append(base)
    base = base.args[0]
  if all(node.op == ExprOp.RESHAPE for node in movements):
    return base if len(base.shape) == 1 else base.vec(), 0, 0 if repeated else width
  indices = np.arange(base.size).reshape(base.shape)
  for node in reversed(movements):
    if node.op == ExprOp.SLICE:
      indices = indices[node.attrs["index"]]
    elif node.op == ExprOp.RESHAPE:
      indices = indices.reshape(node.shape)
    else:
      indices = indices.transpose(node.attrs["axes"])
  count = 1 if repeated else length
  rows = indices.reshape(count, width)
  if count > 0 and width > 0:
    start = int(rows[0, 0])
    stride = 0 if repeated or count == 1 else int(rows[1, 0] - start)
    expected = start + np.arange(count)[:, None] * stride + np.arange(width)
    if stride >= 0 and np.array_equal(rows, expected):
      return base if len(base.shape) == 1 else base.vec(), start, stride
  return value if len(value.shape) == 1 else value.vec(), 0, 0 if repeated else width


class _Mapped(Function):
  def __init__(self, source: Function, length: int) -> None:
    self._source = source
    self._length = length

    def strip_axis(skeleton):
      if isinstance(skeleton, tuple):
        return tuple(strip_axis(part) for part in skeleton)
      if not skeleton.shape or skeleton.shape[0] != length:
        raise ValueError(f"vmap needs leading axis {length}, got {skeleton.shape}")
      return TensorType(skeleton.shape[1:], skeleton.dtype, skeleton.diff)

    def transform(concrete: ConcreteFunction) -> ConcreteFunction:
      if length in concrete._maps:
        return concrete._maps[length]
      inputs = _batch_tree(concrete.input_tree, length)
      outputs = _batch_tree(concrete.output_tree, length)

      def body(*args):
        actuals = inputs.flatten_symbolic(args, "vmap inputs")
        specs = [
          (actual if len(actual.shape) == 1 else actual.vec(), 0, formal.size) for actual, formal in zip(actuals, concrete.inputs, strict=True)
        ]
        values = tuple(_mapped_call(concrete, length, specs, i).reshape((length, *out.shape)) for i, out in enumerate(concrete.outputs))
        return outputs.unflatten(values)

      concrete._maps[length] = ConcreteFunction(f"{concrete.name}_vmap{length}", body, inputs, outputs)
      return concrete._maps[length]

    registry = lift(
      source,
      None if source.inputs is None else _batch_tree(source.inputs, length),
      None if source.outputs is None else _batch_tree(source.outputs, length),
      transform,
      name=f"{source.name}_vmap{length}",
      source_skeleton=strip_axis,
    )
    self._registry = registry
    self.name, self.inputs, self.outputs, self.instances = registry.name, registry.inputs, registry.outputs, registry.instances
    self._fn = registry._fn

  def instantiate(self, shapes: tuple[int | tuple[int, ...] | TensorType, ...] | None = None, /) -> ConcreteFunction:
    return self._registry.instantiate(shapes)

  def _bind(self, skeleton: Any, what: str) -> ConcreteFunction:
    return self._registry._bind(skeleton, what)

  def _arguments(self, args: tuple[Any, ...]) -> tuple[ConcreteFunction, list[Any], list[tuple[int, int]], Any]:
    actuals = []
    layouts = []

    def resolve(tree: Tree | None, value: Any) -> Any:
      if isinstance(tree, _G) or (tree is None and isinstance(value, tuple)):
        parts = tree.parts if isinstance(tree, _G) else (None,) * len(value)
        if not isinstance(value, tuple) or len(value) != len(parts):
          raise ValueError(f"{self.name}: arguments do not match the declared parameter trees")
        return tuple(resolve(part, item) for part, item in zip(parts, value, strict=True))
      decl = Ellipsis if tree is None else tree.decls[0]
      marker = value if isinstance(value, (_Broadcast, _Window)) else None
      value = marker.value if marker is not None else value
      if tree is None and not isinstance(value, (Expr, np.ndarray)):
        raise TypeError("bare mapped function leaves must be Expr or ndarray")
      actual = value if isinstance(value, Expr) else np.asarray(value)
      repeated = isinstance(marker, _Broadcast)
      if isinstance(marker, _Window):
        if decl is Ellipsis:
          raise TypeError("window needs a declared slice shape; instantiate the callee or use a leading-axis view")
        if len(actual.shape) != 1:
          raise ValueError("window needs a rank-1 outer tensor")
        shape, start, stride = decl.shape, marker.start, marker.stride
      else:
        if repeated:
          shape = actual.shape
        elif actual.shape and actual.shape[0] == self._length and (decl is Ellipsis or actual.shape[1:] == decl.shape):
          shape = actual.shape[1:]
        elif decl is not Ellipsis and len(actual.shape) == 1 and actual.size == self._length * decl.size:
          shape = decl.shape
        elif decl is not Ellipsis and actual.size == decl.size:
          shape, repeated = decl.shape, True
        else:
          raise ValueError(f"{self.name}: cannot map shape {actual.shape}; use leading axis {self._length}, broadcast, or window")
        width = int(np.prod(shape, dtype=int))
        if isinstance(actual, Expr):
          outer = actual.reshape(shape if repeated else (self._length, *shape))
          actual, start, stride = _view_window(outer, self._length, width, repeated=repeated)
        else:
          actual = actual.reshape(-1)
          start, stride = 0, 0 if repeated else width
      if decl is not Ellipsis and shape != decl.shape:
        raise ValueError(f"{self.name}: expected slice shape {decl.shape}, got {shape}")
      width = int(np.prod(shape, dtype=int))
      if self._length > 0 and start + (self._length - 1) * stride + width > actual.size:
        raise ValueError("vmap window reads past outer tensor")
      expected_dtype = TensorType(()).dtype if decl is Ellipsis else decl.dtype
      if isinstance(actual, Expr) and actual.type.dtype != expected_dtype:
        raise ValueError(f"{self.name}: expected dtype {expected_dtype}, got {actual.type.dtype}")
      actuals.append(actual)
      layouts.append((start, stride))
      return TensorType(shape, expected_dtype, True if decl is Ellipsis else decl.diff)

    skeleton = resolve(self._source.inputs, args)
    concrete = self._source._bind(skeleton, self.name)
    return concrete, actuals, layouts, skeleton

  @overload
  def __call__[*Ns](self: Function[Any, tuple[*Ns], Any, Any], *args: *Ns) -> Any: ...
  @overload
  def __call__[*Ss](self: Function[tuple[*Ss], Any, Any, Any], *args: *Ss) -> Any: ...
  def __call__(self, *args: Any) -> Any:
    leaves = [leaf.value if isinstance(leaf, (_Broadcast, _Window)) else leaf for leaf in _leaves(args)]
    if leaves and all(isinstance(leaf, Expr) for leaf in leaves):
      return self.symbolic_call(*args)
    if any(isinstance(leaf, Expr) for leaf in leaves):
      raise TypeError(f"{self.name}: cannot mix symbolic and numerical leaves")
    return self.numerical_call(*args)

  def _mapped_instance(self, skeleton: Any) -> ConcreteFunction:
    def add_axis(item):
      return tuple(add_axis(part) for part in item) if isinstance(item, tuple) else TensorType((self._length, *item.shape), item.dtype, item.diff)

    return self._registry._bind(add_axis(skeleton), self.name)

  def symbolic_call(self, *args: Any) -> Any:
    concrete, actuals, layouts, skeleton = self._arguments(args)
    if not all(isinstance(actual, Expr) for actual in actuals):
      raise ValueError(f"{self.name}: symbolic_call needs Expr leaves")
    self._mapped_instance(skeleton)
    specs = [(actual, *layout) for actual, layout in zip(actuals, layouts, strict=True)]
    outputs = tuple(_mapped_call(concrete, self._length, specs, i).reshape((self._length, *out.shape)) for i, out in enumerate(concrete.outputs))
    return concrete.output_tree.unflatten(outputs)

  def numerical_call(self, *args: Any) -> Any:
    concrete, actuals, layouts, skeleton = self._arguments(args)

    if any(isinstance(actual, Expr) for actual in actuals):
      raise ValueError(f"{self.name}: numerical_call needs numerical leaves")
    mapped = self._mapped_instance(skeleton)
    arrays = tuple(
      np.asarray([actual[start + i * stride : start + i * stride + formal.size] for i in range(self._length)]).reshape((self._length, *formal.shape))
      for actual, (start, stride), formal in zip(actuals, layouts, concrete.inputs, strict=True)
    )
    return mapped.numerical_call(*mapped.input_tree.unflatten(arrays))


def vmap[SI, NI, SO, NO](callee: Function[SI, NI, SO, NO] | ConcreteFunction[SI, NI, SO, NO], length: int, /) -> Function[SI, NI, SO, NO]:
  """Map independent calls over a leading axis, preserving parameter and output trees.

  Each output leaf has shape ``(length, *callee_shape)``. Use ``broadcast`` for a whole
  shared tensor and ``window`` for rank-1 overlapping slices of a declared width.
  Affine views are read from their base in place. Other views preserve their element order.
  """
  if not isinstance(callee, Function | ConcreteFunction):
    raise TypeError("vmap callee must be an scaly Function")
  if length < 0:
    raise ValueError("vmap length must be non-negative")
  source = Function._from_instance(callee) if isinstance(callee, ConcreteFunction) else callee
  return cast(Function[SI, NI, SO, NO], _Mapped(source, length))
