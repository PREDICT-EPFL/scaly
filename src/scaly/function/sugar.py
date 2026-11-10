"""Typed mapped callables, argument markers, affine views, and the internal VMAP builder."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, cast, overload

import numpy as np
from scipy import sparse

from ..ir.expr import Expr, ExprOp, as_expr, common_lowering, stop_gradient, pack, values
from ..ir.types import SparsityPattern, TensorType
from .model import Function, _Derived, _traced, as_concrete
from .tree import Tree, _G, _Leaf, _leaves
from .concrete import ConcreteFunction


def _mapped_call(callee: Any, length: int, inputs: Any, output: int = 0) -> Expr:
  """Create an ``ExprOp.VMAP`` node: ``length`` independent calls of ``callee``.

  ``inputs`` holds one ``(outer, start, stride)`` per callee input, in order. The i-th call reads
  ``outer[start + i*stride : start + i*stride + formal.type.storage_size]`` from a rank-1 ``outer``, and
  ``stride=0`` broadcasts one slice to every call. The verifier refuses a window that reads past its
  outer when the function holding the node is built. Formals and outputs may be rank-2; the node
  has shape ``(length * callee.outputs[output].type.storage_size,)``, with the calls' outputs concatenated flat.
  Compact sparse windows use canonical pattern order; lowering applies the callee leaf's layout.
  """
  callee = as_concrete(callee)
  out_expr = callee.outputs[output]
  outers = tuple(as_expr(outer) for outer, _, _ in inputs)
  diff = out_expr.type.diff and any(o.type.diff for o in outers)
  return Expr(
    ExprOp.VMAP,
    outers,
    TensorType((length * out_expr.type.storage_size,), out_expr.type.dtype, diff=diff),
    attrs={
      "callee": callee,
      "output": int(output),
      "length": int(length),
      "starts": tuple(int(start) for _, start, _ in inputs),
      "strides": tuple(int(stride) for _, _, stride in inputs),
      "slice_size": int(out_expr.type.storage_size),
    },
    lowering=common_lowering(*outers) if outers else "auto",
  )


def _mapped_output(callee: ConcreteFunction, length: int, specs: Any, output: int) -> Expr:
  result = _mapped_call(callee, length, specs, output)
  out = callee.outputs[output]
  shape = (length, *out.shape)
  return result.reshape(shape) if out.type.pattern is None else pack(result.reshape((length, *out.type.storage_shape)), out.type.pattern, shape=shape)


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
  start, stride = int(start), int(stride)
  if start < 0 or stride < 0:
    raise ValueError("window start and stride must be non-negative")
  return cast(T, _Window(value, start, stride))


def _batch_tree(tree: Tree, length: int) -> Tree:
  if isinstance(tree, _G):
    return _G(tuple(_batch_tree(part, length) for part in tree.parts), public=False)
  decl = tree.decls[0]
  return _Leaf(tree.names[0], None if decl is None else replace(decl, shape=(length, *decl.shape)), layout=tree.layouts[0] or "csc")


def _view_window(value: Expr, length: int, width: int, *, repeated: bool = False) -> tuple[Expr, int, int]:
  base = value
  movements = []
  stopped = False
  while base.op in (ExprOp.SLICE, ExprOp.RESHAPE, ExprOp.TRANSPOSE, ExprOp.STOP_GRADIENT):
    if base.op == ExprOp.STOP_GRADIENT:
      stopped = True
    else:
      movements.append(base)
    base = base.args[0]
  if stopped:
    base = stop_gradient(base)
  if all(node.op == ExprOp.RESHAPE for node in movements):
    return base if len(base.shape) == 1 else base.vec(), 0, 0 if repeated else width
  count = 1 if repeated else length
  if count == 0 or width == 0:
    return value if len(value.shape) == 1 else value.vec(), 0, 0 if repeated else width
  coords = list(np.unravel_index(np.arange(count * width), value.shape)) if value.shape else []
  for node in movements:
    src = node.args[0]
    if node.op == ExprOp.SLICE:
      selected = iter(coords)
      coords = [
        np.full(count * width, item if item >= 0 else dim + item, dtype=np.int64)
        if isinstance(item, int)
        else item.indices(dim)[0] + item.indices(dim)[2] * next(selected)
        for dim, item in zip(src.shape, node.attrs["index"], strict=True)
      ]
    elif node.op == ExprOp.RESHAPE:
      flat = np.ravel_multi_index(tuple(coords), node.shape) if node.shape else np.zeros(count * width, dtype=np.int64)
      coords = list(np.unravel_index(flat, src.shape)) if src.shape else []
    else:
      coords = [coords[node.attrs["axes"].index(dim)] for dim in range(len(src.shape))]
  indices = np.ravel_multi_index(tuple(coords), base.shape) if base.shape else np.zeros(count * width, dtype=np.int64)
  rows = indices.reshape(count, width)
  if count > 0 and width > 0:
    start = int(rows[0, 0])
    stride = 0 if repeated or count == 1 else int(rows[1, 0] - start)
    expected = start + np.arange(count)[:, None] * stride + np.arange(width)
    if stride >= 0 and np.array_equal(rows, expected):
      return base if len(base.shape) == 1 else base.vec(), start, stride
  return value if len(value.shape) == 1 else value.vec(), 0, 0 if repeated else width


@dataclass(frozen=True, eq=False, repr=False)
class _Mapped(_Derived):
  _length: int

  def _arguments(self, args: tuple[Any, ...]) -> tuple[ConcreteFunction, list[Any], list[tuple[int, int]], Any]:
    actuals = []
    layouts = []

    def resolve(tree: Tree | None, value: Any) -> Any:
      if isinstance(tree, _G) or (tree is None and isinstance(value, tuple)):
        parts = tree.parts if isinstance(tree, _G) else (None,) * len(value)
        if not isinstance(value, tuple) or len(value) != len(parts):
          raise ValueError(f"{self.name}: arguments do not match the declared parameter trees")
        return tuple(resolve(part, item) for part, item in zip(parts, value, strict=True))
      decl = None if tree is None else tree.decls[0]
      marker = value if isinstance(value, (_Broadcast, _Window)) else None
      value = marker.value if marker is not None else value
      if tree is None and not isinstance(value, (Expr, np.ndarray)):
        raise TypeError("bare mapped function leaves must be Expr or ndarray")
      pattern = None if decl is None else decl.pattern
      is_expr = isinstance(value, Expr)
      actual: Any = value if is_expr or sparse.issparse(value) else np.asarray(value)
      repeated = isinstance(marker, _Broadcast)
      if isinstance(marker, _Window):
        if decl is None:
          raise TypeError("window needs a declared slice shape; instantiate the callee or use a leading-axis view")
        if len(actual.shape) != 1:
          raise ValueError("window needs a rank-1 outer tensor")
        shape, start, stride = decl.shape, marker.start, marker.stride
      else:
        actual_size = actual.type.storage_size if is_expr else actual.size
        if repeated:
          shape = decl.shape if decl is not None and pattern is not None and actual.shape == decl.storage_shape else actual.shape
        elif actual.shape and actual.shape[0] == self._length and (decl is None or actual.shape[1:] == decl.shape):
          shape = actual.shape[1:]
        elif decl is not None and pattern is not None and actual.shape == (self._length, *decl.storage_shape):
          shape = decl.shape
        elif decl is not None and len(actual.shape) == 1 and actual_size == self._length * decl.storage_size:
          shape = decl.shape
        elif decl is not None and (actual.shape == decl.shape or actual.shape == decl.storage_shape):
          shape, repeated = decl.shape, True
        elif decl is not None and actual_size == decl.storage_size:
          shape, repeated = decl.shape, True
        else:
          raise ValueError(f"{self.name}: cannot map shape {actual.shape}; use leading axis {self._length}, broadcast, or window")
        width = decl.storage_size if decl is not None else int(np.prod(shape, dtype=int))
        if is_expr:
          if actual.type.pattern is not pattern:
            raise ValueError(f"{self.name}: different stored pattern")
          outer = values(actual) if pattern is not None else actual.reshape(shape if repeated else (self._length, *shape))
          actual, start, stride = _view_window(outer, self._length, width, repeated=repeated)
        else:
          if pattern is not None:
            layout = cast(_Leaf, tree).layouts[0]
            assert layout is not None
            actual = layout.numerical_values(actual, shape if repeated else (self._length, *shape))[..., layout.inverse_permutation]
          actual = actual.reshape(-1)
          start, stride = 0, 0 if repeated else width
      if decl is not None and shape != decl.shape:
        raise ValueError(f"{self.name}: expected slice shape {decl.shape}, got {shape}")
      width = decl.storage_size if decl is not None else int(np.prod(shape, dtype=int))
      if self._length > 0 and start + (self._length - 1) * stride + width > actual.size:
        raise ValueError("vmap window reads past outer tensor")
      expected_dtype = TensorType(()).dtype if decl is None else decl.dtype
      if isinstance(actual, Expr) and actual.type.dtype != expected_dtype:
        raise ValueError(f"{self.name}: expected dtype {expected_dtype}, got {actual.type.dtype}")
      actuals.append(actual)
      layouts.append((start, stride))
      return TensorType(shape, expected_dtype, True if decl is None else decl.diff, pattern)

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
      return tuple(add_axis(part) for part in item) if isinstance(item, tuple) else replace(item, shape=(self._length, *item.shape))

    return self._bind(add_axis(skeleton), self.name)

  def sparsity(self, *args: Any, of: str | None = None) -> SparsityPattern | None:
    """Return the callee's per-iteration pattern for each leading output slice.

    Arguments use the same leading-axis, broadcast and window rules as a mapped call.
    Nested maps keep the underlying matrix pattern, applied independently at each iteration.
    """
    if not args and self.inputs is not None:
      return self._source.sparsity(of=of)
    concrete, _, _, _ = self._arguments(args)
    return self._source.sparsity(*concrete.input_tree.symbols(), of=of)

  def symbolic_call(self, *args: Any) -> Any:
    concrete, actuals, layouts, skeleton = self._arguments(args)
    if not all(isinstance(actual, Expr) for actual in actuals):
      raise ValueError(f"{self.name}: symbolic_call needs Expr leaves")
    self._mapped_instance(skeleton)
    specs = [(actual, *layout) for actual, layout in zip(actuals, layouts, strict=True)]
    outputs = tuple(_mapped_output(concrete, self._length, specs, i) for i in range(len(concrete.outputs)))
    return concrete.output_tree.unflatten(outputs)

  def numerical_call(self, *args: Any) -> Any:
    concrete, actuals, layouts, skeleton = self._arguments(args)

    if any(isinstance(actual, Expr) for actual in actuals):
      raise ValueError(f"{self.name}: numerical_call needs numerical leaves")
    mapped = self._mapped_instance(skeleton)
    arrays = []
    for actual, (start, stride), formal, layout in zip(actuals, layouts, concrete.inputs, concrete.input_tree.layouts, strict=True):
      array = np.asarray([actual[start + i * stride : start + i * stride + formal.type.storage_size] for i in range(self._length)]).reshape(
        (self._length, *formal.type.storage_shape)
      )
      arrays.append(array if layout is None else array[..., layout.permutation])

    return mapped.numerical_call(*mapped.input_tree.unflatten(tuple(arrays)))


def vmap[SI, NI, SO, NO](callee: Function[SI, NI, SO, NO] | ConcreteFunction[SI, NI, SO, NO], length: int, /) -> Function[SI, NI, SO, NO]:
  """Map independent calls over a leading axis, preserving parameter and output trees.

  Each output leaf has shape ``(length, *callee_shape)``. Use ``broadcast`` for a whole
  shared tensor and ``window`` for rank-1 overlapping slices of a declared width.
  Affine views are read from their base in place. Other views preserve their element order.
  """
  if not isinstance(callee, Function | ConcreteFunction):
    raise TypeError("vmap callee must be an scaly Function")
  length = int(length)
  if length < 0:
    raise ValueError("vmap length must be non-negative")
  source = Function._from_instance(callee) if isinstance(callee, ConcreteFunction) else callee

  def strip_axis(skeleton):
    if isinstance(skeleton, tuple):
      return tuple(strip_axis(part) for part in skeleton)
    if not skeleton.shape or skeleton.shape[0] != length:
      raise ValueError(f"vmap needs leading axis {length}, got {skeleton.shape}")
    return replace(skeleton, shape=skeleton.shape[1:])

  def transform(concrete: ConcreteFunction) -> ConcreteFunction:
    if length in concrete._memo.maps:
      return concrete._memo.maps[length]
    inputs = _batch_tree(concrete.input_tree, length)
    outputs = _batch_tree(concrete.output_tree, length)

    def body(*args):
      actuals = inputs.flatten_symbolic(args, "vmap inputs")
      specs = [
        (values(actual).vec() if actual.type.pattern is not None else actual if len(actual.shape) == 1 else actual.vec(), 0, formal.type.storage_size)
        for actual, formal in zip(actuals, concrete.inputs, strict=True)
      ]
      results = tuple(_mapped_output(concrete, length, specs, i) for i in range(len(concrete.outputs)))
      return outputs.unflatten(results)

    concrete._memo.maps[length] = ConcreteFunction._trace(f"{concrete.name}_vmap{length}", body, inputs, outputs)
    return concrete._memo.maps[length]

  mapped = _Mapped(
    f"{source.name}_vmap{length}",
    source._fn,
    None if source.inputs is None else _batch_tree(source.inputs, length),
    None if source.outputs is None else _batch_tree(source.outputs, length),
    source,
    transform,
    strip_axis,
    length,
  )
  return cast(Function[SI, NI, SO, NO], _traced(mapped))
