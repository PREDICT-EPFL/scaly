"""Typed mapped callables, carried loops, conditional calls, argument markers and affine views."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, cast, overload

import numpy as np
from scipy import sparse

from ..ir.expr import Expr, ExprOp, as_expr, common_lowering, stop_gradient, pack, values
from ..ir.expr_spec import loop_result_type
from ..ir.layout import bind_sparse
from ..ir.types import CanonicalPattern, TensorType
from .model import Function, _Derived, _traced, as_concrete
from .tree import Tree, _G, _Leaf, _PatternHole, _leaves
from .concrete import ConcreteFunction


def _mapped_call(callee: Any, length: int, inputs: Any, output: int = 0) -> Expr:
  """Create an independent ``ExprOp.LOOP`` node: ``length`` independent calls of ``callee``.

  ``inputs`` holds one ``(outer, start, stride)`` per callee input, in order. The i-th call reads
  ``outer[start + i*stride : start + i*stride + formal.type.storage_size]`` from a rank-1 ``outer``, and
  ``stride=0`` broadcasts one slice to every call. The verifier refuses a window that reads past its
  outer when the function holding the node is built. Formals and outputs may be rank-2; the node
  has shape ``(length * callee.results[output].type.storage_size,)``, with the calls' results concatenated flat.
  Internal compact sparse windows use canonical pattern order; lowering applies the callee leaf's layout.
  """
  callee = as_concrete(callee)
  out_expr = callee.results[output]
  outers = tuple(as_expr(outer) for outer, _, _ in inputs)
  diff = out_expr.type.diff and any(o.type.diff for o in outers)
  return Expr(
    ExprOp.LOOP,
    outers,
    TensorType((length * out_expr.type.storage_size,), out_expr.type.dtype, diff=diff),
    attrs={
      "callee": callee,
      "output": int(output),
      "length": int(length),
      "starts": tuple(int(start) for _, start, _ in inputs),
      "strides": tuple(int(stride) for _, _, stride in inputs),
      "n_carry": 0,
      "cond": None,
      "index": False,
      "reverse": False,
      "in_place": (),
    },
    lowering=common_lowering(*outers) if outers else "auto",
  )


def _stacked(result: Expr, out: Expr, length: int) -> Expr:
  shape = (length, *out.shape)
  return result.reshape(shape) if out.type.pattern is None else pack(result.reshape((length, *out.type.storage_shape)), out.type.pattern, shape=shape)


def _mapped_output(callee: ConcreteFunction, length: int, specs: Any, output: int) -> Expr:
  return _stacked(_mapped_call(callee, length, specs, output), callee.outputs[output], length)


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
  layout = tree._layout_decls[0] if tree.has_holes else tree.layouts[0]
  return _Leaf(tree.names[0], None if decl is None else replace(decl, shape=(length, *decl.shape)), layout=layout)


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


def _map_arguments(source: Function, length: int, name: str, args: tuple[Any, ...]) -> tuple[ConcreteFunction, list[Any], list[tuple[int, int]], Any]:
  """Bind ``source`` to per-iteration slices of ``args`` and return each slice's flat outer, start and stride."""
  actuals = []
  layouts = []
  canonical_inputs: dict[int, np.ndarray] = {}

  def resolve(tree: Tree | None, value: Any) -> Any:
    if isinstance(tree, _G) or (tree is None and isinstance(value, tuple)):
      parts = tree.parts if isinstance(tree, _G) else (None,) * len(value)
      if not isinstance(value, tuple) or len(value) != len(parts):
        raise ValueError(f"{name}: arguments do not match the declared parameter trees")
      return tuple(resolve(part, item) for part, item in zip(parts, value, strict=True))
    decl = None if tree is None else tree.decls[0]
    marker = value if isinstance(value, (_Broadcast, _Window)) else None
    value = marker.value if marker is not None else value
    if tree is None and not isinstance(value, (Expr, np.ndarray)) and not sparse.issparse(value):
      raise TypeError("bare mapped function leaves must be Expr, ndarray or SciPy sparse")
    is_expr = isinstance(value, Expr)
    actual: Any = value if is_expr or sparse.issparse(value) else np.asarray(value)
    if tree is None and (sparse.issparse(actual) or is_expr and actual.type.pattern is not None):
      decl = _PatternHole(shape_hole=True)
    repeated = isinstance(marker, _Broadcast)
    canonical = None
    if isinstance(decl, _PatternHole):
      if isinstance(marker, _Window):
        raise TypeError("window cannot bind a pattern hole; instantiate the callee first")
      pattern, canonical = bind_sparse(actual, decl.dtype.numpy()) if sparse.issparse(actual) else (actual.type.pattern if is_expr else None, None)
      if decl.shape_hole and pattern is None:
        decl = None
      else:
        shape = (
          actual.shape[1:]
          if decl.shape_hole and not repeated and len(actual.shape) > 2 and actual.shape[0] == length
          else actual.shape
          if decl.shape_hole
          else decl.shape
        )
        decl = TensorType(shape, decl.dtype, decl.diff, pattern)
      if canonical is not None:
        canonical_inputs[len(actuals)] = canonical
        actual = cast(Any, canonical)
    pattern = None if decl is None else decl.pattern
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
      elif actual.shape and actual.shape[0] == length and (decl is None or actual.shape[1:] == decl.shape):
        shape = actual.shape[1:]
      elif decl is not None and pattern is not None and actual.shape == (length, *decl.storage_shape):
        shape = decl.shape
      elif decl is not None and len(actual.shape) == 1 and actual_size == length * decl.storage_size:
        shape = decl.shape
      elif decl is not None and (actual.shape == decl.shape or actual.shape == decl.storage_shape):
        shape, repeated = decl.shape, True
      elif decl is not None and actual_size == decl.storage_size:
        shape, repeated = decl.shape, True
      else:
        raise ValueError(f"{name}: cannot map shape {actual.shape}; use leading axis {length}, broadcast, or window")
      width = decl.storage_size if decl is not None else int(np.prod(shape, dtype=int))
      if is_expr:
        if actual.type.pattern is not pattern:
          raise ValueError(f"{name}: different stored pattern")
        outer = values(actual) if pattern is not None else actual.reshape(shape if repeated else (length, *shape))
        actual, start, stride = _view_window(outer, length, width, repeated=repeated)
      else:
        if pattern is not None and canonical is None:
          layout = cast(_Leaf, tree).layouts[0]
          assert layout is not None and decl is not None
          actual = layout.numerical_values(actual, shape if repeated else (length, *shape), decl.dtype.numpy())
        actual = actual.reshape(-1)
        start, stride = 0, 0 if repeated else width
    if decl is not None and shape != decl.shape:
      raise ValueError(f"{name}: expected slice shape {decl.shape}, got {shape}")
    width = decl.storage_size if decl is not None else int(np.prod(shape, dtype=int))
    if length > 0 and start + (length - 1) * stride + width > actual.size:
      raise ValueError("vmap window reads past outer tensor")
    expected_dtype = TensorType(()).dtype if decl is None else decl.dtype
    if isinstance(actual, Expr) and actual.type.dtype != expected_dtype:
      raise ValueError(f"{name}: expected dtype {expected_dtype}, got {actual.type.dtype}")
    if isinstance(marker, _Window) and isinstance(actual, Expr) and pattern is not None:
      layout = cast(_Leaf, tree).layouts[0]
      assert layout is not None and decl is not None
      if not np.array_equal(layout.permutation, np.arange(pattern.nnz)):
        slots = np.arange(width).reshape(decl.storage_shape)[..., layout.inverse_permutation].reshape(-1)
        actual = actual.gather(start + np.arange(length)[:, None] * stride + slots).vec()
        start, stride = 0, width
    actuals.append(actual)
    layouts.append((start, stride))
    return TensorType(shape, expected_dtype, True if decl is None else decl.diff, pattern)

  skeleton = resolve(source.inputs, args)
  concrete = source._bind(skeleton, name)
  for index, canonical in canonical_inputs.items():
    layout = concrete.input_tree.layouts[index]
    assert layout is not None
    actuals[index] = canonical[..., layout.permutation].reshape(-1)
  return concrete, actuals, layouts, skeleton


@dataclass(frozen=True, eq=False, repr=False)
class _Mapped(_Derived):
  _length: int

  def _arguments(self, args: tuple[Any, ...]) -> tuple[ConcreteFunction, list[Any], list[tuple[int, int]], Any]:
    return _map_arguments(self._source, self._length, self.name, args)

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

  def sparsity(self, *args: Any, of: str | None = None) -> CanonicalPattern | None:
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
    for actual, (start, stride), formal in zip(actuals, layouts, concrete.inputs, strict=True):
      slices = [actual[start + i * stride : start + i * stride + formal.type.storage_size] for i in range(self._length)]
      arrays.append(np.asarray(slices, dtype=actual.dtype).reshape((self._length, *formal.type.storage_shape)))

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


def _cond_node(branches: tuple[ConcreteFunction, ...], args: tuple[Expr, ...], output: int) -> Expr:
  type_ = branches[0].outputs[output].type
  diff = type_.dtype.is_floating and any(branch.outputs[output].type.diff for branch in branches) and any(arg.type.diff for arg in args[1:])
  return Expr(ExprOp.COND, args, replace(type_, diff=diff), attrs={"branches": branches, "output": output}, lowering=common_lowering(*args))


def cond(selector: Any, branches: tuple[Function | ConcreteFunction, ...], *operands: Any) -> Any:
  """Call the branch selected by a scalar boolean or integer, with ``operands`` as its inputs.

  False selects branch zero and True selects branch one. A boolean requires exactly two
  branches. Integer selectors clamp to ``[0, len(branches) - 1]``. Only the selected branch
  runs, including its prints. Branches must have matching input and output structures,
  shapes, dtypes and stored patterns. Results follow the first branch's output tree.
  The selector has no derivative. Derivatives follow the selected branch; structural
  sparsity is the union of the branches.
  """
  if not isinstance(branches, tuple) or not branches or not all(isinstance(fn, Function | ConcreteFunction) for fn in branches):
    raise TypeError("cond branches must be a nonempty tuple of scaly Functions")
  selector = selector if isinstance(selector, Expr) else Expr.const(np.asarray(selector))
  if selector.shape or not (selector.type.dtype.is_bool or selector.type.dtype.is_integer):
    raise TypeError("cond selector must be a boolean or integer scalar")
  if selector.type.dtype.is_bool and len(branches) != 2:
    raise ValueError("a boolean cond selector requires two branches")
  concrete = []
  actuals = None
  for fn in branches:
    source = Function._from_instance(fn) if isinstance(fn, ConcreteFunction) else fn
    branch, prepared = source._prepare_arguments(_symbolic(operands), "cond")
    leaves = branch.input_tree.flatten_symbolic(prepared, "cond")
    if actuals is None:
      actuals = tuple(leaves)
    elif tuple(leaves) != actuals:
      raise ValueError("cond branches must have matching input structures")
    if concrete and branch.output_tree.unflatten(tuple(range(len(branch.outputs)))) != concrete[0].output_tree.unflatten(
      tuple(range(len(concrete[0].outputs)))
    ):
      raise ValueError("cond branches must have matching output structures")
    if concrete and any(
      (out.shape, out.type.dtype, out.type.pattern) != (ref.shape, ref.type.dtype, ref.type.pattern)
      for out, ref in zip(branch.outputs, concrete[0].outputs, strict=True)
    ):
      raise ValueError("cond branches must have matching output shapes, dtypes and stored patterns")
    concrete.append(branch)
  assert actuals is not None
  functions = tuple(concrete)
  results = tuple(_cond_node(functions, (selector, *actuals), i) for i in range(len(functions[0].outputs)))
  return functions[0].output_tree.unflatten(results)


def _loop_node(callee: ConcreteFunction, cond: ConcreteFunction | None, args: tuple[Expr, ...], attrs: dict[str, Any], output: int) -> Expr:
  """Create result ``output`` of an ``ExprOp.LOOP`` node; ``attrs`` holds every attribute but the functions and ``output``."""
  attrs = {**attrs, "callee": callee, "cond": cond, "output": output}
  type_ = loop_result_type(attrs, output)
  assert type_ is not None
  diff = type_.dtype.is_floating and any(arg.type.diff for arg in args)
  return Expr(ExprOp.LOOP, args, replace(type_, diff=diff), attrs=attrs, lowering=common_lowering(*args) if args else "auto")


def _whole(item: Any) -> Any:
  """Mark every leaf of a tree to be read whole by each step."""
  return tuple(_whole(part) for part in item) if isinstance(item, tuple) else broadcast(item)


def _symbolic(item: Any) -> Any:
  """Make every leaf of a tree of loop slices an expression, keeping its marker."""
  if isinstance(item, tuple):
    return tuple(_symbolic(part) for part in item)
  if isinstance(item, _Broadcast):
    return _Broadcast(_symbolic(item.value))
  if isinstance(item, _Window):
    return _Window(_symbolic(item.value), item.start, item.stride)
  return item if isinstance(item, Expr) else Expr.const(item)


def _loop(
  body: Any,
  cond: Any,
  init: Any,
  xs: tuple[Any, ...],
  length: int,
  trip: Any,
  index: bool,
  reverse: bool,
  what: str,
  in_place: tuple[int | str, ...] = (),
) -> tuple[Any, Any, Expr]:
  if not isinstance(body, Function | ConcreteFunction) or cond is not None and not isinstance(cond, Function | ConcreteFunction):
    raise TypeError(f"{what} body and condition must be scaly Functions")
  length = int(length)
  if length < 0:
    raise ValueError(f"{what}: {'max_iter' if cond is not None else 'length'} must be non-negative")
  inits = init if isinstance(init, tuple) else (init,)
  leaves = [as_expr(leaf) for leaf in _leaves(inits)]
  step = (broadcast(Expr.const(np.int64(0))),) if index else ()
  source = Function._from_instance(body) if isinstance(body, ConcreteFunction) else body
  callee, actuals, layouts, _ = _map_arguments(source, length, what, (*_whole(inits), *step, *_symbolic(xs)))
  n_carry, skip = len(leaves), len(leaves) + int(index)
  outers = tuple(actuals[skip:])
  trips = () if trip is None else (trip if isinstance(trip, Expr) else Expr.const(np.int64(trip)),)
  if cond is not None:
    cond = Function._from_instance(cond) if isinstance(cond, ConcreteFunction) else cond
    cond = cond._bind(callee.input_tree.unflatten(callee.input_tree.types), what)
  carry_names = {name: k for k, name in enumerate(callee.input_names[:n_carry])}
  for name in in_place:
    if isinstance(name, str) and name not in carry_names:
      raise ValueError(f"{what}: in_place names unknown carry {name!r}")
  attrs = {
    "n_carry": n_carry,
    "length": length,
    "starts": tuple(start for start, _ in layouts[skip:]),
    "strides": tuple(stride for _, stride in layouts[skip:]),
    "index": bool(index),
    "reverse": bool(reverse),
    "in_place": tuple(carry_names[k] if isinstance(k, str) else k for k in in_place),
  }
  args = (*leaves, *outers, *trips)
  parts = callee.output_tree.parts if isinstance(callee.output_tree, _G) else (callee.output_tree,)
  if sum(part.size for part in parts[: len(inits)]) != n_carry:
    parts = (callee.output_tree,)
  if sum(part.size for part in parts[: len(inits)]) != n_carry:
    raise ValueError(f"{what}: the body must return its {n_carry} carry leaves first")
  n_ys = len(callee.outputs) - n_carry
  finals = tuple(_loop_node(callee, cond, args, attrs, i) for i in range(n_carry))
  ys = tuple(_stacked(_loop_node(callee, cond, args, attrs, n_carry + i), callee.outputs[n_carry + i], length) for i in range(n_ys))
  carry = _G(cast(_G, callee.input_tree).parts[: len(inits)], public=False).unflatten(finals)
  stacked = _G(tuple(parts[len(inits) :]), public=False).unflatten(ys)
  return carry if isinstance(init, tuple) else carry[0], stacked, _loop_node(callee, cond, args, attrs, len(callee.results))


def scan(
  body: Any,
  init: Any,
  xs: tuple[Any, ...] = (),
  *,
  length: int,
  trip: Any = None,
  index: bool = False,
  reverse: bool = False,
  in_place: tuple[int | str, ...] = (),
) -> tuple[Any, Any]:
  """Run ``body`` for ``length`` steps, threading a carry and stacking per-step outputs.

  ``body`` takes one parameter per entry of ``init`` (one carry when ``init`` is not a tuple),
  then the step number as an int64 scalar when ``index`` is true, then one parameter per entry of
  ``xs``. It returns the new carries first, each with its input's shape and dtype, followed by
  any per-step outputs. Each entry of ``xs`` is sliced per step as ``vmap`` slices its arguments:
  a leading axis of ``length``, ``broadcast`` for a value shared by every step, or ``window``.

  Returns ``(carry, ys)``: the final carry, shaped like ``init``, and a tuple of the per-step
  outputs, each with a leading axis of ``length``. ``trip``, an int64 scalar, runs only the first
  ``min(trip, length)`` steps; the unused entries of the stacked outputs are zero. With ``reverse``
  the steps run from the last to the first, and step ``k`` still reads slice ``k`` and writes
  output ``k``. ``in_place`` names carry positions or input names that lowering must prove safe
  to update in place. Forward derivatives follow the same steps as the primal loop.
  Reverse derivatives raise ``NotImplementedError``.
  """
  carry, ys, _ = _loop(body, None, init, tuple(xs), length, trip, index, reverse, "scan", in_place)
  return carry, ys


def while_loop(cond: Any, body: Any, init: Any, *, max_iter: int, params: tuple[Any, ...] = (), index: bool = False) -> tuple[Any, Expr]:
  """Apply ``body`` to a carry while ``cond`` holds, at most ``max_iter`` times.

  ``body`` and ``cond`` take the same parameters: one per entry of ``init`` (one carry when
  ``init`` is not a tuple), then the step number as an int64 scalar when ``index`` is true, then
  one per entry of ``params``, which every step reads whole. ``cond`` returns a bool scalar and is
  evaluated before each step; ``body`` returns the new carries, each with its input's shape and
  dtype. Returns ``(carry, count)``: the final carry, shaped like ``init``, and the number of
  steps taken as an int64 scalar. Forward derivatives take the same steps, with the condition
  evaluated on primal carries. Reverse derivatives raise ``NotImplementedError``.
  """
  carry, ys, count = _loop(body, cond, init, _whole(tuple(params)), max_iter, None, index, False, "while_loop")
  if ys:
    raise ValueError("while_loop: the body must return only its carries")
  return carry, count
