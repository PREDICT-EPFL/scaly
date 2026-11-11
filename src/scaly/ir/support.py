"""Lazy value support over flat logical coordinates, using the fixed pattern algebra."""

from __future__ import annotations

from collections.abc import Callable, Set, Iterator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping, cast

import numpy as np

from .types import CanonicalPattern, _uniform_value, stored_indices

if TYPE_CHECKING:
  from .expr import Expr, ExprOp, OpInfo


@dataclass(frozen=True, slots=True, eq=False)
class _SupportSet(Set[int]):
  _entries: np.ndarray
  _hash_cache: int | None = field(default=None, init=False, repr=False, compare=False)

  def __init__(self, entries: np.ndarray):
    values = np.unique(np.asarray(entries, dtype=np.int64))
    object.__setattr__(self, "_entries", np.frombuffer(values.tobytes(), dtype=np.int64))
    object.__setattr__(self, "_hash_cache", None)

  def __len__(self) -> int:
    return len(self._entries)

  def __iter__(self) -> Iterator[int]:
    return map(int, self._entries)

  def __contains__(self, value: object) -> bool:
    if not isinstance(value, int | np.integer):
      return False
    i = np.searchsorted(self._entries, value)
    return bool(i < len(self) and self._entries[i] == value)

  def __eq__(self, other: object) -> bool:
    if isinstance(other, _SupportSet):
      return bool(np.array_equal(self._entries, other._entries))
    return isinstance(other, Set) and len(self) == len(other) and all(i in other for i in self)

  def __hash__(self) -> int:
    if self._hash_cache is None:
      object.__setattr__(self, "_hash_cache", hash(frozenset(self)))
    assert self._hash_cache is not None
    return self._hash_cache

  def __and__(self, other):
    return _make_support(np.intersect1d(self._entries, support_indices(other), assume_unique=True))

  def __or__(self, other):
    return _make_support(np.union1d(self._entries, support_indices(other)))

  __rand__ = __and__
  __ror__ = __or__

  @classmethod
  def _from_iterable(cls, values):
    return _make_support(np.fromiter(values, dtype=np.int64))


type Support = frozenset[int] | _SupportSet | None


def support_indices(support: frozenset[int] | _SupportSet) -> np.ndarray:
  """Read known support as a coordinate array without copying an existing compact cache."""
  return support._entries if isinstance(support, _SupportSet) else np.sort(np.fromiter(support, dtype=np.int64, count=len(support)))


def _make_support(entries: np.ndarray) -> frozenset[int] | _SupportSet:
  return frozenset(map(int, entries)) if len(entries) <= 1024 else _SupportSet(entries)


_SPECIALIZATION_LIMIT = 32


def compute_support(root: Expr, rules: Mapping[ExprOp, OpInfo]) -> Support:
  """Compute and cache support without recursing through operands or callees."""
  if root._support_cache is not None:
    return root._support_cache[0]
  _run_support(_walk_support((root,), rules, None), rules)
  return _cached(root)


def specialize_support(callee: Any, arguments: tuple[Support, ...], rules: Mapping[ExprOp, OpInfo]) -> Callable[[Expr], Support]:
  """Return a node-support lookup for the callee, using its own supports beyond the cache bound."""

  def request():
    return (yield callee, arguments)

  return _run_support(request(), rules)


def _run_support(walk, rules: Mapping[ExprOp, OpInfo]) -> Callable[[Expr], Support]:
  """Suspend callee walks so deep call nesting uses no Python recursion."""
  pending = [walk]
  result = None
  while pending:
    try:
      callee, arguments = pending[-1].send(result)
    except StopIteration as done:
      result = done.value
      pending.pop()
      continue
    cache = callee._memo.supports
    if arguments in cache:
      result = cache[arguments].__getitem__
    elif len(cache) >= _SPECIALIZATION_LIMIT:
      pending.append(_walk_support(callee.results, rules, None))
      result = None
    else:
      pending.append(_specialized_support(callee, arguments, rules))
      result = None
  assert result is not None
  return result


def _specialized_support(callee: Any, arguments: tuple[Support, ...], rules: Mapping[ExprOp, OpInfo]):
  context = {formal: _bound_support(formal, support) for formal, support in zip(callee.inputs, arguments, strict=True)}
  result = yield from _walk_support(callee.results, rules, context)
  callee._memo.supports[arguments] = context
  return result


def _walk_support(roots: Sequence[Expr], rules: Mapping[ExprOp, OpInfo], context: dict[Expr, Support] | None):
  support_of = _cached if context is None else context.__getitem__
  stack = [(root, False) for root in reversed(roots)]
  while stack:
    node, visited = stack.pop()
    if (context is None and node._support_cache is not None) or (context is not None and node in context):
      continue
    if context is not None and not node.args:
      yield from _walk_support((node,), rules, None)
      context[node] = _cached(node)
      continue
    info = rules[cast("ExprOp", node.op)]
    rule = info.support_rule or info.pattern_rule
    if not visited:
      stack.append((node, True))
      stack.extend((arg, False) for arg in reversed(node.args))
      continue
    result = (yield from _call_support(node, rule, support_of)) if rule in {"call", "vmap"} else _rule(node, rule, support_of)
    result = _bound_support(node, result)
    if context is None:
      object.__setattr__(node, "_support_cache", (result,))
    else:
      context[node] = result
  return support_of


def _bound_support(node: Expr, result: Support) -> Support:
  if node.size == 0:
    return frozenset()
  if node.type.pattern is not None:
    bound = _make_support(stored_indices(node.type))
    return bound if result is None else result & bound
  return result


def _call_support(node: Expr, op: str, support_of: Callable[[Expr], Support]):
  callee, output = node.attrs["callee"], node.attrs["output"]
  if op == "call":
    context = yield callee, tuple(support_of(arg) for arg in node.args)
    return context(callee.results[output])
  length = node.attrs["length"]
  if not length:
    return frozenset()
  groups: dict[tuple[Support, ...], list[int]] = {}
  windows = []
  for arg, formal, start, stride in zip(node.args, callee.inputs, node.attrs["starts"], node.attrs["strides"], strict=True):
    support = support_of(arg)
    indices = None if support is None else support_indices(support)
    logical = stored_indices(formal.type) if formal.type.pattern is not None else None
    windows.append((indices, formal.type.storage_size, logical, start, stride))
  for stage in range(length):
    arguments = []
    for indices, width, logical, start, stride in windows:
      support = None
      if indices is not None:
        offset = start + stage * stride
        lo, hi = np.searchsorted(indices, (offset, offset + width))
        entries = indices[lo:hi] - offset
        support = _make_support(entries if logical is None else logical[entries])
      arguments.append(support)
    groups.setdefault(tuple(arguments), []).append(stage)
  tiles = []
  result_expr = callee.results[output]
  for arguments, stages in groups.items():
    context = yield callee, arguments
    support = context(result_expr)
    if result_expr.type.pattern is not None and support is not None:
      support = _make_support(np.flatnonzero(np.isin(stored_indices(result_expr.type), support_indices(support))))
    tiles.append((stages, support))
  if all(support is None for _, support in tiles):
    return None
  width = result_expr.type.storage_size
  return _make_support(
    np.concatenate(
      [
        (np.asarray(stages, dtype=np.int64)[:, None] * width + (np.arange(width) if support is None else support_indices(support))).reshape(-1)
        for stages, support in tiles
      ]
    )
  )


def _cached(node: Expr) -> Support:
  assert node._support_cache is not None
  return node._support_cache[0]


def _broadcast_support(node: Expr, shape: tuple[int, ...], support: Support) -> Support:
  if support is None or node.shape == shape:
    return support
  if len(support) == node.size and int(np.prod(shape, dtype=np.int64)) > 1024:
    return None
  return _make_support(broadcast_entries(support_indices(support), node.shape, shape)[0])


def _pattern(entries: np.ndarray, shape: tuple[int, int], batches: int) -> CanonicalPattern:
  """Place each batch's support in one diagonal block."""
  rows, cols = shape
  return CanonicalPattern(
    (batches * rows, batches * cols),
    tuple(map(int, entries // cols)),
    tuple(map(int, entries // (rows * cols) * cols + entries % cols)),
  )


def _combine(a: Support, b: Support, *, intersection: bool) -> Support:
  if a is None or b is None:
    return (b if a is None else a) if intersection else None
  if a is b or a == b:
    return a
  result = a & b if intersection else a | b
  if isinstance(result, frozenset) and len(result) > 1024:
    result = _make_support(support_indices(result))
  return a if result == a else b if result == b else result


def _constant_condition(condition: Expr, entries: np.ndarray, shape: tuple[int, ...]) -> np.ndarray | None:
  """Read a constant condition only at the requested broadcast coordinates."""
  source = condition.args[0] if condition.op == "pack" else condition
  if source.op != "const":
    return None
  assert source.value is not None
  logical = project_entries(entries, condition.shape, shape)
  if condition.type.pattern is None:
    return source.value.reshape(-1)[logical]
  stored = stored_indices(condition.type)
  slots = np.searchsorted(stored, logical)
  keep = slots < len(stored)
  keep[keep] &= stored[slots[keep]] == logical[keep]
  selected = np.zeros(len(entries), dtype=bool)
  selected[keep] = source.value.reshape(-1)[slots[keep]]
  return selected


def _rule(node: Expr, op: str | None, support_of: Callable[[Expr], Support]) -> Support:
  if op == "pack":
    support = support_of(node.args[0])
    indices = stored_indices(node.type)
    return _make_support(indices if support is None else indices[support_indices(support)])
  if op == "values":
    support = support_of(node.args[0])
    if support is None:
      return None
    return _make_support(np.flatnonzero(np.isin(stored_indices(node.args[0].type), support_indices(support))))
  if op == "const":
    assert node.value is not None
    if (value := _uniform_value(node.value)) is not None:
      return frozenset() if value == 0 else _make_support(np.arange(node.size)) if node.size <= 1024 else None
    return _make_support(np.flatnonzero(node.value != 0))
  if op == "preserve":
    return support_of(node.args[0])
  if op == "union":
    a, b = (_broadcast_support(arg, node.shape, support_of(arg)) for arg in node.args)
    return _combine(a, b, intersection=False)
  if op == "intersection":
    a, b = node.args
    sa, sb = support_of(a), support_of(b)
    if sa is None or sb is None:
      return _broadcast_support(b, node.shape, sb) if sa is None else _broadcast_support(a, node.shape, sa)
    if a.shape == b.shape:
      return _combine(sa, sb, intersection=True)
    cost_a = len(sa) * (node.size // a.size) if a.size else 0
    cost_b = len(sb) * (node.size // b.size) if b.size else 0
    if cost_b < cost_a:
      a, b, sa, sb = b, a, sb, sa
    entries = broadcast_entries(support_indices(sa), a.shape, node.shape)[0]
    keep = np.isin(project_entries(entries, b.shape, node.shape), support_indices(sb))
    if a.shape == node.shape and np.all(keep):
      return sa
    return _make_support(entries[keep])
  if op == "select":
    a, b = (_broadcast_support(arg, node.shape, support_of(arg)) for arg in node.args[1:])
    if a is None and b is None:
      return None
    condition = node.args[0]
    if condition.op == "const":
      assert condition.value is not None
      if (uniform := _uniform_value(condition.value)) is not None:
        return a if uniform else b
    combined = _combine(a, b, intersection=False)
    source = condition.args[0] if condition.op == "pack" else condition
    if source.op != "const":
      return combined
    entries = np.arange(node.size) if combined is None else support_indices(combined)
    chosen = _constant_condition(condition, entries, node.shape)
    if chosen is not None:
      return _make_support(
        entries[
          (chosen & (True if a is None else np.isin(entries, support_indices(a))))
          | (~chosen & (True if b is None else np.isin(entries, support_indices(b))))
        ]
      )
    return combined
  if op == "numerator":
    return _broadcast_support(node.args[0], node.shape, support_of(node.args[0]))
  if op == "reduce":
    src = node.args[0]
    axes = node.attrs["axes"]
    if node.attrs["kind"] != "add" and any(src.shape[axis] == 0 for axis in axes):
      return None
    support = support_of(src)
    return None if support is None else _make_support(reduce_entries(support_indices(support), src.shape, axes))
  if op in {"transpose", "slice", "gather"}:
    support = support_of(node.args[0])
    if support is None:
      return None
    entries = support_indices(support)
    if op == "transpose":
      entries = transpose_entries(entries, node.args[0].shape, node.attrs["axes"])
    elif op == "slice":
      entries, _ = slice_entries(entries, node.args[0].shape, node.attrs["index"])
    else:
      entries = np.flatnonzero(np.isin(node.attrs["indices"].reshape(-1), entries))
    return _make_support(entries)
  if op == "scatter":
    support = support_of(node.args[0])
    indices = node.attrs["indices"].reshape(-1)
    return _make_support(indices if support is None else indices[support_indices(support)])
  if op in {"stack", "concat"}:
    if all(support_of(arg) is None for arg in node.args):
      return None
    chunks = []
    offset = 0
    axis = node.attrs.get("axis", 0)
    for i, arg in enumerate(node.args):
      support = support_of(arg)
      entries = support_indices(support) if support is not None else np.arange(arg.size)
      coords = list(np.unravel_index(entries, arg.shape)) if arg.shape else []
      if op == "stack":
        coords.insert(axis, np.full(len(entries), i))
      else:
        coords[axis] = coords[axis] + offset
        offset += arg.shape[axis]
      chunks.append(np.ravel_multi_index(tuple(coords), node.shape))
    return _make_support(np.concatenate(chunks))
  if op == "matmul":
    x, y = node.args
    if not node.size or x.shape[-1] == 0:
      return frozenset()
    a, b = support_of(x), support_of(y)
    if a is None and b is None:
      return None
    m, k = (1, x.shape[0]) if len(x.shape) == 1 else x.shape[-2:]
    n = 1 if len(y.shape) == 1 else y.shape[-1]
    batch = node.shape[: len(node.shape) - (len(x.shape) > 1) - (len(y.shape) > 1)]
    batches = node.size // (m * n)
    ae = None if a is None else broadcast_entries(support_indices(a), (*x.shape[:-2], m, k), (*batch, m, k))[0]
    be = None if b is None else broadcast_entries(support_indices(b), (*y.shape[:-2], k, n), (*batch, k, n))[0]
    if ae is None or be is None:
      if ae is not None:
        rows = np.unique(ae // k)
        entries = (rows[:, None] * n + np.arange(n)).reshape(-1)
      else:
        assert be is not None
        cols = np.unique(be // (k * n) * n + be % n)
        entries = (cols[:, None] // n * (m * n) + np.arange(m) * n + cols[:, None] % n).reshape(-1)
      if len(entries) == node.size:
        return frozenset(range(node.size)) if node.size <= 1024 else None
      return _make_support(entries)
    result = _pattern(ae, (m, k), batches).matmul(_pattern(be, (k, n), batches))
    return _make_support(np.asarray(result.rows, dtype=np.int64) * n + np.asarray(result.cols, dtype=np.int64) % n)
  return None


def reduce_entries(entries: np.ndarray, shape: tuple[int, ...], axes: tuple[int, ...]) -> np.ndarray:
  """Project input coordinates onto the axes retained by a reduction."""
  return project_entries(entries, tuple(1 if axis in axes else dim for axis, dim in enumerate(shape)), shape)


def transpose_entries(entries: np.ndarray, shape: tuple[int, ...], axes: tuple[int, ...]) -> np.ndarray:
  """Permute only the given flat coordinates."""
  if not shape:
    return entries
  coords = np.unravel_index(entries, shape)
  return np.ravel_multi_index(tuple(coords[axis] for axis in axes), tuple(shape[axis] for axis in axes))


def slice_entries(entries: np.ndarray, shape: tuple[int, ...], index: tuple[Any, ...]) -> tuple[np.ndarray, np.ndarray]:
  """Map selected coordinates into a slice and return their selection mask."""
  coords = np.unravel_index(entries, shape) if shape else ()
  keep = np.ones(len(entries), dtype=bool)
  output = np.zeros(len(entries), dtype=np.int64)
  for dim, item, coord in zip(shape, index, coords, strict=True):
    if isinstance(item, int):
      keep &= coord == item % dim
    else:
      start, stop, step = item.indices(dim)
      count = len(range(start, stop, step))
      offset = coord - start
      keep &= (offset % step == 0) & (offset // step >= 0) & (offset // step < count)
      output = output * count + offset // step
  return output[keep], keep


def slice_indices(shape: tuple[int, ...], index: tuple[Any, ...], entries: np.ndarray | None = None) -> np.ndarray:
  """Build source indices over the selected region without enumerating the source."""
  axes = [range(*item.indices(dim)) for dim, item in zip(shape, index, strict=True) if isinstance(item, slice)]
  if entries is not None:
    output_shape = tuple(len(axis) for axis in axes)
    coords = iter(np.unravel_index(entries, output_shape) if output_shape else ())
    result = np.zeros(len(entries), dtype=np.int64)
    for dim, item in zip(shape, index, strict=True):
      result = result * dim + (item % dim if isinstance(item, int) else item.indices(dim)[0] + item.indices(dim)[2] * next(coords))
    return result
  result = np.zeros(tuple(len(axis) for axis in axes), dtype=np.int64)
  axis = 0
  for dim, item in zip(shape, index, strict=True):
    result *= dim
    if isinstance(item, int):
      result += item % dim
    else:
      values = np.arange(*item.indices(dim), dtype=np.int64)
      result += values.reshape((1,) * axis + (len(values),) + (1,) * (len(axes) - axis - 1))
      axis += 1
  return result


def broadcast_entries(entries: np.ndarray, in_shape: tuple[int, ...], out_shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
  """Broadcast given coordinates and return the originating entry for each result."""
  if not len(entries):
    return entries, np.empty(0, dtype=np.int64)
  padded = (1,) * (len(out_shape) - len(in_shape)) + in_shape
  source = np.arange(len(entries))
  output = np.zeros(len(entries), dtype=np.int64)
  coords = np.unravel_index(entries, padded) if padded else ()
  for small, large, coord in zip(padded, out_shape, coords, strict=True):
    if small == 1 and large != 1:
      output = (output[:, None] * large + np.arange(large)).reshape(-1)
      source = np.repeat(source, large)
    else:
      output = output * large + coord[source]
  return output, source


def project_entries(entries: np.ndarray, in_shape: tuple[int, ...], out_shape: tuple[int, ...]) -> np.ndarray:
  """Project result coordinates onto an operand's broadcast axes."""
  coords = np.unravel_index(entries, out_shape) if out_shape else ()
  padded = (1,) * (len(out_shape) - len(in_shape)) + in_shape
  source = np.zeros(len(entries), dtype=np.int64)
  for dim, coord in zip(padded, coords, strict=True):
    source = source * dim + (0 if dim == 1 else coord)
  return source
