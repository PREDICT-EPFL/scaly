"""Lazy value support over flat logical coordinates, using the fixed pattern algebra."""

from __future__ import annotations

from collections.abc import Set, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, cast

import numpy as np

from .types import CanonicalPattern

if TYPE_CHECKING:
  from .expr import Expr, ExprOp, OpInfo


@dataclass(frozen=True, slots=True, eq=False)
class _SupportSet(Set[int]):
  _entries: np.ndarray

  def __init__(self, entries: np.ndarray):
    values = np.unique(np.asarray(entries, dtype=np.int64))
    object.__setattr__(self, "_entries", np.frombuffer(values.tobytes(), dtype=np.int64))

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

  def __and__(self, other):
    return _make_support(np.intersect1d(self._entries, support_indices(other), assume_unique=True))

  def __or__(self, other):
    return _make_support(np.union1d(self._entries, support_indices(other)))

  __rand__ = __and__
  __ror__ = __or__


type Support = frozenset[int] | _SupportSet | None


def support_indices(support: frozenset[int] | _SupportSet) -> np.ndarray:
  """Read known support as a coordinate array without copying an existing compact cache."""
  return support._entries if isinstance(support, _SupportSet) else np.fromiter(support, dtype=np.int64, count=len(support))


def _make_support(entries: np.ndarray) -> frozenset[int] | _SupportSet:
  return frozenset(map(int, entries)) if len(entries) <= 1024 else _SupportSet(entries)


def compute_support(root: Expr, rules: Mapping[ExprOp, OpInfo]) -> Support:
  """Compute and cache support without recursing through operands or callees."""
  stack = [(root, False)]
  while stack:
    node, visited = stack.pop()
    if node._support_cache is not None:
      continue
    info = rules[cast("ExprOp", node.op)]
    rule = info.pattern_rule or info.support_rule
    if not visited:
      stack.append((node, True))
      children = node.args
      if rule in {"call", "vmap"} and (rule == "call" or node.attrs["length"]):
        children = (*children, node.attrs["callee"].results[node.attrs["output"]])
      stack.extend((arg, False) for arg in reversed(children) if arg._support_cache is None)
      continue
    result = _rule(node, rule)
    if node.size == 0:
      result = frozenset()
    elif (pattern := node.type.pattern) is not None:
      matrix_size = pattern.shape[0] * pattern.shape[1]
      bound = _make_support(
        np.fromiter(
          (
            batch * matrix_size + r * pattern.shape[1] + c
            for batch in range(node.size // matrix_size)
            for r, c in zip(pattern.rows, pattern.cols, strict=True)
          ),
          dtype=np.int64,
        )
      )
      result = bound if result is None else result & bound
    object.__setattr__(node, "_support_cache", (result,))
  return _cached(root)


def _cached(node: Expr) -> Support:
  assert node._support_cache is not None
  return node._support_cache[0]


def _entries(mask: np.ndarray) -> frozenset[int] | _SupportSet:
  return _make_support(np.flatnonzero(mask))


def broadcast_support(node: Expr, shape: tuple[int, ...]) -> Support:
  """Return a node's support in the coordinates of a broadcast result."""
  support = node.support
  if support is None or node.shape == shape:
    return support
  return _make_support(broadcast_entries(support_indices(support), node.shape, shape)[0])


def _pattern(support: Support, shape: tuple[int, int]) -> CanonicalPattern:
  entries = range(shape[0] * shape[1]) if support is None else support
  return CanonicalPattern(shape, tuple(i // shape[1] for i in entries), tuple(i % shape[1] for i in entries))


def _combine(a: Support, b: Support, *, intersection: bool) -> Support:
  if a is None or b is None:
    return (b if a is None else a) if intersection else None
  if a is b or a == b:
    return a
  result = a & b if intersection else a | b
  return a if result == a else b if result == b else result


def _rule(node: Expr, op: str | None) -> Support:
  if op == "const":
    assert node.value is not None
    if node.value.size and all(stride == 0 or dim <= 1 for stride, dim in zip(node.value.strides, node.shape, strict=True)):
      return frozenset() if node.value.flat[0] == 0 else _make_support(np.arange(node.size)) if node.size <= 1024 else None
    return _entries(node.value != 0)
  if op == "preserve":
    return _cached(node.args[0])
  if op in {"union", "intersection"}:
    a, b = (broadcast_support(arg, node.shape) for arg in node.args)
    return _combine(a, b, intersection=op == "intersection")
  if op == "numerator":
    return broadcast_support(node.args[0], node.shape)
  if op == "sum":
    return frozenset() if _cached(node.args[0]) == frozenset() else frozenset({0})
  if op in {"transpose", "slice", "gather"}:
    if _cached(node.args[0]) is None:
      return None
    support = _cached(node.args[0])
    assert support is not None
    entries = support_indices(support)
    if op == "transpose":
      entries = transpose_entries(entries, node.args[0].shape, node.attrs["axes"])
    elif op == "slice":
      entries, _ = slice_entries(entries, node.args[0].shape, node.attrs["index"])
    else:
      entries = np.flatnonzero(np.isin(node.attrs["indices"].reshape(-1), entries))
    return _make_support(entries)
  if op == "scatter":
    support = _cached(node.args[0])
    indices = node.attrs["indices"].reshape(-1)
    return _make_support(indices if support is None else indices[support_indices(support)])
  if op in {"stack", "concat"}:
    if all(_cached(arg) is None for arg in node.args):
      return None
    result = frozenset()
    offset = 0
    axis = node.attrs.get("axis", 0)
    for i, arg in enumerate(node.args):
      support = _cached(arg)
      entries = support_indices(support) if support is not None else np.arange(arg.size)
      coords = list(np.unravel_index(entries, arg.shape)) if arg.shape else []
      if op == "stack":
        coords.insert(axis, np.full(len(entries), i))
      else:
        coords[axis] = coords[axis] + offset
        offset += arg.shape[axis]
      result = result | _make_support(np.ravel_multi_index(tuple(coords), node.shape))
    return result
  if op == "matmul":
    x, y = node.args
    if x.shape[-1] == 0:
      return frozenset()
    a, b = _cached(x), _cached(y)
    if a is None and b is None:
      return None
    xshape = (1, x.shape[0]) if len(x.shape) == 1 else (x.shape[0], x.shape[1])
    yshape = (y.shape[0], 1) if len(y.shape) == 1 else (y.shape[0], y.shape[1])
    if a is None or b is None:
      known = b if a is None else a
      assert known is not None
      entries = support_indices(known)
      rows = np.arange(xshape[0]) if a is None else np.unique(entries // xshape[1])
      cols = np.unique(entries % yshape[1]) if a is None else np.arange(yshape[1])
      if len(rows) == xshape[0] and len(cols) == yshape[1]:
        return frozenset(range(node.size)) if node.size <= 1024 else None
      return _make_support((rows[:, None] * yshape[1] + cols).reshape(-1))
    result = _pattern(a, xshape).matmul(_pattern(b, yshape))
    return _make_support(np.asarray(result.rows, dtype=np.int64) * result.shape[1] + np.asarray(result.cols, dtype=np.int64))
  if op in {"call", "vmap"}:
    if op == "vmap" and not node.attrs["length"]:
      return frozenset()
    output = node.attrs["callee"].results[node.attrs["output"]]
    support = _cached(output)
    if op == "call" or support is None:
      return support
    return frozenset(stage * output.size + i for stage in range(node.attrs["length"]) for i in support)
  return None


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
