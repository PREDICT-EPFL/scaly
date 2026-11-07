"""Lazy value support over flat logical coordinates, using the fixed pattern algebra."""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, cast

import numpy as np

from .types import CanonicalPattern

if TYPE_CHECKING:
  from .expr import Expr, ExprOp, OpInfo

type Support = frozenset[int] | None


def compute_support(root: Expr, rules: Mapping[ExprOp, OpInfo]) -> Support:
  stack = [(root, False)]
  while stack:
    node, visited = stack.pop()
    if node._support_cache is not None:
      continue
    info = rules.get(cast("ExprOp", node.op))
    rule = None if info is None else info.pattern_rule or info.support_rule
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
      bound = frozenset(
        batch * matrix_size + r * pattern.shape[1] + c
        for batch in range(node.size // matrix_size)
        for r, c in zip(pattern.rows, pattern.cols, strict=True)
      )
      result = bound if result is None else result & bound
    object.__setattr__(node, "_support_cache", (result,))
  return _cached(root)


def _cached(node: Expr) -> Support:
  assert node._support_cache is not None
  return node._support_cache[0]


def _mask(node: Expr) -> np.ndarray:
  support = _cached(node)
  if support is None:
    return np.ones(node.shape, dtype=bool)
  mask = np.zeros(node.size, dtype=bool)
  mask[list(support)] = True
  return mask.reshape(node.shape)


def _entries(mask: np.ndarray) -> frozenset[int]:
  return frozenset(map(int, np.flatnonzero(mask)))


def broadcast_support(node: Expr, shape: tuple[int, ...]) -> Support:
  """Return a node's support in the coordinates of a broadcast result."""
  support = node.support
  return support if support is None or node.shape == shape else _entries(np.broadcast_to(_mask(node), shape))


def _pattern(support: Support, shape: tuple[int, int]) -> CanonicalPattern:
  entries = range(shape[0] * shape[1]) if support is None else support
  return CanonicalPattern(shape, tuple(i // shape[1] for i in entries), tuple(i % shape[1] for i in entries))


def _combine(a: Support, b: Support, size: int, *, intersection: bool) -> Support:
  if a is None or b is None:
    return (b if a is None else a) if intersection else None
  x, y = _pattern(a, (1, size)), _pattern(b, (1, size))
  result = x.intersection(y) if intersection else x.union(y)
  return frozenset(result.cols)


def _rule(node: Expr, op: str | None) -> Support:
  if op == "const":
    assert node.value is not None
    return _entries(node.value != 0)
  if op == "preserve":
    return _cached(node.args[0])
  if op in {"union", "intersection"}:
    a, b = (broadcast_support(arg, node.shape) for arg in node.args)
    return _combine(a, b, node.size, intersection=op == "intersection")
  if op == "numerator":
    return broadcast_support(node.args[0], node.shape)
  if op == "sum":
    return frozenset() if _cached(node.args[0]) == frozenset() else frozenset({0})
  if op in {"transpose", "slice", "gather"}:
    if _cached(node.args[0]) is None:
      return None
    mask = _mask(node.args[0])
    if op == "transpose":
      mask = mask.transpose(node.attrs["axes"])
    elif op == "slice":
      mask = mask[node.attrs["index"]]
    else:
      mask = mask.reshape(-1)[node.attrs["indices"]]
    return _entries(mask)
  if op == "scatter":
    support = _cached(node.args[0])
    indices = node.attrs["indices"].reshape(-1)
    return frozenset(map(int, indices if support is None else indices[list(support)]))
  if op in {"stack", "concat"}:
    if all(_cached(arg) is None for arg in node.args):
      return None
    masks = [_mask(arg) for arg in node.args]
    combine = np.stack if op == "stack" else np.concatenate
    return _entries(combine(masks, axis=node.attrs.get("axis", 0)))
  if op == "matmul":
    x, y = node.args
    if x.shape[-1] == 0:
      return frozenset()
    a, b = _cached(x), _cached(y)
    if a is None and b is None:
      return None
    xshape = (1, x.shape[0]) if len(x.shape) == 1 else (x.shape[0], x.shape[1])
    yshape = (y.shape[0], 1) if len(y.shape) == 1 else (y.shape[0], y.shape[1])
    result = _pattern(a, xshape).matmul(_pattern(b, yshape))
    return frozenset(r * result.shape[1] + c for r, c in zip(result.rows, result.cols, strict=True))
  if op in {"call", "vmap"}:
    if op == "vmap" and not node.attrs["length"]:
      return frozenset()
    output = node.attrs["callee"].results[node.attrs["output"]]
    support = _cached(output)
    if op == "call" or support is None:
      return support
    return frozenset(stage * output.size + i for stage in range(node.attrs["length"]) for i in support)
  return None
