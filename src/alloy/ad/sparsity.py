"""Structural (pattern-only) sparsity analysis.

Which entries of a Jacobian can be nonzero, tracked symbolically through the graph without any
values and without AD. Turning a pattern into values is ``sparse.py``; keeping the dependency
one-way is what lets AD ask this module for a pattern.
"""

from __future__ import annotations

import numpy as np

from ..ir.expr import COMMON_ELEMENTWISE_BINARY, COMMON_ELEMENTWISE_UNARY, Expr, ExprOp
from ..ir.types import SparsityType, broadcast_shape


def _depends_on(expr: Expr, wrt: Expr, memo: dict[tuple[int, int], bool]) -> bool:
  key = (expr.id, wrt.id)
  if key not in memo:
    memo[key] = expr.id == wrt.id or any(_depends_on(arg, wrt, memo) for arg in expr.args)
  return memo[key]


def jacobian_sparsity(expr: Expr, wrt: Expr) -> SparsityType:
  """Estimate structural sparsity of ``d vec(expr) / d vec(wrt)``.

  This is purely symbolic: it tracks element dependencies through the graph without using
  numerical values. It is conservative for nonsmooth elementwise ops and block ops, but exact
  for the structural/arithmetic subset currently implemented here.
  """

  return SparsityType.from_mask(_jac_mask(expr, wrt, {}))


def column_coloring(sparsity: SparsityType) -> tuple[int, ...]:
  """Assign each column a color, greedily, so no two columns sharing a row get the same one.

  Columns of one color can be recovered from a single forward pass, so the number of colors is
  the number of passes a compact Jacobian costs.
  """
  mask = sparsity.to_mask()
  ncols = sparsity.shape[1]
  conflicts = (mask.T.astype(np.int8) @ mask.astype(np.int8)) != 0
  colors: list[int] = []
  for col in range(ncols):
    used = {colors[other] for other in range(col) if conflicts[col, other]}
    color = 0
    while color in used:
      color += 1
    colors.append(color)
  return tuple(colors)


def color_groups(colors: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
  """Invert a coloring into the column indices belonging to each color."""
  if not colors:
    return ()
  return tuple(tuple(i for i, c in enumerate(colors) if c == color) for color in range(max(colors) + 1))


def _jac_mask(expr: Expr, wrt: Expr, memo: dict[int, np.ndarray]) -> np.ndarray:
  if expr.id in memo:
    return memo[expr.id]
  mask = _jac_mask_uncached(expr, wrt, memo)
  memo[expr.id] = mask
  return mask


def _jac_mask_uncached(expr: Expr, wrt: Expr, memo: dict[int, np.ndarray]) -> np.ndarray:
  if expr.op == ExprOp.INPUT:
    return np.eye(wrt.size, dtype=bool) if expr.id == wrt.id else np.zeros((expr.size, wrt.size), dtype=bool)
  if expr.op == ExprOp.CONST:
    return np.zeros((expr.size, wrt.size), dtype=bool)
  if expr.op in COMMON_ELEMENTWISE_UNARY:
    return _jac_mask(expr.args[0], wrt, memo)
  if expr.op in COMMON_ELEMENTWISE_BINARY:
    x, y = expr.args
    return _broadcast_mask(_jac_mask(x, wrt, memo), x.shape, expr.shape) | _broadcast_mask(_jac_mask(y, wrt, memo), y.shape, expr.shape)
  if expr.op == ExprOp.SUM:
    return np.any(_jac_mask(expr.args[0], wrt, memo), axis=0, keepdims=True)
  if expr.op == ExprOp.RESHAPE:
    return _jac_mask(expr.args[0], wrt, memo)
  if expr.op == ExprOp.TRANSPOSE:
    order = np.arange(expr.size).reshape(expr.args[0].shape).transpose(expr.attrs["axes"]).reshape(-1)
    return _jac_mask(expr.args[0], wrt, memo)[order]
  if expr.op == ExprOp.SLICE:
    order = np.arange(expr.args[0].size).reshape(expr.args[0].shape)[expr.attrs["index"]].reshape(-1)
    return _jac_mask(expr.args[0], wrt, memo)[order]
  if expr.op == ExprOp.GATHER:
    return _jac_mask(expr.args[0], wrt, memo)[expr.attrs["indices"].reshape(-1)]
  if expr.op == ExprOp.SCATTER:
    ret = np.zeros((expr.size, wrt.size), dtype=bool)
    ret[expr.attrs["indices"].reshape(-1)] = _jac_mask(expr.args[0], wrt, memo)
    return ret
  if expr.op == ExprOp.STACK:
    return _stack_mask(expr, wrt, memo)
  if expr.op == ExprOp.CONCAT:
    return _concat_mask(expr, wrt, memo)
  if expr.op == ExprOp.MATMUL:
    return _matmul_mask(expr, wrt, memo)
  if expr.op == ExprOp.CALL:
    return _call_mask(expr, wrt, memo)
  if expr.op == ExprOp.MAP:
    return _map_mask(expr, wrt, memo)
  if expr.op == ExprOp.SOLVER_CALL:
    # Solver outputs are treated as non-differentiable opaque calls. Implicit
    # function theorem AD through them is future work.
    return np.zeros((expr.size, wrt.size), dtype=bool)
  raise NotImplementedError(f"jacobian sparsity for op {expr.op!r} is not implemented")


def _broadcast_mask(mask: np.ndarray, in_shape: tuple[int, ...], out_shape: tuple[int, ...]) -> np.ndarray:
  if in_shape == out_shape:
    return mask
  if not in_shape:
    return np.repeat(mask, int(np.prod(out_shape, dtype=int)), axis=0)
  _ = broadcast_shape(in_shape, out_shape)
  source = np.arange(int(np.prod(in_shape, dtype=int))).reshape(in_shape)
  source = np.broadcast_to(source, out_shape).reshape(-1)
  return mask[source]


def _stack_mask(expr: Expr, wrt: Expr, memo: dict[int, np.ndarray]) -> np.ndarray:
  base = expr.args[0].shape
  axis = expr.attrs.get("axis", 0)
  child_index = np.stack([np.full(base, i, dtype=np.int64) for i in range(len(expr.args))], axis=axis).reshape(-1)
  elem_index = np.stack([np.arange(expr.args[i].size, dtype=np.int64).reshape(base) for i in range(len(expr.args))], axis=axis).reshape(-1)
  child_masks = [_jac_mask(arg, wrt, memo) for arg in expr.args]
  return np.stack([child_masks[int(child)][int(elem)] for child, elem in zip(child_index, elem_index, strict=True)], axis=0)


def _concat_mask(expr: Expr, wrt: Expr, memo: dict[int, np.ndarray]) -> np.ndarray:
  axis = expr.attrs.get("axis", 0)
  child_index = np.concatenate([np.full(arg.shape, i, dtype=np.int64) for i, arg in enumerate(expr.args)], axis=axis).reshape(-1)
  elem_index = np.concatenate([np.arange(arg.size, dtype=np.int64).reshape(arg.shape) for arg in expr.args], axis=axis).reshape(-1)
  child_masks = [_jac_mask(arg, wrt, memo) for arg in expr.args]
  return np.stack([child_masks[int(child)][int(elem)] for child, elem in zip(child_index, elem_index, strict=True)], axis=0)


def _matmul_mask(expr: Expr, wrt: Expr, memo: dict[int, np.ndarray]) -> np.ndarray:
  x, y = expr.args
  xm, ym = _jac_mask(x, wrt, memo), _jac_mask(y, wrt, memo)
  rows: list[np.ndarray] = []
  if len(x.shape) == 1 and len(y.shape) == 1:
    rows.append(np.any(xm, axis=0) | np.any(ym, axis=0))
  elif len(x.shape) == 2 and len(y.shape) == 1:
    for i in range(x.shape[0]):
      x_rows = [i * x.shape[1] + k for k in range(x.shape[1])]
      y_rows = list(range(y.shape[0]))
      rows.append(np.any(xm[x_rows], axis=0) | np.any(ym[y_rows], axis=0))
  elif len(x.shape) == 1 and len(y.shape) == 2:
    for j in range(y.shape[1]):
      x_rows = list(range(x.shape[0]))
      y_rows = [k * y.shape[1] + j for k in range(y.shape[0])]
      rows.append(np.any(xm[x_rows], axis=0) | np.any(ym[y_rows], axis=0))
  elif len(x.shape) == 2 and len(y.shape) == 2:
    for i in range(x.shape[0]):
      for j in range(y.shape[1]):
        x_rows = [i * x.shape[1] + k for k in range(x.shape[1])]
        y_rows = [k * y.shape[1] + j for k in range(y.shape[0])]
        rows.append(np.any(xm[x_rows], axis=0) | np.any(ym[y_rows], axis=0))
  else:  # pragma: no cover - matmul construction rejects this today
    raise NotImplementedError(f"matmul sparsity for {x.shape} @ {y.shape}")
  return np.stack(rows, axis=0)


def _call_mask(expr: Expr, wrt: Expr, memo: dict[int, np.ndarray]) -> np.ndarray:
  callee = expr.attrs["callee"]
  callee_out = callee.outputs[expr.attrs["output"]]
  ret = np.zeros((expr.size, wrt.size), dtype=bool)
  for formal, actual in zip(callee.inputs, expr.args, strict=True):
    callee_dep = _jac_mask(callee_out, formal, {})
    actual_dep = _jac_mask(actual, wrt, memo)
    ret |= (callee_dep.astype(np.int8) @ actual_dep.astype(np.int8)) != 0
  return ret


def _map_mask(expr: Expr, wrt: Expr, memo: dict[int, np.ndarray]) -> np.ndarray:
  callee = expr.attrs["callee"]
  callee_out = callee.outputs[expr.attrs["output"]]
  length = expr.attrs["length"]
  starts = expr.attrs["starts"]
  strides = expr.attrs["strides"]
  slice_size = expr.attrs["slice_size"]
  ret = np.zeros((expr.size, wrt.size), dtype=bool)
  # The per-iteration callee dependency tile is the same for every it — only the actual-input
  # mask rows change, since each iteration slices a different range of the outer dependency mask.
  for formal_idx, actual_outer in enumerate(expr.args):
    formal = callee.inputs[formal_idx]
    callee_dep = _jac_mask(callee_out, formal, {})  # (slice_size, formal.size)
    outer_dep = _jac_mask(actual_outer, wrt, memo)  # (outer_size, wrt.size)
    start = starts[formal_idx]
    stride = strides[formal_idx]
    formal_size = formal.size
    callee_dep_i8 = callee_dep.astype(np.int8)
    for it in range(length):
      row0 = start + it * stride
      contribution = (callee_dep_i8 @ outer_dep[row0 : row0 + formal_size].astype(np.int8)) != 0
      ret[it * slice_size : (it + 1) * slice_size] |= contribution
  return ret
