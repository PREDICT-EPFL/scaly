"""Structural (pattern-only) sparsity analysis.

Which entries of a Jacobian can be nonzero, tracked symbolically through the graph without any
values and without AD. Turning a pattern into values is ``sparse.py``; keeping the dependency
one-way is what lets AD ask this module for a pattern.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from ..ir.expr import COMMON_ELEMENTWISE_BINARY, COMMON_ELEMENTWISE_UNARY, Expr, ExprOp
from ..ir.types import SparsityPattern, broadcast_shape


def _depends_on(expr: Expr, wrt: Expr, memo: dict[tuple[int, int], bool]) -> bool:
  stack = [(expr, False)]
  while stack:
    node, visited = stack.pop()
    key = (node.id, wrt.id)
    if key in memo:
      continue
    if node.id == wrt.id:
      memo[key] = True
    elif visited:
      memo[key] = any(memo[(arg.id, wrt.id)] for arg in node.args)
    else:
      stack.append((node, True))
      stack.extend((arg, False) for arg in reversed(node.args) if (arg.id, wrt.id) not in memo)
  return memo[(expr.id, wrt.id)]


def jacobian_sparsity(expr: Expr, wrt: Expr) -> SparsityPattern:
  """Estimate structural sparsity of ``d vec(expr) / d vec(wrt)``.

  This is purely symbolic: it tracks element dependencies through the graph without using
  numerical values. It is conservative for nonsmooth elementwise ops and block ops, but exact
  for the structural/arithmetic subset currently implemented here.
  """
  return _mask_sparsity(_jac_mask(expr, wrt, {}))


def column_coloring(sparsity: SparsityPattern) -> tuple[int, ...]:
  """Assign each column a color, greedily, so no two columns sharing a row get the same one.

  Columns of one color can be recovered from a single forward pass, so the number of colors is
  the number of passes a compact Jacobian costs.
  """
  row_ptr, col_ind, _ = sparsity.to_csr()
  col_ptr, row_ind, _ = sparsity.to_csc()
  colors: list[int] = []
  for col in range(sparsity.shape[1]):
    used = {colors[other] for row in row_ind[col_ptr[col] : col_ptr[col + 1]] for other in col_ind[row_ptr[row] : row_ptr[row + 1]] if other < col}
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


def _empty(shape: tuple[int, int]) -> sparse.csr_array:
  return sparse.csr_array(shape, dtype=bool)


def _mask_sparsity(mask: sparse.csr_array) -> SparsityPattern:
  mask.sort_indices()
  coo = mask.tocoo()
  return SparsityPattern(mask.shape, tuple(int(x) for x in coo.row), tuple(int(x) for x in coo.col))


def _or(x: sparse.csr_array, y: sparse.csr_array) -> sparse.csr_array:
  return sparse.csr_array(x + y, dtype=bool)


def _compose(outer: sparse.csr_array, inner: sparse.csr_array) -> sparse.csr_array:
  return sparse.csr_array(outer @ inner, dtype=bool)


def _incidence(shape: tuple[int, int], rows: np.ndarray, cols: np.ndarray) -> sparse.csr_array:
  return sparse.csr_array((np.ones(rows.size, dtype=bool), (rows, cols)), shape=shape)


def _jac_mask(expr: Expr, wrt: Expr, memo: dict[tuple[int, int], sparse.csr_array]) -> sparse.csr_array:
  stack = [(expr, wrt, 0)]
  while stack:
    node, variable, phase = stack.pop()
    key = (node.id, variable.id)
    if key in memo:
      continue
    if phase == 0:
      stack.append((node, variable, 1))
      if node.op != ExprOp.VMAP or node.attrs["length"]:
        stack.extend((arg, variable, 0) for arg in reversed(node.args) if (arg.id, variable.id) not in memo)
    elif phase == 1 and node.op in (ExprOp.CALL, ExprOp.VMAP):
      stack.append((node, variable, 2))
      callee = node.attrs["callee"]
      callee_out = callee.outputs[node.attrs["output"]]
      # Actual masks must be known before requesting callee masks for active formals.
      if node.op == ExprOp.CALL or node.attrs["length"]:
        for formal, actual in reversed(tuple(zip(callee.inputs, node.args, strict=True))):
          if memo[(actual.id, variable.id)].nnz and (callee_out.id, formal.id) not in memo:
            stack.append((callee_out, formal, 0))
    else:
      memo[key] = _jac_mask_uncached(node, variable, memo)

  return memo[(expr.id, wrt.id)]


def _jac_mask_uncached(expr: Expr, wrt: Expr, memo: dict[tuple[int, int], sparse.csr_array]) -> sparse.csr_array:
  if expr.op == ExprOp.INPUT:
    return sparse.eye_array(wrt.size, format="csr", dtype=bool) if expr.id == wrt.id else _empty((expr.size, wrt.size))
  if expr.op == ExprOp.CONST:
    return _empty((expr.size, wrt.size))
  if expr.op in COMMON_ELEMENTWISE_UNARY:
    return _jac_mask(expr.args[0], wrt, memo)
  if expr.op in COMMON_ELEMENTWISE_BINARY:
    x, y = expr.args
    return _or(_broadcast_mask(_jac_mask(x, wrt, memo), x.shape, expr.shape), _broadcast_mask(_jac_mask(y, wrt, memo), y.shape, expr.shape))
  if expr.op == ExprOp.SUM:
    child = _jac_mask(expr.args[0], wrt, memo)
    incidence = _incidence((1, child.shape[0]), np.zeros(child.shape[0], dtype=np.int64), np.arange(child.shape[0]))
    return _compose(incidence, child)
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
    child = _jac_mask(expr.args[0], wrt, memo)
    indices = expr.attrs["indices"].reshape(-1)
    return _compose(_incidence((expr.size, child.shape[0]), indices, np.arange(child.shape[0])), child)
  if expr.op == ExprOp.STACK:
    return _stack_mask(expr, wrt, memo)
  if expr.op == ExprOp.CONCAT:
    return _concat_mask(expr, wrt, memo)
  if expr.op == ExprOp.MATMUL:
    return _matmul_mask(expr, wrt, memo)
  if expr.op == ExprOp.CALL:
    return _call_mask(expr, wrt, memo)
  if expr.op == ExprOp.VMAP:
    return _vmap_mask(expr, wrt, memo)
  if expr.op == ExprOp.SOLVER_CALL:
    if any(_jac_mask(arg, wrt, memo).nnz for arg in expr.args):
      raise NotImplementedError("active derivative through SOLVER_CALL is not implemented")
    return _empty((expr.size, wrt.size))
  raise NotImplementedError(f"jacobian sparsity for op {expr.op!r} is not implemented")


def _broadcast_mask(mask: sparse.csr_array, in_shape: tuple[int, ...], out_shape: tuple[int, ...]) -> sparse.csr_array:
  if in_shape == out_shape:
    return mask
  _ = broadcast_shape(in_shape, out_shape)
  if not in_shape:
    return mask[np.zeros(int(np.prod(out_shape, dtype=int)), dtype=np.int64)]
  source = np.arange(int(np.prod(in_shape, dtype=int))).reshape(in_shape)
  return mask[np.broadcast_to(source, out_shape).reshape(-1)]


def _combine_children(
  expr: Expr, wrt: Expr, memo: dict[tuple[int, int], sparse.csr_array], child_index: np.ndarray, elem_index: np.ndarray
) -> sparse.csr_array:
  offsets = np.cumsum([0, *(arg.size for arg in expr.args[:-1])])
  children = sparse.vstack([_jac_mask(arg, wrt, memo) for arg in expr.args], format="csr")
  return children[offsets[child_index] + elem_index]


def _stack_mask(expr: Expr, wrt: Expr, memo: dict[tuple[int, int], sparse.csr_array]) -> sparse.csr_array:
  base = expr.args[0].shape
  axis = expr.attrs.get("axis", 0)
  child = np.stack([np.full(base, i, dtype=np.int64) for i in range(len(expr.args))], axis=axis).reshape(-1)
  elem = np.stack([np.arange(arg.size, dtype=np.int64).reshape(base) for arg in expr.args], axis=axis).reshape(-1)
  return _combine_children(expr, wrt, memo, child, elem)


def _concat_mask(expr: Expr, wrt: Expr, memo: dict[tuple[int, int], sparse.csr_array]) -> sparse.csr_array:
  axis = expr.attrs.get("axis", 0)
  child = np.concatenate([np.full(arg.shape, i, dtype=np.int64) for i, arg in enumerate(expr.args)], axis=axis).reshape(-1)
  elem = np.concatenate([np.arange(arg.size, dtype=np.int64).reshape(arg.shape) for arg in expr.args], axis=axis).reshape(-1)
  return _combine_children(expr, wrt, memo, child, elem)


def _matmul_mask(expr: Expr, wrt: Expr, memo: dict[tuple[int, int], sparse.csr_array]) -> sparse.csr_array:
  x, y = expr.args
  xm, ym = _jac_mask(x, wrt, memo), _jac_mask(y, wrt, memo)
  x_rows: list[int] = []
  y_rows: list[int] = []
  out_rows: list[int] = []
  if len(x.shape) == 1 and len(y.shape) == 1:
    entries = ((0, k, k) for k in range(x.shape[0]))
  elif len(x.shape) == 2 and len(y.shape) == 1:
    entries = ((i, i * x.shape[1] + k, k) for i in range(x.shape[0]) for k in range(x.shape[1]))
  elif len(x.shape) == 1 and len(y.shape) == 2:
    entries = ((j, k, k * y.shape[1] + j) for j in range(y.shape[1]) for k in range(x.shape[0]))
  elif len(x.shape) == 2 and len(y.shape) == 2:
    entries = (
      (i * y.shape[1] + j, i * x.shape[1] + k, k * y.shape[1] + j) for i in range(x.shape[0]) for j in range(y.shape[1]) for k in range(x.shape[1])
    )
  else:  # pragma: no cover
    raise NotImplementedError(f"matmul sparsity for {x.shape} @ {y.shape}")
  for out, xr, yr in entries:
    out_rows.append(out)
    x_rows.append(xr)
    y_rows.append(yr)
  out = np.asarray(out_rows, dtype=np.int64)
  return _or(
    _compose(_incidence((expr.size, x.size), out, np.asarray(x_rows, dtype=np.int64)), xm),
    _compose(_incidence((expr.size, y.size), out, np.asarray(y_rows, dtype=np.int64)), ym),
  )


def _call_mask(expr: Expr, wrt: Expr, memo: dict[tuple[int, int], sparse.csr_array]) -> sparse.csr_array:
  callee = expr.attrs["callee"]
  callee_out = callee.outputs[expr.attrs["output"]]
  ret = _empty((expr.size, wrt.size))
  for formal, actual in zip(callee.inputs, expr.args, strict=True):
    actual_dep = _jac_mask(actual, wrt, memo)
    if actual_dep.nnz:
      ret = _or(ret, _compose(_jac_mask(callee_out, formal, memo), actual_dep))
  return ret


def _vmap_mask(expr: Expr, wrt: Expr, memo: dict[tuple[int, int], sparse.csr_array]) -> sparse.csr_array:
  callee = expr.attrs["callee"]
  callee_out = callee.outputs[expr.attrs["output"]]
  length = expr.attrs["length"]
  starts = expr.attrs["starts"]
  strides = expr.attrs["strides"]
  ret = _empty((expr.size, wrt.size))
  if not length:
    return ret
  for formal_idx, actual_outer in enumerate(expr.args):
    formal = callee.inputs[formal_idx]
    outer_dep = _jac_mask(actual_outer, wrt, memo)
    if not outer_dep.nnz:
      continue
    callee_dep = _jac_mask(callee_out, formal, memo)
    start, stride = starts[formal_idx], strides[formal_idx]
    window_cols = np.repeat(start + np.arange(length) * stride, formal.size) + np.tile(np.arange(formal.size), length)
    windows = _incidence((length * formal.size, actual_outer.size), np.arange(length * formal.size), window_cols)
    tiled = sparse.kron(sparse.eye_array(length, dtype=bool), callee_dep, format="csr")
    ret = _or(ret, _compose(tiled, _compose(windows, outer_dep)))
  return ret


def star_coloring(sparsity: SparsityPattern) -> tuple[int, ...]:
  """Greedily star-color a square sparsity pattern in column order.

  The pattern is treated as an undirected graph. A valid coloring is proper, and no simple path
  of three edges has only two colors. This is the coloring needed to recover a symmetric Hessian
  from compressed forward products. It is deliberately not distance-2 coloring.
  """
  rows, cols = sparsity.shape
  if rows != cols:
    raise ValueError(f"star coloring requires a square sparsity pattern, got {sparsity.shape}")

  neighbors = [set() for _ in range(rows)]
  for row, col in zip(sparsity.rows, sparsity.cols, strict=True):
    if row != col:
      neighbors[row].add(col)
      neighbors[col].add(row)

  colors = [-1] * rows
  for vertex in range(rows):
    color = 0
    while _star_color_conflicts(vertex, color, colors, neighbors):
      color += 1
    colors[vertex] = color
  return tuple(colors)


def _star_color_conflicts(vertex: int, color: int, colors: list[int], neighbors: list[set[int]]) -> bool:
  """Return whether adding ``vertex`` with ``color`` creates a two-colored three-edge path."""
  if any(colors[neighbor] == color for neighbor in neighbors[vertex] if colors[neighbor] >= 0):
    return True

  # The new vertex is an endpoint: vertex-u-w-x has colors c,d,c,d. Excluding u from the final
  # neighbor check keeps this a simple path rather than the backtracking walk vertex-u-w-u.
  for first in neighbors[vertex]:
    first_color = colors[first]
    if first_color < 0 or first_color == color:
      continue
    for middle in neighbors[first]:
      if colors[middle] != color:
        continue
      if any(last != first and last != vertex and colors[last] == first_color for last in neighbors[middle]):
        return True

  # The new vertex is internal: first-vertex-middle-last has colors d,c,d,c. Two distinct
  # neighbors of the vertex must share d, and the second one must have a c-colored neighbor.
  by_color: dict[int, list[int]] = {}
  for neighbor in neighbors[vertex]:
    neighbor_color = colors[neighbor]
    if neighbor_color >= 0:
      by_color.setdefault(neighbor_color, []).append(neighbor)
  for same_color_neighbors in by_color.values():
    if len(same_color_neighbors) < 2:
      continue
    for middle in same_color_neighbors:
      if any(last not in same_color_neighbors and last != vertex and colors[last] == color for last in neighbors[middle]):
        return True
  return False


def _symmetrize_sparsity(sparsity: SparsityPattern) -> SparsityPattern:
  """Return the undirected union of a square structural pattern and its transpose."""
  if sparsity.shape[0] != sparsity.shape[1]:
    raise ValueError(f"symmetric sparsity requires a square pattern, got {sparsity.shape}")
  rows = np.asarray(sparsity.rows, dtype=np.int64)
  cols = np.asarray(sparsity.cols, dtype=np.int64)
  mask = sparse.csr_array((np.ones(rows.size, dtype=bool), (rows, cols)), shape=sparsity.shape)
  return _mask_sparsity(mask.maximum(mask.T))
