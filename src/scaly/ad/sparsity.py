"""Structural (pattern-only) sparsity analysis.

Which entries of a Jacobian can be nonzero, tracked symbolically through the graph without any
values and without AD. Turning a pattern into values is ``sparse.py``; keeping the dependency
one-way is what lets AD ask this module for a pattern.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse

from ..ir.expr import COMMON_ELEMENTWISE_BINARY, COMMON_ELEMENTWISE_UNARY, OP_INFO, Expr, ExprOp
from ..ir.support import Support, broadcast_entries, transpose_entries, slice_entries, support_indices, project_entries
from ..ir.types import SparsityPattern, broadcast_shape


def _depends_on(expr: Expr, wrt: Expr, memo: dict[tuple[int, int], bool], *, through_stops: bool = True) -> bool:
  """Trace value dependencies, or derivative activity when ``through_stops`` is false."""
  stack = [(expr, False)]
  while stack:
    node, visited = stack.pop()
    key = (node.id, wrt.id)
    if key in memo:
      continue
    if node.id == wrt.id:
      memo[key] = True
    elif not through_stops and node.op == ExprOp.STOP_GRADIENT:
      memo[key] = False
    elif visited:
      memo[key] = any(memo[(arg.id, wrt.id)] for arg in node.args)
    else:
      stack.append((node, True))
      stack.extend((arg, False) for arg in reversed(node.args) if (arg.id, wrt.id) not in memo)
  return memo[(expr.id, wrt.id)]


def _check_wrt(wrt: Expr) -> None:
  if wrt.op != ExprOp.INPUT:
    raise ValueError("wrt must be an independent input from sc.sym or a Function declaration")


def _prunable_support(expr: Expr) -> Support:
  """Known zeros that can be removed without dropping custom derivative paths."""
  support = expr.support
  if support is None or len(support) == expr.size:
    return support
  pending = [(expr, False)]
  while pending:
    node, visited = pending.pop()
    if node._custom_derivative_cache is not None:
      continue
    if node.op == ExprOp.STOP_GRADIENT:
      object.__setattr__(node, "_custom_derivative_cache", False)
      continue
    children = node.args
    if node.op in {ExprOp.CALL, ExprOp.VMAP}:
      callee, output = node.attrs["callee"], node.attrs["output"]
      if not visited and _declared(callee, output):
        object.__setattr__(node, "_custom_derivative_cache", True)
        continue
      children = (*children, callee.results[output])
    if visited:
      object.__setattr__(node, "_custom_derivative_cache", any(child._custom_derivative_cache for child in children))
    else:
      pending.append((node, True))
      pending.extend((child, False) for child in children if child._custom_derivative_cache is None)
  if expr._custom_derivative_cache:
    return None
  return support


def jacobian_sparsity(expr: Expr, wrt: Expr) -> SparsityPattern:
  """Estimate structural sparsity of ``d vec(expr) / d vec(wrt)``.

  ``wrt`` must be an independent input from ``sc.sym`` or a Function declaration.
  This is purely symbolic: it tracks element dependencies through the graph without using
  numerical values. It is conservative for nonsmooth elementwise ops and block ops, but exact
  for the structural/arithmetic subset currently implemented here.
  """
  _check_wrt(wrt)
  return _mask_sparsity(_jac_mask(expr, wrt, {}))


def column_coloring(sparsity: SparsityPattern) -> tuple[int, ...]:
  """Assign each column a color, greedily, so no two columns sharing a row get the same one.

  Columns of one color can be recovered from a single forward pass, so the number of colors is
  the number of passes a compact Jacobian costs.
  """
  rows, inverse = np.unique(np.asarray(sparsity.rows, dtype=np.int64), return_inverse=True)
  mask = _incidence((len(rows), sparsity.shape[1]), inverse, np.asarray(sparsity.cols, dtype=np.int64))
  csr, csc = mask.tocsr(), mask.tocsc()
  row_ptr, col_ind = csr.indptr, csr.indices
  col_ptr, row_ind = csc.indptr, csc.indices
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


def _empty(shape: tuple[int, int]) -> sparse.coo_array:
  return sparse.coo_array(shape, dtype=bool)


def _mask_sparsity(mask: sparse.coo_array) -> SparsityPattern:
  mask.sum_duplicates()
  return SparsityPattern(mask.shape, tuple(map(int, mask.row)), tuple(map(int, mask.col)))


def _or(x: sparse.coo_array, y: sparse.coo_array) -> sparse.coo_array:
  return _incidence(x.shape, np.concatenate((x.row, y.row)), np.concatenate((x.col, y.col)))


def _compose(outer: sparse.coo_array, inner: sparse.coo_array) -> sparse.coo_array:
  if not outer.nnz or not inner.nnz:
    return _empty((outer.shape[0], inner.shape[1]))
  rows, ri = np.unique(outer.row, return_inverse=True)
  cols, ci = np.unique(inner.col, return_inverse=True)
  domain, inverse = np.unique(np.concatenate((outer.col, inner.row)), return_inverse=True)
  a = sparse.csr_array((outer.data, (ri, inverse[: outer.nnz])), shape=(len(rows), len(domain)))
  b = sparse.csr_array((inner.data, (inverse[outer.nnz :], ci)), shape=(len(domain), len(cols)))
  product = (a @ b).tocoo()
  return _incidence((outer.shape[0], inner.shape[1]), rows[product.row], cols[product.col])


def _incidence(shape: tuple[int, int], rows: np.ndarray, cols: np.ndarray) -> sparse.coo_array:
  result = sparse.coo_array((np.ones(rows.size, dtype=bool), (rows, cols)), shape=shape)
  result.sum_duplicates()
  return result


def _supported_rows(mask: sparse.coo_array, support: Support) -> sparse.coo_array:
  if support is None:
    return mask
  keep = np.isin(mask.row, support_indices(support))
  return _incidence(mask.shape, mask.row[keep], mask.col[keep])


def _take_rows(mask: sparse.coo_array, indices: np.ndarray) -> sparse.coo_array:
  indices = np.asarray(indices).reshape(-1)
  incidence = _incidence((len(indices), mask.shape[0]), np.arange(len(indices)), indices)
  return _compose(incidence, mask)


def _jac_mask(expr: Expr, wrt: Expr, memo: dict[tuple[int, int], sparse.coo_array]) -> sparse.coo_array:
  stack = [(expr, wrt, 0)]
  while stack:
    node, variable, phase = stack.pop()
    key = (node.id, variable.id)
    if key in memo:
      continue
    if phase == 0:
      stack.append((node, variable, 1))
      if node.op != ExprOp.STOP_GRADIENT and (node.op != ExprOp.VMAP or node.attrs["length"]):
        stack.extend((arg, variable, 0) for arg in reversed(node.args) if (arg.id, variable.id) not in memo)
    elif phase == 1 and node.op in (ExprOp.CALL, ExprOp.VMAP):
      stack.append((node, variable, 2))
      callee = node.attrs["callee"]
      callee_out = callee.results[node.attrs["output"]]
      # Actual masks must be known before requesting callee masks for active formals.
      if (node.op == ExprOp.CALL or node.attrs["length"]) and not _declared(callee, node.attrs["output"]):
        for formal, actual in reversed(tuple(zip(callee.inputs, node.args, strict=True))):
          if memo[(actual.id, variable.id)].nnz and (callee_out.id, formal.id) not in memo:
            stack.append((callee_out, formal, 0))
    else:
      memo[key] = _jac_mask_uncached(node, variable, memo)

  return memo[(expr.id, wrt.id)]


def _jac_mask_uncached(expr: Expr, wrt: Expr, memo: dict[tuple[int, int], sparse.coo_array]) -> sparse.coo_array:
  if expr.op == ExprOp.INPUT:
    if expr.id != wrt.id:
      return _empty((expr.size, wrt.size))
    entries = support_indices(expr.support) if expr.support is not None else np.arange(wrt.size)
    return _incidence((wrt.size, wrt.size), entries, entries)
  if expr.op in {ExprOp.CONST, ExprOp.STOP_GRADIENT}:
    return _empty((expr.size, wrt.size))
  if expr.op in COMMON_ELEMENTWISE_UNARY:
    return _jac_mask(expr.args[0], wrt, memo)
  if expr.op in COMMON_ELEMENTWISE_BINARY:
    x, y = expr.args
    rule = OP_INFO[ExprOp(expr.op)]
    xs = x if rule.pattern_rule == "intersection" or rule.support_rule == "numerator" else None
    ys = y if rule.pattern_rule == "intersection" else None
    xm = _broadcast_mask(_jac_mask(x, wrt, memo), x.shape, expr.shape, ys)
    ym = _broadcast_mask(_jac_mask(y, wrt, memo), y.shape, expr.shape, xs)
    return _or(xm, ym)
  if expr.op == ExprOp.SUM:
    child = _jac_mask(expr.args[0], wrt, memo)
    return _incidence((1, wrt.size), np.zeros(child.nnz, dtype=np.int64), child.col)
  if expr.op in {ExprOp.RESHAPE, ExprOp.PRINT}:
    return _jac_mask(expr.args[0], wrt, memo)
  if expr.op == ExprOp.TRANSPOSE:
    child = _jac_mask(expr.args[0], wrt, memo)
    rows = transpose_entries(child.row, expr.args[0].shape, expr.attrs["axes"])
    return _incidence((expr.size, wrt.size), rows, child.col)
  if expr.op == ExprOp.SLICE:
    child = _jac_mask(expr.args[0], wrt, memo)
    rows, keep = slice_entries(child.row, expr.args[0].shape, expr.attrs["index"])
    return _incidence((expr.size, wrt.size), rows, child.col[keep])
  if expr.op == ExprOp.GATHER:
    return _take_rows(_jac_mask(expr.args[0], wrt, memo), expr.attrs["indices"])
  if expr.op == ExprOp.SCATTER:
    child = _jac_mask(expr.args[0], wrt, memo)
    indices = expr.attrs["indices"].reshape(-1)
    return _incidence((expr.size, wrt.size), indices[child.row], child.col)
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


def _broadcast_mask(
  mask: sparse.coo_array, in_shape: tuple[int, ...], out_shape: tuple[int, ...], coefficient: Expr | None = None
) -> sparse.coo_array:
  shape = (int(np.prod(out_shape, dtype=np.int64)), mask.shape[1])
  if not mask.nnz:
    return _empty(shape)
  support = coefficient.support if coefficient is not None else None
  if coefficient is not None and support is not None and len(support) == coefficient.size:
    support = None
  if in_shape == out_shape and (support is None or coefficient is not None and coefficient.shape == out_shape):
    return _supported_rows(mask, support)
  _ = broadcast_shape(in_shape, out_shape)
  if coefficient is not None and support is not None:
    if not len(support):
      return _empty(shape)
    bound_cost = len(support) * (shape[0] // coefficient.size)
    mask_cost = mask.nnz * (shape[0] // int(np.prod(in_shape, dtype=np.int64)))
    if bound_cost < mask_cost:
      rows = broadcast_entries(support_indices(support), coefficient.shape, out_shape)[0]
      source = project_entries(rows, in_shape, out_shape)
      return _compose(_incidence((shape[0], mask.shape[0]), rows, source), mask)
  rows, source = (mask.row, np.arange(mask.nnz)) if in_shape == out_shape else broadcast_entries(mask.row, in_shape, out_shape)
  if coefficient is not None and support is not None:
    keep = np.isin(project_entries(rows, coefficient.shape, out_shape), support_indices(support))
    rows, source = rows[keep], source[keep]
  return _incidence(shape, rows, mask.col[source])


def _stack_mask(expr: Expr, wrt: Expr, memo: dict[tuple[int, int], sparse.coo_array]) -> sparse.coo_array:
  return _combine_children(expr, wrt, memo, stack=True)


def _concat_mask(expr: Expr, wrt: Expr, memo: dict[tuple[int, int], sparse.coo_array]) -> sparse.coo_array:
  return _combine_children(expr, wrt, memo, stack=False)


def _combine_children(expr: Expr, wrt: Expr, memo: dict[tuple[int, int], sparse.coo_array], *, stack: bool) -> sparse.coo_array:
  rows, cols = [], []
  offset = 0
  axis = expr.attrs.get("axis", 0)
  for i, arg in enumerate(expr.args):
    mask = _jac_mask(arg, wrt, memo)
    coords = list(np.unravel_index(mask.row, arg.shape)) if arg.shape else []
    if stack:
      coords.insert(axis, np.full(mask.nnz, i))
    else:
      coords[axis] = coords[axis] + offset
      offset += arg.shape[axis]
    rows.append(np.ravel_multi_index(tuple(coords), expr.shape))
    cols.append(mask.col)
  return _incidence((expr.size, wrt.size), np.concatenate(rows), np.concatenate(cols))


def _matmul_mask(expr: Expr, wrt: Expr, memo: dict[tuple[int, int], sparse.coo_array]) -> sparse.coo_array:
  x, y = expr.args
  xm, ym = _jac_mask(x, wrt, memo), _jac_mask(y, wrt, memo)
  m, k = (1, x.shape[0]) if len(x.shape) == 1 else x.shape
  n = 1 if len(y.shape) == 1 else y.shape[1]

  def contribution(mask: sparse.coo_array, other: Expr, *, left: bool) -> sparse.coo_array:
    if not mask.nnz:
      return _empty((expr.size, wrt.size))
    active, inverse = np.unique(mask.row, return_inverse=True)
    support = other.support
    if support is None:
      count = n if left else m
      source = np.repeat(np.arange(len(active)), count)
      destination = np.tile(np.arange(count), len(active))
      output = (active[source] // k) * n + destination if left else destination * n + active[source] % n
    else:
      entries = support_indices(support)
      # Boolean join over contraction coordinates and compact active entry labels.
      if left:
        a = _incidence((len(active), k), np.arange(len(active)), active % k).tocsr()
        b = _incidence((k, n), entries // n, entries % n).tocsr()
        joined = (a @ b).tocoo()
        source = joined.row
        output = (active[source] // k) * n + joined.col
      else:
        a = _incidence((m, k), entries // k, entries % k).tocsr()
        b = _incidence((k, len(active)), active // n, np.arange(len(active))).tocsr()
        joined = (a @ b).tocoo()
        source = joined.col
        output = joined.row * n + active[source] % n
    local = _incidence((expr.size, len(active)), output, source)
    compressed = _incidence((len(active), wrt.size), inverse, mask.col)
    return _compose(local, compressed)

  return _or(contribution(xm, y, left=True), contribution(ym, x, left=False))


def _declared(callee: Any, output: int) -> bool:
  """Whether the pattern of ``callee``'s result ``output`` is declared rather than read from its body."""
  return callee.rules is not None and output < len(callee.outputs)


def _callee_mask(callee: Any, output: int, k: int, memo: dict[tuple[int, int], sparse.coo_array]) -> sparse.coo_array:
  """The pattern of ``callee``'s result ``output`` in its input ``k``.

  A function with custom rules has the pattern it declares, or a dense one; its body need not
  describe the derivative its rules compute.
  """
  out, formal = callee.results[output], callee.inputs[k]
  if not _declared(callee, output):
    return _jac_mask(out, formal, memo)
  if callee.rules.sparsity is None:
    return sparse.coo_array(np.ones((out.size, formal.size), dtype=bool))
  pattern = callee.rules.sparsity[output][k]
  return _incidence(pattern.shape, np.asarray(pattern.rows, dtype=np.int64), np.asarray(pattern.cols, dtype=np.int64))


def _call_mask(expr: Expr, wrt: Expr, memo: dict[tuple[int, int], sparse.coo_array]) -> sparse.coo_array:
  callee = expr.attrs["callee"]
  ret = _empty((expr.size, wrt.size))
  for k, actual in enumerate(expr.args):
    actual_dep = _jac_mask(actual, wrt, memo)
    if actual_dep.nnz:
      ret = _or(ret, _compose(_callee_mask(callee, expr.attrs["output"], k, memo), actual_dep))
  return ret


def _vmap_mask(expr: Expr, wrt: Expr, memo: dict[tuple[int, int], sparse.coo_array]) -> sparse.coo_array:
  callee = expr.attrs["callee"]
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
    callee_dep = _callee_mask(callee, expr.attrs["output"], formal_idx, memo)
    start, stride = starts[formal_idx], strides[formal_idx]
    window_cols = np.repeat(start + np.arange(length) * stride, formal.size) + np.tile(np.arange(formal.size), length)
    windows = _incidence((length * formal.size, actual_outer.size), np.arange(length * formal.size), window_cols)
    tiled = sparse.kron(sparse.eye_array(length, dtype=bool), callee_dep, format="coo")
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
  mask = sparse.coo_array((np.ones(rows.size, dtype=bool), (rows, cols)), shape=sparsity.shape)
  return _mask_sparsity(_or(mask, sparse.coo_array(mask.T)))
