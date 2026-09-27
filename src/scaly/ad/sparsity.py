"""Structural (pattern-only) sparsity analysis.

Which entries of a Jacobian can be nonzero, tracked symbolically through the graph without any
values and without AD. Turning a pattern into values is ``sparse.py``; keeping the dependency
one-way is what lets AD ask this module for a pattern.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse

from ..ir.expr import COMMON_ELEMENTWISE_BINARY, COMMON_ELEMENTWISE_UNARY, PREDICATE_OPS, Expr, ExprOp, independent
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
  (expr,), (wrt,), _ = independent((expr,), (wrt,))
  return _mask_sparsity(_jac_mask(expr, wrt, {}))


def column_coloring(sparsity: SparsityType) -> tuple[int, ...]:
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


def _mask_sparsity(mask: sparse.csr_array) -> SparsityType:
  mask.sort_indices()
  coo = mask.tocoo()
  return SparsityType(mask.shape, tuple(int(x) for x in coo.row), tuple(int(x) for x in coo.col))


def _or(x: sparse.csr_array, y: sparse.csr_array) -> sparse.csr_array:
  return sparse.csr_array(x + y, dtype=bool)


def _compose(outer: sparse.csr_array, inner: sparse.csr_array) -> sparse.csr_array:
  return sparse.csr_array(outer @ inner, dtype=bool)


def _incidence(shape: tuple[int, int], rows: np.ndarray, cols: np.ndarray) -> sparse.csr_array:
  return sparse.csr_array((np.ones(rows.size, dtype=bool), (rows, cols)), shape=shape)


def _row_blocks(nrows: int, out_width: int, in_width: int) -> tuple[np.ndarray, np.ndarray]:
  """Coordinates of a block-diagonal all-ones pattern: row ``b`` of ``out_width`` entries against
  row ``b`` of ``in_width`` entries, for ``nrows`` rows."""
  b = np.arange(nrows)[:, None, None]
  r = b * out_width + np.arange(out_width)[None, :, None] + 0 * np.arange(in_width)[None, None, :]
  c = b * in_width + np.arange(in_width)[None, None, :] + 0 * np.arange(out_width)[None, :, None]
  return r.reshape(-1), c.reshape(-1)


def _sparse_ldl_reads(expr: Expr) -> sparse.csr_array:
  """Which matrix entries each entry of a ``sparse_ldl_factor`` result reads: column ``j`` of ``L`` and
  ``D[j]`` come from the columns of the elimination subtree of ``j`` (``j`` and every column whose
  path up the tree, parent = first row below the diagonal, passes through ``j``)."""
  a = expr.attrs
  n = a["a_ptr"].size - 1
  l_ptr, l_rows = a["l_ptr"], a["l_rows"]
  counts = np.diff(l_ptr)
  parent = np.where(counts > 0, l_rows[np.minimum(l_ptr[:-1], max(l_rows.size - 1, 0))], -1)
  anc_rows, anc_cols = [], []
  for k in range(n):  # k lies in the subtree of each of its ancestors, itself included
    j = k
    while j >= 0:
      anc_rows.append(j)
      anc_cols.append(k)
      j = int(parent[j])
  subtree = _incidence((n, n), np.asarray(anc_rows, dtype=np.int64), np.asarray(anc_cols, dtype=np.int64))
  reads = _incidence((n, expr.args[0].size), np.repeat(np.arange(n), np.diff(a["a_ptr"])), a["a_src"])
  column = np.concatenate([np.repeat(np.arange(n), counts), np.arange(n)])  # the column of each factor entry
  return _compose(_compose(_incidence((expr.size, n), np.arange(expr.size), column), subtree), reads)


def _jac_mask(expr: Expr, wrt: Expr, memo: dict[int, sparse.csr_array]) -> sparse.csr_array:
  if expr.id in memo:
    return memo[expr.id]
  mask = _jac_mask_uncached(expr, wrt, memo)
  memo[expr.id] = mask
  return mask


def _jac_mask_uncached(expr: Expr, wrt: Expr, memo: dict[int, sparse.csr_array]) -> sparse.csr_array:
  if expr.op == ExprOp.INPUT:
    return sparse.eye_array(wrt.size, format="csr", dtype=bool) if expr.id == wrt.id else _empty((expr.size, wrt.size))
  if expr.op == ExprOp.CONST or expr.op in PREDICATE_OPS:
    return _empty((expr.size, wrt.size))
  if expr.op == ExprOp.COPYSIGN:
    # The sign operand only flips the result, so its derivative is zero wherever it exists.
    return _broadcast_mask(_jac_mask(expr.args[0], wrt, memo), expr.args[0].shape, expr.shape)
  if expr.op == ExprOp.SELECT:
    _, a, b = expr.args
    return _or(_broadcast_mask(_jac_mask(a, wrt, memo), a.shape, expr.shape), _broadcast_mask(_jac_mask(b, wrt, memo), b.shape, expr.shape))
  if expr.op == ExprOp.CAST:
    return _jac_mask(expr.args[0], wrt, memo) if expr.type.diff else _empty((expr.size, wrt.size))
  if expr.op in COMMON_ELEMENTWISE_UNARY:
    return _jac_mask(expr.args[0], wrt, memo)
  if expr.op in COMMON_ELEMENTWISE_BINARY:
    x, y = expr.args
    return _or(_broadcast_mask(_jac_mask(x, wrt, memo), x.shape, expr.shape), _broadcast_mask(_jac_mask(y, wrt, memo), y.shape, expr.shape))
  if expr.op in {ExprOp.SUM, ExprOp.MAX, ExprOp.MIN}:
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
  if expr.op in {ExprOp.RAGGED_ADD, ExprOp.RAGGED_DOT}:
    # Run-time ranges: any output entry may depend on any entry of the floating operands (and a
    # ragged_add's entry on its own base entry).
    floats = [expr.args[1], expr.args[4]] if expr.op == ExprOp.RAGGED_ADD else [expr.args[0], expr.args[1]]
    mask = _jac_mask(expr.args[0], wrt, memo) if expr.op == ExprOp.RAGGED_ADD else _empty((expr.size, wrt.size))
    for arg in floats:
      dense = _incidence((expr.size, arg.size), np.repeat(np.arange(expr.size), arg.size), np.tile(np.arange(arg.size), expr.size))
      mask = _or(mask, _compose(dense, _jac_mask(arg, wrt, memo)))
    return mask
  if expr.op == ExprOp.SPARSE_LDL:
    return _compose(_sparse_ldl_reads(expr), _jac_mask(expr.args[0], wrt, memo))
  if expr.op == ExprOp.SPARSE_LDL_SOLVE:
    # Every unknown may depend on every entry of the factor and of the right-hand side.
    mask = _empty((expr.size, wrt.size))
    for arg in expr.args:
      dense = _incidence((expr.size, arg.size), np.repeat(np.arange(expr.size), arg.size), np.tile(np.arange(arg.size), expr.size))
      mask = _or(mask, _compose(dense, _jac_mask(arg, wrt, memo)))
    return mask
  if expr.op in {ExprOp.CHOLESKY, ExprOp.LDL}:
    # Every entry of the lower triangle of the factor may depend on every entry the factorization reads.
    a = expr.args[0]
    n = a.shape[0]
    lower = np.flatnonzero(np.tril(np.ones((n, n), dtype=bool)).reshape(-1))
    rows, cols = np.repeat(lower, lower.size), np.tile(lower, lower.size)
    return _compose(_incidence((expr.size, a.size), rows, cols), _jac_mask(a, wrt, memo))
  if expr.op == ExprOp.TRISOLVE:
    # Column c of the solution may depend on all of column c of the right-hand side and on every
    # entry of the triangle the solve reads.
    t, b = expr.args
    n = t.shape[0]
    m = 1 if len(b.shape) == 1 else b.shape[1]
    tri = np.tril(np.ones((n, n), dtype=bool)) if expr.attrs["lower"] else np.triu(np.ones((n, n), dtype=bool))
    if expr.attrs["unit"]:
      np.fill_diagonal(tri, False)
    read = np.flatnonzero(tri.reshape(-1))
    t_rows, t_cols = np.repeat(np.arange(expr.size), read.size), np.tile(read, expr.size)
    r, r2, col = np.meshgrid(np.arange(n), np.arange(n), np.arange(m), indexing="ij")
    b_rows, b_cols = (r * m + col).reshape(-1), (r2 * m + col).reshape(-1)
    from_t = _compose(_incidence((expr.size, t.size), t_rows, t_cols), _jac_mask(t, wrt, memo))
    return _or(from_t, _compose(_incidence((expr.size, b.size), b_rows, b_cols), _jac_mask(b, wrt, memo)))
  if expr.op == ExprOp.TAKE:
    # The index is known at run time only: lane j of row b may read any entry of row b.
    x = expr.args[0]
    n, lanes = x.shape[-1], expr.args[1].size
    rows, cols = _row_blocks(x.size // n if n else 0, lanes, n)
    return _compose(_incidence((expr.size, x.size), rows, cols), _jac_mask(x, wrt, memo))
  if expr.op in {ExprOp.PUT_ADD, ExprOp.PUT}:
    # Every entry keeps its base entry (a put may replace it; the pattern stays conservative) and may
    # receive any value of its row.
    base, _, values = expr.args
    n, lanes = base.shape[-1], values.shape[-1]
    rows, cols = _row_blocks(base.size // n if n else 0, n, lanes)
    return _or(_jac_mask(base, wrt, memo), _compose(_incidence((expr.size, values.size), rows, cols), _jac_mask(values, wrt, memo)))
  if expr.op in {ExprOp.INDEX_ADD, ExprOp.INDEX_SET}:
    base, values = expr.args
    indices = expr.attrs["indices"]
    kept = _jac_mask(base, wrt, memo)
    if expr.op == ExprOp.INDEX_SET:
      keep = np.ones(expr.size, dtype=bool)
      keep[indices] = False
      kept = _compose(_incidence((expr.size, expr.size), np.flatnonzero(keep), np.flatnonzero(keep)), kept)
    child = _jac_mask(values, wrt, memo)
    return _or(kept, _compose(_incidence((expr.size, values.size), indices, np.arange(values.size)), child))
  if expr.op in {ExprOp.SCATTER, ExprOp.SEGMENT_MAX, ExprOp.SEGMENT_MIN}:
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
  if expr.op == ExprOp.SCAN:
    return _scan_mask(expr, wrt, memo)
  if expr.op == ExprOp.WHILE:
    return _while_mask(expr, wrt, memo)
  if expr.op == ExprOp.SOLVER_CALL:
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


def _combine_children(expr: Expr, wrt: Expr, memo: dict[int, sparse.csr_array], child_index: np.ndarray, elem_index: np.ndarray) -> sparse.csr_array:
  offsets = np.cumsum([0, *(arg.size for arg in expr.args[:-1])])
  children = sparse.vstack([_jac_mask(arg, wrt, memo) for arg in expr.args], format="csr")
  return children[offsets[child_index] + elem_index]


def _stack_mask(expr: Expr, wrt: Expr, memo: dict[int, sparse.csr_array]) -> sparse.csr_array:
  base = expr.args[0].shape
  axis = expr.attrs.get("axis", 0)
  child = np.stack([np.full(base, i, dtype=np.int64) for i in range(len(expr.args))], axis=axis).reshape(-1)
  elem = np.stack([np.arange(arg.size, dtype=np.int64).reshape(base) for arg in expr.args], axis=axis).reshape(-1)
  return _combine_children(expr, wrt, memo, child, elem)


def _concat_mask(expr: Expr, wrt: Expr, memo: dict[int, sparse.csr_array]) -> sparse.csr_array:
  axis = expr.attrs.get("axis", 0)
  child = np.concatenate([np.full(arg.shape, i, dtype=np.int64) for i, arg in enumerate(expr.args)], axis=axis).reshape(-1)
  elem = np.concatenate([np.arange(arg.size, dtype=np.int64).reshape(arg.shape) for arg in expr.args], axis=axis).reshape(-1)
  return _combine_children(expr, wrt, memo, child, elem)


def _matmul_mask(expr: Expr, wrt: Expr, memo: dict[int, sparse.csr_array]) -> sparse.csr_array:
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


def _call_mask(expr: Expr, wrt: Expr, memo: dict[int, sparse.csr_array]) -> sparse.csr_array:
  callee, output = expr.attrs["callee"], int(expr.attrs["output"])
  ret = _empty((expr.size, wrt.size))
  for k, actual in enumerate(expr.args):
    outer = _jac_mask(actual, wrt, memo)
    if outer.nnz:  # an argument that does not depend on ``wrt`` needs no pattern of the callee
      ret = _or(ret, _compose(_callee_mask(callee, output, k), outer))
  return ret


def _callee_mask(callee: Any, output: int, k: int) -> sparse.csr_array:
  """The pattern of ``callee``'s output ``output`` in its input ``k``: the one its
  ``custom_derivative(sparsity=...)`` gives, else the body's."""
  out, formal = callee.outputs[output], callee.inputs[k]
  override = getattr(callee, "custom_sparsity", None)
  if override is None:
    return _jac_mask(out, formal, {})
  return _override_mask(override(output, k), (out.size, formal.size))


def _override_mask(pattern: Any, shape: tuple[int, int]) -> sparse.csr_array:
  """A pattern given by ``custom_derivative(sparsity=...)`` as a boolean CSR array of ``shape``."""
  if pattern is None:
    return _empty(shape)
  if isinstance(pattern, SparsityType):
    mask = _incidence(pattern.shape, np.asarray(pattern.rows, dtype=np.int64), np.asarray(pattern.cols, dtype=np.int64))
  elif sparse.issparse(pattern):
    mask = sparse.csr_array(pattern, dtype=bool)
  else:
    mask = sparse.csr_array(np.asarray(pattern, dtype=bool))
  if mask.shape != shape:
    raise ValueError(f"a custom sparsity pattern of shape {mask.shape} does not fit an output/input pair of shape {shape}")
  return mask


def _vmap_mask(expr: Expr, wrt: Expr, memo: dict[int, sparse.csr_array]) -> sparse.csr_array:
  callee, output = expr.attrs["callee"], int(expr.attrs["output"])
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
    callee_dep = _callee_mask(callee, output, formal_idx)
    start, stride = starts[formal_idx], strides[formal_idx]
    window_cols = np.repeat(start + np.arange(length) * stride, formal.size) + np.tile(np.arange(formal.size), length)
    windows = _incidence((length * formal.size, actual_outer.size), np.arange(length * formal.size), window_cols)
    tiled = sparse.kron(sparse.eye_array(length, dtype=bool), callee_dep, format="csr")
    ret = _or(ret, _compose(tiled, _compose(windows, outer_dep)))
  return ret


def _scan_mask(expr: Expr, wrt: Expr, memo: dict[int, sparse.csr_array]) -> sparse.csr_array:
  """Step the carry's dependence set through the loop: ``R_{k+1} = C R_k ∪ X_k``, where ``C`` is the
  body's carry-to-carry pattern and ``X_k`` what step ``k``'s slices bring in. With no sliced input
  that changes from step to step the sequence of patterns is eventually periodic (there are finitely
  many), so the walk stops at the first repeat and reads the final pattern off the cycle."""
  callee, length, output = expr.attrs["callee"], int(expr.attrs["length"]), int(expr.attrs["output"])
  starts, strides = expr.attrs["starts"], expr.attrs["strides"]
  xs = callee.inputs[1:]
  init, outers = expr.args[0], expr.args[1:]
  reach = _jac_mask(init, wrt, memo)
  outer_masks = [_jac_mask(outer, wrt, memo) for outer in outers]
  if not reach.nnz and not any(m.nnz for m in outer_masks):
    # Nothing the loop reads depends on ``wrt``: skip walking its steps.
    return _empty((expr.size, wrt.size))
  step = _callee_mask(callee, 0, 0)
  from_xs = [_callee_mask(callee, 0, i + 1) for i in range(len(xs))]
  y_carry = _callee_mask(callee, output, 0) if output > 0 else None
  y_xs = [_callee_mask(callee, output, i + 1) for i in range(len(xs))] if output > 0 else []
  rows: list[sparse.csr_array] = []
  # Only a slice that brings in dependence changes from step to step; the step number brings none.
  moving = any(stride != 0 and m.nnz for stride, m in zip(strides, outer_masks, strict=True))
  seen: dict[tuple[bytes, bytes, bytes], int] = {}
  history: list[sparse.csr_array] = []
  for k in range(length):
    if output == 0 and not moving:
      reach.sort_indices()
      key = (reach.indptr.tobytes(), reach.indices.tobytes(), np.asarray(reach.shape, dtype=np.int64).tobytes())
      if key in seen:
        first = seen[key]
        return history[first + (length - first) % (k - first)]
      seen[key] = k
      history.append(reach)
    slices = [m[start + k * stride + np.arange(x.size)] for m, x, start, stride in zip(outer_masks, xs, starts, strides, strict=True)]
    if output == -1:
      rows.append(reach)
    elif y_carry is not None:
      y = _compose(y_carry, reach)
      for dep, sl in zip(y_xs, slices, strict=True):
        y = _or(y, _compose(dep, sl))
      rows.append(y)
    nxt = _compose(step, reach)
    for dep, sl in zip(from_xs, slices, strict=True):
      nxt = _or(nxt, _compose(dep, sl))
    reach = sparse.csr_array(nxt, dtype=bool)
  if output == 0:
    return reach
  return sparse.vstack(rows, format="csr") if rows else _empty((0, wrt.size))


def _while_mask(expr: Expr, wrt: Expr, memo: dict[int, sparse.csr_array]) -> sparse.csr_array:
  """The carry after any number of steps up to ``max_iter``: the union of the step pattern's powers
  applied to the initial carry, with the params' pattern brought in again at every step, grown until
  it stops changing. Every stored carry gets the same union; the step count has none."""
  output, max_iter = int(expr.attrs["output"]), int(expr.attrs["max_iter"])
  if output == 1:
    return _empty((1, wrt.size))
  body = expr.attrs["callee"]
  first = 1 + int(bool(expr.attrs.get("index", False)))
  reach = _jac_mask(expr.args[0], wrt, memo)
  inject = _empty((reach.shape[0], wrt.size))
  for i, param in enumerate(expr.args[1:]):
    param_mask = _jac_mask(param, wrt, memo)
    if param_mask.nnz:
      inject = _or(inject, _compose(_callee_mask(body, 0, first + i), param_mask))
  if not reach.nnz and not inject.nnz:
    return _empty((expr.size, wrt.size))
  step = _callee_mask(body, 0, 0)
  frontier = reach
  for _ in range(max_iter):
    frontier = _or(_compose(step, frontier), inject)
    grown = _or(reach, frontier)
    if grown.nnz == reach.nnz:
      break
    reach = grown
  return reach if output == 0 else sparse.vstack([reach] * max_iter, format="csr") if max_iter else _empty((0, wrt.size))


def star_coloring(sparsity: SparsityType) -> tuple[int, ...]:
  """Greedily star-color a square sparsity pattern in column order.

  The pattern is treated as an undirected graph. A valid coloring is proper, and no simple path
  of three edges has only two colors. This is the coloring needed to recover a symmetric Hessian
  from compressed forward products; it is deliberately not distance-2 coloring.
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


def _symmetrize_sparsity(sparsity: SparsityType) -> SparsityType:
  """Return the undirected union of a square structural pattern and its transpose."""
  if sparsity.shape[0] != sparsity.shape[1]:
    raise ValueError(f"symmetric sparsity requires a square pattern, got {sparsity.shape}")
  rows = np.asarray(sparsity.rows, dtype=np.int64)
  cols = np.asarray(sparsity.cols, dtype=np.int64)
  mask = sparse.csr_array((np.ones(rows.size, dtype=bool), (rows, cols)), shape=sparsity.shape)
  return _mask_sparsity(mask.maximum(mask.T))
