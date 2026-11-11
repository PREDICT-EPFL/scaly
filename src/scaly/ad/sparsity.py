"""Structural (pattern-only) sparsity analysis.

Which entries of a Jacobian can be nonzero, tracked symbolically through the graph without any
values and without AD. Turning a pattern into values is ``sparse.py``; keeping the dependency
one-way is what lets AD ask this module for a pattern.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import sparse

from ..ir.expr import COMMON_ELEMENTWISE_BINARY, COMMON_ELEMENTWISE_TERNARY, COMMON_ELEMENTWISE_UNARY, OP_INFO, Expr, ExprOp, callees_of
from ..ir.support import (
  reduce_entries,
  Support,
  broadcast_entries,
  transpose_entries,
  slice_entries,
  support_indices,
  project_entries,
  specialize_support,
  _constant_condition,
)
from ..ir.types import SparsityPattern, TensorType, broadcast_shape, stored_indices
from .rules import ELEMENTWISE, zero_partial


def _stored_slots(type_: TensorType, entries: np.ndarray) -> np.ndarray:
  return entries if type_.pattern is None else np.searchsorted(stored_indices(type_), entries)


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
    if node.op == ExprOp.LOOP:
      if not visited and node.attrs["callee"].rules is not None:
        object.__setattr__(node, "_custom_derivative_cache", True)
        continue
      children = (*children, *node.attrs["callee"].results)
    elif node.op == ExprOp.COND:
      output = node.attrs["output"]
      if not visited and any(_declared(branch, output) for branch in callees_of(node)):
        object.__setattr__(node, "_custom_derivative_cache", True)
        continue
      children = (*children, *(branch.outputs[output] for branch in callees_of(node)))
    elif callees_of(node):
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


@dataclass
class _JacContext:
  masks: dict[tuple[int, int], sparse.coo_array]
  support_of: Callable[[Expr], Support] = lambda node: node.support
  specializations: dict[Callable[[Expr], Support], _JacContext] = field(default_factory=dict)
  calls: dict[Expr, list[tuple[np.ndarray, _JacContext]]] = field(default_factory=dict)
  conditional_calls: dict[Expr, tuple[Expr, ...]] = field(default_factory=dict)


def _call_contexts(expr: Expr, context: _JacContext) -> list[tuple[np.ndarray, _JacContext]]:
  if expr in context.calls:
    return context.calls[expr]
  callee = expr.attrs["callee"]
  groups: dict[tuple[Support, ...], list[int]] = {}
  if expr.op == ExprOp.CALL:
    groups[tuple(context.support_of(arg) for arg in expr.args)] = [0]
  else:
    windows = []
    for arg, formal, start, stride in zip(expr.args, callee.inputs, expr.attrs["starts"], expr.attrs["strides"], strict=True):
      support = context.support_of(arg)
      windows.append((None if support is None else support_indices(support), formal.type, start, stride))
    for stage in range(expr.attrs["length"]):
      arguments = []
      for indices, type_, start, stride in windows:
        support = None
        if indices is not None:
          offset = start + stage * stride
          lo, hi = np.searchsorted(indices, (offset, offset + type_.storage_size))
          entries = indices[lo:hi] - offset
          support = frozenset(map(int, entries if type_.pattern is None else stored_indices(type_)[entries]))
        arguments.append(support)
      groups.setdefault(tuple(arguments), []).append(stage)
  contexts = []
  for arguments, stages in groups.items():
    support_of = specialize_support(callee, arguments, OP_INFO)
    if support_of not in context.specializations:
      context.specializations[support_of] = _JacContext({}, support_of, context.specializations)
    contexts.append((np.asarray(stages, dtype=np.int64), context.specializations[support_of]))
  context.calls[expr] = contexts
  return contexts


def _jac_mask(expr: Expr, wrt: Expr, memo: dict[tuple[int, int], sparse.coo_array] | _JacContext) -> sparse.coo_array:
  root = _JacContext(memo) if isinstance(memo, dict) else memo
  stack = [(expr, wrt, 0, root)]
  while stack:
    node, variable, phase, context = stack.pop()
    masks = context.masks
    key = (node.id, variable.id)
    if key in masks:
      continue
    if phase == 0:
      stack.append((node, variable, 1, context))
      if node.op != ExprOp.STOP_GRADIENT and (node.op != ExprOp.VMAP or node.attrs["length"]):
        stack.extend((arg, variable, 0, context) for arg in reversed(node.args) if (arg.id, variable.id) not in masks)
    elif phase == 1 and node.op == ExprOp.COND:
      stack.append((node, variable, 2, context))
      stack.extend((call, variable, 0, context) for call in _cond_calls(node, context))
    elif phase == 1 and callees_of(node) and node.op not in (ExprOp.EXTERN, ExprOp.LOOP):
      stack.append((node, variable, 2, context))
      callee = node.attrs["callee"]
      callee_out = callee.results[node.attrs["output"]]
      # Actual masks must be known before requesting callee masks for active formals.
      if (
        (node.op == ExprOp.CALL or node.attrs["length"])
        and not _declared(callee, node.attrs["output"])
        and any(masks[(arg.id, variable.id)].nnz for arg in node.args)
      ):
        for _, child in _call_contexts(node, context):
          for formal, actual in reversed(tuple(zip(callee.inputs, node.args, strict=True))):
            if masks[(actual.id, variable.id)].nnz and (callee_out.id, formal.id) not in child.masks:
              stack.append((callee_out, formal, 0, child))
    else:
      masks[key] = _jac_mask_uncached(node, variable, context)

  return root.masks[(expr.id, wrt.id)]


def _jac_mask_uncached(expr: Expr, wrt: Expr, memo: _JacContext) -> sparse.coo_array:
  if expr.op == ExprOp.INPUT:
    if expr.id != wrt.id:
      return _empty((expr.size, wrt.size))
    entries = support_indices(expr.support) if expr.support is not None else np.arange(wrt.size)
    return _incidence((wrt.size, wrt.size), entries, entries)
  # A bool or integer value, a predicate's or a cast's, has no derivative.
  if (
    expr.op in {ExprOp.CONST, ExprOp.STOP_GRADIENT}
    or not expr.type.dtype.is_floating
    or expr.op in ELEMENTWISE
    and ELEMENTWISE[expr.op].partials == (zero_partial,)
  ):
    return _empty((expr.size, wrt.size))
  if expr.op == ExprOp.PACK:
    child = _jac_mask(expr.args[0], wrt, memo)
    return _incidence((expr.size, wrt.size), stored_indices(expr.type)[child.row], child.col)
  if expr.op == ExprOp.VALUES:
    return _take_rows(_jac_mask(expr.args[0], wrt, memo), stored_indices(expr.args[0].type))
  if expr.op in COMMON_ELEMENTWISE_UNARY:
    return _jac_mask(expr.args[0], wrt, memo)
  if expr.op in COMMON_ELEMENTWISE_BINARY:
    x, y = expr.args
    rule = OP_INFO[ExprOp(expr.op)]
    xs = x if rule.pattern_rule == "intersection" or rule.support_rule == "numerator" else None
    ys = y if rule.pattern_rule == "intersection" else None
    xm = _broadcast_mask(_jac_mask(x, wrt, memo), x.shape, expr.shape, ys, memo.support_of(ys) if ys is not None else None)
    ym = _broadcast_mask(_jac_mask(y, wrt, memo), y.shape, expr.shape, xs, memo.support_of(xs) if xs is not None else None)
    return _or(xm, ym)
  if expr.op in COMMON_ELEMENTWISE_TERNARY:
    condition, x, y = expr.args
    masks = []
    for branch, arg in ((True, x), (False, y)):
      mask = _broadcast_mask(_jac_mask(arg, wrt, memo), arg.shape, expr.shape)
      chosen = _constant_condition(condition, mask.row, expr.shape)
      if chosen is not None:
        keep = chosen == branch
        mask = _incidence(mask.shape, mask.row[keep], mask.col[keep])
      masks.append(mask)
    return _or(*masks)
  if expr.op == ExprOp.REDUCE:
    child = _jac_mask(expr.args[0], wrt, memo)
    rows = reduce_entries(child.row, expr.args[0].shape, expr.attrs["axes"])
    return _incidence((expr.size, wrt.size), rows, child.col)
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
  if expr.op == ExprOp.EXTERN:
    ret = _empty((expr.size, wrt.size))
    for arg in expr.args:
      mask = _jac_mask(arg, wrt, memo)
      rows, cols = stored_indices(expr.type), stored_indices(arg.type)
      dense = _incidence((expr.size, arg.size), np.repeat(rows, len(cols)), np.tile(cols, len(rows)))
      ret = _or(ret, _compose(dense, mask))
    return ret
  if expr.op == ExprOp.COND:
    result = _empty((expr.size, wrt.size))
    for call in _cond_calls(expr, memo):
      result = _or(result, _jac_mask(call, wrt, memo))
    return result
  if expr.op == ExprOp.LOOP:
    return _loop_mask(expr, wrt, memo)
  if expr.op == ExprOp.SOLVER_CALL:
    if any(_jac_mask(arg, wrt, memo).nnz for arg in expr.args):
      raise NotImplementedError("active derivative through SOLVER_CALL is not implemented")
    return _empty((expr.size, wrt.size))
  raise NotImplementedError(f"jacobian sparsity for op {expr.op!r} is not implemented")


def _broadcast_mask(
  mask: sparse.coo_array, in_shape: tuple[int, ...], out_shape: tuple[int, ...], coefficient: Expr | None = None, support: Support = None
) -> sparse.coo_array:
  shape = (int(np.prod(out_shape, dtype=np.int64)), mask.shape[1])
  if not mask.nnz:
    return _empty(shape)
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


def _stack_mask(expr: Expr, wrt: Expr, memo: _JacContext) -> sparse.coo_array:
  return _combine_children(expr, wrt, memo, stack=True)


def _concat_mask(expr: Expr, wrt: Expr, memo: _JacContext) -> sparse.coo_array:
  return _combine_children(expr, wrt, memo, stack=False)


def _combine_children(expr: Expr, wrt: Expr, memo: _JacContext, *, stack: bool) -> sparse.coo_array:
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


def _matmul_mask(expr: Expr, wrt: Expr, memo: _JacContext) -> sparse.coo_array:
  x, y = expr.args
  xm, ym = _jac_mask(x, wrt, memo), _jac_mask(y, wrt, memo)
  if max(len(x.shape), len(y.shape)) > 2:
    batch = expr.shape[: len(expr.shape) - (len(x.shape) > 1) - (len(y.shape) > 1)]
    m, k = (1, x.shape[0]) if len(x.shape) == 1 else x.shape[-2:]
    n = 1 if len(y.shape) == 1 else y.shape[-1]
    xi = np.broadcast_to(np.arange(x.size).reshape((*x.shape[:-2], m, k)), (*batch, m, k))
    yi = np.broadcast_to(np.arange(y.size).reshape((*y.shape[:-2], k, n)), (*batch, k, n))
    output = np.arange(expr.size).reshape((*batch, m, n))
    xs, ys = memo.support_of(x), memo.support_of(y)
    rows, xc, yc = [], [], []
    for inner in range(k):
      a = np.broadcast_to(xi[..., :, inner, None], output.shape).reshape(-1)
      b = np.broadcast_to(yi[..., inner, None, :], output.shape).reshape(-1)
      rows.append(output.reshape(-1))
      xc.append(a)
      yc.append(b)
    if not rows:
      return _empty((expr.size, wrt.size))
    row, a, b = np.concatenate(rows), np.concatenate(xc), np.concatenate(yc)
    left = np.ones(len(row), dtype=bool) if ys is None else np.isin(b, support_indices(ys))
    right = np.ones(len(row), dtype=bool) if xs is None else np.isin(a, support_indices(xs))
    return _or(_compose(_incidence((expr.size, x.size), row[left], a[left]), xm), _compose(_incidence((expr.size, y.size), row[right], b[right]), ym))
  m, k = (1, x.shape[0]) if len(x.shape) == 1 else x.shape
  n = 1 if len(y.shape) == 1 else y.shape[1]

  def contribution(mask: sparse.coo_array, other: Expr, *, left: bool) -> sparse.coo_array:
    if not mask.nnz:
      return _empty((expr.size, wrt.size))
    support = memo.support_of(other)
    if expr.attrs.get("sampled") is not None:
      requested = stored_indices(expr.type)
      assert support is not None
      entries = support_indices(support)
      axis = requested % n if left else requested // n
      coefficient = _incidence((n, k), entries % n, entries // n) if left else _incidence((m, k), entries // k, entries % k)
      joined = (_incidence((len(requested), n if left else m), np.arange(len(requested)), axis).tocsr() @ coefficient.tocsr()).tocoo()
      sources = (requested[joined.row] // n) * k + joined.col if left else joined.col * n + requested[joined.row] % n
      local = _incidence((expr.size, x.size if left else y.size), requested[joined.row], sources)
      return _compose(local, mask)
    active, inverse = np.unique(mask.row, return_inverse=True)
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


def _callee_mask(callee: Any, output: int, k: int, memo: dict[tuple[int, int], sparse.coo_array] | _JacContext) -> sparse.coo_array:
  """The pattern of ``callee``'s result ``output`` in its input ``k``.

  A function with custom rules has the pattern it declares, or a dense one; its body need not
  describe the derivative its rules compute.
  """
  out, formal = callee.results[output], callee.inputs[k]
  if not _declared(callee, output):
    return _jac_mask(out, formal, memo)
  if callee.rules.sparsity is None:
    rows, cols = stored_indices(out.type), stored_indices(formal.type)
    return _incidence((out.size, formal.size), np.repeat(rows, len(cols)), np.tile(cols, len(rows)))
  pattern = callee.rules.sparsity[output][k]
  rows, cols = np.asarray(pattern.rows, dtype=np.int64), np.asarray(pattern.cols, dtype=np.int64)
  keep = np.ones(len(rows), dtype=bool)
  if out.type.pattern is not None:
    keep &= np.isin(rows, stored_indices(out.type))
  if formal.type.pattern is not None:
    keep &= np.isin(cols, stored_indices(formal.type))
  return _incidence(pattern.shape, rows[keep], cols[keep])


def _cond_calls(expr: Expr, context: _JacContext) -> tuple[Expr, ...]:
  if expr not in context.conditional_calls:
    context.conditional_calls[expr] = tuple(
      Expr(ExprOp.CALL, expr.args[1:], branch.outputs[expr.attrs["output"]].type, attrs={"callee": branch, "output": expr.attrs["output"]})
      for branch in expr.attrs["branches"]
    )
  return context.conditional_calls[expr]


def _call_mask(expr: Expr, wrt: Expr, memo: _JacContext) -> sparse.coo_array:
  callee = expr.attrs["callee"]
  ret = _empty((expr.size, wrt.size))
  for k, actual in enumerate(expr.args):
    actual_dep = _jac_mask(actual, wrt, memo)
    if actual_dep.nnz:
      child = memo if _declared(callee, expr.attrs["output"]) else _call_contexts(expr, memo)[0][1]
      ret = _or(ret, _compose(_callee_mask(callee, expr.attrs["output"], k, child), actual_dep))
  return ret


def _vmap_mask(expr: Expr, wrt: Expr, memo: _JacContext) -> sparse.coo_array:
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
    start, stride = starts[formal_idx], strides[formal_idx]
    output = callee.results[expr.attrs["output"]]
    contexts = [(np.arange(length, dtype=np.int64), memo)] if _declared(callee, expr.attrs["output"]) else _call_contexts(expr, memo)
    for stages, child in contexts:
      callee_dep = _callee_mask(callee, expr.attrs["output"], formal_idx, child)
      rows = _stored_slots(output.type, callee_dep.row)
      cols = _stored_slots(formal.type, callee_dep.col)
      iterations = stages[:, None]
      windows = _incidence(
        (expr.size, actual_outer.size),
        (iterations * output.type.storage_size + rows).reshape(-1),
        (start + iterations * stride + cols).reshape(-1),
      )
      ret = _or(ret, _compose(windows, outer_dep))
  return ret


def _loop_mask(expr: Expr, wrt: Expr, memo: _JacContext) -> sparse.coo_array:
  """A boolean fixed point over the carries, each step reading the union of its slices over all steps.

  Stacked outputs read the converged carries and their own step's slices; trajectories repeat the
  converged carries at every slot.
  """
  callee, n_carry, length, output = expr.attrs["callee"], expr.attrs["n_carry"], expr.attrs["length"], expr.attrs["output"]
  n_ys, skip = len(callee.outputs) - n_carry, n_carry + expr.attrs["index"]
  support_of = specialize_support(callee, (None,) * len(callee.inputs), OP_INFO)
  child = memo.specializations.setdefault(support_of, _JacContext({}, support_of, memo.specializations))
  steps = np.arange(length, dtype=np.int64)[:, None]
  outers = []
  for k, (outer, formal, start, stride) in enumerate(
    zip(expr.args[n_carry:], callee.inputs[skip:], expr.attrs["starts"], expr.attrs["strides"], strict=False)
  ):
    mask = _jac_mask(outer, wrt, memo)
    if mask.nnz and length:
      outers.append((skip + k, formal, start, stride, outer.size, mask))

  def windows(
    size: int, rows: np.ndarray, formal: Expr, start: int, stride: int, cols: np.ndarray, outer_size: int, mask: sparse.coo_array
  ) -> sparse.coo_array:
    """Compose ``mask`` with the reads of formal entries ``cols`` at every step, landing in ``rows``."""
    at = start + steps * stride + _stored_slots(formal.type, cols)
    return _compose(_incidence((size, outer_size), np.broadcast_to(rows, at.shape).reshape(-1), at.reshape(-1)), mask)

  union = {
    k: windows(formal.size, (e := stored_indices(formal.type)), formal, start, stride, e, size, mask)
    for k, formal, start, stride, size, mask in outers
  }
  carries = [_jac_mask(init, wrt, memo) for init in expr.args[:n_carry]]

  def step(out: int, inputs: dict[int, sparse.coo_array]) -> sparse.coo_array:
    result = _empty((callee.results[out].size, wrt.size))
    for k, mask in inputs.items():
      if mask.nnz:
        result = _or(result, _compose(_callee_mask(callee, out, k, child), mask))
    return result

  for _ in range(length):
    grown = [_or(carry, step(j, {**dict(enumerate(carries)), **union})) for j, carry in enumerate(carries)]
    if all(new.nnz == old.nnz for new, old in zip(grown, carries, strict=True)):
      break
    carries = grown
  if n_carry <= output < n_carry + n_ys:
    out_type = callee.results[output].type
    own = step(output, dict(enumerate(carries)))
    rows = steps * out_type.storage_size + _stored_slots(out_type, own.row)
    result = _incidence((expr.size, wrt.size), rows.reshape(-1), np.broadcast_to(own.col, rows.shape).reshape(-1))
    for k, formal, start, stride, size, mask in outers:
      local = _callee_mask(callee, output, k, child)
      result = _or(
        result, windows(expr.size, steps * out_type.storage_size + _stored_slots(out_type, local.row), formal, start, stride, local.col, size, mask)
      )
    return result
  if output < n_carry:
    return carries[output]
  carry, carry_type = carries[output - n_carry - n_ys - 1], callee.inputs[output - n_carry - n_ys - 1].type
  rows = steps * carry_type.storage_size + _stored_slots(carry_type, carry.row)
  return _incidence((expr.size, wrt.size), rows.reshape(-1), np.broadcast_to(carry.col, rows.shape).reshape(-1))


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
