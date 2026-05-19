from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .expr import Expr, concat, gather, map_, scatter
from .ops import COMMON_ELEMENTWISE_BINARY, COMMON_ELEMENTWISE_UNARY, Ops
from .types import SparsityType, broadcast_shape


@dataclass(frozen=True, slots=True)
class SparseJacobian:
  sparsity: SparsityType
  values: Expr

  @property
  def flat_indices(self) -> np.ndarray:
    return np.asarray(self.sparsity.rows, dtype=np.int64) * self.sparsity.shape[1] + np.asarray(self.sparsity.cols, dtype=np.int64)

  def to_dense(self) -> Expr:
    from .expr import scatter

    return scatter(self.values, self.flat_indices, self.sparsity.shape)


def jacobian_sparsity(expr: Expr, wrt: Expr) -> SparsityType:
  """Estimate structural sparsity of ``d vec(expr) / d vec(wrt)``.

  This is purely symbolic: it tracks element dependencies through the graph without using
  numerical values. It is conservative for nonsmooth elementwise ops and block ops, but exact
  for the structural/arithmetic subset currently implemented here.
  """

  return SparsityType.from_mask(_jac_mask(expr, wrt, {}))


def sparse_jacobian_reference(expr: Expr, wrt: Expr) -> SparseJacobian:
  """Reference compact Jacobian path: build dense ``J`` and gather nonzeros."""

  from .ad import jacobian

  sparsity = jacobian_sparsity(expr, wrt)
  dense = jacobian(expr, wrt)
  flat = np.asarray(sparsity.rows, dtype=np.int64) * wrt.size + np.asarray(sparsity.cols, dtype=np.int64)
  return SparseJacobian(sparsity, gather(dense, flat))


def sparse_jacobian_colored(expr: Expr, wrt: Expr) -> SparseJacobian:
  """Compact sparse Jacobian values from graph-colored compressed JVPs.

  This avoids constructing the full dense Jacobian before gathering. The current
  IR still materializes one JVP graph per color; preserving the color axis as a
  real loop/batch dimension is a later lowering task.
  """

  from .ad import jvp_many
  from .rewrite import cse, simplify_cse_fixpoint

  expr = cse(expr)
  sparsity = jacobian_sparsity(expr, wrt)
  if sparsity.nnz == 0:
    return SparseJacobian(sparsity, Expr.const(np.zeros((0,), dtype=np.float64)))

  colors = column_coloring(sparsity)
  ncolors = max(colors) + 1 if colors else 0
  seeds = np.zeros((ncolors, wrt.size), dtype=np.float64)
  for col, color in enumerate(colors):
    seeds[color, col] = 1.0
  compressed = jvp_many(expr, wrt, Expr.const(seeds.reshape((ncolors, *wrt.shape)))).reshape((ncolors, expr.size)).T
  rows = np.asarray(sparsity.rows, dtype=np.int64)
  cols = np.asarray(sparsity.cols, dtype=np.int64)
  color_of_nnz = np.asarray([colors[int(col)] for col in cols], dtype=np.int64)
  return SparseJacobian(sparsity, simplify_cse_fixpoint(gather(compressed, rows * ncolors + color_of_nnz)))


def sparse_jacobian(expr: Expr, wrt: Expr) -> SparseJacobian:
  structured = _sparse_jacobian_structured(expr, wrt)
  if structured is not None:
    return structured
  return sparse_jacobian_colored(expr, wrt)


def _sparse_jacobian_structured(expr: Expr, wrt: Expr) -> SparseJacobian | None:
  """Try to compute the compact sparse Jacobian by splitting along axis 0 and routing every
  rank-1 ``Ops.MAP`` piece through per-formal local coloring + const-seed JVPs.

  Returns ``None`` if the expression's structure prevents this decomposition (any non-MAP piece
  that itself depends on ``wrt`` would force the fallback, which we handle by mixing the
  structured per-piece MAP path with the global-colored path for the rest).
  """

  pieces = _split_axis0_pieces(expr)
  if pieces is None or not any(piece.op == Ops.MAP for piece, _ in pieces):
    return None

  from .rewrite import simplify_cse_fixpoint

  global_rows: list[int] = []
  global_cols: list[int] = []
  global_values: list[Expr] = []
  for piece, row_offset in pieces:
    if piece.op == Ops.MAP:
      sj = _sparse_jacobian_map(piece, wrt)
    else:
      sj = sparse_jacobian_colored(piece, wrt)
    nnz_piece = sj.sparsity.nnz
    if nnz_piece == 0:
      continue
    global_rows.extend(r + row_offset for r in sj.sparsity.rows)
    global_cols.extend(sj.sparsity.cols)
    global_values.append(sj.values)
  total_rows = int(expr.shape[0]) if expr.shape else expr.size
  sparsity = SparsityType((expr.size, wrt.size), tuple(global_rows), tuple(global_cols))
  if sparsity.nnz == 0:
    return SparseJacobian(sparsity, Expr.const(np.zeros((0,), dtype=np.float64)))
  values = global_values[0] if len(global_values) == 1 else concat(global_values, axis=0)
  _ = total_rows  # documentation: piece row offsets cover [0, total_rows)
  return SparseJacobian(sparsity, simplify_cse_fixpoint(values))


def _split_axis0_pieces(expr: Expr) -> list[tuple[Expr, int]] | None:
  """Split ``expr`` into a list of ``(piece, row_offset)`` along axis 0.

  Only handles rank-1 expressions whose outermost producer is a single op or a ``CONCAT``
  along axis 0 of rank-1 pieces. Returns ``None`` for anything else.
  """

  if len(expr.shape) != 1:
    return None
  if expr.op != Ops.CONCAT:
    return [(expr, 0)]
  if expr.attrs.get("axis", 0) != 0:
    return None
  pieces: list[tuple[Expr, int]] = []
  offset = 0
  for arg in expr.args:
    if len(arg.shape) != 1:
      return None
    pieces.append((arg, offset))
    offset += arg.shape[0]
  return pieces


def _sparse_jacobian_map(map_expr: Expr, wrt: Expr) -> SparseJacobian:
  """Compute compact sparse Jacobian of a rank-1 ``Ops.MAP`` w.r.t. ``wrt`` using per-formal
  local coloring and const-seed JVPs wrapped in MAPs.

  Falls back to ``sparse_jacobian_colored`` if any formal's actual outer tensor depends on
  ``wrt`` through computation rather than being ``wrt`` itself.
  """

  from .ad import _CALL_JVP_MANY_CONST_CACHE, _call_jvp_many_const_function, _depends_on

  callee = map_expr.attrs["callee"]
  output_idx = map_expr.attrs["output"]
  length = map_expr.attrs["length"]
  starts = map_expr.attrs["starts"]
  strides = map_expr.attrs["strides"]
  slice_size = map_expr.attrs["slice_size"]
  callee_out = callee.outputs[output_idx]

  global_sparsity = jacobian_sparsity(map_expr, wrt)
  if global_sparsity.nnz == 0 or length == 0:
    return SparseJacobian(global_sparsity, Expr.const(np.zeros((global_sparsity.nnz,), dtype=np.float64)))

  dep_memo: dict[tuple[int, int], bool] = {}
  direct_formals: list[int] = []
  for f_idx, actual in enumerate(map_expr.args):
    if actual.id == wrt.id:
      direct_formals.append(f_idx)
    elif _depends_on(actual, wrt, dep_memo):
      # Indirect dependency would require a chain-rule through actual_outer; fall back.
      return sparse_jacobian_colored(map_expr, wrt)

  if not direct_formals:
    return SparseJacobian(global_sparsity, Expr.const(np.zeros((global_sparsity.nnz,), dtype=np.float64)))

  rows_arr = np.asarray(global_sparsity.rows, dtype=np.int64)
  cols_arr = np.asarray(global_sparsity.cols, dtype=np.int64)
  nnz = global_sparsity.nnz
  it_global = rows_arr // slice_size
  lr_global = rows_arr % slice_size

  pieces: list[Expr] = []
  _ = _CALL_JVP_MANY_CONST_CACHE  # kept for diagnostics; ensures cache is initialised on import
  for f_idx in direct_formals:
    formal = callee.inputs[f_idx]
    local_mask = _jac_mask(callee_out, formal, {})
    if not local_mask.any():
      continue
    local_sparsity = SparsityType.from_mask(local_mask)
    local_colors = column_coloring(local_sparsity)
    if not local_colors:
      continue
    c_f = max(local_colors) + 1
    seed_f = np.zeros((c_f, formal.size), dtype=np.float64)
    for j, c in enumerate(local_colors):
      seed_f[c, j] = 1.0
    seed_f_shaped = seed_f.reshape((c_f, *formal.shape)) if formal.shape != (formal.size,) else seed_f
    inner_fn, arg_indices, active = _call_jvp_many_const_function(callee, output_idx, f_idx, seed_f_shaped)
    active_count = len(active)
    if active_count == 0:
      continue
    primal_specs = [(map_expr.args[i], starts[i], strides[i]) for i in arg_indices]
    mapped_flat = map_(inner_fn, length, primal_specs)
    # Build a per-formal contribution map for each nnz.
    formal_size = formal.size
    start_f = starts[f_idx]
    stride_f = strides[f_idx]
    lc_arr = cols_arr - (start_f + it_global * stride_f)
    in_window = (lc_arr >= 0) & (lc_arr < formal_size)
    contributes = np.zeros(nnz, dtype=bool)
    if in_window.any():
      valid = np.flatnonzero(in_window)
      contributes[valid] = local_mask[lr_global[valid], lc_arr[valid]]
    if not contributes.any():
      continue
    contrib_idx = np.flatnonzero(contributes)
    active_to_pos = {c: i for i, c in enumerate(active)}
    local_colors_arr = np.asarray(local_colors, dtype=np.int64)
    color_at = local_colors_arr[lc_arr[contrib_idx]]
    pos_at = np.asarray([active_to_pos[int(c)] for c in color_at], dtype=np.int64)
    flat_indices = it_global[contrib_idx] * (active_count * slice_size) + pos_at * slice_size + lr_global[contrib_idx]
    gathered = gather(mapped_flat, flat_indices)
    if contributes.all():
      piece_values = gathered
    else:
      piece_values = scatter(gathered, contrib_idx, (nnz,))
    pieces.append(piece_values)

  if not pieces:
    return SparseJacobian(global_sparsity, Expr.const(np.zeros((nnz,), dtype=np.float64)))
  values = pieces[0]
  for p in pieces[1:]:
    values = values + p
  return SparseJacobian(global_sparsity, values)


def sparse_hessian(expr: Expr, wrt: Expr) -> SparseJacobian:
  from .ad import gradient
  from .rewrite import simplify

  if expr.size != 1:
    raise ValueError("sparse_hessian expects a scalar expression")
  return sparse_jacobian(simplify(gradient(expr, wrt).reshape((wrt.size,))), wrt)


def column_coloring(sparsity: SparsityType) -> tuple[int, ...]:
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
  if expr.op == Ops.INPUT:
    return np.eye(wrt.size, dtype=bool) if expr.id == wrt.id else np.zeros((expr.size, wrt.size), dtype=bool)
  if expr.op == Ops.CONST:
    return np.zeros((expr.size, wrt.size), dtype=bool)
  if expr.op in COMMON_ELEMENTWISE_UNARY:
    return _jac_mask(expr.args[0], wrt, memo)
  if expr.op in COMMON_ELEMENTWISE_BINARY:
    x, y = expr.args
    return _broadcast_mask(_jac_mask(x, wrt, memo), x.shape, expr.shape) | _broadcast_mask(_jac_mask(y, wrt, memo), y.shape, expr.shape)
  if expr.op == Ops.SUM:
    return np.any(_jac_mask(expr.args[0], wrt, memo), axis=0, keepdims=True)
  if expr.op == Ops.RESHAPE:
    return _jac_mask(expr.args[0], wrt, memo)
  if expr.op == Ops.TRANSPOSE:
    order = np.arange(expr.size).reshape(expr.args[0].shape).transpose(expr.attrs["axes"]).reshape(-1)
    return _jac_mask(expr.args[0], wrt, memo)[order]
  if expr.op == Ops.SLICE:
    order = np.arange(expr.args[0].size).reshape(expr.args[0].shape)[expr.attrs["index"]].reshape(-1)
    return _jac_mask(expr.args[0], wrt, memo)[order]
  if expr.op == Ops.GATHER:
    return _jac_mask(expr.args[0], wrt, memo)[expr.attrs["indices"].reshape(-1)]
  if expr.op == Ops.SCATTER:
    ret = np.zeros((expr.size, wrt.size), dtype=bool)
    ret[expr.attrs["indices"].reshape(-1)] = _jac_mask(expr.args[0], wrt, memo)
    return ret
  if expr.op == Ops.STACK:
    return _stack_mask(expr, wrt, memo)
  if expr.op == Ops.CONCAT:
    return _concat_mask(expr, wrt, memo)
  if expr.op == Ops.MATMUL:
    return _matmul_mask(expr, wrt, memo)
  if expr.op == Ops.CALL:
    return _call_mask(expr, wrt, memo)
  if expr.op == Ops.MAP:
    return _map_mask(expr, wrt, memo)
  if expr.op == Ops.SOLVER_CALL:
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
