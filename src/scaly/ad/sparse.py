"""Compact sparse Jacobians and Hessians: AD-driven construction over a structural pattern.

The pattern itself — which entries can be nonzero — comes from ``sparsity.py``, which knows
nothing about AD. This module turns a pattern into values: graph coloring, compressed JVPs, and
the structured VMAP decomposition that keeps a multistage Jacobian from materializing densely.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np

from ..function.sugar import _mapped_call
from ..ir.expr import Expr, ExprOp, concat, gather, pack, scatter, values as stored_values
from ..passes.expr import cse, simplify, simplify_cse_fixpoint
from .derivatives import gradient, jacobian
from .calls import _call_jvp_many_const_function
from .forward import _pushforward, jvp_many
from .sparsity import (
  _callee_mask,
  _check_wrt,
  _depends_on,
  _incidence,
  _mask_sparsity,
  _stored_slots,
  _symmetrize_sparsity,
  column_coloring,
  jacobian_sparsity,
  star_coloring,
)
from ..ir.types import SparsityPattern, stored_indices


Triangle = Literal["full", "lower", "upper"]
_TRIANGLES: tuple[Triangle, ...] = ("full", "lower", "upper")


def _validate_triangle(triangle: object) -> Triangle:
  """Validate and type-narrow a sparse-Hessian layout selection."""
  if not isinstance(triangle, str) or triangle not in _TRIANGLES:
    raise ValueError(f"triangle must be one of {_TRIANGLES}, got {triangle!r}")
  return triangle


@dataclass(frozen=True, slots=True)
class _Compact:
  """Compact derivative values in the coordinate order of ``sparsity``, before they become a sparse leaf.

  ``coloring_width`` is the number of forward-mode directions the values were computed from, or
  ``None`` when they did not come from a coloring. A colored result also keeps the compressed
  JVP batch and its recovery indices, so selecting a triangle or reordering gathers from it directly.
  """

  sparsity: SparsityPattern
  values: Expr
  coloring_width: int | None = None
  _compressed: Expr | None = field(default=None, compare=False, repr=False)
  _recovery: np.ndarray | None = field(default=None, compare=False, repr=False)

  def _select(self, keep: np.ndarray, sparsity: SparsityPattern) -> _Compact:
    """The entries at ``keep``, in that order, gathered straight from the compressed batch when there is one."""
    source = self._compressed if self._compressed is not None else self.values
    recovery = self._recovery[keep] if self._recovery is not None else keep
    return _Compact(sparsity, simplify_cse_fixpoint(gather(source, recovery)), self.coloring_width, source, recovery)

  def triangle(self, triangle: Triangle) -> _Compact:
    """Select one triangle from a symmetric sparse matrix without rebuilding its JVP batch."""
    triangle = _validate_triangle(triangle)
    if triangle == "full":
      return self
    if self.sparsity.shape[0] != self.sparsity.shape[1]:
      raise ValueError(f"triangle selection requires a square sparsity pattern, got {self.sparsity.shape}")
    rows = np.asarray(self.sparsity.rows, dtype=np.int64)
    cols = np.asarray(self.sparsity.cols, dtype=np.int64)
    keep = np.flatnonzero(rows >= cols if triangle == "lower" else rows <= cols)
    return self._select(keep, SparsityPattern(self.sparsity.shape, tuple(map(int, rows[keep])), tuple(map(int, cols[keep]))))

  def leaf(self) -> Expr:
    """The values as a sparse tensor over the canonical pattern."""
    logical_size = self.sparsity.shape[0] * self.sparsity.shape[1]
    if logical_size > np.iinfo(np.int64).max:
      raise ValueError(f"sparse derivative logical size {logical_size} exceeds int64; see #447")
    pattern, order = self.sparsity.canonicalize()
    values = self.values
    if order != tuple(range(len(order))):
      source = self._compressed if self._compressed is not None else self.values
      indices = np.asarray(order, dtype=np.int64)
      recovery = self._recovery[indices] if self._recovery is not None else indices
      values = simplify_cse_fixpoint(gather(source, recovery))
    return pack(values, pattern, shape=pattern.shape)


def sparse_jacobian_reference(expr: Expr, wrt: Expr) -> Expr:
  """Reference sparse Jacobian path: build the dense Jacobian and keep its structural nonzeros.

  ``wrt`` must be an independent input from ``sc.sym`` or a Function declaration.
  """

  sparsity = jacobian_sparsity(expr, wrt)
  dense = jacobian(expr, wrt)
  flat = np.asarray(sparsity.rows, dtype=np.int64) * wrt.size + np.asarray(sparsity.cols, dtype=np.int64)
  return _Compact(sparsity, gather(dense, flat)).leaf()


def sparse_jacobian_colored(expr: Expr, wrt: Expr) -> Expr:
  """Sparse Jacobian from graph-colored compressed JVPs over the whole pattern.

  ``wrt`` must be an independent input from ``sc.sym`` or a Function declaration.
  """
  return _colored(expr, wrt).leaf()


def _colored(expr: Expr, wrt: Expr) -> _Compact:
  _check_wrt(wrt)
  expr = cse(expr)
  sparsity = jacobian_sparsity(expr, wrt)
  coloring_pattern = (
    SparsityPattern(sparsity.shape[:1] + (wrt.type.storage_size,), sparsity.rows, tuple(map(int, _stored_slots(wrt.type, np.asarray(sparsity.cols)))))
    if wrt.type.pattern is not None
    else sparsity
  )
  colors = column_coloring(coloring_pattern)
  return _sparse_jacobian_colored(expr, wrt, sparsity, colors)


def _sparse_jacobian_colored(
  expr: Expr,
  wrt: Expr,
  sparsity: SparsityPattern,
  colors: tuple[int, ...],
  recovery_indices: np.ndarray | None = None,
) -> _Compact:
  """Build compact values from a coloring and an optional compressed-row recovery table."""
  if sparsity.nnz == 0:
    return _Compact(sparsity, Expr.const(np.zeros((0,), dtype=expr.type.dtype.numpy()), dtype=expr.type.dtype), 0)
  ncolors = max(colors) + 1 if colors else 0
  seeds = np.zeros((ncolors, wrt.type.storage_size), dtype=wrt.type.dtype.numpy())
  for col, color in enumerate(colors):
    seeds[color, col] = 1.0
  seed_expr = Expr.const(seeds.reshape((ncolors, *wrt.type.storage_shape)), dtype=wrt.type.dtype)
  if wrt.type.pattern is not None:
    seed_expr = pack(seed_expr, wrt.type.pattern, shape=(ncolors, *wrt.shape))
  compressed = stored_values(jvp_many(expr, wrt, seed_expr)).reshape((ncolors, expr.type.storage_size)).T
  rows = np.asarray(sparsity.rows, dtype=np.int64)
  cols = np.asarray(sparsity.cols, dtype=np.int64)
  if recovery_indices is None:
    rows = _stored_slots(expr.type, rows)
    cols = _stored_slots(wrt.type, cols)
    flat = rows * ncolors + np.asarray([colors[int(col)] for col in cols], dtype=np.int64)
  else:
    flat = recovery_indices
  return _Compact(sparsity, simplify_cse_fixpoint(gather(compressed, flat)), ncolors, compressed, flat)


def _star_recovery_indices(sparsity: SparsityPattern, colors: tuple[int, ...]) -> np.ndarray:
  """Build constant compressed-row indices for recovering a symmetric pattern by star coloring."""
  if sparsity.shape[0] != sparsity.shape[1]:
    raise ValueError(f"star recovery requires a square sparsity pattern, got {sparsity.shape}")
  if len(colors) != sparsity.shape[1]:
    raise ValueError(f"star recovery colors have length {len(colors)}, expected {sparsity.shape[1]}")
  ncolors = max(colors) + 1 if colors else 0
  neighbors = [set() for _ in range(sparsity.shape[0])]
  for row, col in zip(sparsity.rows, sparsity.cols, strict=True):
    if row != col:
      neighbors[row].add(col)
      neighbors[col].add(row)
  flat: list[int] = []
  for row, col in zip(sparsity.rows, sparsity.cols, strict=True):
    if row == col:
      flat.append(row * ncolors + colors[row])
      continue
    forward_color = colors[col]
    forward_unique = sum(colors[neighbor] == forward_color for neighbor in neighbors[row]) == 1
    reverse_color = colors[row]
    reverse_unique = sum(colors[neighbor] == reverse_color for neighbor in neighbors[col]) == 1
    if forward_unique:
      flat.append(row * ncolors + forward_color)
    elif reverse_unique:
      flat.append(col * ncolors + reverse_color)
    else:
      raise ValueError(f"star coloring cannot recover Hessian entry ({row}, {col})")
  return np.asarray(flat, dtype=np.int64)


def sparse_jacobian(expr: Expr, wrt: Expr) -> Expr:
  """Sparse Jacobian: a sparse tensor of shape ``(expr.size, wrt.size)`` storing only its structural nonzeros.

  ``wrt`` must be an independent input from ``sc.sym`` or a Function declaration.
  Takes the structured path when ``expr`` is a ``VMAP`` (or a concatenation of them) over
  ``wrt`` or a constant gather of it. It colors the callee's small local pattern instead of the
  whole matrix, so the cost tracks the callee rather than the iteration count. Otherwise falls back
  to coloring the global pattern.
  """
  return _sparse_jacobian(expr, wrt).leaf()


def _sparse_jacobian(expr: Expr, wrt: Expr) -> _Compact:
  _check_wrt(wrt)
  structured = _sparse_jacobian_structured(expr, wrt)
  if structured is not None:
    return structured
  return _colored(expr, wrt)


def _sparse_jacobian_structured(expr: Expr, wrt: Expr) -> _Compact | None:
  """Try to compute the compact sparse Jacobian by splitting along axis 0 and routing every
  rank-1 ``ExprOp.VMAP`` piece through per-formal local coloring + const-seed JVPs.

  Returns ``None`` if the expression's structure prevents this decomposition (any non-VMAP piece
  that itself depends on ``wrt`` would force the fallback, which we handle by mixing the
  structured per-piece VMAP path with the global-colored path for the rest).
  """

  pieces = _split_axis0_pieces(expr)
  if pieces is None or not any(piece.op == ExprOp.VMAP for piece, _ in pieces):
    return None

  global_rows: list[int] = []
  global_cols: list[int] = []
  global_values: list[Expr] = []
  coloring_width = 0
  for piece, row_offset in pieces:
    if piece.op == ExprOp.VMAP:
      sj = _sparse_jacobian_vmap(piece, wrt)
    else:
      sj = _colored(piece, wrt)
    if sj.coloring_width is not None:
      coloring_width += sj.coloring_width
    nnz_piece = sj.sparsity.nnz
    if nnz_piece == 0:
      continue
    global_rows.extend(r + row_offset for r in sj.sparsity.rows)
    global_cols.extend(sj.sparsity.cols)
    global_values.append(sj.values)
  sparsity = SparsityPattern((expr.size, wrt.size), tuple(global_rows), tuple(global_cols))
  if sparsity.nnz == 0:
    return _Compact(sparsity, Expr.const(np.zeros((0,), dtype=expr.type.dtype.numpy()), dtype=expr.type.dtype), coloring_width)
  # Flatten piece values that are themselves axis-0 CONCATs so their producers write straight
  # into the single output concat instead of materializing an intermediate nnz-sized buffer.
  flat = [a for v in global_values for a in (v.args if v.op == ExprOp.CONCAT and v.attrs.get("axis", 0) == 0 else (v,))]
  values = flat[0] if len(flat) == 1 else concat(flat, axis=0)
  return _Compact(sparsity, simplify_cse_fixpoint(values), coloring_width)


def _split_axis0_pieces(expr: Expr) -> list[tuple[Expr, int]] | None:
  """Split ``expr`` into a list of ``(piece, row_offset)`` along axis 0.

  Reshapes preserve flat row order, including the leading-axis wrappers of typed maps.
  Split their rank-1 producers and axis-0 concatenations; return ``None`` for other layouts.
  """

  while expr.op == ExprOp.RESHAPE:
    expr = expr.args[0]
  if len(expr.shape) != 1:
    return None
  if expr.op != ExprOp.CONCAT:
    return [(expr, 0)]
  if expr.attrs.get("axis", 0) != 0:
    return None
  pieces: list[tuple[Expr, int]] = []
  offset = 0
  for arg in expr.args:
    nested = _split_axis0_pieces(arg)
    if nested is None:
      return None
    pieces.extend((piece, offset + relative) for piece, relative in nested)
    offset += arg.size
  return pieces


def _sparse_jacobian_vmap(vmap_expr: Expr, wrt: Expr) -> _Compact:
  """Compute compact sparse Jacobian of a rank-1 ``ExprOp.VMAP`` w.r.t. ``wrt`` using per-formal
  local coloring and const-seed JVPs wrapped in VMAPs.

  Falls back to ``sparse_jacobian_colored`` if any formal's actual outer tensor depends on
  ``wrt`` through computation rather than being ``wrt`` itself or a constant gather of it.
  """

  callee = vmap_expr.attrs["callee"]
  output_idx = vmap_expr.attrs["output"]
  length = vmap_expr.attrs["length"]
  starts = vmap_expr.attrs["starts"]
  strides = vmap_expr.attrs["strides"]
  slice_size = vmap_expr.attrs["slice_size"]

  global_sparsity = jacobian_sparsity(vmap_expr, wrt)
  if global_sparsity.nnz == 0 or length == 0:
    return _Compact(global_sparsity, Expr.const(np.zeros((global_sparsity.nnz,), dtype=vmap_expr.type.dtype.numpy()), dtype=vmap_expr.type.dtype), 0)

  dep_memo: dict[tuple[int, int], bool] = {}
  direct_formals: dict[int, np.ndarray] = {}
  for f_idx, actual in enumerate(vmap_expr.args):
    while actual.op == ExprOp.RESHAPE:
      actual = actual.args[0]
    source = actual.args[0] if actual.op == ExprOp.GATHER else actual
    while source.op == ExprOp.RESHAPE:
      source = source.args[0]
    if source.id == wrt.id:
      direct_formals[f_idx] = actual.attrs["indices"].reshape(-1) if actual.op == ExprOp.GATHER else np.arange(actual.size)
    elif _depends_on(actual, wrt, dep_memo):
      # Indirect dependency would require a chain-rule through actual_outer; fall back.
      return _colored(vmap_expr, wrt)

  if not direct_formals:
    return _Compact(global_sparsity, Expr.const(np.zeros((global_sparsity.nnz,), dtype=vmap_expr.type.dtype.numpy()), dtype=vmap_expr.type.dtype), 0)

  rows_arr = np.asarray(global_sparsity.rows, dtype=np.int64)
  cols_arr = np.asarray(global_sparsity.cols, dtype=np.int64)
  nnz = global_sparsity.nnz
  global_coordinates = rows_arr * wrt.size + cols_arr

  pieces: list[tuple[Expr, np.ndarray]] = []  # (gathered piece values, nnz slots they cover)
  coloring_width = 0
  for f_idx, column_map in direct_formals.items():
    formal = callee.inputs[f_idx]
    local_mask = _callee_mask(callee, output_idx, f_idx, {})
    out = callee.results[output_idx]
    if out.type.pattern is not None or formal.type.pattern is not None:
      local_mask = _incidence(
        (out.type.storage_size, formal.type.storage_size), _stored_slots(out.type, local_mask.row), _stored_slots(formal.type, local_mask.col)
      )
    if not local_mask.nnz:
      continue
    local_sparsity = _mask_sparsity(local_mask)
    local_colors = column_coloring(local_sparsity)
    if not local_colors:
      continue
    c_f = max(local_colors) + 1
    coloring_width += c_f
    seed_f = np.zeros((c_f, formal.type.storage_size), dtype=formal.type.dtype.numpy())
    for j, c in enumerate(local_colors):
      seed_f[c, j] = 1.0
    seed_f_shaped = seed_f.reshape((c_f, *formal.type.storage_shape))
    inner_fn, arg_indices, active = _call_jvp_many_const_function(callee, output_idx, f_idx, seed_f_shaped, pushforward=_pushforward)
    active_count = len(active)
    if active_count == 0:
      continue
    primal_specs = [(vmap_expr.args[i], starts[i], strides[i]) for i in arg_indices]
    mapped_flat = _mapped_call(inner_fn, length, primal_specs)
    # Build a per-formal contribution map for each nnz.
    start_f = starts[f_idx]
    stride_f = strides[f_idx]
    iterations = np.repeat(np.arange(length), local_mask.nnz)
    local_rows, local_cols = np.tile(local_mask.row, length), np.tile(local_mask.col, length)
    columns = column_map[start_f + iterations * stride_f + local_cols]
    coordinates = (iterations * slice_size + local_rows) * wrt.size + columns
    slots = np.searchsorted(global_coordinates, coordinates)
    valid = np.flatnonzero(slots < nnz)
    valid = valid[global_coordinates[slots[valid]] == coordinates[valid]]
    valid = valid[np.argsort(slots[valid], kind="stable")]
    contrib_idx = slots[valid]
    iterations, local_rows, local_cols = iterations[valid], local_rows[valid], local_cols[valid]
    if not contrib_idx.size:
      continue
    active_to_pos = {c: i for i, c in enumerate(active)}
    local_colors_arr = np.asarray(local_colors, dtype=np.int64)
    color_at = local_colors_arr[local_cols]
    pos_at = np.asarray([active_to_pos[int(c)] for c in color_at], dtype=np.int64)
    flat_indices = iterations * (active_count * slice_size) + pos_at * slice_size + local_rows
    pieces.append((gather(mapped_flat, flat_indices), contrib_idx))

  if not pieces:
    return _Compact(global_sparsity, Expr.const(np.zeros((nnz,), dtype=vmap_expr.type.dtype.numpy()), dtype=vmap_expr.type.dtype), coloring_width)
  perm = np.concatenate([idx for _, idx in pieces])
  if perm.size == nnz and np.bincount(perm, minlength=nnz).max() == 1:
    # Piece supports exactly partition the nnz (verified: every slot covered exactly once), so
    # instead of scattering each piece into a full-nnz buffer and summing, emit the gathered
    # values back to back and permute the pattern to match — no full-nnz temporaries at all.
    values = pieces[0][0] if len(pieces) == 1 else concat([g for g, _ in pieces], axis=0)
    permuted = SparsityPattern(global_sparsity.shape, tuple(int(r) for r in rows_arr[perm]), tuple(int(c) for c in cols_arr[perm]))
    return _Compact(permuted, values, coloring_width)
  values = None
  for gathered, contrib_idx in pieces:
    piece_values = gathered if np.array_equal(contrib_idx, np.arange(nnz)) else scatter(gathered, contrib_idx, (nnz,))
    values = piece_values if values is None else values + piece_values
  assert values is not None
  return _Compact(global_sparsity, values, coloring_width)


def sparse_hessian(expr: Expr, wrt: Expr, *, triangle: Triangle = "full") -> Expr:
  """Sparse Hessian of a scalar expression, a sparse tensor of shape ``(wrt.size, wrt.size)``.

  ``wrt`` must be an independent input from ``sc.sym`` or a Function declaration. ``triangle``
  selects the entries stored: ``"full"`` keeps the whole symmetric pattern, ``"lower"`` and
  ``"upper"`` keep one triangle with the diagonal. All of them come from one star-colored JVP batch.
  """
  return _sparse_hessian(expr, wrt, triangle=triangle).leaf()


def _sparse_hessian(expr: Expr, wrt: Expr, *, triangle: Triangle = "full") -> _Compact:
  triangle = _validate_triangle(triangle)
  if expr.size != 1:
    raise ValueError("sparse_hessian expects a scalar expression")
  _check_wrt(wrt)
  if wrt.type.pattern is not None:
    grad = stored_values(gradient(expr, wrt))
    relation = jacobian_sparsity(grad, wrt)
    compact = _symmetrize_sparsity(
      SparsityPattern((grad.size, wrt.type.storage_size), relation.rows, tuple(map(int, _stored_slots(wrt.type, np.asarray(relation.cols)))))
    )
    entries = stored_indices(wrt.type)
    cols = tuple(map(int, entries[np.asarray(compact.cols, dtype=np.int64)]))
    relation = SparsityPattern((grad.size, wrt.size), compact.rows, cols)
    result = _sparse_jacobian_colored(grad, wrt, relation, column_coloring(compact))
    rows = entries[np.asarray(compact.rows, dtype=np.int64)]
    sparsity = SparsityPattern((wrt.size, wrt.size), tuple(map(int, rows)), cols)
    return replace(result, sparsity=sparsity).triangle(triangle)
  gradient_expr = cse(simplify(gradient(expr, wrt).reshape((wrt.size,))))
  sparsity = _symmetrize_sparsity(jacobian_sparsity(gradient_expr, wrt))
  colors = star_coloring(sparsity)
  recovery = _star_recovery_indices(sparsity, colors)
  return _sparse_jacobian_colored(gradient_expr, wrt, sparsity, colors, recovery).triangle(triangle)
