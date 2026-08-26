"""Compact sparse Jacobians and Hessians: AD-driven construction over a structural pattern.

The pattern itself — which entries can be nonzero — comes from ``sparsity.py``, which knows
nothing about AD. This module turns a pattern into values: graph coloring, compressed JVPs, and
the structured VMAP decomposition that keeps a multistage Jacobian from materializing densely.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import numpy as np

from ..function.sugar import vmap
from ..ir.expr import Expr, ExprOp, concat, gather, scatter
from ..passes.expr import cse, simplify, simplify_cse_fixpoint
from .derivatives import gradient, jacobian
from .forward import _call_jvp_many_const_function, jvp_many
from .sparsity import _depends_on, _jac_mask, _mask_sparsity, _symmetrize_sparsity, column_coloring, jacobian_sparsity, star_coloring
from ..ir.types import SparsityType


Triangle = Literal["full", "lower", "upper"]
_TRIANGLES: tuple[Triangle, ...] = ("full", "lower", "upper")


def _validate_triangle(triangle: object) -> Triangle:
  """Validate and type-narrow a sparse-Hessian layout selection."""
  if not isinstance(triangle, str) or triangle not in _TRIANGLES:
    raise ValueError(f"triangle must be one of {_TRIANGLES}, got {triangle!r}")
  return cast(Triangle, triangle)


@dataclass(frozen=True, slots=True)
class SparseJacobian:
  sparsity: SparsityType
  values: Expr
  coloring_width: int | None = None

  @property
  def flat_indices(self) -> np.ndarray:
    return np.asarray(self.sparsity.rows, dtype=np.int64) * self.sparsity.shape[1] + np.asarray(self.sparsity.cols, dtype=np.int64)

  def to_dense(self) -> Expr:
    return scatter(self.values, self.flat_indices, self.sparsity.shape)


def sparse_jacobian_reference(expr: Expr, wrt: Expr) -> SparseJacobian:
  """Reference compact Jacobian path: build dense ``J`` and gather nonzeros."""

  sparsity = jacobian_sparsity(expr, wrt)
  dense = jacobian(expr, wrt)
  flat = np.asarray(sparsity.rows, dtype=np.int64) * wrt.size + np.asarray(sparsity.cols, dtype=np.int64)
  return SparseJacobian(sparsity, gather(dense, flat))


def sparse_jacobian_colored(expr: Expr, wrt: Expr) -> SparseJacobian:
  """Compact sparse Jacobian values from graph-colored compressed JVPs."""
  expr = cse(expr)
  sparsity = jacobian_sparsity(expr, wrt)
  colors = column_coloring(sparsity)
  return _sparse_jacobian_colored(expr, wrt, sparsity, colors)


def _sparse_jacobian_colored(
  expr: Expr,
  wrt: Expr,
  sparsity: SparsityType,
  colors: tuple[int, ...],
  recovery_indices: np.ndarray | None = None,
) -> SparseJacobian:
  """Build compact values from a coloring and an optional compressed-row recovery table."""
  if sparsity.nnz == 0:
    return SparseJacobian(sparsity, Expr.const(np.zeros((0,), dtype=np.float64)), 0)
  ncolors = max(colors) + 1 if colors else 0
  seeds = np.zeros((ncolors, wrt.size), dtype=np.float64)
  for col, color in enumerate(colors):
    seeds[color, col] = 1.0
  compressed = jvp_many(expr, wrt, Expr.const(seeds.reshape((ncolors, *wrt.shape)))).reshape((ncolors, expr.size)).T
  rows = np.asarray(sparsity.rows, dtype=np.int64)
  cols = np.asarray(sparsity.cols, dtype=np.int64)
  if recovery_indices is None:
    flat = rows * ncolors + np.asarray([colors[int(col)] for col in cols], dtype=np.int64)
  else:
    flat = recovery_indices
  values = simplify_cse_fixpoint(gather(compressed, flat))
  return SparseJacobian(sparsity, values, ncolors)


def _star_recovery_indices(sparsity: SparsityType, colors: tuple[int, ...]) -> np.ndarray:
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


def sparse_jacobian(expr: Expr, wrt: Expr) -> SparseJacobian:
  """Compact sparse Jacobian: the structural pattern plus an expression for its nonzeros only.

  Takes the structured path when ``expr`` is a ``VMAP`` (or a concatenation of them) over exactly
  ``wrt``, coloring the callee's small local pattern instead of the whole matrix, so the cost
  tracks the callee rather than the iteration count. Otherwise falls back to coloring the global
  pattern. The value order is the pattern's ``(rows, cols)`` order, which is not necessarily
  sorted.
  """
  structured = _sparse_jacobian_structured(expr, wrt)
  if structured is not None:
    return structured
  return sparse_jacobian_colored(expr, wrt)


def _sparse_jacobian_structured(expr: Expr, wrt: Expr) -> SparseJacobian | None:
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
      sj = sparse_jacobian_colored(piece, wrt)
    if sj.coloring_width is not None:
      coloring_width += sj.coloring_width
    nnz_piece = sj.sparsity.nnz
    if nnz_piece == 0:
      continue
    global_rows.extend(r + row_offset for r in sj.sparsity.rows)
    global_cols.extend(sj.sparsity.cols)
    global_values.append(sj.values)
  total_rows = int(expr.shape[0]) if expr.shape else expr.size
  sparsity = SparsityType((expr.size, wrt.size), tuple(global_rows), tuple(global_cols))
  if sparsity.nnz == 0:
    return SparseJacobian(sparsity, Expr.const(np.zeros((0,), dtype=np.float64)), coloring_width)
  # Flatten piece values that are themselves axis-0 CONCATs so their producers write straight
  # into the single output concat instead of materializing an intermediate nnz-sized buffer.
  flat = [a for v in global_values for a in (v.args if v.op == ExprOp.CONCAT and v.attrs.get("axis", 0) == 0 else (v,))]
  values = flat[0] if len(flat) == 1 else concat(flat, axis=0)
  _ = total_rows  # documentation: piece row offsets cover [0, total_rows)
  return SparseJacobian(sparsity, simplify_cse_fixpoint(values), coloring_width)


def _split_axis0_pieces(expr: Expr) -> list[tuple[Expr, int]] | None:
  """Split ``expr`` into a list of ``(piece, row_offset)`` along axis 0.

  Only handles rank-1 expressions whose outermost producer is a single op or a ``CONCAT``
  along axis 0 of rank-1 pieces. Returns ``None`` for anything else.
  """

  if len(expr.shape) != 1:
    return None
  if expr.op != ExprOp.CONCAT:
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


def _sparse_jacobian_vmap(vmap_expr: Expr, wrt: Expr) -> SparseJacobian:
  """Compute compact sparse Jacobian of a rank-1 ``ExprOp.VMAP`` w.r.t. ``wrt`` using per-formal
  local coloring and const-seed JVPs wrapped in VMAPs.

  Falls back to ``sparse_jacobian_colored`` if any formal's actual outer tensor depends on
  ``wrt`` through computation rather than being ``wrt`` itself.
  """

  callee = vmap_expr.attrs["callee"]
  output_idx = vmap_expr.attrs["output"]
  length = vmap_expr.attrs["length"]
  starts = vmap_expr.attrs["starts"]
  strides = vmap_expr.attrs["strides"]
  slice_size = vmap_expr.attrs["slice_size"]
  callee_out = callee.outputs[output_idx]

  global_sparsity = jacobian_sparsity(vmap_expr, wrt)
  if global_sparsity.nnz == 0 or length == 0:
    return SparseJacobian(global_sparsity, Expr.const(np.zeros((global_sparsity.nnz,), dtype=np.float64)), 0)

  dep_memo: dict[tuple[int, int], bool] = {}
  direct_formals: list[int] = []
  for f_idx, actual in enumerate(vmap_expr.args):
    if actual.id == wrt.id:
      direct_formals.append(f_idx)
    elif _depends_on(actual, wrt, dep_memo):
      # Indirect dependency would require a chain-rule through actual_outer; fall back.
      return sparse_jacobian_colored(vmap_expr, wrt)

  if not direct_formals:
    return SparseJacobian(global_sparsity, Expr.const(np.zeros((global_sparsity.nnz,), dtype=np.float64)), 0)

  rows_arr = np.asarray(global_sparsity.rows, dtype=np.int64)
  cols_arr = np.asarray(global_sparsity.cols, dtype=np.int64)
  nnz = global_sparsity.nnz
  it_global = rows_arr // slice_size
  lr_global = rows_arr % slice_size

  pieces: list[tuple[Expr, np.ndarray]] = []  # (gathered piece values, nnz slots they cover)
  coloring_width = 0
  for f_idx in direct_formals:
    formal = callee.inputs[f_idx]
    local_mask = _jac_mask(callee_out, formal, {})
    if not local_mask.nnz:
      continue
    local_sparsity = _mask_sparsity(local_mask)
    local_colors = column_coloring(local_sparsity)
    if not local_colors:
      continue
    c_f = max(local_colors) + 1
    coloring_width += c_f
    seed_f = np.zeros((c_f, formal.size), dtype=np.float64)
    for j, c in enumerate(local_colors):
      seed_f[c, j] = 1.0
    seed_f_shaped = seed_f.reshape((c_f, *formal.shape)) if formal.shape != (formal.size,) else seed_f
    inner_fn, arg_indices, active = _call_jvp_many_const_function(callee, output_idx, f_idx, seed_f_shaped)
    active_count = len(active)
    if active_count == 0:
      continue
    primal_specs = [(vmap_expr.args[i], starts[i], strides[i]) for i in arg_indices]
    mapped_flat = vmap(inner_fn, length, primal_specs)
    # Build a per-formal contribution map for each nnz.
    formal_size = formal.size
    start_f = starts[f_idx]
    stride_f = strides[f_idx]
    lc_arr = cols_arr - (start_f + it_global * stride_f)
    in_window = (lc_arr >= 0) & (lc_arr < formal_size)
    contributes = np.zeros(nnz, dtype=bool)
    if in_window.any():
      valid = np.flatnonzero(in_window)
      contributes[valid] = np.asarray(local_mask[lr_global[valid], lc_arr[valid]]).reshape(-1)
    if not contributes.any():
      continue
    contrib_idx = np.flatnonzero(contributes)
    active_to_pos = {c: i for i, c in enumerate(active)}
    local_colors_arr = np.asarray(local_colors, dtype=np.int64)
    color_at = local_colors_arr[lc_arr[contrib_idx]]
    pos_at = np.asarray([active_to_pos[int(c)] for c in color_at], dtype=np.int64)
    flat_indices = it_global[contrib_idx] * (active_count * slice_size) + pos_at * slice_size + lr_global[contrib_idx]
    pieces.append((gather(mapped_flat, flat_indices), contrib_idx))

  if not pieces:
    return SparseJacobian(global_sparsity, Expr.const(np.zeros((nnz,), dtype=np.float64)), coloring_width)
  perm = np.concatenate([idx for _, idx in pieces])
  if perm.size == nnz and np.bincount(perm, minlength=nnz).max() == 1:
    # Piece supports exactly partition the nnz (verified: every slot covered exactly once), so
    # instead of scattering each piece into a full-nnz buffer and summing, emit the gathered
    # values back to back and permute the pattern to match — no full-nnz temporaries at all.
    values = pieces[0][0] if len(pieces) == 1 else concat([g for g, _ in pieces], axis=0)
    permuted = SparsityType(global_sparsity.shape, tuple(int(r) for r in rows_arr[perm]), tuple(int(c) for c in cols_arr[perm]))
    return SparseJacobian(permuted, values, coloring_width)
  values = None
  for gathered, contrib_idx in pieces:
    piece_values = gathered if contrib_idx.size == nnz else scatter(gathered, contrib_idx, (nnz,))
    values = piece_values if values is None else values + piece_values
  assert values is not None
  return SparseJacobian(global_sparsity, values, coloring_width)


def sparse_hessian(expr: Expr, wrt: Expr, *, triangle: Triangle = "full") -> SparseJacobian:
  """Compact sparse Hessian of a scalar expression using one global star-colored JVP batch."""
  triangle = _validate_triangle(triangle)
  if expr.size != 1:
    raise ValueError("sparse_hessian expects a scalar expression")
  gradient_expr = cse(simplify(gradient(expr, wrt).reshape((wrt.size,))))
  sparsity = _symmetrize_sparsity(jacobian_sparsity(gradient_expr, wrt))
  colors = star_coloring(sparsity)
  recovery = _star_recovery_indices(sparsity, colors)
  if triangle != "full":
    rows = np.asarray(sparsity.rows, dtype=np.int64)
    cols = np.asarray(sparsity.cols, dtype=np.int64)
    keep = rows >= cols if triangle == "lower" else rows <= cols
    sparsity = SparsityType(
      sparsity.shape,
      tuple(int(row) for row in rows[keep]),
      tuple(int(col) for col in cols[keep]),
    )
    recovery = recovery[keep]
  return _sparse_jacobian_colored(gradient_expr, wrt, sparsity, colors, recovery)
