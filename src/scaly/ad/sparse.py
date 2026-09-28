"""Compact sparse Jacobians and Hessians: AD-driven construction over a structural pattern.

The pattern itself — which entries can be nonzero — comes from ``sparsity.py``, which knows
nothing about AD. This module turns a pattern into values: graph coloring, compressed JVPs, and
the structured VMAP decomposition that keeps a multistage Jacobian from materializing densely.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np
from scipy import sparse

from ..function.sugar import vmap
from ..ir.expr import Expr, ExprOp, concat, gather, independent, scatter, substitute
from ..passes.expr import cse, simplify, simplify_cse_fixpoint
from .derivatives import gradient, jacobian
from .forward import _call_jvp_many_function, jvp_many
from .sparsity import _callee_mask, _depends_on, _mask_sparsity, _symmetrize_sparsity, column_coloring, jacobian_sparsity, star_coloring
from ..ir.types import SparsityType


Triangle = Literal["full", "lower", "upper"]
_TRIANGLES: tuple[Triangle, ...] = ("full", "lower", "upper")


def _validate_triangle(triangle: object) -> Triangle:
  """Validate and type-narrow a sparse-Hessian layout selection."""
  if not isinstance(triangle, str) or triangle not in _TRIANGLES:
    raise ValueError(f"triangle must be one of {_TRIANGLES}, got {triangle!r}")
  return triangle


@dataclass(frozen=True, slots=True)
class SparseJacobian:
  sparsity: SparsityType
  values: Expr
  coloring_width: int | None = None
  _compressed: Expr | None = field(default=None, compare=False, repr=False)
  _recovery: np.ndarray | None = field(default=None, compare=False, repr=False)

  @property
  def flat_indices(self) -> np.ndarray:
    return np.asarray(self.sparsity.rows, dtype=np.int64) * self.sparsity.shape[1] + np.asarray(self.sparsity.cols, dtype=np.int64)

  def to_dense(self) -> Expr:
    return scatter(self.values, self.flat_indices, self.sparsity.shape)

  def triangle(self, triangle: Triangle) -> SparseJacobian:
    """Select one triangle from a symmetric sparse matrix without rebuilding its JVP batch."""
    triangle = _validate_triangle(triangle)
    if triangle == "full":
      return self
    if self.sparsity.shape[0] != self.sparsity.shape[1]:
      raise ValueError(f"triangle selection requires a square sparsity pattern, got {self.sparsity.shape}")
    rows = np.asarray(self.sparsity.rows, dtype=np.int64)
    cols = np.asarray(self.sparsity.cols, dtype=np.int64)
    keep = rows >= cols if triangle == "lower" else rows <= cols
    sparsity = SparsityType(
      self.sparsity.shape,
      tuple(int(row) for row in rows[keep]),
      tuple(int(col) for col in cols[keep]),
    )
    source = self._compressed if self._compressed is not None else self.values
    recovery = self._recovery[keep] if self._recovery is not None else np.flatnonzero(keep)
    values = simplify_cse_fixpoint(gather(source, recovery))
    return SparseJacobian(sparsity, values, self.coloring_width, source, recovery)


def _at_inputs(fn, expr: Expr, wrt: Expr, *args, **kwargs) -> SparseJacobian | None:
  """``fn(expr, wrt)`` with ``wrt`` made an input (see ``independent``), its values mapped back; None
  when ``wrt`` already is one."""
  if wrt.op == ExprOp.INPUT:
    return None
  (expr,), (at,), back = independent((expr,), (wrt,))
  sj = fn(expr, at, *args, **kwargs)
  compressed = None if sj._compressed is None else substitute(sj._compressed, back)
  return replace(sj, values=substitute(sj.values, back), _compressed=compressed)


def sparse_jacobian_reference(expr: Expr, wrt: Expr) -> SparseJacobian:
  """Reference compact Jacobian path: build dense ``J`` and gather nonzeros."""
  if (moved := _at_inputs(sparse_jacobian_reference, expr, wrt)) is not None:
    return moved

  sparsity = jacobian_sparsity(expr, wrt)
  dense = jacobian(expr, wrt)
  flat = np.asarray(sparsity.rows, dtype=np.int64) * wrt.size + np.asarray(sparsity.cols, dtype=np.int64)
  return SparseJacobian(sparsity, gather(dense, flat))


def sparse_jacobian_colored(expr: Expr, wrt: Expr) -> SparseJacobian:
  """Compact sparse Jacobian values from graph-colored compressed JVPs."""
  if (moved := _at_inputs(sparse_jacobian_colored, expr, wrt)) is not None:
    return moved
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
  return SparseJacobian(
    sparsity, values, ncolors, compressed if recovery_indices is not None else None, flat if recovery_indices is not None else None
  )


def _star_recovery_indices(sparsity: SparsityType, colors: tuple[int, ...]) -> np.ndarray:
  """Build constant compressed-row indices for recovering a symmetric pattern by star coloring."""
  if sparsity.shape[0] != sparsity.shape[1]:
    raise ValueError(f"star recovery requires a square sparsity pattern, got {sparsity.shape}")
  if len(colors) != sparsity.shape[1]:
    raise ValueError(f"star recovery colors have length {len(colors)}, expected {sparsity.shape[1]}")
  ncolors = max(colors) + 1 if colors else 0
  n = sparsity.shape[0]
  rows, cols = np.asarray(sparsity.rows, dtype=np.int64), np.asarray(sparsity.cols, dtype=np.int64)
  color = np.asarray(colors, dtype=np.int64)
  off = rows != cols
  ends = (np.concatenate([rows[off], cols[off]]), np.concatenate([cols[off], rows[off]]))
  adjacency = sparse.csr_array((np.ones(ends[0].size), ends), shape=(n, n)) > 0  # each edge once, either orientation given
  onehot = sparse.csr_array((np.ones(n), (np.arange(n), color)), shape=(n, ncolors))
  counts = sparse.csr_array(adjacency.astype(np.float64) @ onehot).tocoo()  # neighbours of each vertex in each colour
  keys = counts.row.astype(np.int64) * ncolors + counts.col
  order = np.argsort(keys)
  keys, tally = keys[order], counts.data[order]

  def neighbours_in(vertex: np.ndarray, colour: np.ndarray) -> np.ndarray:
    query = vertex * ncolors + colour
    at = np.minimum(np.searchsorted(keys, query), max(keys.size - 1, 0))
    return np.where(keys[at] == query, tally[at], 0.0) if keys.size else np.zeros(query.size)

  forward = neighbours_in(rows, color[cols]) == 1  # the row's only neighbour of the column's colour
  reverse = neighbours_in(cols, color[rows]) == 1
  stuck = off & ~forward & ~reverse
  if stuck.any():
    k = int(np.argmax(stuck))
    raise ValueError(f"star coloring cannot recover Hessian entry ({rows[k]}, {cols[k]})")
  return np.where(~off | forward, rows * ncolors + color[np.where(off, cols, rows)], cols * ncolors + color[rows]).astype(np.int64)


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


def _window_offset(actual: Expr, wrt: Expr) -> int | None:
  """Where ``actual`` starts in a vector ``wrt`` when it is ``wrt`` or a contiguous run of it (a slice
  with step 1, possibly reshaped to a vector), else ``None``. An NLP's variable leaves are such runs
  of its one variable vector, so a map over them keeps the structured path."""
  if actual.id == wrt.id:
    return 0
  if actual.op == ExprOp.RESHAPE:  # a reshape keeps the flat order
    return _window_offset(actual.args[0], wrt)
  if actual.op != ExprOp.SLICE or actual.args[0].id != wrt.id or len(wrt.shape) != 1:
    return None
  (index,) = actual.attrs["index"]
  if not isinstance(index, slice):
    return None
  start, _, step = index.indices(wrt.shape[0])
  return start if step == 1 else None


def _sparse_jacobian_vmap(vmap_expr: Expr, wrt: Expr) -> SparseJacobian:
  """Compute compact sparse Jacobian of a rank-1 ``ExprOp.VMAP`` w.r.t. ``wrt`` using per-formal
  local coloring and const-seed JVPs wrapped in VMAPs.

  Falls back to ``sparse_jacobian_colored`` if any formal's actual outer tensor depends on
  ``wrt`` through computation rather than being ``wrt`` itself or a contiguous run of it.
  """

  callee = vmap_expr.attrs["callee"]
  output_idx = vmap_expr.attrs["output"]
  length = vmap_expr.attrs["length"]
  starts = vmap_expr.attrs["starts"]
  strides = vmap_expr.attrs["strides"]
  slice_size = vmap_expr.attrs["slice_size"]

  if getattr(callee, "custom_jvp", None) is not None:
    # The local seeded tangents below differentiate the body; a rule must be honored instead.
    return sparse_jacobian_colored(vmap_expr, wrt)
  global_sparsity = jacobian_sparsity(vmap_expr, wrt)
  if global_sparsity.nnz == 0 or length == 0:
    return SparseJacobian(global_sparsity, Expr.const(np.zeros((global_sparsity.nnz,), dtype=np.float64)), 0)

  dep_memo: dict[tuple[int, int], bool] = {}
  offsets: dict[int, int] = {}  # formal -> where its outer tensor starts in wrt
  for f_idx, actual in enumerate(vmap_expr.args):
    offset = _window_offset(actual, wrt)
    if offset is not None:
      offsets[f_idx] = offset
    elif _depends_on(actual, wrt, dep_memo):
      # Indirect dependency would require a chain-rule through actual_outer; fall back.
      return sparse_jacobian_colored(vmap_expr, wrt)
  direct_formals = list(offsets)

  if not direct_formals:
    return SparseJacobian(global_sparsity, Expr.const(np.zeros((global_sparsity.nnz,), dtype=np.float64)), 0)

  rows_arr = np.asarray(global_sparsity.rows, dtype=np.int64)
  cols_arr = np.asarray(global_sparsity.cols, dtype=np.int64)
  nnz = global_sparsity.nnz
  it_global = rows_arr // slice_size
  lr_global = rows_arr % slice_size

  # Each formal's columns are colored on their own; all the colors become rows of one block seed,
  # so a single mapped pass differentiates every formal and computes the primal once.
  colored: list[tuple[int, Any, np.ndarray, int]] = []  # (formal, local mask, local colors, first seed row)
  coloring_width = 0
  for f_idx in direct_formals:
    local_mask = _callee_mask(callee, output_idx, f_idx)
    if not local_mask.nnz:
      continue
    local_colors = column_coloring(_mask_sparsity(local_mask))
    if not local_colors:
      continue
    colored.append((f_idx, local_mask, np.asarray(local_colors, dtype=np.int64), coloring_width))
    coloring_width += max(local_colors) + 1
  if not colored:
    return SparseJacobian(global_sparsity, Expr.const(np.zeros((nnz,), dtype=np.float64)), coloring_width)
  constants = []
  for f_idx, _, local_colors, first in colored:
    formal = callee.inputs[f_idx]
    seed = np.zeros((coloring_width, formal.size), dtype=np.float64)
    seed[first + local_colors, np.arange(formal.size)] = 1.0
    constants.append(seed.reshape((coloring_width, *formal.shape)))
  formals = tuple(f_idx for f_idx, *_ in colored)
  inner_fn, arg_indices, seed_indices, active = _call_jvp_many_function(callee, output_idx, formals, coloring_width, tuple(constants))
  assert not seed_indices
  primal_specs = [(vmap_expr.args[i], starts[i], strides[i]) for i in arg_indices]
  mapped_flat = vmap(inner_fn, length, primal_specs)
  active_to_pos = {c: i for i, c in enumerate(active)}
  active_count = len(active)
  pieces: list[tuple[Expr, np.ndarray]] = []  # (gathered piece values, nnz slots they cover)
  for f_idx, local_mask, local_colors, first in colored:
    # Which nnz this formal contributes, and where each sits in the mapped output.
    formal_size = callee.inputs[f_idx].size
    lc_arr = cols_arr - (offsets[f_idx] + starts[f_idx] + it_global * strides[f_idx])
    in_window = (lc_arr >= 0) & (lc_arr < formal_size)
    contributes = np.zeros(nnz, dtype=bool)
    if in_window.any():
      valid = np.flatnonzero(in_window)
      contributes[valid] = np.asarray(local_mask[lr_global[valid], lc_arr[valid]]).reshape(-1)
    if not contributes.any():
      continue
    contrib_idx = np.flatnonzero(contributes)
    pos_at = np.asarray([active_to_pos[int(first + c)] for c in local_colors[lc_arr[contrib_idx]]], dtype=np.int64)
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
  if (moved := _at_inputs(sparse_hessian, expr, wrt, triangle=triangle)) is not None:
    return moved
  gradient_expr = cse(simplify(gradient(expr, wrt).reshape((wrt.size,))))
  sparsity = _symmetrize_sparsity(jacobian_sparsity(gradient_expr, wrt))
  colors = star_coloring(sparsity)
  recovery = _star_recovery_indices(sparsity, colors)
  return _sparse_jacobian_colored(gradient_expr, wrt, sparsity, colors, recovery).triangle(triangle)
