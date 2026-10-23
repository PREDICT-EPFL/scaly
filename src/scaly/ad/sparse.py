"""Compact sparse Jacobians and Hessians: AD-driven construction over a structural pattern.

The pattern itself — which entries can be nonzero — comes from ``sparsity.py``, which knows
nothing about AD. This module turns a pattern into values: graph coloring, compressed JVPs, and
the structured VMAP decomposition that keeps a multistage Jacobian from materializing densely.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Literal

import numpy as np

from ..function.sugar import _mapped_call
from ..ir.expr import Expr, ExprOp, concat, gather, scatter
from ..passes.expr import cse, simplify, simplify_cse_fixpoint
from .derivatives import gradient, jacobian
from .calls import _call_jvp_many_const_function
from .forward import _pushforward, jvp_many
from .sparsity import _callee_mask, _check_wrt, _depends_on, _mask_sparsity, _symmetrize_sparsity, column_coloring, jacobian_sparsity, star_coloring
from ..ir.types import SparsityPattern


Triangle = Literal["full", "lower", "upper"]
_TRIANGLES: tuple[Triangle, ...] = ("full", "lower", "upper")


def _validate_triangle(triangle: object) -> Triangle:
  """Validate and type-narrow a sparse-Hessian layout selection."""
  if not isinstance(triangle, str) or triangle not in _TRIANGLES:
    raise ValueError(f"triangle must be one of {_TRIANGLES}, got {triangle!r}")
  return triangle


@dataclass(frozen=True, slots=True)
class SparseJacobian:
  """A sparse derivative: the structural ``sparsity`` and the ``values`` of its entries, in the same order.

  ``coloring_width`` is the number of forward-mode directions the values were computed from, or
  ``None`` when they did not come from a coloring.
  """

  sparsity: SparsityPattern
  values: Expr
  coloring_width: int | None = None
  _compressed: Expr | None = field(default=None, compare=False, repr=False)
  _recovery: np.ndarray | None = field(default=None, compare=False, repr=False)

  @property
  def flat_indices(self) -> np.ndarray:
    """Row-major positions of the entries in the dense matrix."""
    return np.asarray(self.sparsity.rows, dtype=np.int64) * self.sparsity.shape[1] + np.asarray(self.sparsity.cols, dtype=np.int64)

  def to_dense(self) -> Expr:
    """Scatter the values into a dense matrix expression."""
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
    sparsity = SparsityPattern(
      self.sparsity.shape,
      tuple(int(row) for row in rows[keep]),
      tuple(int(col) for col in cols[keep]),
    )
    source = self._compressed if self._compressed is not None else self.values
    recovery = self._recovery[keep] if self._recovery is not None else np.flatnonzero(keep)
    values = simplify_cse_fixpoint(gather(source, recovery))
    return SparseJacobian(sparsity, values, self.coloring_width, source, recovery)


def sparse_jacobian_reference(expr: Expr, wrt: Expr) -> SparseJacobian:
  """Reference compact Jacobian path: build dense ``J`` and gather nonzeros.

  ``wrt`` must be an independent input from ``sc.sym`` or a Function declaration.
  """

  sparsity = jacobian_sparsity(expr, wrt)
  dense = jacobian(expr, wrt)
  flat = np.asarray(sparsity.rows, dtype=np.int64) * wrt.size + np.asarray(sparsity.cols, dtype=np.int64)
  return SparseJacobian(sparsity, gather(dense, flat))


def sparse_jacobian_colored(expr: Expr, wrt: Expr) -> SparseJacobian:
  """Compact sparse Jacobian values from graph-colored compressed JVPs.

  ``wrt`` must be an independent input from ``sc.sym`` or a Function declaration.
  """
  _check_wrt(wrt)
  expr = cse(expr)
  sparsity = jacobian_sparsity(expr, wrt)
  colors = column_coloring(sparsity)
  return _sparse_jacobian_colored(expr, wrt, sparsity, colors)


def _sparse_jacobian_colored(
  expr: Expr,
  wrt: Expr,
  sparsity: SparsityPattern,
  colors: tuple[int, ...],
  recovery_indices: np.ndarray | None = None,
) -> SparseJacobian:
  """Build compact values from a coloring and an optional compressed-row recovery table."""
  if sparsity.nnz == 0:
    return SparseJacobian(sparsity, Expr.const(np.zeros((0,), dtype=expr.type.dtype.numpy()), dtype=expr.type.dtype), 0)
  ncolors = max(colors) + 1 if colors else 0
  seeds = np.zeros((ncolors, wrt.size), dtype=wrt.type.dtype.numpy())
  for col, color in enumerate(colors):
    seeds[color, col] = 1.0
  compressed = jvp_many(expr, wrt, Expr.const(seeds.reshape((ncolors, *wrt.shape)), dtype=wrt.type.dtype)).reshape((ncolors, expr.size)).T
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


def sparse_jacobian(expr: Expr, wrt: Expr) -> SparseJacobian:
  """Compact sparse Jacobian: the structural pattern plus an expression for its nonzeros only.

  ``wrt`` must be an independent input from ``sc.sym`` or a Function declaration.
  Takes the structured path when ``expr`` is a ``VMAP`` (or a concatenation of them) over exactly
  ``wrt``, coloring the callee's small local pattern instead of the whole matrix, so the cost
  tracks the callee rather than the iteration count. Otherwise falls back to coloring the global
  pattern. The value order is the pattern's ``(rows, cols)`` order, which is not necessarily
  sorted.
  """
  _check_wrt(wrt)
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
  sparsity = SparsityPattern((expr.size, wrt.size), tuple(global_rows), tuple(global_cols))
  if sparsity.nnz == 0:
    return SparseJacobian(sparsity, Expr.const(np.zeros((0,), dtype=expr.type.dtype.numpy()), dtype=expr.type.dtype), coloring_width)
  # Flatten piece values that are themselves axis-0 CONCATs so their producers write straight
  # into the single output concat instead of materializing an intermediate nnz-sized buffer.
  flat = [a for v in global_values for a in (v.args if v.op == ExprOp.CONCAT and v.attrs.get("axis", 0) == 0 else (v,))]
  values = flat[0] if len(flat) == 1 else concat(flat, axis=0)
  return SparseJacobian(sparsity, simplify_cse_fixpoint(values), coloring_width)


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

  global_sparsity = jacobian_sparsity(vmap_expr, wrt)
  if global_sparsity.nnz == 0 or length == 0:
    return SparseJacobian(
      global_sparsity, Expr.const(np.zeros((global_sparsity.nnz,), dtype=vmap_expr.type.dtype.numpy()), dtype=vmap_expr.type.dtype), 0
    )

  dep_memo: dict[tuple[int, int], bool] = {}
  direct_formals: list[int] = []
  for f_idx, actual in enumerate(vmap_expr.args):
    while actual.op == ExprOp.RESHAPE:
      actual = actual.args[0]
    if actual.id == wrt.id:
      direct_formals.append(f_idx)
    elif _depends_on(actual, wrt, dep_memo):
      # Indirect dependency would require a chain-rule through actual_outer; fall back.
      return sparse_jacobian_colored(vmap_expr, wrt)

  if not direct_formals:
    return SparseJacobian(
      global_sparsity, Expr.const(np.zeros((global_sparsity.nnz,), dtype=vmap_expr.type.dtype.numpy()), dtype=vmap_expr.type.dtype), 0
    )

  rows_arr = np.asarray(global_sparsity.rows, dtype=np.int64)
  cols_arr = np.asarray(global_sparsity.cols, dtype=np.int64)
  nnz = global_sparsity.nnz
  it_global = rows_arr // slice_size
  lr_global = rows_arr % slice_size

  pieces: list[tuple[Expr, np.ndarray]] = []  # (gathered piece values, nnz slots they cover)
  coloring_width = 0
  for f_idx in direct_formals:
    formal = callee.inputs[f_idx]
    local_mask = _callee_mask(callee, output_idx, f_idx, {})
    if not local_mask.nnz:
      continue
    local_sparsity = _mask_sparsity(local_mask)
    local_colors = column_coloring(local_sparsity)
    if not local_colors:
      continue
    c_f = max(local_colors) + 1
    coloring_width += c_f
    seed_f = np.zeros((c_f, formal.size), dtype=formal.type.dtype.numpy())
    for j, c in enumerate(local_colors):
      seed_f[c, j] = 1.0
    seed_f_shaped = seed_f.reshape((c_f, *formal.shape)) if formal.shape != (formal.size,) else seed_f
    inner_fn, arg_indices, active = _call_jvp_many_const_function(callee, output_idx, f_idx, seed_f_shaped, pushforward=_pushforward)
    active_count = len(active)
    if active_count == 0:
      continue
    primal_specs = [(vmap_expr.args[i], starts[i], strides[i]) for i in arg_indices]
    mapped_flat = _mapped_call(inner_fn, length, primal_specs)
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
    return SparseJacobian(
      global_sparsity, Expr.const(np.zeros((nnz,), dtype=vmap_expr.type.dtype.numpy()), dtype=vmap_expr.type.dtype), coloring_width
    )
  perm = np.concatenate([idx for _, idx in pieces])
  if perm.size == nnz and np.bincount(perm, minlength=nnz).max() == 1:
    # Piece supports exactly partition the nnz (verified: every slot covered exactly once), so
    # instead of scattering each piece into a full-nnz buffer and summing, emit the gathered
    # values back to back and permute the pattern to match — no full-nnz temporaries at all.
    values = pieces[0][0] if len(pieces) == 1 else concat([g for g, _ in pieces], axis=0)
    permuted = SparsityPattern(global_sparsity.shape, tuple(int(r) for r in rows_arr[perm]), tuple(int(c) for c in cols_arr[perm]))
    return SparseJacobian(permuted, values, coloring_width)
  values = None
  for gathered, contrib_idx in pieces:
    piece_values = gathered if contrib_idx.size == nnz else scatter(gathered, contrib_idx, (nnz,))
    values = piece_values if values is None else values + piece_values
  assert values is not None
  return SparseJacobian(global_sparsity, values, coloring_width)


def sparse_hessian(expr: Expr, wrt: Expr, *, triangle: Triangle = "full") -> SparseJacobian:
  """Compact sparse Hessian of a scalar expression using one global star-colored JVP batch.

  ``wrt`` must be an independent input from ``sc.sym`` or a Function declaration.
  """
  triangle = _validate_triangle(triangle)
  if expr.size != 1:
    raise ValueError("sparse_hessian expects a scalar expression")
  _check_wrt(wrt)
  if os.environ.get("SCALY_LOCAL_HESSIAN") == "1":
    local = _separable_hessian(expr, wrt, triangle)
    if local is not None:
      return local
  gradient_expr = cse(simplify(gradient(expr, wrt).reshape((wrt.size,))))
  sparsity = _symmetrize_sparsity(jacobian_sparsity(gradient_expr, wrt))
  colors = star_coloring(sparsity)
  recovery = _star_recovery_indices(sparsity, colors)
  return _sparse_jacobian_colored(gradient_expr, wrt, sparsity, colors, recovery).triangle(triangle)


def _keep(rows: np.ndarray, cols: np.ndarray, triangle: Triangle) -> np.ndarray:
  return np.ones(rows.shape, dtype=bool) if triangle == "full" else (rows >= cols if triangle == "lower" else rows <= cols)


def _separable_hessian(expr: Expr, wrt: Expr, triangle: Triangle = "full") -> SparseJacobian | None:
  """Hessian of a sum of mapped terms and a remainder: one local Hessian per stage, scattered into the nonzeros."""
  memo: dict[tuple[int, int], bool] = {}
  mapped: list[tuple[Expr, Expr | None]] = []
  rest: list[Expr] = []
  _split_scalar(expr, wrt, None, mapped, rest, memo)
  if not mapped:
    return None
  terms: list[tuple[np.ndarray, np.ndarray, Expr]] = []
  width = 0
  for vmap, weights in mapped:
    term = _mapped_hessian(vmap, weights, wrt, memo, triangle)
    if term is None:
      rest.append(_weighted_sum(vmap, weights))
      continue
    rows, cols, values, w = term
    width = max(width, w)
    if rows.size:
      terms.append((rows, cols, values))
  if not terms:
    return None
  if rest:
    remainder = rest[0]
    for r in rest[1:]:
      remainder = remainder + r
    support = np.asarray(sorted(set(jacobian_sparsity(remainder.reshape((1,)), wrt).cols)), dtype=np.int64)
    if support.size:
      v = Expr.sym("remainder_v", (support.size,))
      mask = np.ones(wrt.size)
      mask[support] = 0.0
      flat_wrt = wrt.reshape((wrt.size,))
      embedded = (flat_wrt * Expr.const(mask) + scatter(v, support, (wrt.size,))).reshape(wrt.shape)
      from ..ir.expr import substitute
      local = sparse_hessian(substitute(remainder, {wrt: embedded}), v)
      if local.sparsity.nnz:
        rows = support[np.asarray(local.sparsity.rows, dtype=np.int64)]
        cols = support[np.asarray(local.sparsity.cols, dtype=np.int64)]
        keep = _keep(rows, cols, triangle)
        values = substitute(gather(local.values, np.flatnonzero(keep)), {v: gather(flat_wrt, support)})
        width = max(width, local.coloring_width or 0)
        terms.append((rows[keep], cols[keep], values))
  n = wrt.size
  flat = np.unique(np.concatenate([rows * n + cols for rows, cols, _ in terms]))
  sparsity = SparsityPattern((n, n), tuple(int(v) for v in flat // n), tuple(int(v) for v in flat % n))
  values: Expr | None = None
  for rows, cols, term_values in terms:
    positions = np.searchsorted(flat, rows * n + cols)
    if positions.size == flat.size and np.array_equal(positions, np.arange(flat.size)):
      part = term_values
    else:
      part = scatter(term_values, positions, (flat.size,))
    values = part if values is None else values + part
  assert values is not None
  sj = SparseJacobian(sparsity, simplify_cse_fixpoint(values), width)
  return sj if bool(_keep(flat // n, flat % n, triangle).all()) else sj.triangle(triangle)


def _weighted_sum(y: Expr, weights: Expr | None) -> Expr:
  return y.sum() if weights is None else (weights * y).sum()


def _split_scalar(expr: Expr, wrt: Expr, scale: Expr | None, mapped: list, rest: list, memo: dict) -> None:
  if not _depends_on(expr, wrt, memo):
    return
  op = expr.op
  if op == ExprOp.ADD and all(a.shape == () for a in expr.args):
    for arg in expr.args:
      _split_scalar(arg, wrt, scale, mapped, rest, memo)
    return
  if op == ExprOp.SUB and all(a.shape == () for a in expr.args):
    _split_scalar(expr.args[0], wrt, scale, mapped, rest, memo)
    _split_scalar(expr.args[1], wrt, -1.0 if scale is None else -scale, mapped, rest, memo)
    return
  if op == ExprOp.NEG:
    _split_scalar(expr.args[0], wrt, -1.0 if scale is None else -scale, mapped, rest, memo)
    return
  if op == ExprOp.MUL and all(a.shape == () for a in expr.args):
    a, b = expr.args
    if not _depends_on(a, wrt, memo):
      _split_scalar(b, wrt, a if scale is None else scale * a, mapped, rest, memo)
      return
    if not _depends_on(b, wrt, memo):
      _split_scalar(a, wrt, b if scale is None else scale * b, mapped, rest, memo)
      return
  if op == ExprOp.SUM:
    y = expr.args[0]
    weights = None if scale is None else scale * Expr.const(np.ones(y.shape))
    _split_weighted(y, wrt, weights, mapped, rest, memo)
    return
  rest.append(expr if scale is None else scale * expr)


def _split_weighted(y: Expr, wrt: Expr, weights: Expr | None, mapped: list, rest: list, memo: dict) -> None:
  """Split ``sum(weights * y)`` with ``weights`` independent of ``wrt``."""
  if not _depends_on(y, wrt, memo):
    return
  op = y.op
  if op == ExprOp.RESHAPE:
    inner = y.args[0]
    _split_weighted(inner, wrt, None if weights is None else weights.reshape(inner.shape), mapped, rest, memo)
    return
  if op == ExprOp.CONCAT and len(y.shape) == 1:
    offset = 0
    for arg in y.args:
      _split_weighted(arg, wrt, None if weights is None else weights[offset : offset + arg.size], mapped, rest, memo)
      offset += arg.size
    return
  if op in (ExprOp.ADD, ExprOp.SUB) and all(a.shape == y.shape for a in y.args):
    _split_weighted(y.args[0], wrt, weights, mapped, rest, memo)
    neg = -1.0 * (Expr.const(np.ones(y.shape)) if weights is None else weights)
    _split_weighted(y.args[1], wrt, weights if op == ExprOp.ADD else neg, mapped, rest, memo)
    return
  if op == ExprOp.MUL and all(a.shape == y.shape for a in y.args):
    a, b = y.args
    for c, other in ((a, b), (b, a)):
      if not _depends_on(c, wrt, memo):
        _split_weighted(other, wrt, c if weights is None else weights * c, mapped, rest, memo)
        return
  if op == ExprOp.VMAP and y.attrs["callee"].rules is None:
    mapped.append((y, weights))
    return
  rest.append(_weighted_sum(y, weights))


def _mapped_hessian(vmap: Expr, weights: Expr | None, wrt: Expr, memo: dict, triangle: Triangle = "full"):
  callee = vmap.attrs["callee"]
  output = vmap.attrs["output"]
  length = vmap.attrs["length"]
  starts = vmap.attrs["starts"]
  strides = vmap.attrs["strides"]
  slice_size = vmap.attrs["slice_size"]
  active: list[int] = []
  for i, actual in enumerate(vmap.args):
    while actual.op == ExprOp.RESHAPE:
      actual = actual.args[0]
    if actual.id == wrt.id:
      active.append(i)
    elif _depends_on(actual, wrt, memo):
      return None
  if not active or length == 0:
    return None
  _, _, local, _ = _local_hessian_function(callee, output, tuple(active), None)
  local_global: list[int] = []
  local_stride: list[int] = []
  for a in active:
    size = callee.inputs[a].size
    local_global.extend(starts[a] + j for j in range(size))
    local_stride.extend([strides[a]] * size)
  base = np.asarray(local_global, dtype=np.int64)
  step = np.asarray(local_stride, dtype=np.int64)
  lr = np.asarray(local.rows, dtype=np.int64)
  lc = np.asarray(local.cols, dtype=np.int64)
  it = np.arange(length, dtype=np.int64)[:, None]
  rows = base[lr][None, :] + it * step[lr][None, :]
  cols = base[lc][None, :] + it * step[lc][None, :]
  keep = _keep(rows, cols, triangle).any(axis=0)
  kept = tuple(int(k) for k in np.flatnonzero(keep))
  if not kept:
    return (np.zeros(0, np.int64), np.zeros(0, np.int64), Expr.const(np.zeros(0)), 0)
  fn, arg_indices, _, local_width = _local_hessian_function(callee, output, tuple(active), kept)
  lam = Expr.const(np.ones(slice_size)) if weights is None else weights.reshape((weights.size,))
  specs = [(vmap.args[i], starts[i], strides[i]) for i in arg_indices]
  specs.append((lam, 0, 0 if weights is None else slice_size))
  values = _mapped_call(fn, length, specs)
  return rows[:, keep].reshape(-1), cols[:, keep].reshape(-1), values, local_width


def _local_hessian_function(callee, output: int, active: tuple[int, ...], kept: tuple[int, ...] | None):
  from ..function.concrete import ConcreteFunction
  key = ("local_hessian", output, active, kept)
  cache = callee._memo.helpers
  if key not in cache:
    sizes = [callee.inputs[a].size for a in active]
    v = Expr.sym("v", (sum(sizes),))
    parts = {}
    offset = 0
    for a, size in zip(active, sizes, strict=True):
      parts[callee.inputs[a]] = v[offset : offset + size].reshape(callee.inputs[a].shape)
      offset += size
    lam_name = f"lam:{callee.result_names[output]}"
    lam = Expr.sym(lam_name, callee.outputs[output].shape)
    from ..ir.expr import substitute
    body = substitute(callee.results[output], parts)
    local = sparse_hessian((lam * body).sum(), v)
    flat_formals = concat([callee.inputs[a].reshape((callee.inputs[a].size,)) for a in active])
    local_values = local.values if kept is None else gather(local.values, np.asarray(kept, dtype=np.int64))
    values = callee._inherit_lowering(simplify_cse_fixpoint(substitute(local_values, {v: flat_formals})))
    dep: dict[tuple[int, int], bool] = {}
    arg_indices = tuple(i for i, inp in enumerate(callee.inputs) if _depends_on(values, inp, dep))
    inputs = tuple(callee.inputs[i] for i in arg_indices) + (lam,)
    names = tuple(callee.input_names[i] for i in arg_indices) + (lam_name,)
    import hashlib
    digest = hashlib.sha1(repr((output, active, kept, callee._effective_lowering)).encode()).hexdigest()[:10]
    fn = ConcreteFunction._from_exprs(f"{callee.name}_hess_{digest}", inputs, [values], names, ["hess"], role="forward")
    cache[key] = (fn, arg_indices, local.sparsity, local.coloring_width or 0)
  return cache[key]
