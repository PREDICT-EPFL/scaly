"""Structured sparsity descriptors (Phase 6 first slice).

Today ``SparsityType`` materializes a sparse pattern as a flat COO ``(rows, cols)``
table. That's correct but pays an ``O(nnz)`` table cost: a Jacobian that's
``N`` copies of a small per-stage block ends up with ``O(N * nnz_per_stage)``
constant data and many redundant entries.

This module adds a structured layer above ``SparsityType`` that can describe:

- ``DenseStructure``: a dense block of some shape.
- ``COOStructure``: an explicit list of nonzero coordinates (same as
  ``SparsityType``).
- ``TiledStructure``: ``length`` copies of a base structured pattern placed
  along (row, col) offsets ``(row_stride * i, col_stride * i)`` — the natural
  shape of MAP-based Jacobians where the same per-stage Jacobian repeats.
- ``BlockDiagonalStructure``: a block-diagonal layout of arbitrary sub-patterns
  (useful for batched MLPs and multistage OCPs whose stages differ slightly).

``materialize`` flattens any structured descriptor to a ``SparsityType`` for
backward-compatible callers (e.g. solver headers, the legacy renderer).
Future Program IR sparse assembly can iterate the structure directly instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import SparsityType


@dataclass(frozen=True, slots=True)
class DenseStructure:
  shape: tuple[int, int]

  @property
  def nnz(self) -> int:
    return self.shape[0] * self.shape[1]


@dataclass(frozen=True, slots=True)
class COOStructure:
  shape: tuple[int, int]
  rows: tuple[int, ...]
  cols: tuple[int, ...]

  def __post_init__(self) -> None:
    if len(self.rows) != len(self.cols):
      raise ValueError("rows and cols must have the same length")

  @property
  def nnz(self) -> int:
    return len(self.rows)


@dataclass(frozen=True, slots=True)
class TiledStructure:
  """``length`` copies of ``base`` placed at row offsets ``i*row_stride`` and column offsets ``i*col_stride``."""

  base: "StructuredSparsity"
  length: int
  row_stride: int
  col_stride: int
  shape: tuple[int, int]

  def __post_init__(self) -> None:
    if self.length < 0:
      raise ValueError(f"tiled length must be non-negative, got {self.length}")
    if self.shape[0] < 0 or self.shape[1] < 0:
      raise ValueError(f"tiled shape must be non-negative, got {self.shape}")

  @property
  def nnz(self) -> int:
    return self.length * self.base.nnz


@dataclass(frozen=True, slots=True)
class BlockDiagonalStructure:
  """Block-diagonal stack of arbitrary sub-structures with per-block row/col offsets."""

  blocks: tuple["StructuredSparsity", ...]
  row_offsets: tuple[int, ...]
  col_offsets: tuple[int, ...]
  shape: tuple[int, int]

  def __post_init__(self) -> None:
    if not (len(self.blocks) == len(self.row_offsets) == len(self.col_offsets)):
      raise ValueError("blocks, row_offsets, col_offsets must have the same length")

  @property
  def nnz(self) -> int:
    return sum(b.nnz for b in self.blocks)


StructuredSparsity = DenseStructure | COOStructure | TiledStructure | BlockDiagonalStructure


def materialize(structure: StructuredSparsity) -> SparsityType:
  """Flatten a structured descriptor to a COO ``SparsityType`` for legacy/ABI consumers."""
  if isinstance(structure, DenseStructure):
    return SparsityType.dense(structure.shape)
  if isinstance(structure, COOStructure):
    return SparsityType(structure.shape, structure.rows, structure.cols)
  if isinstance(structure, TiledStructure):
    base_sp = materialize(structure.base)
    rows: list[int] = []
    cols: list[int] = []
    for i in range(structure.length):
      dr = i * structure.row_stride
      dc = i * structure.col_stride
      rows.extend(r + dr for r in base_sp.rows)
      cols.extend(c + dc for c in base_sp.cols)
    return SparsityType(structure.shape, tuple(rows), tuple(cols))
  if isinstance(structure, BlockDiagonalStructure):
    rows = []
    cols = []
    for blk, dr, dc in zip(structure.blocks, structure.row_offsets, structure.col_offsets, strict=True):
      blk_sp = materialize(blk)
      rows.extend(r + dr for r in blk_sp.rows)
      cols.extend(c + dc for c in blk_sp.cols)
    return SparsityType(structure.shape, tuple(rows), tuple(cols))
  raise TypeError(f"unknown structured sparsity: {structure!r}")


def from_sparsity(sp: SparsityType) -> COOStructure:
  """Lift a materialized ``SparsityType`` into a ``COOStructure`` (identity wrap)."""
  return COOStructure(sp.shape, sp.rows, sp.cols)


def detect_tiled(sp: SparsityType, length: int) -> TiledStructure | None:
  """Try to recognize ``sp`` as ``length`` copies of a smaller base pattern.

  Heuristic: if the nnz divides into ``length`` equal-size groups whose
  ``(row - i*row_stride, col - i*col_stride)`` matches the first group's
  pattern, return a ``TiledStructure``. Otherwise return ``None``.

  This is intentionally conservative — it only fires when the pattern is
  unambiguously tiled in row-major nnz order. Real callers should prefer
  building ``TiledStructure`` directly when MAP structure is known
  upstream (see ``alloy.sparsity._sparse_jacobian_map``).
  """
  if length <= 0 or sp.nnz % length != 0:
    return None
  per = sp.nnz // length
  if per == 0:
    return None
  base_rows = sp.rows[:per]
  base_cols = sp.cols[:per]
  if not base_rows:
    return None
  row_stride = (sp.rows[per] - base_rows[0]) if length > 1 else 0
  col_stride = (sp.cols[per] - base_cols[0]) if length > 1 else 0
  for i in range(length):
    chunk_rows = sp.rows[i * per : (i + 1) * per]
    chunk_cols = sp.cols[i * per : (i + 1) * per]
    if tuple(r - i * row_stride for r in chunk_rows) != base_rows:
      return None
    if tuple(c - i * col_stride for c in chunk_cols) != base_cols:
      return None
  base = COOStructure((max(base_rows) + 1 if base_rows else 0, max(base_cols) + 1 if base_cols else 0), base_rows, base_cols)
  return TiledStructure(base, length, row_stride, col_stride, sp.shape)


def to_mask(structure: StructuredSparsity) -> np.ndarray:
  """Materialize the structured pattern into a dense boolean mask (debug helper)."""
  return materialize(structure).to_mask()


__all__ = [
  "BlockDiagonalStructure",
  "COOStructure",
  "DenseStructure",
  "StructuredSparsity",
  "TiledStructure",
  "detect_tiled",
  "from_sparsity",
  "materialize",
  "to_mask",
]
