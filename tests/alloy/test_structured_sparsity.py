"""Phase 6 first slice: structured sparsity descriptors and materialization."""

from __future__ import annotations

import pytest

import alloy as al
from alloy.structured_sparsity import (
  BlockDiagonalStructure,
  COOStructure,
  DenseStructure,
  TiledStructure,
  detect_tiled,
  from_sparsity,
  materialize,
  to_mask,
)


def test_dense_materialize_round_trip() -> None:
  d = DenseStructure((3, 4))
  sp = materialize(d)
  assert sp.shape == (3, 4)
  assert sp.nnz == 12


def test_coo_materialize_identity() -> None:
  c = COOStructure((3, 3), (0, 1, 2), (1, 2, 0))
  sp = materialize(c)
  assert sp.rows == (0, 1, 2)
  assert sp.cols == (1, 2, 0)
  # round-trip via from_sparsity
  c2 = from_sparsity(sp)
  assert c2 == c


def test_tiled_materialize_expands_to_repeated_block() -> None:
  base = COOStructure((2, 2), (0, 1), (0, 1))
  tile = TiledStructure(base, length=3, row_stride=2, col_stride=2, shape=(6, 6))
  sp = materialize(tile)
  assert sp.nnz == 6
  assert sp.rows == (0, 1, 2, 3, 4, 5)
  assert sp.cols == (0, 1, 2, 3, 4, 5)


def test_tiled_materialize_supports_zero_length() -> None:
  base = COOStructure((1, 1), (0,), (0,))
  tile = TiledStructure(base, length=0, row_stride=1, col_stride=1, shape=(0, 0))
  sp = materialize(tile)
  assert sp.nnz == 0


def test_block_diagonal_materialize() -> None:
  b1 = COOStructure((2, 2), (0, 1), (0, 1))
  b2 = COOStructure((3, 3), (0, 1, 2), (0, 1, 2))
  block = BlockDiagonalStructure((b1, b2), row_offsets=(0, 2), col_offsets=(0, 2), shape=(5, 5))
  sp = materialize(block)
  assert sp.nnz == 5
  assert sp.rows == (0, 1, 2, 3, 4)
  assert sp.cols == (0, 1, 2, 3, 4)


def test_detect_tiled_recognizes_repeated_block_diagonal() -> None:
  base = COOStructure((2, 2), (0, 1), (0, 1))
  big = materialize(TiledStructure(base, length=4, row_stride=2, col_stride=2, shape=(8, 8)))
  found = detect_tiled(big, length=4)
  assert found is not None
  assert found.length == 4
  assert found.row_stride == 2
  assert found.col_stride == 2
  inner = found.base
  assert isinstance(inner, COOStructure)
  assert inner.rows == base.rows


def test_detect_tiled_returns_none_when_not_uniform() -> None:
  # 3x3 identity has 3 nonzeros; not a valid length=2 tile
  sp = al.SparsityType((3, 3), (0, 1, 2), (0, 1, 2))
  assert detect_tiled(sp, length=2) is None


def test_to_mask_matches_materialized_sparsity() -> None:
  base = COOStructure((2, 2), (0, 1), (1, 0))
  tile = TiledStructure(base, length=3, row_stride=2, col_stride=2, shape=(6, 6))
  mask = to_mask(tile)
  assert mask.shape == (6, 6)
  assert int(mask.sum()) == 6


def test_unknown_structure_type_raises() -> None:
  with pytest.raises(TypeError):
    materialize("not a structure")  # ty: ignore[invalid-argument-type]


def test_structured_sparsity_is_part_of_public_alloy_surface() -> None:
  # Verify the module is importable from alloy.* (callers can find the new API).
  from alloy import structured_sparsity as ss

  assert hasattr(ss, "TiledStructure")
  assert hasattr(ss, "materialize")


def test_structured_jacobian_sparsity_recognizes_map_tile() -> None:
  """MAP-based jacobians yield a ``TiledStructure`` whose tile matches the per-stage local mask."""
  from alloy.sparsity import structured_jacobian_sparsity

  x = al.sym("x", 2)
  stage = al.Function("stage", [x], [(x * x).sum()], ["x"], ["y"])
  z = al.sym("z", 6)
  mapped = al.map_(stage, length=3, inputs=[(z, 0, 2)])
  struct = structured_jacobian_sparsity(mapped, z)
  assert isinstance(struct, TiledStructure)
  assert struct.length == 3
  assert struct.row_stride == 1  # stage output size
  assert struct.col_stride == 2  # stage input size
  # materializing must match the flat jacobian_sparsity
  flat = materialize(struct)
  from alloy.sparsity import jacobian_sparsity

  ref = jacobian_sparsity(mapped, z)
  assert flat.rows == ref.rows
  assert flat.cols == ref.cols


def test_structured_jacobian_sparsity_falls_back_to_coo() -> None:
  """A scalar elementwise graph has no MAP — descriptor falls back to a plain COOStructure."""
  from alloy.sparsity import structured_jacobian_sparsity

  x = al.sym("x", 4)
  y = (x * x).sum()
  struct = structured_jacobian_sparsity(y, x)
  assert isinstance(struct, COOStructure)
