"""The public surface of ``scaly.linalg``: its names, and that ``sc.linalg`` is the package, loaded on first use."""

from __future__ import annotations

import importlib

import scaly as sc


def test_the_linalg_surface() -> None:
  linalg = importlib.import_module("scaly.linalg")
  assert sc.linalg is linalg
  assert linalg.__all__ == [
    "CostModel",
    "LinalgOptions",
    "Segment",
    "S",
    "SparseLDL",
    "SparseMatrix",
    "SymbolicLDL",
    "analyze",
    "banded",
    "cho_solve",
    "cholesky",
    "ldl",
    "ldl_solve",
    "ldl_unpack",
    "lu",
    "lu_solve",
    "ordering",
    "solve",
    "solve_triangular",
    "sparse_ldl",
    "stagewise",
  ]
  # The sparse tree spec and matrix are linalg's own; the core names neither.
  assert "linalg" not in sc.__all__ and not hasattr(sc, "S") and not hasattr(sc, "SparseMatrix")
  ops = importlib.import_module("scaly.linalg.ops")
  assert {"cholesky", "ldl", "lu", "solve_triangular", "ragged_add", "ragged_dot", "sparse_ldl_factor", "sparse_ldl_solve"} <= set(ops.__all__)
