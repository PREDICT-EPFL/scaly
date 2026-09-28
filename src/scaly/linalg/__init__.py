"""Linear algebra as Scaly values: sparse matrices with static patterns, and the factorizations
built on them. Everything here is generated code, built from the expression ops ``linalg.ops``
registers (the factorizations, triangular solves and ragged runs) and the core's own."""

from . import banded, stagewise
from .options import LinalgOptions
from .dense import cho_solve, cholesky, ldl, ldl_solve, ldl_unpack, lu, lu_solve, solve, solve_triangular
from .sparse_factor import SparseLDL, sparse_ldl
from .sparse import S, SparseMatrix
from .symbolic import CostModel, Segment, SymbolicLDL, analyze, ordering

__all__ = [
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
