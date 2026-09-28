"""The ``linalg`` option namespace: when factorizations and solves become straight-line code."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..utils.options import register_option_namespace


@dataclass(frozen=True, slots=True)
class LinalgOptions:
  """The ``linalg`` namespace. It shapes generated code, never a derivative: a derivative takes each
  factorization's choice from the node it differentiates.

  Attributes:
    dense_unroll: the largest order at which ``cholesky``, ``ldl`` and ``solve_triangular`` become
      straight-line code instead of loops.
    sparse_unroll: the most multiply-adds and divisions a sparse ``L D L^T`` may take and still be
      generated as straight-line code instead of loops over its columns
      (``SparseLDL(schedule="auto")``). Straight-line code runs several times faster but costs
      about a millisecond of generation per operation.
  """

  dense_unroll: int = 8
  sparse_unroll: int = 1000


def _count(value: Any) -> bool:
  return isinstance(value, int) and not isinstance(value, bool) and value >= 0


register_option_namespace("linalg", LinalgOptions(), affects_derivatives=False, checks={"dense_unroll": _count, "sparse_unroll": _count})

__all__ = ["LinalgOptions"]
