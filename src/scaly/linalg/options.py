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
    dense_unroll: the largest order at which ``cholesky``, ``ldl``, ``lu`` and ``solve_triangular``
      become straight-line code instead of loops, recorded in the graph; None, the default, leaves
      the choice to the target when the graph is rendered, which makes it by the size of the
      straight-line body (``Target.straight_line_ops``).
    sparse_unroll: the most multiply-adds and divisions a sparse ``L D L^T`` may take and still be
      generated as straight-line code instead of loops over its columns
      (``SparseLDL(schedule="auto")``). Straight-line code runs several times faster but costs
      about a millisecond of generation per operation.
  """

  dense_unroll: int | None = None
  sparse_unroll: int = 1000


def _count(value: Any) -> bool:
  return isinstance(value, int) and not isinstance(value, bool) and value >= 0


register_option_namespace(
  "linalg", LinalgOptions(), affects_derivatives=False, checks={"dense_unroll": lambda v: v is None or _count(v), "sparse_unroll": _count}
)

__all__ = ["LinalgOptions"]
