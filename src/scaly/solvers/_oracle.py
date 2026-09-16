"""Shared utilities for assembling oracle Scaly ``Function``s for QP/NLP backends."""

from __future__ import annotations

from typing import Iterable

from ..ir.expr import Expr, ExprOp, topo


def collect_free_inputs(exprs: Iterable[Expr]) -> tuple[Expr, ...]:
  """Return all unique ``ExprOp.INPUT`` exprs reachable from ``exprs`` in deterministic order.

  Deterministic ordering uses (name, id) so that a stable parameter signature is produced
  regardless of construction order.
  """
  seen: set[int] = set()
  result: list[Expr] = []
  for node in topo(exprs):
    if node.op == ExprOp.INPUT and node.id not in seen:
      seen.add(node.id)
      result.append(node)
  result.sort(key=lambda e: (e.name or "", e.id))
  return tuple(result)
