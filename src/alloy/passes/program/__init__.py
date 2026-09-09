"""Ordered Program IR optimization pipeline and its observer interface."""

from __future__ import annotations

from collections.abc import Callable

from ...ir.program import ProgramNode
from .combine_scatter_sums import combine_scatter_sums
from .fold_arith import fold_arith
from .fuse_elementwise import fuse_elementwise
from .hoist_invariant import hoist_invariant
from .pack_workspace import WORKSPACE_SPILL_THRESHOLD, pack_workspace
from .scalarize import scalarize_program
from .unroll_unit_loops import unroll_unit_loops

PassFn = Callable[[ProgramNode], ProgramNode]
ProgramObserver = Callable[[str, ProgramNode], None]
PASS_PIPELINE: tuple[tuple[str, PassFn], ...] = (
  ("hoist_invariant", hoist_invariant),
  ("scalarize", scalarize_program),
  ("combine_scatter_sums", combine_scatter_sums),
  ("fuse_elementwise", fuse_elementwise),
  ("fold_arith", fold_arith),
  ("unroll_unit_loops", unroll_unit_loops),
  ("pack_workspace", pack_workspace),
)


def optimize_program(prog: ProgramNode, observe: ProgramObserver | None = None) -> ProgramNode:
  """Run the Program IR optimization pipeline in its declared order."""
  for name, fn in PASS_PIPELINE:
    prog = fn(prog)
    if observe is not None:
      observe(f"pass:{name}", prog)
  return prog


__all__ = [
  "PASS_PIPELINE",
  "WORKSPACE_SPILL_THRESHOLD",
  "ProgramObserver",
  "combine_scatter_sums",
  "fold_arith",
  "fuse_elementwise",
  "hoist_invariant",
  "optimize_program",
  "pack_workspace",
  "unroll_unit_loops",
]
