"""Ordered Program IR optimization pipeline and its observer interface."""

from __future__ import annotations

from collections.abc import Callable

from ...ir.program import ProgramNode
from ...ir.program_spec import verify_program
from ._common import prune_procedures
from .coalesce_stores import coalesce_stores
from .combine_scatter_sums import combine_scatter_sums
from .delinearize_loops import delinearize_loops
from .fold_arith import fold_arith
from .fuse_elementwise import fuse_elementwise
from .hoist_invariant import hoist_invariant
from .pack_workspace import WORKSPACE_SPILL_THRESHOLD, pack_workspace
from .prepare_scalar import prepare_scalar_expressions
from .scalarize import scalarize_program
from .unroll_unit_loops import unroll_unit_loops

PassFn = Callable[[ProgramNode], ProgramNode]
ProgramObserver = Callable[[str, ProgramNode], None]
PASS_PIPELINE: tuple[tuple[str, PassFn], ...] = (
  ("hoist_invariant", hoist_invariant),
  ("scalarize", scalarize_program),
  ("prune_procedures", prune_procedures),
  ("combine_scatter_sums", combine_scatter_sums),
  ("fuse_elementwise", fuse_elementwise),
  ("fold_arith", fold_arith),
  ("unroll_unit_loops", unroll_unit_loops),
  ("fold_arith_after_unroll", fold_arith),
  ("delinearize_loops", delinearize_loops),
  ("pack_workspace", pack_workspace),
  ("coalesce_stores", coalesce_stores),
  ("prepare_scalar", prepare_scalar_expressions),
)


def optimize_program(prog: ProgramNode, observe: ProgramObserver | None = None) -> ProgramNode:
  """Run the Program IR optimization pipeline in its declared order."""
  verify_program(prog)
  for name, fn in PASS_PIPELINE:
    prog = fn(prog)
    if observe is not None:
      observe(f"pass:{name}", prog)
  return prog


__all__ = [
  "PASS_PIPELINE",
  "WORKSPACE_SPILL_THRESHOLD",
  "ProgramObserver",
  "coalesce_stores",
  "combine_scatter_sums",
  "delinearize_loops",
  "fold_arith",
  "fuse_elementwise",
  "hoist_invariant",
  "optimize_program",
  "pack_workspace",
  "prepare_scalar_expressions",
  "prune_procedures",
  "unroll_unit_loops",
]
