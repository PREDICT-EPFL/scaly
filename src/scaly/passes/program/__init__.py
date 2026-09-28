"""Ordered Program IR optimization pipeline and its observer interface."""

from __future__ import annotations

from typing import Literal

from collections.abc import Callable

from ...ir.program import ProgramNode
from ...ir.program_spec import verify_program
from ._common import prune_procedures
from .coalesce_stores import coalesce_stores
from .combine_scatter_sums import combine_scatter_sums
from .fold_arith import fold_arith
from .fold_tiles import fold_tiles
from .fuse_elementwise import fuse_elementwise
from .fuse_ranges import fuse_ranges
from .hoist_invariant import hoist_invariant
from .hoist_reciprocals import hoist_reciprocals
from .pack_workspace import pack_workspace
from .prepare_scalar import prepare_scalar_expressions
from .scalarize import scalarize_program
from .unroll_unit_loops import unroll_unit_loops
from .widen_ranges import widen_ranges

PassFn = Callable[[ProgramNode], ProgramNode]
ProgramObserver = Callable[[str, ProgramNode], None]
PASS_PIPELINE: tuple[tuple[str, PassFn], ...] = (
  ("hoist_invariant", hoist_invariant),
  ("scalarize", scalarize_program),
  ("fold_tiles", fold_tiles),
  ("fuse_ranges", fuse_ranges),
  ("prune_procedures", prune_procedures),
  ("combine_scatter_sums", combine_scatter_sums),
  ("fuse_elementwise", fuse_elementwise),
  ("fold_arith", fold_arith),
  ("unroll_unit_loops", unroll_unit_loops),
  ("fold_arith_after_unroll", fold_arith),
  ("pack_workspace", pack_workspace),
  ("coalesce_stores", coalesce_stores),
  ("prepare_scalar", prepare_scalar_expressions),
)


def optimize_program(
  prog: ProgramNode, observe: ProgramObserver | None = None, *, reciprocal: bool = False, lanes: Literal["auto"] | Literal[1, 2, 4, 8] = 1
) -> ProgramNode:
  """Run the Program IR optimization pipeline in its declared order."""
  verify_program(prog)
  for name, fn in PASS_PIPELINE:
    if name == "prepare_scalar" and lanes != 1:
      continue
    if name == "pack_workspace":
      if reciprocal:
        prog = hoist_reciprocals(prog)
        if observe is not None:
          observe("pass:hoist_reciprocals", prog)
      if lanes != 1:
        prog = prepare_scalar_expressions(prog)
        prog = widen_ranges(prog, lanes=lanes)
        if observe is not None:
          observe("pass:widen_ranges", prog)
    prog = fn(prog)
    if observe is not None:
      observe(f"pass:{name}", prog)
  return prog


__all__ = ["PASS_PIPELINE", "ProgramObserver", "optimize_program"]
