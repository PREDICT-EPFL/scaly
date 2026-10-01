"""Ordered Program IR optimization pipeline, the slots extensions insert passes at, and its observer interface."""

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


# Passes extensions insert, as (after or before, anchor, name, pass), in the order inserted.
_INSERTED: list[tuple[bool, str, str, PassFn]] = []


def _names() -> list[str]:
  return [name for name, _ in pipeline()]


def _insert(after: bool, anchor: str, name: str, fn: PassFn) -> None:
  names = _names()
  if anchor not in names:
    raise ValueError(f"no pass {anchor!r} to insert {name!r} at; the pipeline is {names}")
  if name in names:
    raise ValueError(f"a pass named {name!r} is already in the pipeline")
  _INSERTED.append((after, anchor, name, fn))


def insert_after(anchor: str, name: str, fn: PassFn) -> None:
  """Run the pass ``fn``, named ``name``, right after the pass ``anchor`` (a ``PASS_PIPELINE`` name
  or one inserted before). Its name is what an observer sees as ``pass:<name>``.

  The pipeline is part of the JIT's key for a library it built before: ``fn`` by its module and
  name, with the files of its package. A pass that is a closure, a bound method or a partial
  application carries state no file holds, and with one inserted every Function is rendered at
  each start, as before there was a key."""
  _insert(True, anchor, name, fn)


def insert_before(anchor: str, name: str, fn: PassFn) -> None:
  """Run the pass ``fn``, named ``name``, right before the pass ``anchor``."""
  _insert(False, anchor, name, fn)


def pipeline() -> tuple[tuple[str, PassFn], ...]:
  """The passes in the order they run: ``PASS_PIPELINE`` with the inserted ones at their slots."""
  passes = list(PASS_PIPELINE)
  for after, anchor, name, fn in _INSERTED:
    at = next(i for i, (n, _) in enumerate(passes) if n == anchor)
    passes.insert(at + 1 if after else at, (name, fn))
  return tuple(passes)


def optimize_program(prog: ProgramNode, observe: ProgramObserver | None = None) -> ProgramNode:
  """Run the Program IR optimization pipeline in its declared order."""
  verify_program(prog)
  for name, fn in pipeline():
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
  "insert_after",
  "insert_before",
  "optimize_program",
  "pack_workspace",
  "pipeline",
  "prepare_scalar_expressions",
  "prune_procedures",
  "unroll_unit_loops",
]
