"""Solver queries over a ``ConcreteFunction`` graph.

What a caller needs to know before rendering or compiling: whether a ConcreteFunction is a solver, which
inner Functions its descriptor drives, which backends it transitively reaches, and the native
flags that reaching them costs. Lowering and codegen both ask these questions, so they live with
the solvers rather than inside either consumer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..ir.expr import callees_of
from ..function.concrete import ConcreteFunction
from .model import ExternalOracle

if TYPE_CHECKING:
  from .model import SolverDescriptor


def is_solver_function(fun: ConcreteFunction) -> bool:
  """Return whether ``fun`` is a solver's own concrete graph."""
  desc = fun.descriptor
  return isinstance(getattr(desc, "backend", None), str)


def solver_descriptor(fun: ConcreteFunction) -> SolverDescriptor:
  return fun.descriptor


def solver_callees(fun: ConcreteFunction) -> list[ConcreteFunction]:
  """Return the inner Functions a solver ConcreteFunction depends on at codegen time."""
  if not is_solver_function(fun):
    return []
  desc = solver_descriptor(fun)
  out: list[ConcreteFunction] = []
  stage = () if desc.stage_oracles is None else (desc.stage_oracles.jac, desc.stage_oracles.hess)
  for cand in (desc.oracle, desc.base, desc.grad, desc.jac, desc.hess, desc.bounds, *stage):
    if isinstance(cand, ConcreteFunction) and cand not in out:
      out.append(cand)
  return out


def solver_backends_used(fun: ConcreteFunction) -> tuple[str, ...]:
  """Sorted names of every solver backend reachable from ``fun`` (through
  callees and solver oracle Functions)."""
  found: set[str] = set()
  seen: set[int] = set()

  def visit(fn: ConcreteFunction) -> None:
    if id(fn) in seen:
      return
    seen.add(id(fn))
    if is_solver_function(fn):
      found.add(solver_descriptor(fn).backend)
      for callee in solver_callees(fn):
        visit(callee)
      return
    for node in fn.nodes:
      for callee in callees_of(node):
        visit(callee)

  visit(fun)
  return tuple(sorted(found))


def external_oracles(fun: ConcreteFunction) -> tuple[ExternalOracle, ...]:
  """External descriptor oracles in call order, deduplicated by identity."""
  if not is_solver_function(fun):
    return ()
  desc = solver_descriptor(fun)
  out: list[ExternalOracle] = []
  for oracle in (desc.base, desc.grad, desc.jac, desc.hess, desc.bounds):
    if isinstance(oracle, ExternalOracle) and oracle not in out:
      out.append(oracle)
  return tuple(out)
