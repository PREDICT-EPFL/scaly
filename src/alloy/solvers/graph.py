"""Solver queries over a ``Function`` graph.

What a caller needs to know before rendering or compiling: whether a Function is a solver, which
inner Functions its descriptor drives, which backends it transitively reaches, and the native
flags that reaching them costs. Lowering and codegen both ask these questions, so they live with
the solvers rather than inside either consumer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..ir.expr import ExprOp, topo
from ..function import Function
from .paths import backend_compile_flags
from .solver_function import ExternalOracle

if TYPE_CHECKING:
  from .solver_function import SolverDescriptor


def is_solver_function(fun: Function) -> bool:
  desc = getattr(fun, "descriptor", None)
  return isinstance(getattr(desc, "backend", None), str)


def solver_descriptor(fun: Function) -> SolverDescriptor:
  return fun.descriptor  # ty: ignore[unresolved-attribute]


def solver_callees(fun: Function) -> list[Function]:
  """Return the inner Functions a solver Function depends on at codegen time."""
  if not is_solver_function(fun):
    return []
  desc = solver_descriptor(fun)
  out: list[Function] = []
  for cand in (desc.oracle, desc.base, desc.grad, desc.jac, desc.hess, desc.bounds):
    if isinstance(cand, Function) and cand not in out:
      out.append(cand)
  return out


def solver_backends_used(fun: Function) -> tuple[str, ...]:
  """Sorted names of every solver backend reachable from ``fun`` (through
  CALL/VMAP callees, SOLVER_CALL nodes, and solver oracle Functions)."""
  found: set[str] = set()
  seen: set[int] = set()

  def visit(fn: Function) -> None:
    if id(fn) in seen:
      return
    seen.add(id(fn))
    if is_solver_function(fn):
      found.add(solver_descriptor(fn).backend)
      for callee in solver_callees(fn):
        visit(callee)
      return
    for node in topo(fn.outputs):
      if node.op in {ExprOp.CALL, ExprOp.VMAP}:
        visit(node.attrs["callee"])
      elif node.op == ExprOp.SOLVER_CALL:
        found.add(node.attrs["solver"].backend)

  visit(fun)
  return tuple(sorted(found))


def external_oracles(fun: Function) -> tuple[ExternalOracle, ...]:
  """External descriptor oracles in call order, deduplicated by identity."""
  if not is_solver_function(fun):
    return ()
  desc = solver_descriptor(fun)
  out: list[ExternalOracle] = []
  for oracle in (desc.base, desc.grad, desc.jac, desc.hess, desc.bounds):
    if isinstance(oracle, ExternalOracle) and oracle not in out:
      out.append(oracle)
  return tuple(out)


def solver_compile_flags(fun: Function, *, rpath: bool = True) -> list[str]:
  """Compiler/linker flags an AOT consumer needs for ``fun``.

  Returns ``[]`` when ``fun`` does not transitively reach any solver.
  Otherwise: plugin package ``-I`` / ``-L`` paths, each reached backend's
  ``link_flags``, and an ``-Wl,-rpath`` pointing at the vendored lib
  directories so the resulting binary finds the shared libs at load time
  without ``LD_LIBRARY_PATH`` / ``DYLD_LIBRARY_PATH`` overrides.

  Set ``rpath=False`` if the consumer plans to bundle the libs elsewhere and
  will set the rpath / install_name themselves.
  """
  names = solver_backends_used(fun)
  if not names:
    return []
  return backend_compile_flags(names, rpath=rpath)
