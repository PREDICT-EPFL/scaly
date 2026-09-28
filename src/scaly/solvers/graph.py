"""Solver queries over a ``Function`` graph.

Whether a Function is a solver, which inner Functions its descriptor drives, which backends a graph
transitively reaches, and the native flags that reaching them costs. The compiler asks none of
these: it reaches solvers only through the extern-callee protocol (``scaly.function.extern``).
They are for tools that consume generated code, such as a build script that links it.
"""

from __future__ import annotations

from ..function import ConcreteFunction, Function
from ..function.extern import extern_functions
from .model import ExternalOracle, SolverDescriptor
from .paths import backend_compile_flags


def is_solver_function(fun: Function) -> bool:
  return isinstance(fun.extern, SolverDescriptor)


def solver_descriptor(fun: Function) -> SolverDescriptor:
  """The descriptor of a solver Function; ``TypeError`` for any other."""
  if not isinstance(fun.extern, SolverDescriptor):
    raise TypeError(f"{fun.name!r} is not a solver Function")
  return fun.extern


def solver_callees(fun: ConcreteFunction) -> list[ConcreteFunction]:
  """Return the inner Functions a solver Function depends on at codegen time."""
  return list(fun.extern.dependencies()) if isinstance(fun.extern, SolverDescriptor) else []


def solver_backends_used(fun: Function) -> tuple[str, ...]:
  """Sorted names of every solver backend reachable from ``fun``, through calls, maps and the
  Functions a solver drives."""
  return tuple(sorted({fn.extern.backend for fn in extern_functions(fun.concrete) if isinstance(fn.extern, SolverDescriptor)}))


def external_oracles(fun: ConcreteFunction) -> tuple[ExternalOracle, ...]:
  """External descriptor oracles in call order, deduplicated by identity."""
  return fun.extern.external_oracles() if isinstance(fun.extern, SolverDescriptor) else ()


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
