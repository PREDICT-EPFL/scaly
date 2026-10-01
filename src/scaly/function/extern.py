"""The extern-callee protocol: a Function whose body is C written by someone other than the compiler.

An ``EXTERN_CALL`` node reads one output of such a callee. Its ``extern`` attribute, which the
Function carries as ``Function.extern``, answers what lowering, code generation and the JIT ask:
which Functions its C calls (lowered as usual), which C sources it adds, the C that defines it, and
what compiling and loading the translation unit takes. Solvers implement it (``scaly.opt.external``);
nothing in the compiler knows what a solver is.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ..ir.expr import Expr, ExprOp, callees_of, topo
from ..ir.types import TensorType
from ..utils.names import c_ident
from .model import ConcreteFunction
from .tree import Tree, _G, flat_tree, param_list

if TYPE_CHECKING:
  import ctypes


class HasRawSymbol(Protocol):
  """Anything that names the C function it defines."""

  @property
  def raw_symbol(self) -> str: ...


@dataclass(frozen=True, slots=True)
class ExternSource:
  """C source placed in the translation unit ahead of every extern body.

  ``source`` defines ``raw_symbol`` with the generated kernels' flat-buffer convention: one
  ``const double*`` per input, one ``double*`` per output, and a trailing ``double*`` workspace of
  ``workspace_size`` doubles. Sources are emitted once per translation unit; two different sources
  for one ``raw_symbol`` are an error."""

  raw_symbol: str
  source: str
  workspace_size: int = 0


@dataclass(frozen=True, slots=True)
class ExternRenderCtx:
  """What an extern callee's ``render`` is handed: ``symbol`` prefixes every static it declares
  (several callees share one translation unit), and ``raw_symbol`` is the function it must define,
  ``static void <raw_symbol>(const double* in0, ..., double* out0, ..., double* w)``."""

  symbol: str
  raw_symbol: str

  @staticmethod
  def raw_symbol_of(fun: ConcreteFunction | HasRawSymbol) -> str:
    """The C symbol of a dependency's kernel, or of anything naming its own (an ``ExternSource``)."""
    return f"{c_ident(fun.name)}_raw" if isinstance(fun, ConcreteFunction) else fun.raw_symbol


@dataclass(frozen=True, slots=True)
class ExternState:
  """Per-callee state a compiled library exposes after a call, read through a C accessor.

  The library exports ``int <accessor>(<ctype>* out)``; ``decode`` turns the filled structure into
  what the compiled Function returns for it, and raises ``ValueError`` when it is not valid (never
  run, a layout mismatch)."""

  accessor: str
  ctype: type[ctypes.Structure]
  decode: Callable[[Any], Any]


LinkResolver = Callable[[Sequence[str]], Sequence[str]]
"""Compiler and linker flags for a set of library names; may raise when a library is missing."""


@dataclass(frozen=True, slots=True)
class BuildRequirements:
  """What a translation unit holding an extern callee needs beyond the generated C.

  ``includes`` are ``#include`` lines for the source. ``header_types`` and ``source_blocks`` are
  blocks of C: type definitions the header needs (it includes ``<stdint.h>`` ahead of them) and
  definitions the source needs after the ABI status codes; identical blocks from several callees
  are emitted once. ``declarations`` are further exported prototypes for the header.
  ``libraries`` are resolved to flags by ``link_flags``: callees sharing a resolver are resolved
  together, once, with the sorted union of their library names. ``isolated`` asks the JIT to load
  the library in its own linker namespace where the platform has one. ``versions`` are the
  ``(distribution, version)`` pairs of what the build links but the source does not show (a vendored
  library): they join the JIT cache key, so an upgrade never reuses a library built against another."""

  includes: tuple[str, ...] = ()
  header_types: tuple[tuple[str, ...], ...] = ()
  source_blocks: tuple[tuple[str, ...], ...] = ()
  declarations: tuple[str, ...] = ()
  libraries: tuple[str, ...] = ()
  link_flags: LinkResolver | None = None
  isolated: bool = False
  versions: tuple[tuple[str, str], ...] = ()


@runtime_checkable
class ExternCallee(Protocol):
  """The attribute of an ``EXTERN_CALL`` node and of its Function (``Function.extern``).

  ``dependencies``, ``extern_sources``, ``render`` and ``build_requirements`` are asked every time
  a Function holding the callee is built, also when its library is cached: the JIT's key for the
  library is a digest of their answers (``codegen/structure.py``). They must be cheap, change
  nothing, and answer the same for the same callee; ``render`` runs under the default options."""

  def dependencies(self) -> tuple[ConcreteFunction, ...]:
    """The Functions the C calls; they lower to ``<name>_raw`` kernels in the same translation unit."""
    ...

  def extern_sources(self) -> tuple[ExternSource, ...]:
    """C sources the body calls that the compiler did not generate."""
    ...

  def render(self, fun: ConcreteFunction, ctx: ExternRenderCtx) -> list[str]:
    """The C lines defining ``ctx.raw_symbol`` for ``fun``, plus any statics and exported helpers."""
    ...

  def build_requirements(self, fun: ConcreteFunction) -> BuildRequirements:
    """Includes, preamble, declarations and link flags for a translation unit holding ``fun``."""
    ...

  def state(self, fun: ConcreteFunction) -> ExternState | None:
    """The state the library exposes for ``fun`` after a call, or ``None``."""
    ...


def extern_function(
  name: str,
  callee: ExternCallee,
  inputs: Sequence[tuple[str, tuple[int, ...]]],
  outputs: Sequence[tuple[str, tuple[int, ...]]],
  *,
  input_tree: _G | None = None,
  output_tree: Tree[Any, Any] | None = None,
) -> ConcreteFunction[Any, Any, Any, Any]:
  """A Function named ``name`` whose body is ``callee``: one ``EXTERN_CALL`` node per output.

  ``inputs`` and ``outputs`` are ``(name, shape)`` pairs in call order. ``input_tree`` is its
  parameter list and ``output_tree`` its output tree; without them both are flat."""
  input_exprs = tuple(Expr.sym(leaf, shape, diff=False) for leaf, shape in inputs)
  output_exprs = tuple(
    Expr(ExprOp.EXTERN_CALL, input_exprs, TensorType(shape, diff=False), attrs={"extern": callee, "name": name, "output": i, "output_name": leaf})
    for i, (leaf, shape) in enumerate(outputs)
  )
  input_names, output_names = tuple(leaf for leaf, _ in inputs), tuple(leaf for leaf, _ in outputs)
  in_tree = input_tree or param_list(flat_tree(input_names, tuple(e.type for e in input_exprs)))
  out_tree = output_tree or flat_tree(output_names, tuple(e.type for e in output_exprs))
  function = ConcreteFunction.from_exprs(name, input_exprs, output_exprs, input_names, output_names)._with_trees(in_tree, out_tree)
  function.extern = callee
  return function


def extern_functions(fun: ConcreteFunction) -> tuple[ConcreteFunction, ...]:
  """Every Function with an extern body that ``fun`` reaches, itself included, in depth-first order.

  Two different extern Functions spelled the same in C are refused: each renders one ``_raw``."""
  found: dict[str, ConcreteFunction] = {}
  seen: set[int] = set()

  def visit(fn: ConcreteFunction) -> None:
    if id(fn) in seen:
      return
    seen.add(id(fn))
    if fn.extern is not None:
      symbol = c_ident(fn.name)
      if symbol in found and found[symbol] is not fn:
        raise ValueError(f"duplicate extern symbol {symbol!r} in one generated translation unit")
      found[symbol] = fn
      for dependency in fn.extern.dependencies():
        visit(dependency)
      return
    for node in topo(fn.outputs):
      for callee in callees_of(node):
        visit(callee)

  visit(fun)
  return tuple(found.values())
