"""Frozen C library build descriptions and concrete external function declarations."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cached_property
from pathlib import Path
from typing import Any, ClassVar, Mapping, Sequence

from ..ir.expr import Expr, ExprOp
from ..utils.names import NameScope, c_ident
from .concrete import ConcreteFunction
from .tree import Tree, parameter_list


def _files(value: Mapping[str, str] | Sequence[tuple[str, str]]) -> tuple[tuple[str, str], ...]:
  items = tuple(sorted(dict(value).items()))
  for name, source in items:
    if not name or Path(name).is_absolute() or ".." in Path(name).parts or "\\" in name or not isinstance(source, str):
      raise ValueError("library files need relative names without '..' and C source text")
  return items


@dataclass(frozen=True, slots=True)
class CLibrary:
  """Describe C sources compiled separately from a generated Scaly translation unit.

  ``sources`` and ``headers`` map relative file names to their text. The description copies them
  into immutable tuples. ``include_dirs`` names additional header directories. ``defines`` maps
  macro names to values, or ``None`` for a macro without a value. ``compile_flags`` uses clang's
  dialect and applies only to these sources. ``link`` contains library names such as ``"m"``,
  resolved by the selected toolchain. ``versions`` records user-supplied dependency versions.
  Source, headers, flags and resolved library content all contribute to the JIT cache key.
  """

  name: str
  sources: Mapping[str, str] | tuple[tuple[str, str], ...] = ()
  headers: Mapping[str, str] | tuple[tuple[str, str], ...] = ()
  include_dirs: tuple[str, ...] = ()
  defines: Mapping[str, str | int | None] | tuple[tuple[str, str | int | None], ...] = ()
  compile_flags: tuple[str, ...] = ()
  link: tuple[str, ...] = ()
  versions: Mapping[str, str] | tuple[tuple[str, str], ...] = ()

  def __post_init__(self) -> None:
    if not self.name:
      raise ValueError("CLibrary needs a non-empty name")
    object.__setattr__(self, "sources", _files(self.sources))
    object.__setattr__(self, "headers", _files(self.headers))
    if set(dict(self.sources)) & set(dict(self.headers)):
      raise ValueError("library source and header file names must be distinct")
    object.__setattr__(self, "include_dirs", tuple(str(Path(directory).resolve()) for directory in self.include_dirs))
    object.__setattr__(self, "defines", tuple(sorted((name, None if value is None else str(value)) for name, value in dict(self.defines).items())))
    object.__setattr__(self, "compile_flags", tuple(self.compile_flags))
    object.__setattr__(self, "link", tuple(self.link))
    object.__setattr__(self, "versions", tuple(sorted(dict(self.versions).items())))
    for name in self.link:
      if not name or name.startswith("-") or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_+-." for c in name):
        raise ValueError(f"link needs library names, not linker flags or paths: {name!r}")


@dataclass(frozen=True, eq=False)
class _Extern:
  library: CLibrary
  name: str
  input_tree: Tree[Any, Any]
  output_tree: Tree[Any, Any]
  workspace: int
  nodes: ClassVar[tuple[Expr, ...]] = ()
  descriptor: ClassVar[None] = None
  rules: ClassVar[None] = None

  @cached_property
  def inputs(self) -> tuple[Expr, ...]:
    return self.input_tree.flatten_symbolic(self.input_tree.symbols(), self.name)

  @cached_property
  def outputs(self) -> tuple[Expr, ...]:
    return self.output_tree.flatten_symbolic(self.output_tree.symbols(), self.name)

  @property
  def results(self) -> tuple[Expr, ...]:
    return self.outputs

  @property
  def input_names(self) -> tuple[str, ...]:
    return self.input_tree.names

  @property
  def output_names(self) -> tuple[str, ...]:
    return self.output_tree.names

  @property
  def signature(self) -> tuple:
    return self.input_tree.types, self.input_tree.layouts, self.output_tree.types, self.output_tree.layouts, self.workspace

  def _call(self, actuals: Sequence[Expr]) -> tuple[Expr, ...]:
    return tuple(
      Expr(
        ExprOp.EXTERN,
        tuple(actuals),
        replace(out.type, diff=out.type.diff and any(arg.type.diff for arg in actuals)),
        attrs={"callee": self, "output": i},
      )
      for i, out in enumerate(self.outputs)
    )

  @cached_property
  def _function(self) -> ConcreteFunction:
    return ConcreteFunction.build(f"{self.name}_extern", self.input_tree, self.inputs, self.output_tree, self._call(self.inputs))

  def symbolic_call(self, *args: Any) -> Any:
    return self.output_tree.unflatten(self._call(self.input_tree.flatten_symbolic(args, self.name)))

  def __call__(self, *args: Any) -> Any:
    return self.symbolic_call(*args) if self.input_tree.is_symbolic(args) else self._function(*args)


def extern(library: CLibrary, symbol: str, *, inputs: tuple[Tree[Any, Any], ...], outputs: Tree[Any, Any], workspace: int = 0) -> _Extern:
  """Declare a concrete external C function and return its callable.

  ``inputs`` gives one ``sc.arg`` or ``sc.group`` tree per positional parameter; ``outputs`` is
  the output tree. Every leaf must have a concrete shape, dtype and stored pattern. The C symbol
  returns ``void`` and takes one ``const T*`` per input leaf, one ``T*`` per output leaf, then
  ``double* w`` with ``workspace`` doubles of scratch. Sparse buffers use their declared layout.
  Leaf dtypes must be supported by Scaly's generated calling convention.

  Calls must be deterministic and reentrant: read-only inputs, fully written outputs, no retained
  pointers, hidden state or aliased buffers. Report failure as an ordinary output. A symbolic call
  builds one EXTERN node per output, sharing one invocation. A numerical call compiles a wrapper.
  Active differentiation raises unless a wrapping Function has ``sc.custom_derivative`` rules.
  """
  if not isinstance(library, CLibrary):
    raise TypeError("extern needs a CLibrary")
  if c_ident(symbol) != symbol or NameScope().reserved(symbol):
    raise ValueError(f"extern needs a non-reserved C symbol, got {symbol!r}")
  if type(workspace) is not int or workspace < 0:
    raise ValueError("extern workspace must be a nonnegative number of doubles")
  parameters = parameter_list(tuple(inputs))
  if parameters.has_holes or outputs.has_holes:
    raise ValueError("extern signatures must be concrete, without shape holes")
  if not outputs.size:
    raise ValueError("extern needs at least one output")
  return _Extern(library, symbol, parameters, outputs, workspace)
