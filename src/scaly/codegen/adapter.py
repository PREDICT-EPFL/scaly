"""Output adapters: named layers over a rendered module, such as the C++ header or CasADi's query
functions, registered by the modules that implement them and found by name through the
``scaly.adapters`` entry points.

An adapter never changes what a function computes. It may replace the header (the C++ one), add
definitions and prototypes to it, reorder the sparse tables it describes, check that a function
fits its layout, reserve workspace, wrap the entry (a gather of compact outputs), and append source
after the entry (query functions). ``codegen/aot.py`` and ``codegen/c.py`` apply the hooks and know
no adapter by name.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from ..function import ConcreteFunction
  from ..ir.types import SparsityType

ADAPTER_ENTRY_POINTS = "scaly.adapters"
"""The entry-point group naming the module that registers each adapter."""


@dataclass(frozen=True, slots=True)
class HeaderSpec:
  """What a header renders from. ``types`` are the extern callees' type definitions, ``defines``
  and ``declarations`` what the adapters and callees add (prototypes in C spelling), and
  ``sparsities`` the patterns the sparse tables describe, one per output."""

  fun: ConcreteFunction
  sz_w: int
  typed_buffers: bool
  types: tuple[str, ...]
  defines: tuple[str, ...]
  declarations: tuple[str, ...]
  sparsities: tuple[SparsityType | None, ...]


@dataclass(frozen=True, slots=True)
class EntryHook:
  """How an adapter wraps the entry: ``ptr`` redirects the body's writes to an output (by name) to
  another C pointer, ``setup`` runs after the null checks and ``epilogue`` before the return."""

  ptr: dict[str, str]
  setup: tuple[str, ...] = ()
  epilogue: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Adapter:
  """One registered adapter; every hook is optional. ``header`` renders the whole header in place
  of the C one, as ``<symbol>.<header_suffix>``, and a header that C cannot include sets
  ``source_includes_header`` to false. ``defines`` go after the ABI status codes in header and
  source. ``extra_workspace`` doubles are reserved past the packed workspace, in adapter order, and
  ``entry_prologue`` receives where its share starts. ``extra_source`` receives the entry's total
  workspace."""

  name: str
  header: Callable[[HeaderSpec], str] | None = None
  header_suffix: str = "h"
  source_includes_header: bool = True
  defines: tuple[str, ...] = ()
  declarations: Callable[[str], Sequence[str]] | None = None
  sparsities: Callable[[ConcreteFunction], tuple[SparsityType | None, ...]] | None = None
  check: Callable[[ConcreteFunction], None] | None = None
  extra_workspace: Callable[[ConcreteFunction], int] | None = None
  entry_prologue: Callable[[ConcreteFunction, int], EntryHook] | None = None
  extra_source: Callable[[ConcreteFunction, int], Sequence[str]] | None = None


_ADAPTERS: dict[str, Adapter] = {}


def register_adapter(
  name: str,
  *,
  header: Callable[[HeaderSpec], str] | None = None,
  header_suffix: str = "h",
  source_includes_header: bool = True,
  defines: Sequence[str] = (),
  declarations: Callable[[str], Sequence[str]] | None = None,
  sparsities: Callable[[ConcreteFunction], tuple[SparsityType | None, ...]] | None = None,
  check: Callable[[ConcreteFunction], None] | None = None,
  extra_workspace: Callable[[ConcreteFunction], int] | None = None,
  entry_prologue: Callable[[ConcreteFunction, int], EntryHook] | None = None,
  extra_source: Callable[[ConcreteFunction, int], Sequence[str]] | None = None,
) -> Adapter:
  """Register the adapter ``name`` with the hooks ``Adapter`` describes; registering a name twice raises."""
  if name in _ADAPTERS:
    raise ValueError(f"output adapter {name!r} is already registered")
  adapter = Adapter(
    name,
    header=header,
    header_suffix=header_suffix,
    source_includes_header=source_includes_header,
    defines=tuple(defines),
    declarations=declarations,
    sparsities=sparsities,
    check=check,
    extra_workspace=extra_workspace,
    entry_prologue=entry_prologue,
    extra_source=extra_source,
  )
  _ADAPTERS[name] = adapter
  return adapter


def available_adapters() -> tuple[str, ...]:
  """Every adapter name that is registered or installed."""
  return tuple(sorted({*_ADAPTERS, *(ep.name for ep in entry_points(group=ADAPTER_ENTRY_POINTS))}))


def get_adapter(name: str) -> Adapter:
  """The adapter ``name``, loading the module its entry point names the first time it is asked for."""
  if name not in _ADAPTERS:
    for ep in entry_points(group=ADAPTER_ENTRY_POINTS, name=name):
      ep.load()
  if name not in _ADAPTERS:
    raise ValueError(f"unknown output adapter {name!r}; available: {', '.join(available_adapters()) or 'none'}")
  return _ADAPTERS[name]


def resolve_adapters(names: Sequence[str]) -> tuple[Adapter, ...]:
  """The adapters ``names`` in order, once each. At most one may render the header and at most one
  may reorder its sparse tables."""
  if isinstance(names, str):
    raise TypeError(f"adapters is a sequence of names, got the string {names!r}; write ({names!r},)")
  adapters = tuple(get_adapter(name) for name in dict.fromkeys(names))
  for hook in ("header", "sparsities"):
    owners = [a.name for a in adapters if getattr(a, hook) is not None]
    if len(owners) > 1:
      raise ValueError(f"output adapters {', '.join(owners)} each set the {hook}; use one of them")
  return adapters


def entry_workspace(fun: ConcreteFunction, sz_w: int, adapters: Sequence[Adapter]) -> int:
  """The ``SZ_W`` an entry needs: the packed spill size plus what the adapters reserve."""
  return sz_w + sum(a.extra_workspace(fun) for a in adapters if a.extra_workspace is not None)


def entry_hooks(fun: ConcreteFunction, sz_w: int, adapters: Sequence[Adapter]) -> list[EntryHook]:
  """Each adapter's entry hook, handed where its reserved workspace starts past ``sz_w``."""
  hooks: list[EntryHook] = []
  offset = sz_w
  for a in adapters:
    if a.entry_prologue is not None:
      hooks.append(a.entry_prologue(fun, offset))
    if a.extra_workspace is not None:
      offset += a.extra_workspace(fun)
  return hooks
