"""User-settable conventions that shape the graphs Scaly builds: ``sc.options``, ``sc.set_options``.

An option is read when a graph is built, never when it is rendered. A derivative built under one
setting therefore carries its choice in its own structure, which reaches the generated C and the JIT
cache key with no global state involved at render time.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, fields, replace
from typing import Any, Literal

type Nonsmooth = Literal["split", "first", "error"]


@dataclass(frozen=True, slots=True)
class OptionNamespace:
  """Options a package declares under one name, ``sc.options(<name>=dict(...))``: ``defaults`` is a
  frozen dataclass of them, ``checks`` validates a field's value (a tuple of allowed values or a
  predicate). ``affects_derivatives`` says whether a derivative built under one value can differ
  from one built under another; only such namespaces reach derivative helper names and caches."""

  name: str
  defaults: Any
  affects_derivatives: bool
  checks: tuple[tuple[str, Any], ...] = ()


_NAMESPACES: dict[str, OptionNamespace] = {}


def _declared(name: str) -> OptionNamespace | None:
  """The namespace ``name``, importing ``scaly.<name>`` first when it is not declared yet: a
  package declares its namespace when imported, and ``sc.options(<name>=...)`` may come first."""
  if name not in _NAMESPACES and name.isidentifier():
    try:
      importlib.import_module(f"scaly.{name}")
    except ModuleNotFoundError as error:
      if error.name != f"scaly.{name}":
        raise  # the package is there and fails to import: that is not an unknown option
  return _NAMESPACES.get(name)


def register_option_namespace(name: str, defaults: Any, *, affects_derivatives: bool, checks: dict[str, Any] | None = None) -> OptionNamespace:
  """Declare the option namespace ``name`` with the frozen dataclass ``defaults``; declaring a name twice raises."""
  if name in _NAMESPACES or name in {f.name for f in fields(Options)}:
    raise ValueError(f"option namespace {name!r} is already declared")
  namespace = OptionNamespace(name, defaults, affects_derivatives, tuple((checks or {}).items()))
  _NAMESPACES[name] = namespace
  return namespace


@dataclass(frozen=True, slots=True)
class Options:
  """The conventions in force: the compiler's own, and the namespaces packages declare.

  Attributes:
    nonsmooth: the derivative of ``maximum``, ``minimum``, ``reduce_max`` and ``reduce_min`` where
      arguments tie. ``"split"`` shares it equally among the tied arguments, ``"first"`` gives it
      all to the first (the left operand, or the lowest index), and ``"error"`` refuses to
      differentiate these operations at all. Away from ties every convention gives the same value.
      ``floor`` and ``ceil`` have a zero derivative, exact but at their jumps, under ``"split"``
      and ``"first"``; ``"error"`` refuses them too.
    max_trajectory: the most values reverse mode may store for the carries of one loop. A loop over
      a large carry (a factorization's) differentiated through its steps would exceed any
      reasonable memory; building such a derivative raises instead and names the loop, since an
      implicit rule (``sc.custom_derivative``) is the fix.
    changed: the namespaces set to other than their defaults, by name, as ``(name, values)`` pairs;
      read one with ``namespace``.
  """

  nonsmooth: Nonsmooth = "split"
  max_trajectory: int = 50_000_000
  changed: tuple[tuple[str, Any], ...] = ()

  def namespace(self, name: str) -> Any:
    """The values of the option namespace ``name`` in force."""
    for key, values in self.changed:
      if key == name:
        return values
    namespace = _declared(name)
    if namespace is None:
      raise KeyError(f"unknown option namespace {name!r}; declared: {sorted(_NAMESPACES)}")
    return namespace.defaults

  def derivative_key(self) -> tuple[Any, ...]:
    """What a derivative built under these options depends on: the compiler's own and the
    namespaces that declare they affect derivatives."""
    return (self.nonsmooth, self.max_trajectory, tuple((k, v) for k, v in self.changed if _NAMESPACES[k].affects_derivatives))


def _count(value: Any) -> bool:
  return isinstance(value, int) and not isinstance(value, bool) and value >= 0


_CHOICES: dict[str, Any] = {"nonsmooth": ("split", "first", "error"), "max_trajectory": _count}
_default = Options()
_current: ContextVar[Options | None] = ContextVar("scaly_options", default=None)


def _check(label: str, value: Any, check: Any) -> None:
  if callable(check) and not check(value):
    raise ValueError(f"option {label}={value!r} must be a non-negative integer")
  if not callable(check) and value not in check:
    raise ValueError(f"option {label}={value!r} is not one of {check}")


def _updated(base: Options, changes: dict[str, Any]) -> Options:
  core = {f.name for f in fields(Options)} - {"changed"}
  plain: dict[str, Any] = {}
  changed = dict(base.changed)
  for name, value in changes.items():
    if name in core:
      _check(name, value, _CHOICES[name])
      plain[name] = value
    elif (namespace := _declared(name)) is not None:
      if not isinstance(value, dict):
        raise TypeError(f"option namespace {name!r} takes a dict of its options, got {value!r}")
      known = {f.name for f in fields(namespace.defaults)}
      checks = dict(namespace.checks)
      for key, item in value.items():
        if key not in known:
          raise TypeError(f"unknown option {name}.{key!r}; known: {sorted(known)}")
        if key in checks:
          _check(f"{name}.{key}", item, checks[key])
      values = replace(base.namespace(name), **value)
      if values == namespace.defaults:
        changed.pop(name, None)
      else:
        changed[name] = values
    else:
      raise TypeError(f"unknown scaly option {name!r}; known options: {sorted(core)}, namespaces: {sorted(_NAMESPACES)}")
  return replace(base, **plain, changed=tuple(sorted(changed.items())))


def get_options() -> Options:
  """The options in force here: the innermost ``sc.options`` block, else the process default."""
  current = _current.get()
  return _default if current is None else current


def set_options(**changes: Any) -> None:
  """Change the process default, which every thread sees outside an ``sc.options`` block."""
  global _default
  _default = _updated(_default, changes)


@contextmanager
def options(**changes: Any) -> Iterator[Options]:
  """Apply ``changes`` inside a ``with`` block, for this thread or task only, and restore on exit.
  A namespace's options go in a dict under its name: ``sc.options(linalg=dict(dense_unroll=4))``."""
  token = _current.set(_updated(get_options(), changes))
  try:
    yield get_options()
  finally:
    _current.reset(token)


__all__ = ["OptionNamespace", "Options", "get_options", "options", "register_option_namespace", "set_options"]
