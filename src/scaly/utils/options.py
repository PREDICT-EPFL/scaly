"""User-settable conventions that shape the graphs Scaly builds: ``sc.options``, ``sc.set_options``.

An option is read when a graph is built, never when it is rendered. A derivative built under one
setting therefore carries its choice in its own structure, which reaches the generated C and the JIT
cache key with no global state involved at render time.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, fields, replace
from typing import Any, Literal

type Nonsmooth = Literal["split", "first", "error"]


@dataclass(frozen=True, slots=True)
class Options:
  """The conventions in force.

  Attributes:
    nonsmooth: the derivative of ``maximum``, ``minimum``, ``reduce_max`` and ``reduce_min`` where
      arguments tie. ``"split"`` shares it equally among the tied arguments, ``"first"`` gives it
      all to the first (the left operand, or the lowest index), and ``"error"`` refuses to
      differentiate these operations at all. Away from ties every convention gives the same value.
  """

  nonsmooth: Nonsmooth = "split"


_CHOICES: dict[str, tuple[Any, ...]] = {"nonsmooth": ("split", "first", "error")}
_default = Options()
_current: ContextVar[Options | None] = ContextVar("scaly_options", default=None)


def _updated(base: Options, changes: dict[str, Any]) -> Options:
  known = {f.name for f in fields(Options)}
  for name, value in changes.items():
    if name not in known:
      raise TypeError(f"unknown scaly option {name!r}; known options: {sorted(known)}")
    if value not in _CHOICES[name]:
      raise ValueError(f"option {name}={value!r} is not one of {_CHOICES[name]}")
  return replace(base, **changes)


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
  """Apply ``changes`` inside a ``with`` block, for this thread or task only, and restore on exit."""
  token = _current.set(_updated(get_options(), changes))
  try:
    yield get_options()
  finally:
    _current.reset(token)


__all__ = ["Options", "get_options", "options", "set_options"]
