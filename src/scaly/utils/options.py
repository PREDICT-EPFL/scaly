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
      ``floor`` and ``ceil`` have a zero derivative, exact but at their jumps, under ``"split"``
      and ``"first"``; ``"error"`` refuses them too.
    dense_unroll: the largest order at which ``cholesky``, ``ldl`` and ``solve_triangular`` become
      straight-line code instead of loops.
    sparse_unroll: the most multiply-adds and divisions a sparse ``L D L^T`` may take and still be
      generated as straight-line code instead of loops over its columns
      (``SparseLDL(schedule="auto")``). Straight-line code runs several times faster but costs
      about a millisecond of generation per operation.
    max_trajectory: the most values reverse mode may store for the carries of one loop. A loop over
      a large carry (a factorization's) differentiated through its steps would exceed any
      reasonable memory; building such a derivative raises instead and names the loop, since an
      implicit rule (``sc.custom_derivative``) is the fix.
  """

  nonsmooth: Nonsmooth = "split"
  dense_unroll: int = 8
  sparse_unroll: int = 1000
  max_trajectory: int = 50_000_000


def _count(value: Any) -> bool:
  return isinstance(value, int) and not isinstance(value, bool) and value >= 0


_CHOICES: dict[str, Any] = {"nonsmooth": ("split", "first", "error"), "dense_unroll": _count, "sparse_unroll": _count, "max_trajectory": _count}
_default = Options()
_current: ContextVar[Options | None] = ContextVar("scaly_options", default=None)


def _updated(base: Options, changes: dict[str, Any]) -> Options:
  known = {f.name for f in fields(Options)}
  for name, value in changes.items():
    if name not in known:
      raise TypeError(f"unknown scaly option {name!r}; known options: {sorted(known)}")
    check = _CHOICES[name]
    if callable(check) and not check(value):
      raise ValueError(f"option {name}={value!r} must be a non-negative integer")
    if not callable(check) and value not in check:
      raise ValueError(f"option {name}={value!r} is not one of {check}")
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
