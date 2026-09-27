"""Exact discretization of linear time-invariant systems, and linearization of any model or map at a point."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.linalg import expm

from ..function.api import jacobian
from ..function.model import Function

__all__ = ["foh", "linearize", "zoh"]


def _matrices(a: Any, b: Any, dt: float) -> tuple[np.ndarray, np.ndarray]:
  a, b = np.atleast_2d(np.asarray(a, dtype=np.float64)), np.asarray(b, dtype=np.float64)
  b = b.reshape(a.shape[0], -1)
  if a.shape[0] != a.shape[1] or b.shape[0] != a.shape[0]:
    raise ValueError(f"A must be square and B have as many rows, got {a.shape} and {b.shape}")
  if not float(dt) > 0:
    raise ValueError(f"dt must be positive, got {dt}")
  return a, b


def zoh(a: Any, b: Any, dt: float) -> tuple[np.ndarray, np.ndarray]:
  """``(Ad, Bd)`` with ``x_next = Ad x + Bd u`` exact for ``x' = A x + B u`` and ``u`` held over the
  interval: ``Ad = e^{A dt}``, ``Bd = int_0^dt e^{A s} ds B``, both read off one matrix exponential."""
  a, b = _matrices(a, b, dt)
  n, m = b.shape
  big = np.zeros((n + m, n + m))
  big[:n, :n], big[:n, n:] = a, b
  e = expm(big * dt)
  return e[:n, :n], e[:n, n:]


def foh(a: Any, b: Any, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """``(Ad, B0, B1)`` with ``x_next = Ad x + B0 u + B1 u_next`` exact for ``x' = A x + B u`` and ``u``
  interpolated linearly from ``u`` to ``u_next`` over the interval (a first-order hold), from one
  matrix exponential of ``[[A, B, 0], [0, 0, I/dt], [0, 0, 0]]``."""
  a, b = _matrices(a, b, dt)
  n, m = b.shape
  big = np.zeros((n + 2 * m, n + 2 * m))
  big[:n, :n], big[:n, n : n + m], big[n : n + m, n + m :] = a, b, np.eye(m) / dt
  e = expm(big * dt)
  ramp = e[:n, n + m :]
  return e[:n, :n], e[:n, n : n + m] - ramp, ramp


def linearize(f: Function[Any, Any, Any, Any], *point: Any) -> tuple[np.ndarray, ...]:
  """The Jacobians of ``f``'s one output with respect to each of its input leaves, at ``point`` (one
  argument per parameter, as ``f`` is called): ``(A, B)`` for a model ``f(x, u)`` or a map
  ``F(x, u)``, one more matrix per further leaf. Each is evaluated as generated code; a model's pair
  goes to ``zoh`` for an exact discretization of the linearization."""
  if not isinstance(f, Function):
    raise TypeError(f"linearize needs an sc.Function, got {type(f).__name__}")
  instance = f.concrete if f.is_concrete else f.instantiate(*point)
  if len(point) != len(instance.input_tree.parts):
    raise ValueError(f"{f.name} takes {len(instance.input_tree.parts)} arguments, got {len(point)}")
  return tuple(np.atleast_2d(np.asarray(jacobian(instance, name)(*point))) for name in instance.input_names)
