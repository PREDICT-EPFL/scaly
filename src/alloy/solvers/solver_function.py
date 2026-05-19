"""Common opaque-Function wrapper for QP and NLP solvers.

A :class:`SolverFunction` mimics the public surface of :class:`alloy.Function`
(``input_names``/``output_names``/``__call__``) without being constructed from
an expression graph. The roadmap calls these "opaque ``Function``s": calling
them runs the bound backend, and their ``.tape()`` is a single opaque call.

Inlining a solver inside a larger expression graph is not yet supported; that
will need a dedicated IR op once the safety-filter workload starts composing
filters into trajectories.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True, slots=True)
class SolverStatus:
  code: int
  name: str
  iter: int = 0

  @property
  def ok(self) -> bool:
    # PIQP solved == 1, IPOPT solved == 0 / 1 / 6 (Solve_Succeeded /
    # Solved_To_Acceptable_Level / Feasible_Point_Found). Specific codes
    # belong on the backend; this is convenience.
    return self.code in (0, 1, 6)


class SolverFunction:
  """Opaque callable returned by :func:`alloy.qp` and :func:`alloy.nlp`.

  ``inputs`` documents the call-time signature ``(name, shape)``.
  ``outputs`` documents the result keys.
  ``backend`` is the Python callable that performs the actual solve. It
  receives a dict of named inputs and returns a dict of named outputs plus a
  ``status`` entry of type :class:`SolverStatus`.
  """

  __slots__ = ("name", "input_names", "input_shapes", "output_names", "output_shapes", "_backend", "last_status", "meta")

  def __init__(
    self,
    name: str,
    inputs: Sequence[tuple[str, tuple[int, ...]]],
    outputs: Sequence[tuple[str, tuple[int, ...]]],
    backend: Callable[[dict[str, np.ndarray]], tuple[dict[str, np.ndarray], SolverStatus]],
    meta: Mapping[str, Any] | None = None,
  ) -> None:
    self.name = name
    self.input_names = tuple(n for n, _ in inputs)
    self.input_shapes = tuple(s for _, s in inputs)
    self.output_names = tuple(n for n, _ in outputs)
    self.output_shapes = tuple(s for _, s in outputs)
    self._backend = backend
    self.last_status: SolverStatus | None = None
    self.meta = dict(meta or {})

  def __repr__(self) -> str:
    return f"SolverFunction({self.name!r}, {self.input_names}->{self.output_names})"

  def _resolve_inputs(self, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> dict[str, np.ndarray]:
    if args and kwargs:
      raise TypeError("pass positional inputs or keyword inputs, not both")
    if kwargs:
      missing = [n for n in self.input_names if n not in kwargs]
      extra = [n for n in kwargs if n not in self.input_names]
      if missing or extra:
        parts: list[str] = []
        if missing:
          parts.append(f"missing keyword inputs: {missing}")
        if extra:
          parts.append(f"unexpected keyword inputs: {extra}")
        raise TypeError(", ".join(parts))
      ordered = [kwargs[n] for n in self.input_names]
    elif len(args) != len(self.input_names):
      raise TypeError(f"expected {len(self.input_names)} inputs, got {len(args)}")
    else:
      ordered = list(args)
    resolved: dict[str, np.ndarray] = {}
    for name, shape, val in zip(self.input_names, self.input_shapes, ordered, strict=True):
      arr = np.asarray(val, dtype=np.float64)
      if shape:
        if arr.shape != shape:
          # accept 1D when shape is (k,) and arr is 0-d scalar etc.
          try:
            arr = arr.reshape(shape)
          except ValueError as exc:
            raise ValueError(f"input {name!r}: cannot reshape {arr.shape} to {shape}") from exc
      resolved[name] = np.ascontiguousarray(arr)
    return resolved

  def __call__(self, *args: Any, **kwargs: Any) -> dict[str, np.ndarray]:
    inputs = self._resolve_inputs(args, kwargs)
    outputs, status = self._backend(inputs)
    self.last_status = status
    return outputs
