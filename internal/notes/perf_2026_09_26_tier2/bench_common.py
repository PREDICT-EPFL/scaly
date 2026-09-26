"""Shared timing helpers for the Tier 2 studies: median wall time of a compiled call."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np


def median_us(fn: Callable[[], object], *, repeat: int = 41, inner: int | None = None) -> float:
  """Median microseconds per call, with ``inner`` calls per sample chosen to fill about 2 ms."""
  fn()
  if inner is None:
    t0 = time.perf_counter()
    fn()
    one = max(time.perf_counter() - t0, 1e-7)
    inner = max(1, int(2e-3 / one))
  samples = []
  for _ in range(repeat):
    t0 = time.perf_counter()
    for _ in range(inner):
      fn()
    samples.append((time.perf_counter() - t0) / inner)
  return float(np.median(samples) * 1e6)


def timed(fn: Callable[[], object]) -> tuple[object, float]:
  t0 = time.perf_counter()
  out = fn()
  return out, (time.perf_counter() - t0) * 1e3
