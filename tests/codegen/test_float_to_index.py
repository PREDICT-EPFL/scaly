from __future__ import annotations

import numpy as np

import scaly as sc


def test_clamp_before_cast_keeps_a_non_finite_input_in_range() -> None:
  """A float index is clamped by ``maximum`` and ``minimum`` (C ``fmax`` and ``fmin``) before its
  cast to ``int64``: NaN takes the bound ``fmax`` returns and an infinity the end it saturates at,
  so the cast, undefined in C for either, only ever sees a finite value, and ``take(in_range=True)``
  reads inside its table."""
  table = np.array([10.0, 11.0, 12.0, 13.0])
  x = sc.sym("x")
  clamped = sc.minimum(sc.maximum(x.floor(), 0.0), 3.0)
  cell = sc.cast(clamped, "int64")
  fn = sc.Function._from_exprs(
    "clamp_cast", [x], [clamped, sc.take(sc.const(table), sc.stack([cell]), in_range=True)[0]], ["x"], ["clamped", "value"]
  )
  # The clamped float is checked as well as the read: a NaN reaching the cast is undefined, and
  # AArch64 happens to convert it to 0, which would hide it.
  for xv, expected in ((np.nan, 0), (np.inf, 3), (-np.inf, 0), (1e300, 3), (-1e300, 0), (2.5, 2), (3.0, 3), (-0.0, 0)):
    clamped_v, value = fn(np.array(xv))
    assert (clamped_v, value) == (expected, table[expected]), xv
