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


def test_a_constant_nan_folds_through_the_clamp_as_the_c_does() -> None:
  """Constant folding evaluates ``maximum`` and ``minimum`` as C's ``fmax`` and ``fmin``: a NaN
  operand gives the other one. A point that folds to NaN before a clamped cast (a spline read at a
  constant NaN) therefore leaves a finite constant for the cast, not a cast of NaN in the C."""
  from scaly import interp
  from scaly.codegen import render_c_module
  from scaly.passes.expr import simplify

  folded = simplify(sc.minimum(sc.maximum(sc.const(np.nan), 0.0), 3.0))
  assert folded.op == sc.ExprOp.CONST and float(np.asarray(folded.value)) == 0.0  # fmax(NaN, 0), then fmin(0, 3)
  for op, other in ((sc.minimum, 3.0), (sc.maximum, 0.0)):
    alone = simplify(op(sc.const(np.nan), other))
    assert alone.op == sc.ExprOp.CONST and float(np.asarray(alone.value)) == other

  f = interp.interpolant(np.linspace(0.0, 1.0, 50), np.linspace(0.0, 1.0, 50) ** 2, kind="cubic")
  x = sc.sym("x")
  fn = sc.Function._from_exprs("const_nan_read", [x], [f(sc.const(np.nan)) + 0.0 * x], ["x"], ["y"])
  assert "(int64_t)((double)NAN)" not in render_c_module(fn).body
  assert np.isnan(fn(np.array(0.5)))
