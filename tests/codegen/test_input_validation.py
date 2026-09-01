from __future__ import annotations

import numpy as np

import alloy as al


def test_jit_reports_input_errors() -> None:
  x = al.sym("x", 2)
  y = al.sym("y", 2)
  out = ((x + 2.0) * y).sum()
  f = al.Function._from_exprs("f", [x, y], [out], ["x", "y"], ["out"])
  xv, yv = np.array([1.0, 3.0]), np.array([4.0, 5.0])

  np.testing.assert_allclose(f((xv, yv)), ((xv + 2.0) * yv).sum())

  try:
    _ = f((xv,))
  except ValueError as e:
    assert "does not have the declared structure of ('x', 'y')" in str(e)
  else:  # pragma: no cover
    raise AssertionError("a tree with the wrong leaf count should fail")

  try:
    f((xv.reshape(1, 2), yv))
  except ValueError as e:
    assert "expected shape (2,) for 'x', got (1, 2)" in str(e)
  else:  # pragma: no cover
    raise AssertionError("shape mismatch should fail")
