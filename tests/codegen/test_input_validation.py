from __future__ import annotations

import numpy as np

import scaly as sc


def test_jit_reports_input_errors() -> None:
  @sc.function(sc.group(sc.arg("x", 2), sc.arg("y", 2)), outputs=sc.arg("out"))
  def f(inputs: tuple[sc.Expr, sc.Expr]) -> sc.Expr:
    x, y = inputs
    return ((x + 2.0) * y).sum()

  xv, yv = np.array([1.0, 3.0]), np.array([4.0, 5.0])

  np.testing.assert_allclose(f((xv, yv)), ((xv + 2.0) * yv).sum())

  try:
    _ = f((xv,))  # ty: ignore[no-matching-overload]
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
