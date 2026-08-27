from __future__ import annotations

import numpy as np

import alloy as al


def test_jit_reports_input_errors() -> None:
  x = al.sym("x", 2)
  y = al.sym("y", 2)
  out = ((x + 2.0) * y).sum()
  f = al.Function._from_exprs("f", [x, y], [out], ["x", "y"], ["out"])
  env = {"x": np.array([1.0, 3.0]), "y": np.array([4.0, 5.0])}

  np.testing.assert_allclose(f(**env), ((env["x"] + 2.0) * env["y"]).sum())

  try:
    _ = f(x=env["x"])
  except TypeError as e:
    assert "missing keyword inputs: ['y']" in str(e)
  else:  # pragma: no cover
    raise AssertionError("missing keyword input should fail")

  try:
    _ = f(x=env["x"], y=env["y"], z=env["x"])
  except TypeError as e:
    assert "unexpected keyword inputs: ['z']" in str(e)
  else:  # pragma: no cover
    raise AssertionError("extra keyword input should fail")

  try:
    f(x=env["x"].reshape(1, 2), y=env["y"])
  except ValueError as e:
    assert "input 'x' has shape (1, 2), expected (2,)" in str(e)
  else:  # pragma: no cover
    raise AssertionError("shape mismatch should fail")
