"""Phase 10 first slice: ``Ops.SUM_AXIS`` for axis-aware reductions."""

from __future__ import annotations

import os
import subprocess

import numpy as np
import pytest

import alloy as al
from alloy.codegen.program_c import can_render_program_c
from alloy.ops import Ops
from alloy.spec import verify_expr


def _has_compiler() -> bool:
  cc = os.environ.get("ALLOY_CC", "cc")
  try:
    subprocess.run([cc, "--version"], check=True, capture_output=True)
    return True
  except (FileNotFoundError, subprocess.CalledProcessError):
    return False


def test_sum_axis_construct_and_shape() -> None:
  x = al.sym("x", (3, 4))
  y = x.sum(axis=0)
  assert y.op == Ops.SUM_AXIS
  assert y.shape == (4,)
  assert y.attrs["axes"] == (0,)


def test_sum_axis_supports_tuple_and_negative_axes() -> None:
  x = al.sym("x", (2, 3, 4))
  assert x.sum(axis=(0, 2)).shape == (3,)
  assert x.sum(axis=-1).shape == (2, 3)


def test_sum_axis_verifies() -> None:
  x = al.sym("x", (2, 3))
  y = x.sum(axis=0)
  verify_expr(y)


def test_sum_axis_interpreter_matches_numpy() -> None:
  x = al.sym("x", (3, 4))
  y = x.sum(axis=1)
  fn = al.Function("f_sum_axis", [x], [y], ["x"], ["y"])
  x_val = np.arange(12.0).reshape(3, 4)
  out = fn.eval_interpreter(x_val)[0]
  np.testing.assert_allclose(out, x_val.sum(axis=1))


def test_sum_axis_jacobian_against_dense() -> None:
  x = al.sym("x", (2, 3))
  y = x.sum(axis=1)
  fn = al.Function("f_sumaxis_jac", [x], [y], ["x"], ["y"])
  j = al.jacobian(fn, "x", "y")
  J = j(np.arange(6.0).reshape(2, 3))
  # Each output[i] sums row i of x, so J should be 2x6 with a block-diagonal
  # of ones-of-length-3.
  expected = np.array(
    [
      [1, 1, 1, 0, 0, 0],
      [0, 0, 0, 1, 1, 1],
    ],
    dtype=np.float64,
  )
  np.testing.assert_allclose(J, expected)


def test_sum_axis_can_render_through_program_ir() -> None:
  x = al.sym("x", (2, 3))
  fn = al.Function("f_sumaxis_can", [x], [x.sum(axis=0)], ["x"], ["y"])
  assert can_render_program_c(fn)


def test_sum_axis_jit_matches_interpreter() -> None:
  if not _has_compiler():
    pytest.skip("no C compiler in PATH")
  x = al.sym("x", (3, 4))
  fn = al.Function("f_sumaxis_jit", [x], [x.sum(axis=0)], ["x"], ["y"])
  x_val = np.linspace(-1.0, 1.0, 12).reshape(3, 4)
  ref = fn.eval_interpreter(x_val)[0]
  old = os.environ.get("ALLOY_USE_PROGRAM_IR_C")
  try:
    os.environ["ALLOY_USE_PROGRAM_IR_C"] = "1"
    fn.recompile()
    out = fn(x_val)
  finally:
    if old is None:
      del os.environ["ALLOY_USE_PROGRAM_IR_C"]
    else:
      os.environ["ALLOY_USE_PROGRAM_IR_C"] = old
  np.testing.assert_allclose(out, ref, atol=1e-12, rtol=1e-12)


def test_sum_axis_rejects_out_of_bounds() -> None:
  x = al.sym("x", (2, 3))
  with pytest.raises(ValueError, match="out of bounds"):
    x.sum(axis=5)


def test_sum_axis_rejects_duplicate_axes() -> None:
  x = al.sym("x", (2, 3))
  with pytest.raises(ValueError, match="unique"):
    x.sum(axis=(0, 0))


# --------------------------------------------------------------------------- mean (sugar)


def test_mean_axis_interpreter_matches_numpy() -> None:
  x = al.sym("x", (3, 4))
  fn = al.Function("f_mean_axis", [x], [x.mean(axis=1)], ["x"], ["y"])
  x_val = np.arange(12.0).reshape(3, 4)
  np.testing.assert_allclose(fn.eval_interpreter(x_val)[0], x_val.mean(axis=1))


def test_mean_all_and_tuple_axes_match_numpy() -> None:
  x = al.sym("x", (2, 3, 4))
  x_val = np.linspace(-2.0, 5.0, 24).reshape(2, 3, 4)
  f_all = al.Function("f_mean_all", [x], [x.mean()], ["x"], ["y"])
  f_tup = al.Function("f_mean_tup", [x], [x.mean(axis=(0, 2))], ["x"], ["y"])
  np.testing.assert_allclose(f_all.eval_interpreter(x_val)[0], x_val.mean())
  np.testing.assert_allclose(f_tup.eval_interpreter(x_val)[0], x_val.mean(axis=(0, 2)))


def test_mean_axis_jacobian_against_dense() -> None:
  x = al.sym("x", (2, 3))
  fn = al.Function("f_meanaxis_jac", [x], [x.mean(axis=1)], ["x"], ["y"])
  j = al.jacobian(fn, "x", "y")
  J = j(np.arange(6.0).reshape(2, 3))
  # mean over rows of length 3 -> each output row contributes 1/3 across its block.
  expected = np.array([[1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1]], dtype=np.float64) / 3.0
  np.testing.assert_allclose(J, expected)
