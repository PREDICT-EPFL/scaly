"""A Function may name an output like an input (``(x, y) -> (y, z)``); the generated code keeps them apart."""

from __future__ import annotations

import numpy as np

import scaly as sc
from scaly.codegen import render_c_module


def _clash(name: str) -> sc.Function:
  @sc.function(sc.G(sc.L("x", 2), sc.L("y", 2)), sc.G(sc.L("y", 2), sc.L("z", 2)), name=name)
  def f(inputs):
    x, y = inputs
    return x * 3.0, y + x  # z reads the input y after the output y is written

  return f


def test_an_output_named_like_an_input_is_its_own_buffer() -> None:
  x, y = np.array([1.0, 2.0]), np.array([5.0, 6.0])
  out_y, z = _clash("nc_entry")((x, y))
  np.testing.assert_array_equal(out_y, 3 * x)
  np.testing.assert_array_equal(z, y + x)


def test_an_input_passed_straight_through_under_its_own_name() -> None:
  @sc.function(sc.G(sc.L("x", 2), sc.L("y", 2)), sc.G(sc.L("y", 2), sc.L("z", 2)), name="nc_passthrough")
  def f(inputs):
    x, y = inputs
    return y, 2.0 * x

  out_y, z = f((np.array([1.0, 2.0]), np.array([5.0, 6.0])))
  np.testing.assert_array_equal(out_y, [5.0, 6.0])
  np.testing.assert_array_equal(z, [2.0, 4.0])


def test_a_callee_and_a_loop_body_with_clashing_names() -> None:
  inner = _clash("nc_callee")
  a, b = sc.sym("a", 2), sc.sym("b", 2)
  y, z = inner((a, b))
  c = sc.sym("c", 2)
  body = sc.Function._from_exprs("nc_body", [c, b], [c * 0.5 + b], ["c", "b"], ["c"])  # output named like the carry
  cond = sc.Function._from_exprs("nc_cond", [c, b], [sc.greater(c[0], 1.0)], ["c", "b"], ["go"])
  loop, _ = sc.while_loop(cond, body, z, max_iter=20, params=(b,))
  fn = sc.Function._from_exprs("nc_outer", [a, b], [y, z, loop], ["a", "b"], ["y", "z", "loop"])
  av, bv = np.array([1.0, -2.0]), np.array([0.25, 0.5])
  got_y, got_z, got_loop = fn((av, bv))
  np.testing.assert_array_equal(got_y, 3 * av)
  np.testing.assert_array_equal(got_z, bv + av)
  ref = bv + av
  while ref[0] > 1.0:
    ref = 0.5 * ref + bv
  np.testing.assert_array_equal(got_loop, ref)


def test_the_typed_header_names_both_sides() -> None:
  header = render_c_module(_clash("nc_header")).header
  assert "y_in" in header and "y_out" in header
