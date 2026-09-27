"""The model contract and the discrete maps built over a model's signature."""

from __future__ import annotations

import numpy as np
import pytest

import scaly as sc
from scaly import integrators as si
from scaly.codegen import render_c_module
from scaly.integrators.model import check_model
from scaly.ir.expr import ExprOp, topo


@sc.function(3, 1, output="xdot")
def cubic(x, u):
  return sc.stack([x[1], x[2], -x[0] * x[0] * x[0] + u[0]])


def _ops(fn) -> set[ExprOp | str]:
  return {e.op for e in topo(fn.outputs)}


def test_substeps_unroll_up_to_the_threshold_then_loop() -> None:
  assert ExprOp.SCAN not in _ops(si.rk4(cubic, dt=0.1, steps=si.UNROLL_STEPS))
  assert ExprOp.SCAN in _ops(si.rk4(cubic, dt=0.1, steps=si.UNROLL_STEPS + 1))


def test_the_generated_c_does_not_grow_with_looped_substeps() -> None:
  few, many = (render_c_module(si.rk4(cubic, dt=0.1, steps=n)).source for n in (si.UNROLL_STEPS + 1, 500))
  assert len(few.splitlines()) == len(many.splitlines())


@pytest.mark.parametrize(
  ("declaration", "message"),
  [
    ((sc.L("x", (2, 2)), 1), "must be a float vector"),
    ((sc.G(sc.L("x", 2), sc.L("v", 2)), 1), "first parameter must be the state"),
  ],
)
def test_the_state_must_be_one_vector(declaration, message: str) -> None:
  @sc.function(*declaration)
  def model(x, u):
    return u

  with pytest.raises(ValueError, match=message):
    check_model(model.concrete, "m")


def test_the_output_must_be_the_state_derivative() -> None:
  @sc.function(2, 1, output=sc.G("a", "b"))
  def two(x, u):
    return x, u

  @sc.function(2, 1)
  def short(x, u):
    return u

  for model in (two, short):
    with pytest.raises(ValueError, match="one output, the derivative of 'x' shaped"):
      si.rk4(model, dt=0.1)


def test_a_dt_input_needs_a_free_name() -> None:
  @sc.function(2, (), output="xdot")
  def model(x, dt):
    return x * dt

  assert si.rk4(model, dt=0.1).input_names == ("x", "dt")
  with pytest.raises(ValueError, match="already has an input named 'dt'"):
    si.rk4(model, dt=None)


def test_state_named_z_gives_znext() -> None:
  @sc.function(sc.L("z", 2), output="zdot")
  def model(z):
    return -z

  step = si.rk4(model, dt=0.1)
  assert step.output_names == ("znext",)
  np.testing.assert_allclose(step(np.ones(2)), np.exp(-0.1) * np.ones(2), rtol=1e-6)
