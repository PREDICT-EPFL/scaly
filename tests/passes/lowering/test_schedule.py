"""Accumulator schedules preserve ordered sums, including empty domains and output tails."""

import numpy as np
import pytest

import scaly as sc
from scaly.function.sugar import _mapped_call
from scaly.ir import program as p
from scaly.ir.program import ProgramOp, RangeKind, walk_program
from scaly.passes.lowering import lower_function, main_proc


def _function(shapes, body, lowering="block"):
  @sc.function(sc.group(*(sc.arg(f"x{k}", shape) for k, shape in enumerate(shapes))), outputs=sc.arg("y"), name="accumulator")
  def fn(inputs):
    return body(*inputs).with_lowering(lowering)

  return fn


def _lowered(fn):
  seen = {}
  lower_function(fn, observe=seen.__setitem__)
  return main_proc(seen["lowered"])


_PRODUCTS = [
  ((3, 7), (7,)),
  ((3, 7), (7, 1)),
  ((5, 0), (0,)),
]


@pytest.mark.parametrize(("sa", "sb"), _PRODUCTS)
@pytest.mark.parametrize("lowering", ["block", "scalar"])
def test_product_matches_numpy(sa, sb, lowering, isolated_cache):
  fn = _function((sa, sb), lambda a, b: a @ b, lowering)
  rng = np.random.default_rng(45)
  a, b = (rng.integers(-8, 9, shape).astype(np.float64) for shape in (sa, sb))
  fn.recompile()
  np.testing.assert_array_equal(fn((a, b)), a @ b)


def test_sum_updates_a_scalar_and_stores_once():
  proc = _lowered(_function(((19,),), lambda x: x.sum()))
  assignments = [n for n in walk_program(proc) if n.op == ProgramOp.ASSIGN]
  assert len(assignments) == 2
  initial, update = sorted(assignments, key=lambda n: not n.attrs.get("declare", False))
  assert initial.attrs["target"] == update.attrs["target"]
  assert initial.args == (p.const_float(0),)
  assert update.args[0].op == ProgramOp.ADD
  assert sum(n.op == ProgramOp.STORE for n in walk_program(proc)) == 1
  assert all(n.args[0].attrs["buffer"] != "y" for n in walk_program(proc) if n.op == ProgramOp.LOAD)


def test_unit_stride_reduction_has_four_independent_accumulators_and_a_tail():
  proc = _lowered(_function(((9, 7), (7,)), lambda a, b: a @ b))
  reductions = [n for n in walk_program(proc) if n.op == ProgramOp.FOR and n.args[0].attrs["kind"] == RangeKind.REDUCE]
  assert sorted(len(loop.args) - 1 for loop in reductions) == [1, 4]
  for loop in reductions:
    updates = list(loop.args[1:])
    assert all(n.op == ProgramOp.ASSIGN and not n.attrs.get("declare") for n in updates)
    assert len({n.attrs["target"] for n in updates}) == len(updates)
    assert not any(n.op == ProgramOp.STORE for n in walk_program(loop))


def test_strided_reduction_precedes_the_contiguous_output_axis():
  proc = _lowered(_function(((7,), (7, 9)), lambda a, b: a @ b))
  reduction = next(n for n in walk_program(proc) if n.op == ProgramOp.FOR and n.args[0].attrs["kind"] == RangeKind.REDUCE)
  output = next(n for n in reduction.args[1:] if n.op == ProgramOp.FOR)
  assert output.args[0].attrs["kind"] == RangeKind.GLOBAL
  assert output.args[0].args[1] == p.const_int(9)
  (update,) = output.args[1:]
  assert update.op == ProgramOp.STORE and update.args[1].op == ProgramOp.ADD
  assert update.args[1].args[0].op == ProgramOp.LOAD and update.args[1].args[0].args[0] is update.args[0]


def test_sum_of_special_values_matches_numpy(isolated_cache):
  fn = _function(((5,),), lambda x: x.sum())
  rows = [np.full(5, v) for v in (np.nan, np.inf, -np.inf, -0.0, 0.0)]
  for index in (0, 1, 2, 3, 4):
    row = np.ones(5)
    row[index] = np.nan
    rows.append(row)
  for row in rows:
    np.testing.assert_array_equal(fn((row,)), row.sum())


def test_mapped_sums_keep_scalar_accumulators_in_their_procedure(isolated_cache):
  @sc.function(sc.group(sc.arg("x", 3), sc.arg("weights", 3)), outputs=sc.arg("y"), name="sum_stage")
  def stage(inputs):
    x, weights = inputs
    return (x.sum() * weights.exp().sum()).block()

  @sc.function(sc.group(sc.arg("z", 15), sc.arg("weights", 3)), outputs=sc.arg("y"), name="sum_map")
  def fn(inputs):
    z, weights = inputs
    return _mapped_call(stage, 5, [(z, 0, 3), (weights, 0, 0)])

  z, weights = np.linspace(-2, 3, 15), np.array([0.1, 0.3, -0.2])
  fn.recompile()
  np.testing.assert_allclose(np.asarray(fn((z, weights))), z.reshape(5, 3).sum(axis=1) * np.exp(weights).sum(), rtol=1e-14, atol=1e-14)


def test_strided_product_keeps_automatic_scalarization_and_constant_folding(isolated_cache):
  selection = np.zeros((40, 40))
  selection[np.arange(40), np.arange(40) * 7 % 40] = 1

  @sc.function(sc.group(sc.arg("x", (40, 20))), outputs=sc.arg("y"), name="auto_product_stage")
  def stage(inputs):
    return inputs[0].T @ sc.const(selection)

  @sc.function(sc.group(sc.arg("z", 2400)), outputs=sc.arg("y"), name="auto_product_map")
  def fn(inputs):
    return _mapped_call(stage, 3, [(inputs[0], 0, 800)])

  seen = {}
  lower_function(fn, observe=seen.__setitem__)
  expanded = next(proc for proc in seen["pass:scalarize"].args if proc.attrs["name"] == stage.name)
  assert expanded.attrs.get("scalarized")
  assert not any(n.op == ProgramOp.MUL for n in walk_program(expanded) if n.dtype.is_floating)
  values = np.random.default_rng(45).normal(size=2400)
  fn.recompile()
  expected = values.reshape(3, 40, 20).transpose(0, 2, 1) @ selection
  np.testing.assert_array_equal(np.asarray(fn((values,))).reshape(3, 20, 40), expected)


def test_product_keeps_each_outputs_reference_sum_order(isolated_cache):
  fn = _function(((9, 7), (7,)), lambda a, b: a @ b)
  a, b = np.ones((9, 7)), np.ones(7)
  terms = np.array([1e16, 1.0, -1e16, 3.0, -3.0, 2.0, 0.25])
  a[...] = terms
  expected = np.zeros(9)
  for k in range(7):
    expected += a[:, k] * b[k]
  fn.recompile()
  np.testing.assert_array_equal(fn((a, b)), expected)
