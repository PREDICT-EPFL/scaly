"""Buffer reuse preserves the functional values of indexed loop updates."""

import numpy as np
import pytest

import scaly as sc
from scaly.codegen import render_c_source
from scaly.ir.program import ProgramOp, walk_program
from scaly.passes.lowering import lower_function


def _loop(body, size, length=3, **kwargs):
  @sc.function(sc.arg("initial", size))
  def run(initial):
    result, _ = sc.scan(body, initial, length=length, in_place=(0,), **kwargs)
    return result.with_lowering("block")

  return run


def _inout(fn):
  program = lower_function(fn)
  return [n for n in walk_program(program) if n.op == ProgramOp.BUFFER and n.attrs.get("intent") == "inout"]


def test_last_use():
  @sc.function(sc.arg("carry", 8))
  def step(carry):
    return sc.scatter(carry, [2, 3], sc.gather(carry, [0, 1]) + 1, combine="set")

  fn = _loop(step, 8)
  expected = np.arange(8.0)
  for _ in range(3):
    expected[[2, 3]] = expected[[0, 1]] + 1
  np.testing.assert_array_equal(fn(np.arange(8.0)), expected)
  assert _inout(fn)
  proc = next(proc for proc in lower_function(fn).args if proc.attrs.get("inout_carries"))
  assert proc.attrs["param_count"] == proc.attrs["input_count"] == 1
  assert "_inplace_raw(double* carry, double* w)" in render_c_source(fn)


def test_same_runtime_index():
  @sc.function(sc.arg("carry", 8), sc.arg("pivot", sc.TensorType((), sc.dtypes.int64)))
  def step(carry, pivot):
    return sc.scatter(carry, pivot, sc.gather(carry, pivot).floor() + 1, combine="set")

  @sc.function(sc.arg("initial", 8), sc.arg("pivot", ()))
  def run(initial, pivot):
    result, _ = sc.scan(step, initial, (sc.broadcast(pivot.cast(sc.dtypes.int64)),), length=3, in_place=(0,))
    return result.with_lowering("block")

  expected = np.arange(8.0) + 0.5
  expected[4] = 7
  np.testing.assert_array_equal(run(np.arange(8.0) + 0.5, np.array(4.0)), expected)
  assert _inout(run)
  proc = next(proc for proc in lower_function(run).args if proc.attrs.get("inout_carries"))
  assert not any(node.op == ProgramOp.BUFFER and node.attrs.get("address_space") == "private" for node in walk_program(proc))


def test_disjoint_later_read():
  @sc.function(sc.arg("carry", 8), sc.arg("j", sc.TensorType((), sc.dtypes.int64)))
  def step(carry, j):
    first = sc.scatter(carry, j, sc.gather(carry, j) + 1, combine="set")
    return sc.scatter(first, j + 4, sc.gather(carry, j + 4) + 2, combine="set")

  fn = _loop(step, 8, index=True)
  expected = np.arange(8.0)
  expected[:3] += 1
  expected[4:7] += 2
  np.testing.assert_array_equal(fn(np.arange(8.0)), expected)
  assert _inout(fn)


def test_overlapping_later_gather_refuses():
  @sc.function(sc.arg("carry", 4), sc.arg("j", sc.TensorType((), sc.dtypes.int64)))
  def step(carry, j):
    first = sc.scatter(carry, j, sc.const(10.0), combine="set")
    return sc.scatter(first, j + 1, sc.gather(carry, j) + 1, combine="set")

  with pytest.raises(NotImplementedError, match="carry"):
    lower_function(_loop(step, 4, index=True))


def test_unknown_runtime_index_refuses_disjointness():
  @sc.function(sc.arg("carry", 8), sc.arg("pivot", sc.TensorType((), sc.dtypes.int64)))
  def step(carry, pivot):
    first = sc.scatter(carry, pivot, sc.gather(carry, pivot) + 1, combine="set")
    return sc.scatter(first, [7], sc.gather(carry, [0]) + 2, combine="set")

  @sc.function(sc.arg("initial", 8), sc.arg("pivot", ()))
  def run(initial, pivot):
    result, _ = sc.scan(step, initial, (sc.broadcast(pivot.cast(sc.dtypes.int64)),), length=3, in_place=("carry",))
    return result.with_lowering("block")

  with pytest.raises(NotImplementedError, match="carry"):
    lower_function(run)


@pytest.mark.parametrize(("length", "reverse"), [(0, False), (1, True), (5, True)])
def test_lengths_and_reverse_preserve_input(length, reverse):
  @sc.function(sc.arg("carry", 8), sc.arg("j", sc.TensorType((), sc.dtypes.int64)))
  def step(carry, j):
    return sc.scatter(carry, j, sc.gather(carry, j) + 1, combine="set")

  fn = _loop(step, 8, length=length, index=True, reverse=reverse)
  initial = np.arange(8.0)
  expected = initial.copy()
  expected[:length] += 1
  np.testing.assert_array_equal(fn(initial), expected)
  np.testing.assert_array_equal(initial, np.arange(8.0))


def test_repeated_same_indices_materialize_before_writing():
  @sc.function(sc.arg("carry", 4))
  def step(carry):
    return sc.scatter(carry, [1, 1], sc.gather(carry, [1, 1]) + 1, combine="add")

  fn = _loop(step, 4)
  expected = np.arange(4.0)
  for _ in range(3):
    np.add.at(expected, [1, 1], expected[[1, 1]] + 1)
  np.testing.assert_array_equal(fn(np.arange(4.0)), expected)


def test_returning_old_carry_refuses():
  @sc.function(sc.arg("carry", 4))
  def step(carry):
    return sc.scatter(carry, [0], [5.0], combine="set"), carry

  with pytest.raises(NotImplementedError, match="carry"):
    lower_function(_loop(step, 4))


@pytest.mark.parametrize("source", ["direct", "reshape"])
def test_update_source_alias_refuses(source):
  @sc.function(sc.arg("carry", 4))
  def step(carry):
    values = carry if source == "direct" else carry.reshape((2, 2)).reshape((4,))
    return sc.scatter(carry, [1, 2, 3, 0], values, combine="set")

  with pytest.raises(NotImplementedError, match="carry"):
    lower_function(_loop(step, 4))


def test_alias_created_before_update_is_read_after_it():
  @sc.function(sc.arg("carry", 4))
  def step(carry):
    old = carry[:2]
    first = sc.scatter(carry, [0], sc.gather(carry, [2]), combine="set")
    return sc.scatter(first, [2, 3], old + 1, combine="set")

  with pytest.raises(NotImplementedError, match="carry"):
    lower_function(_loop(step, 4))


def test_scatter_reuses_a_dead_private_base():
  @sc.function(sc.arg("x", 8))
  def run(x):
    base = x + 1
    return sc.scatter(base, [2, 4], sc.gather(base, [0, 1]) + 2, combine="set")

  expected = np.arange(8.0) + 1
  expected[[2, 4]] = expected[[0, 1]] + 2
  np.testing.assert_array_equal(run(np.arange(8.0)), expected)
  source = render_c_source(run)
  assert "zero_" not in source


def test_scatter_keeps_a_live_private_base():
  @sc.function(sc.arg("x", 4))
  def run(x):
    base = x + 1
    updated = sc.scatter(base, [0], [20.0], combine="set")
    return updated, base

  got = run(np.arange(4.0))
  np.testing.assert_array_equal(got[0], [20, 2, 3, 4])
  np.testing.assert_array_equal(got[1], [1, 2, 3, 4])


def test_inout_parameter_and_call_contract():
  from scaly.ir import program as p
  from scaly.ir.program import ProgramNode
  from scaly.ir.program_spec import verify_program
  from scaly.ir.spec import VerifyError
  from scaly.passes.program._common import buffer_refs
  from scaly.passes.program.scalarize import _scalarize_proc, scalarize_program

  carry = p.buffer("carry", sc.dtypes.float64, (4,), intent="inout")
  proc = p.proc("update", [carry], [p.store(p.view(carry, [p.const_int(0)]), p.const_float(2))])
  proc = ProgramNode(proc.op, proc.args, {**proc.attrs, "input_count": 1, "scalarize_mode": "procedure", "lowering": "scalar"}, dtype=proc.dtype)
  actual = p.buffer("actual", sc.dtypes.float64, (4,))
  call = ProgramNode(ProgramOp.CALL, (actual,), {"callee": "update", "n_in": 1, "n_out": 0, "inout": (0,), "returns": ()}, dtype=sc.dtypes.float64)
  caller = p.proc("caller", [actual], [call])
  program = p.program([proc, caller])
  verify_program(program)
  refs = buffer_refs(call)
  assert refs.call_inputs == refs.call_outputs == frozenset({"actual"})
  assert scalarize_program(program).args[0] is proc
  with pytest.raises(NotImplementedError, match="inout"):
    _scalarize_proc(proc, {"update": proc})
  wrong = ProgramNode(call.op, call.args, {**call.attrs, "inout": ()}, dtype=call.dtype)
  with pytest.raises(VerifyError, match="inout"):
    verify_program(p.program([proc, p.proc("caller", [actual], [wrong])]))


def test_inplace_helper_name_does_not_collide_with_user_function():
  @sc.function(sc.arg("carry", 4), name="step")
  def step(carry):
    return sc.scatter(carry, [0], sc.gather(carry, [0]) + 1, combine="set")

  @sc.function(sc.arg("x", 4), name="step_inplace")
  def other(x):
    return x * 3

  @sc.function(sc.arg("initial", 4))
  def run(initial):
    result, _ = sc.scan(step, initial, length=3, in_place=(0,))
    return other(result)

  np.testing.assert_array_equal(run(np.arange(4.0)), [9, 3, 6, 9])
  program = lower_function(run)
  names = [proc.attrs["name"] for proc in program.args]
  assert "step_inplace" in names
  assert len(names) == len(set(names))


def test_floor_regression_numpy_result():
  @sc.function(sc.arg("carry", 3))
  def step(carry):
    first = sc.scatter(carry, [0], sc.gather(carry, [1]), combine="set")
    return sc.scatter(first, [0], sc.gather(carry.floor(), [0]) + 0.5, combine="set")

  @sc.function(sc.arg("initial", 3))
  def run(initial):
    result, _ = sc.scan(step, initial, length=1)
    return result.with_lowering("block")

  initial = np.array([1.5, 2.5, 3.5])
  expected = initial.copy()
  expected[0] = initial[1]
  expected[0] = np.floor(initial[0]) + 0.5
  np.testing.assert_array_equal(run(initial), expected)
  with pytest.raises(NotImplementedError, match="carry"):
    lower_function(_loop(step, 3, length=1))


def test_same_index_matrix_broadcast():
  @sc.function(sc.arg("carry", (2, 4)))
  def step(carry):
    return sc.scatter(carry, [0, 2], sc.gather(carry, [0, 2], axis=1) + sc.const([[1.0], [2.0]]), axis=1, combine="set")

  fn = _loop(step, (2, 4))
  expected = np.arange(8.0).reshape(2, 4)
  for _ in range(3):
    expected[:, [0, 2]] += [[1], [2]]
  np.testing.assert_array_equal(fn(np.arange(8.0).reshape(2, 4)), expected)


def test_disjoint_runtime_axis_indices():
  @sc.function(sc.arg("carry", (2, 8)), sc.arg("j", sc.TensorType((), sc.dtypes.int64)))
  def step(carry, j):
    first = sc.scatter(carry, j, sc.gather(carry, j, axis=1) + 1, axis=1, combine="set")
    return sc.scatter(first, j + 4, sc.gather(carry, j + 4, axis=1) + 2, axis=1, combine="set")

  fn = _loop(step, (2, 8), index=True)
  expected = np.arange(16.0).reshape(2, 8)
  expected[:, :3] += 1
  expected[:, 4:7] += 2
  np.testing.assert_array_equal(fn(np.arange(16.0).reshape(2, 8)), expected)


def test_unchanged_carry_needs_no_step_copy():
  @sc.function(sc.arg("carry", 8))
  def step(carry):
    return carry

  fn = _loop(step, 8)
  np.testing.assert_array_equal(fn(np.arange(8.0)), np.arange(8.0))
  proc = next(proc for proc in lower_function(fn).args if proc.attrs.get("inout_carries"))
  assert proc.attrs["param_count"] == 1
  assert not any(node.op == ProgramOp.STORE for stmt in proc.args[1:] for node in walk_program(stmt))


def test_reused_buffer_drops_old_zero_support():
  @sc.function(sc.arg("x", 4))
  def step(x):
    base = x * sc.const([0.0, 1.0, 0.0, 1.0])
    return sc.scatter(base, [0], [5.0], combine="set")

  @sc.function(sc.arg("x", 4))
  def run(x):
    return step(x)

  np.testing.assert_array_equal(run(np.arange(4.0)), [5, 1, 0, 3])


def test_float_index_cannot_prove_disjointness():
  from unittest.mock import patch

  @sc.function(sc.arg("carry", 3), sc.arg("j", sc.TensorType((), sc.dtypes.int64)))
  def step(carry, j):
    first = sc.scatter(carry, [0], sc.gather(carry, [0]) + 1, combine="set")
    read_index = ((j.cast(sc.dtypes.float64) / 13).exp() <= 3.6976308645666602).cast(sc.dtypes.int64)
    return sc.scatter(first, [2], sc.gather(carry, read_index), combine="set")

  @sc.function(sc.arg("initial", 3), name="float_index_automatic")
  def automatic(initial):
    return sc.scan(step, initial, length=18, index=True)[0]

  @sc.function(sc.arg("initial", 3), name="float_index_reference")
  def reference(initial):
    return sc.scan(step, initial, length=18, index=True)[0]

  initial = np.array([1.0, 2.0, 3.0])
  with patch("scaly.passes.lowering.calls.carry_reuse", return_value={}):
    expected = reference(initial)
  np.testing.assert_array_equal(automatic(initial), expected)
  assert not _inout(automatic)
  with pytest.raises(NotImplementedError, match="carry"):
    lower_function(_loop(step, 3, length=18, index=True))


def test_batched_integer_indices_remain_provable():
  @sc.function(sc.arg("carry", 8), sc.arg("j", sc.TensorType((), sc.dtypes.int64)))
  def step(carry, j):
    index = sc.where(j < 3, j, j % 3)
    first = sc.scatter(carry, index, sc.gather(carry, index) + 1, combine="set")
    return sc.scatter(first, index + 4, sc.gather(carry, index + 4) + 2, combine="set")

  fn = _loop(step, 8, length=6, index=True)
  expected = np.arange(8.0)
  expected[:3] += 2
  expected[4:7] += 4
  np.testing.assert_array_equal(fn(np.arange(8.0)), expected)
  assert _inout(fn)


def test_large_integer_index_constants_remain_exact():
  from unittest.mock import patch

  @sc.function(sc.arg("carry", 3), sc.arg("j", sc.TensorType((), sc.dtypes.int64)))
  def step(carry, j):
    first = sc.scatter(carry, [0], sc.gather(carry, [0]) + 1, combine="set")
    read_index = (np.int64(9007199254740992) < j + np.int64(9007199254740993)).cast(sc.dtypes.int64)
    return sc.scatter(first, [2], sc.gather(carry, read_index), combine="set")

  @sc.function(sc.arg("initial", 3), name="integer_constant_automatic")
  def automatic(initial):
    return sc.scan(step, initial, length=1, index=True, in_place=("carry",))[0].with_lowering("block")

  @sc.function(sc.arg("initial", 3), name="integer_constant_reference")
  def reference(initial):
    return sc.scan(step, initial, length=1, index=True)[0]

  initial = np.array([1.0, 2.0, 3.0])
  with patch("scaly.passes.lowering.calls.carry_reuse", return_value={}):
    expected = reference(initial)
  np.testing.assert_array_equal(automatic(initial), expected)
  np.testing.assert_array_equal(expected, [2, 2, 2])
  assert _inout(automatic)


def test_print_alias_reads_the_entering_carry():
  @sc.function(sc.arg("carry", 3))
  def step(carry):
    printed = sc.print("c={}", carry)
    first = sc.scatter(carry, [0], sc.gather(printed, [2]), combine="set")
    return sc.scatter(first, [1], sc.gather(printed, [0]), combine="set")

  @sc.function(sc.arg("initial", 3))
  def automatic(initial):
    return sc.scan(step, initial, length=1)[0]

  np.testing.assert_array_equal(automatic(np.array([1.0, 2.0, 3.0])), [3, 1, 3])
  assert not _inout(automatic)
  with pytest.raises(NotImplementedError, match="carry"):
    lower_function(_loop(step, 3, length=1))


def test_print_alias_stacks_the_entering_carry():
  @sc.function(sc.arg("carry", 3))
  def step(carry):
    printed = sc.print("{}", carry)
    return sc.scatter(carry, [0], sc.gather(printed, [1]), combine="set"), printed

  @sc.function(sc.arg("initial", 3))
  def automatic(initial):
    return sc.scan(step, initial, length=1)

  final, (stacked,) = automatic(np.array([1.0, 2.0, 3.0]))
  np.testing.assert_array_equal(final, [2, 2, 3])
  np.testing.assert_array_equal(stacked, [[1, 2, 3]])
  assert not _inout(automatic)


def test_inout_carry_is_not_hoisted():
  @sc.function(sc.arg("carry", 4), sc.arg("x", 2))
  def accumulate(carry, x):
    return sc.scatter(carry, [0, 1], sc.gather(carry, [0, 1]) + x, combine="set")

  @sc.function(sc.arg("initial", 4), sc.arg("xs", (3, 2)))
  def run(initial, xs):
    return sc.scan(accumulate, initial, (xs,), length=3, in_place=(0,))[0].with_lowering("block")

  initial, xs = np.arange(4.0), np.arange(6.0).reshape(3, 2)
  expected = initial.copy()
  expected[:2] += xs.sum(axis=0)
  np.testing.assert_array_equal(run(initial, xs), expected)
  assert _inout(run)


def test_disjointness_keeps_steps_separate():
  @sc.function(sc.arg("carry", 3), sc.arg("j", sc.TensorType((), sc.dtypes.int64)))
  def step(carry, j):
    index = j % 2
    first = sc.scatter(carry, index, sc.gather(carry, index) + 1, combine="set")
    return sc.scatter(first, [2], sc.gather(carry, 1 - index).reshape((1,)), combine="set")

  fn = _loop(step, 3, length=6, index=True)
  expected = np.arange(3.0)
  for j in range(6):
    old = expected.copy()
    expected[j % 2] += 1
    expected[2] = old[1 - j % 2]
  np.testing.assert_array_equal(fn(np.arange(3.0)), expected)
  assert _inout(fn)


def test_unknown_carry_name_has_context():
  @sc.function(sc.arg("carry", 3))
  def step(carry):
    return carry

  with pytest.raises(ValueError, match="in_place names unknown carry 'missing'"):
    sc.scan(step, sc.sym("initial", 3), length=1, in_place=("missing",))


def test_trajectory_explains_strict_refusal():
  from scaly.function.concrete import ConcreteFunction
  from scaly.function.model import as_concrete
  from scaly.function.sugar import _loop_node

  @sc.function(sc.arg("carry", 3))
  def step(carry):
    return sc.scatter(carry, [0], [2.0], combine="set")

  initial = sc.sym("initial", 3)
  attrs = {"n_carry": 1, "length": 2, "starts": (), "strides": (), "index": False, "reverse": False, "in_place": (0,)}
  callee = as_concrete(step)
  final = _loop_node(callee, None, (initial,), attrs, 0)
  trajectory = _loop_node(callee, None, (initial,), attrs, 2)
  run = ConcreteFunction._from_exprs("strict_trajectory", [initial], [final, trajectory], ["initial"], ["final", "trajectory"])
  with pytest.raises(NotImplementedError, match="carry.*trajectory is used"):
    lower_function(run)


@pytest.mark.parametrize("hint", ["auto", "block"])
def test_static_unroll_matches_rolled_in_place(hint):
  @sc.function(sc.arg("carry", 4), sc.arg("j", sc.TensorType((), sc.dtypes.int64)))
  def step(carry, j):
    return sc.scatter(carry, j, sc.gather(carry, j) + 2, combine="set")

  @sc.function(sc.arg("initial", 4), name=f"carry_unroll_{hint}")
  def run(initial):
    final, _ = sc.scan(step, initial, length=4, index=True, in_place=("carry",))
    return final.with_lowering(hint)

  initial = np.arange(4.0)
  np.testing.assert_array_equal(run(initial), initial + 2)
  program = lower_function(run)
  carried = any(node.op == ProgramOp.RANGE and node.attrs.get("carried") for node in walk_program(program))
  assert carried == (hint == "block")
  assert bool(_inout(run)) == (hint == "block")


def test_unroll_hint_does_not_bypass_in_place_refusal():
  @sc.function(sc.arg("carry", 3))
  def step(carry):
    printed = sc.print("{}", carry)
    first = sc.scatter(carry, [0], sc.gather(printed, [2]), combine="set")
    return sc.scatter(first, [1], sc.gather(printed, [0]), combine="set")

  @sc.function(sc.arg("initial", 3))
  def run(initial):
    final, _ = sc.scan(step, initial, length=2, in_place=("carry",))
    return final.with_lowering("scalar")

  with pytest.raises(NotImplementedError, match="cannot prove in_place carry 'carry'"):
    lower_function(run)


def test_dynamic_index_keeps_an_in_place_loop_rolled():
  @sc.function(sc.arg("carry", 4), sc.arg("pivot", sc.TensorType((), sc.dtypes.int64)))
  def step(carry, pivot):
    return sc.scatter(carry, pivot, sc.gather(carry, pivot) + 1, combine="set")

  @sc.function(sc.arg("initial", 4), sc.arg("pivot", ()))
  def run(initial, pivot):
    return sc.scan(step, initial, (sc.broadcast(pivot.cast(sc.dtypes.int64)),), length=3, in_place=(0,))[0]

  np.testing.assert_array_equal(run(np.arange(4.0), np.array(2.0)), [0, 1, 5, 3])
  assert _inout(run)
