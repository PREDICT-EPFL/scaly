"""Mapped range fusion preserves scalar sharing, pointer offsets, and memory ordering."""

from collections.abc import Callable

import numpy as np
import pytest

from scaly.function.model import as_concrete
from scaly.function.concrete import ConcreteFunction
from scaly.function.sugar import _mapped_call
import scaly as sc
from scaly.ir import program as p
from scaly.ir.program import ProgramNode, ProgramOp
from scaly.ir.program_spec import verify_program
from scaly.ir.types import dtypes
from scaly.passes.lowering import lower_function, main_proc
from scaly.ir.program import walk_program
from scaly.passes.program.fuse_ranges import fuse_ranges


def _function(name, inputs, outputs):
  return ConcreteFunction._from_exprs(name, inputs, outputs, [x.name for x in inputs], [f"out{i}" for i in range(len(outputs))])


def _body(proc):
  return proc.args[proc.attrs["param_count"] :]


def _assert_fused(prog):
  proc = main_proc(prog)
  assert not any(n.op == ProgramOp.CALL for n in walk_program(proc))
  assert not any(n.op == ProgramOp.BUFFER and n.attrs.get("address_space") == "private" for n in _body(proc))
  assert proc.attrs["sz_w"] == 0
  assert all(n.args[0].attrs.get("mapped") for n in _body(proc) if n.op == ProgramOp.FOR)


@pytest.mark.parametrize("length", [4, 31])
def test_transpose_gather_discards_unused_scalar_outputs(length):
  x = sc.sym("x", 2)
  stage = _function("range_stage", [x], [sc.stack([x[0].sin() + x[1] ** 2, x[0].cos(), x[1].exp(), x[0] * x[1]])])
  z = sc.sym("z", 2 * length + 3)
  mapped = _mapped_call(stage, length, [(z, 1, 2)])
  moved = mapped.reshape((length, 2, 2)).transpose((1, 0, 2)).reshape((4 * length,))
  selected = sc.gather(moved, np.arange(2 * length))
  fn = _function("range_selected", [z], [selected])
  prog = lower_function(fn)
  _assert_fused(prog)
  assert not any(n.op == ProgramOp.EXP for n in walk_program(prog))
  data = np.linspace(-0.8, 0.9, z.size)
  stage_data = data[1 : 1 + 2 * length].reshape(length, 2)
  expected = np.stack([np.sin(stage_data[:, 0]) + stage_data[:, 1] ** 2, np.cos(stage_data[:, 0])], axis=1).reshape(-1)
  np.testing.assert_allclose(fn(data), expected, rtol=1e-14, atol=1e-14)


def test_shared_consumers_schedule_expensive_scalar_once():
  x = sc.sym("x", 1)
  stage = _function("range_shared_stage", [x], [x.sin()])
  z = sc.sym("z", 12)
  mapped = _mapped_call(stage, 12, [(z, 0, 1)])
  fn = _function("range_shared", [z], [mapped + mapped * mapped, mapped * 3])
  prog = lower_function(fn)
  _assert_fused(prog)
  proc = main_proc(prog)
  assert sum(n.op == ProgramOp.SIN for n in walk_program(proc)) == 1
  data = np.linspace(-1, 1, 12)
  expected = np.sin(data)
  first, second = fn(data)
  np.testing.assert_allclose(first, expected + expected * expected)
  np.testing.assert_allclose(second, expected * 3)


def test_chained_mapped_stages_share_their_scalar_values():
  x = sc.sym("x", 2)
  first = _function("range_first", [x], [x.sin()])
  second = _function("range_second", [x], [x * x + 2])
  z = sc.sym("z", 14)
  mapped = _mapped_call(first, 7, [(z, 0, 2)])
  result = _mapped_call(second, 7, [(mapped, 0, 2)])
  fn = _function("range_chained", [z], [result])
  _assert_fused(lower_function(fn))
  data = np.linspace(-1.3, 0.7, 14)
  np.testing.assert_allclose(fn(data), np.sin(data) ** 2 + 2)


def test_shifted_ranges_align_through_input_offsets():
  x = sc.sym("x", 2)
  first = _function("range_offset_first", [x], [x.sin()])
  second = _function("range_offset_second", [x], [x * x])
  z = sc.sym("z", 14)
  a = _mapped_call(first, 7, [(z, 0, 2)])
  b = _mapped_call(second, 6, [(z, 2, 2)])
  out = a + sc.scatter(b, np.arange(2, 14), (14,))
  fn = _function("range_offsets", [z], [out])
  _assert_fused(lower_function(fn))
  data = np.linspace(-0.9, 0.8, 14)
  expected = np.sin(data)
  expected[2:] += data[2:] ** 2
  np.testing.assert_allclose(fn(data), expected)


def test_cross_stage_consumer_keeps_materialized_producer():
  x = sc.sym("x", 1)
  stage = _function("range_divergent_stage", [x], [x.sin()])
  z = sc.sym("z", 9)
  mapped = _mapped_call(stage, 9, [(z, 0, 1)])
  fn = _function("range_divergent", [z], [mapped[:-1] + mapped[1:]])
  prog = lower_function(fn)
  assert any(n.op == ProgramOp.CALL for n in walk_program(main_proc(prog)))
  data = np.linspace(-0.5, 1.2, 9)
  np.testing.assert_allclose(fn(data), np.sin(data[:-1]) + np.sin(data[1:]))


def _raw_program(extra: Callable[[ProgramNode, ProgramNode], list[ProgramNode]] = lambda _x, _tmp: []):
  x = p.buffer("x", dtypes.float64, (5,))
  y = p.buffer("y", dtypes.float64, (5,))
  tmp = p.buffer("tmp", dtypes.float64, (5,), address_space="private")
  a = p.buffer("a", dtypes.float64, (1,))
  b = p.buffer("b", dtypes.float64, (1,))
  callee = p.proc("stage", [a, b], [p.store(p.view(b, [p.const_int(0)]), p.load(p.view(a, [p.const_int(0)])))])
  callee = ProgramNode(callee.op, callee.args, {**callee.attrs, "input_count": 1, "scalarized": True}, callee.dtype)
  i = p.var("i")
  call = ProgramNode(ProgramOp.CALL, (p.view(x, [i]), p.view(tmp, [i])), {"callee": "stage", "n_in": 1, "n_out": 1, "returns": ()})
  mapped = p.for_(p.range_("i", 0, 5), [call])
  copied = p.for_(p.range_("i", 0, 5), [p.store(p.view(y, [i]), p.load(p.view(tmp, [i])))])
  root = p.proc("root", [x, y], [tmp, mapped, *extra(x, tmp), copied])
  root = ProgramNode(root.op, root.args, {**root.attrs, "input_count": 1}, root.dtype)
  return p.program([callee, root])


@pytest.mark.parametrize("through_alias", [False, True])
def test_input_write_between_producer_and_consumer_blocks_motion(through_alias):
  def extra(x, _tmp):
    if not through_alias:
      return [p.store(p.view(x, [p.const_int(0)]), p.const_float(9))]
    alias = ProgramNode(ProgramOp.BUFFER, (), {**x.attrs, "name": "alias", "address_space": "private", "alias_of": "x", "alias_offset": 0}, x.dtype)
    return [alias, p.store(p.view(alias, [p.const_int(0)]), p.const_float(9))]

  prog = _raw_program(extra)
  assert fuse_ranges(prog) is prog


def test_call_escape_keeps_producer_storage():
  prog = _raw_program(lambda _x, tmp: [p.call("opaque", [tmp])])
  assert fuse_ranges(prog) is prog


def test_scalar_name_allocation_reserves_unused_assignment_targets():
  prog = _raw_program(lambda _x, _tmp: [p.assign("v0", p.const_float(1), declare=True)])
  result = fuse_ranges(prog)
  verify_program(result)
  root = result.args[-1]
  assert any(n.op == ProgramOp.FOR and n.args[0].attrs["name"] != "v0" for n in _body(root))


def test_sparse_hessian_lower_triangle_has_only_surviving_stage_stores():
  x = sc.sym("x", 2)
  stage = _function("range_hess_stage", [x], [sc.stack([x[0] * x[1] + x[0].sin() + x[1] ** 3])])
  length = 11

  @sc.function(sc.arg("z", 2 * length), outputs=sc.group(sc.arg("f", ()), sc.arg("g", length)), name="range_hess_base")
  def base(z: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
    return (z * z).sum(), sc.vmap(stage, length)(z).vec()

  hess = base.factory("range_hess", ["z", "lam:f", "lam:g"], [sc.factory.SpHess("gamma", "z", triangle="lower")], aux={"gamma": ["f", "g"]})
  observed = {}
  prog = lower_function(hess, observe=lambda name, program: observed.__setitem__(name, program))
  _assert_fused(prog)
  assert sum(n.op in {ProgramOp.STORE, ProgramOp.STORE_PAIR} for n in walk_program(main_proc(prog))) <= 3
  with pytest.raises(AssertionError):
    _assert_fused(observed["pass:scalarize"])
  data = np.linspace(-0.7, 1.1, 2 * length)
  multipliers = np.linspace(-0.3, 0.8, length)
  objective_weight = np.array(0.9)
  expected = np.eye(2 * length) * (2 * objective_weight)
  for k, weight in enumerate(multipliers):
    expected[2 * k : 2 * k + 2, 2 * k : 2 * k + 2] += weight * np.array([[-np.sin(data[2 * k]), 1], [1, 6 * data[2 * k + 1]]])
  sparsity = as_concrete(hess).output_sparsities[0]
  assert sparsity is not None
  np.testing.assert_allclose(hess(*(data, objective_weight, multipliers)), expected[sparsity.rows, sparsity.cols], rtol=1e-13, atol=1e-14)


def test_cross_store_loop_carried_dependency_keeps_original_schedule():
  prog = _raw_program()
  callee, root = prog.args
  x, y, tmp, mapped, _copied = root.args
  a = p.buffer("a", dtypes.float64, (5,), address_space="private")
  i = p.var("i")
  loop = p.for_(
    p.range_("i", 1, 5),
    [
      p.store(p.view(a, [i]), p.load(p.view(tmp, [p.sub(i, p.const_int(1))]))),
      p.store(p.view(tmp, [i]), p.load(p.view(a, [i]))),
    ],
  )
  modified = ProgramNode(root.op, (x, y, tmp, a, mapped, loop, _copied), root.attrs, root.dtype)
  original = p.program([callee, modified])
  assert fuse_ranges(original) is original


def test_explicit_range_value_keeps_its_original_binding():
  prog = _raw_program()
  callee, root = prog.args
  loop = root.args[-1]
  store = loop.args[1]
  changed = p.store(store.args[0], p.add(store.args[1], p.var("i", dtypes.float64)))
  body = (*root.args[:-1], p.for_(loop.args[0], [changed]))
  original = p.program([callee, ProgramNode(root.op, body, root.attrs, root.dtype)])
  assert fuse_ranges(original) is original


def test_short_upstream_producer_stays_outside_mapped_range():
  prog = _raw_program()
  callee, root = prog.args
  x, y, tmp, mapped, copied = root.args
  fixed = p.buffer("fixed", dtypes.float64, (1,), address_space="private")
  at_zero = p.view(fixed, [p.const_int(0)])
  computed = p.for_(p.range_("j", 0, 1), [p.store(at_zero, ProgramNode(ProgramOp.SIN, (p.load(p.view(x, [p.const_int(0)])),), dtype=dtypes.float64))])
  consumer = copied.args[1]
  copied = p.for_(copied.args[0], [p.store(consumer.args[0], p.add(consumer.args[1], p.load(at_zero)))])
  root = ProgramNode(root.op, (x, y, tmp, fixed, computed, mapped, copied), root.attrs, root.dtype)
  original = p.program([callee, root])
  result = fuse_ranges(original)
  assert result is not original
  root = result.args[-1]
  assert computed in _body(root)
  fused = next(n for n in _body(root) if n.op == ProgramOp.FOR and n.args[0].attrs.get("mapped"))
  assert not any(n.op == ProgramOp.SIN for n in walk_program(fused))


def test_discarded_callee_outputs_do_not_materialize_reused_scratch():
  x = sc.sym("x", 2)
  stage = _function("range_multiple_outputs", [x], [x.sin(), x.exp()])
  z = sc.sym("z", 12)
  mapped = _mapped_call(stage, 6, [(z, 0, 2)], output=0)
  fn = _function("range_one_output", [z], [mapped])
  prog = lower_function(fn)
  _assert_fused(prog)
  assert not any(n.op == ProgramOp.EXP for n in walk_program(prog))
  data = np.linspace(-0.5, 1.2, 12)
  np.testing.assert_allclose(fn(data), np.sin(data))


@pytest.mark.parametrize("nested", [False, True])
def test_scalar_offset_mutation_blocks_mapped_read_motion(nested):
  prog = _raw_program()
  callee, root = prog.args
  _x, y, tmp, mapped, copied = root.args
  x = p.buffer("x", dtypes.float64, (6,))
  call = mapped.args[1]
  offset = p.var("offset")
  call = ProgramNode(call.op, (p.view(x, [p.add(p.var("i"), offset)]), call.args[1]), call.attrs, call.dtype)
  mapped = p.for_(mapped.args[0], [call])
  changed = p.assign("offset", p.const_int(1))
  if nested:
    changed = p.for_(p.range_("j", 0, 1), [changed])
  body = (x, y, tmp, p.assign("offset", p.const_int(0), declare=True), mapped, changed, copied)
  original = p.program([callee, ProgramNode(root.op, body, root.attrs, root.dtype)])
  assert fuse_ranges(original) is original


def test_assembly_rewrite_folds_shared_scalar_graph_once(monkeypatch):
  import importlib

  fusion = importlib.import_module("scaly.passes.program.fuse_ranges")
  source = p.buffer("shared_input", dtypes.float64, (1,))
  load = p.load(p.view(source, [p.const_int(0)]))
  shared = load
  for _ in range(40):
    shared = p.add(p.mul(shared, shared), p.const_float(1.0))
  original = fusion.fold_program
  visits = []

  def counted(node):
    visits.append(node)
    return original(node)

  monkeypatch.setattr(fusion, "fold_program", counted)
  folded = {}
  for index in range(20):
    result = fusion._rewrite(p.add(load, p.const_float(float(index))), {load: shared}, folded)
    assert result is original(p.add(shared, p.const_float(float(index))))
  assert len(visits) <= len(list(walk_program(shared))) + 20
