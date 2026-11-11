"""Mapped range fusion preserves scalar sharing, pointer offsets, and memory ordering."""

from collections.abc import Callable
import importlib

from scaly.ir.expr import _scatter_zero

import numpy as np
import pytest

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
  out = a + _scatter_zero(b, np.arange(2, 14), (14,))
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
  callee = ProgramNode(callee.op, callee.args, {**callee.attrs, "input_count": 1, "scalarized": True}, dtype=callee.dtype)
  i = p.var("i")
  call = ProgramNode(
    ProgramOp.CALL, (p.view(x, [i]), p.view(tmp, [i])), {"callee": "stage", "n_in": 1, "n_out": 1, "returns": ()}, dtype=dtypes.float64
  )
  mapped = p.for_(p.range_("i", 0, 5), [call])
  copied = p.for_(p.range_("i", 0, 5), [p.store(p.view(y, [i]), p.load(p.view(tmp, [i])))])
  root = p.proc("root", [x, y], [tmp, mapped, *extra(x, tmp), copied])
  root = ProgramNode(root.op, root.args, {**root.attrs, "input_count": 1}, dtype=root.dtype)
  return p.program([callee, root])


@pytest.mark.parametrize("through_alias", [False, True])
def test_input_write_between_producer_and_consumer_blocks_motion(through_alias):
  def extra(x, _tmp):
    if not through_alias:
      return [p.store(p.view(x, [p.const_int(0)]), p.const_float(9))]
    alias = ProgramNode(
      ProgramOp.BUFFER, (), {**x.attrs, "name": "alias", "address_space": "private", "alias_of": "x", "alias_offset": 0}, dtype=x.dtype
    )
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


@pytest.mark.parametrize(("objective_size", "max_stores"), [(2, 8), (22, 3)])
def test_sparse_hessian_lower_triangle_has_only_surviving_stage_stores(objective_size, max_stores):
  x = sc.sym("x", 2)
  stage = _function("range_hess_stage", [x], [sc.stack([x[0] * x[1] + x[0].sin() + x[1] ** 3])])
  length = 11

  @sc.function(sc.arg("z", 2 * length), outputs=sc.group(sc.arg("f", ()), sc.arg("g", length)), name="range_hess_base")
  def base(z: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
    return (z[:objective_size] * z[:objective_size]).sum(), sc.vmap(stage, length)(z).vec()

  hess = base.factory("range_hess", ["z", "lam:f", "lam:g"], [sc.factory.SpHess("gamma", "z", triangle="lower")], aux={"gamma": ["f", "g"]})
  observed = {}
  prog = lower_function(hess, observe=lambda name, program: observed.__setitem__(name, program))
  _assert_fused(prog)
  assert sum(n.op in {ProgramOp.STORE, ProgramOp.STORE_PAIR} for n in walk_program(main_proc(prog))) <= max_stores
  with pytest.raises(AssertionError):
    _assert_fused(observed["pass:scalarize"])
  data = np.linspace(-0.7, 1.1, 2 * length)
  multipliers = np.linspace(-0.3, 0.8, length)
  objective_weight = np.array(0.9)
  expected = np.zeros((2 * length, 2 * length))
  expected[np.arange(objective_size), np.arange(objective_size)] = 2 * objective_weight
  for k, weight in enumerate(multipliers):
    expected[2 * k : 2 * k + 2, 2 * k : 2 * k + 2] += weight * np.array([[-np.sin(data[2 * k]), 1], [1, 6 * data[2 * k + 1]]])
  np.testing.assert_allclose(hess(*(data, objective_weight, multipliers)).toarray(), np.tril(expected), rtol=1e-13, atol=1e-14)


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
  modified = ProgramNode(root.op, (x, y, tmp, a, mapped, loop, _copied), root.attrs, dtype=root.dtype)
  original = p.program([callee, modified])
  assert fuse_ranges(original) is original


def test_explicit_range_value_keeps_its_original_binding():
  prog = _raw_program()
  callee, root = prog.args
  loop = root.args[-1]
  store = loop.args[1]
  changed = p.store(store.args[0], p.add(store.args[1], p.var("i", dtypes.float64)))
  body = (*root.args[:-1], p.for_(loop.args[0], [changed]))
  original = p.program([callee, ProgramNode(root.op, body, root.attrs, dtype=root.dtype)])
  assert fuse_ranges(original) is original


@pytest.mark.parametrize("producer_length", [1, 10])
def test_independent_expensive_producer_stays_outside_mapped_range(producer_length):
  prog = _raw_program()
  callee, root = prog.args
  x, y, tmp, mapped, copied = root.args
  fixed = p.buffer("fixed", dtypes.float64, (producer_length,), address_space="private")
  at_zero = p.view(fixed, [p.const_int(0)])
  computed = p.for_(
    p.range_("j", 0, producer_length),
    [p.store(p.view(fixed, [p.var("j")]), ProgramNode(ProgramOp.SIN, (p.load(p.view(x, [p.const_int(0)])),), dtype=dtypes.float64))],
  )
  consumer = copied.args[1]
  copied = p.for_(copied.args[0], [p.store(consumer.args[0], p.add(consumer.args[1], p.load(at_zero)))])
  root = ProgramNode(root.op, (x, y, tmp, fixed, computed, mapped, copied), root.attrs, dtype=root.dtype)
  original = p.program([callee, root])
  result = fuse_ranges(original)
  assert result is not original
  root = result.args[-1]
  assert computed in _body(root)
  fused = next(n for n in _body(root) if n.op == ProgramOp.FOR and n.args[0].attrs.get("mapped"))
  assert not any(n.op == ProgramOp.SIN for n in walk_program(fused))


def test_expensive_initializer_of_shared_assembly_keeps_original_schedule():
  prog = _raw_program()
  callee, root = prog.args
  x, y, tmp, mapped, _copied = root.args
  assembly = p.buffer("assembly", dtypes.float64, (6,), address_space="private")
  at_zero = p.view(assembly, [p.const_int(0)])
  computed = p.for_(p.range_("j", 0, 1), [p.store(at_zero, ProgramNode(ProgramOp.SIN, (p.load(p.view(x, [p.const_int(0)])),), dtype=dtypes.float64))])
  i = p.var("i")
  target = p.view(assembly, [p.add(i, p.const_int(1))])
  assembled = p.for_(p.range_("i", 0, 5), [p.store(target, p.load(p.view(tmp, [i])))])
  copied = p.for_(p.range_("i", 0, 5), [p.store(p.view(y, [i]), p.add(p.load(target), p.load(at_zero)))])
  body = (x, y, tmp, assembly, computed, mapped, assembled, copied)
  original = p.program([callee, ProgramNode(root.op, body, root.attrs, dtype=root.dtype)])
  verify_program(original)
  assert fuse_ranges(original) is original


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("depth", [9, 20])
def test_independent_arithmetic_in_mapped_range_is_bounded(split, depth):
  prog = _raw_program()
  callee, root = prog.args
  x, y, tmp, mapped, copied = root.args
  value = p.load(p.view(x, [p.const_int(0)]))
  factor = p.load(p.view(x, [p.const_int(1)]))
  setup = []
  computed = []
  for k in range(depth):
    value = p.add(p.mul(value, factor), p.const_float(1))
    if split or k == depth - 1:
      fixed = p.buffer(f"fixed{k}", dtypes.float64, (1,), address_space="private")
      at_zero = p.view(fixed, [p.const_int(0)])
      loop = p.for_(p.range_(f"j{k}", 0, 1), [p.store(at_zero, value)])
      setup.extend([fixed, loop])
      computed.append(loop)
      value = p.load(at_zero)
  consumer = copied.args[1]
  copied = p.for_(copied.args[0], [p.store(consumer.args[0], p.add(consumer.args[1], value))])
  body = (x, y, tmp, *setup, mapped, copied)
  original = p.program([callee, ProgramNode(root.op, body, root.attrs, dtype=root.dtype)])
  verify_program(original)
  result = fuse_ranges(original)
  verify_program(result)
  root = result.args[-1]
  assert computed[0] in _body(root)
  fused = next(n for n in _body(root) if n.op == ProgramOp.FOR and n.args[0].attrs.get("mapped"))
  assert sum(n.op in {ProgramOp.ADD, ProgramOp.MUL} and n.dtype.is_floating for n in walk_program(fused)) <= 9


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
  call = ProgramNode(call.op, (p.view(x, [p.add(p.var("i"), offset)]), call.args[1]), call.attrs, dtype=call.dtype)
  mapped = p.for_(mapped.args[0], [call])
  changed = p.assign("offset", p.const_int(1))
  if nested:
    changed = p.for_(p.range_("j", 0, 1), [changed])
  body = (x, y, tmp, p.assign("offset", p.const_int(0), declare=True), mapped, changed, copied)
  original = p.program([callee, ProgramNode(root.op, body, root.attrs, dtype=root.dtype)])
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


def test_shared_seed_outputs_substitute_their_common_graph_once(monkeypatch):
  import importlib

  fusion = importlib.import_module("scaly.passes.program.fuse_ranges")
  depth, width, length = 40, 32, 5
  a, b = (p.buffer(name, dtypes.float64, (size,)) for name, size in (("a", 1), ("b", width)))
  value = p.load(p.view(a, [p.const_int(0)]))
  body = []
  for k in range(depth):
    body.append(p.assign(f"v{k}", ProgramNode(ProgramOp.SIN, (value,), dtype=dtypes.float64), declare=True))
    value = p.var(f"v{k}", dtypes.float64)
  body.extend(p.store(p.view(b, [p.const_int(k)]), p.add(value, p.const_float(k))) for k in range(width))
  callee = p.proc("shared_seed", [a, b], body)
  callee = ProgramNode(callee.op, callee.args, {**callee.attrs, "input_count": 1, "scalarized": True}, dtype=callee.dtype)
  x = p.buffer("x", dtypes.float64, (length,))
  y = p.buffer("y", dtypes.float64, (length * width,))
  tmp = p.buffer("tmp", dtypes.float64, (length * width,), address_space="private")
  i = p.var("i", dtypes.int64)
  call = ProgramNode(
    ProgramOp.CALL,
    (p.view(x, [i]), p.view(tmp, [p.mul(i, p.const_int(width))])),
    {"callee": "shared_seed", "n_in": 1, "n_out": 1},
    dtype=dtypes.float64,
  )
  mapped = p.for_(p.range_("i", 0, length), [call])
  copied = p.for_(p.range_("i", 0, length * width), [p.store(p.view(y, [i]), p.load(p.view(tmp, [i])))])
  root = p.proc("root", [x, y], [tmp, mapped, copied])
  root = ProgramNode(root.op, root.args, {**root.attrs, "input_count": 1}, dtype=root.dtype)
  original = fusion.rebuild_program
  rebuilds = 0

  def counted(node, args):
    nonlocal rebuilds
    rebuilds += 1
    return original(node, args)

  monkeypatch.setattr(fusion, "rebuild_program", counted)
  result = fusion.fuse_ranges(p.program([callee, root]))
  assert not any(node.op == ProgramOp.CALL for node in walk_program(result.args[-1]))
  assert rebuilds < 8 * (depth + width)


def _stencil_gradient(length, measurement):
  x = sc.sym("x", 2)
  step = _function("stencil_step", [x], [x.sin() + x * x])
  z = sc.sym("z", (length, 2))
  residual = z[1:] - sc.vmap(step, length - 1)(z[:-1])
  cost = (residual * residual).sum()
  if measurement:
    cost = cost + (z[:, 0] * z[:, 0]).sum()
  return _function("stencil_grad", [z], [sc.gradient(cost, z)])


@pytest.mark.parametrize("measurement", [False, True])
def test_stencil_rewrite_work_is_independent_of_stage_count(monkeypatch, measurement):
  fusion = importlib.import_module("scaly.passes.program.fuse_ranges")
  original = fusion._rewrite
  counts = []
  for length in (64, 4096):
    calls = []

    def counted(*args):
      calls.append(None)
      return original(*args)

    monkeypatch.setattr(fusion, "_rewrite", counted)
    lower_function(_stencil_gradient(length, measurement))
    counts.append(len(calls))
  assert counts[0] > 0
  assert counts[0] == counts[1]


@pytest.mark.parametrize("measurement", [False, True])
def test_stencil_gradient_matches_numpy(measurement):
  data = np.linspace(-0.7, 0.8, 18).reshape(9, 2)
  residual = data[1:] - (np.sin(data[:-1]) + data[:-1] ** 2)
  expected = np.zeros_like(data)
  expected[1:] += 2 * residual
  expected[:-1] -= 2 * residual * (np.cos(data[:-1]) + 2 * data[:-1])
  if measurement:
    expected[:, 0] += 2 * data[:, 0]
  np.testing.assert_allclose(_stencil_gradient(9, measurement)(data), expected)


def test_two_block_gathers_upstream_of_map_still_fuse():
  length = 8
  x = sc.sym("x", 2)
  step = _function("upstream_step", [x], [x.sin()])
  z = sc.sym("z", (length, 2))
  a, b = sc.sym("a", 2 * length), sc.sym("b", 2 * length)
  mapped = sc.vmap(step, length)(z)
  indices = np.repeat([0, 1], length)
  extra = (sc.gather(a, indices) + sc.gather(b, indices)).reshape((length, 2))
  result = mapped * extra
  fn = _function("upstream_values", [z, a, b], [result])
  _assert_fused(lower_function(fn))
  data = np.linspace(-0.7, 0.8, 2 * length).reshape(length, 2)
  av, bv = np.linspace(0.2, 0.6, 2 * length), np.linspace(-0.8, -0.3, 2 * length)
  expected = np.sin(data) * (av[indices] + bv[indices]).reshape(length, 2)
  np.testing.assert_allclose(fn(data, av, bv), expected)


def _position_dependent_program(case, length=12):
  original = _raw_program()
  callee, root = original.args
  _x, _y, _tmp, mapped, copied = root.args
  x, y = (p.buffer(name, dtypes.float64, (length,)) for name in ("x", "y"))
  tmp = p.buffer("tmp", dtypes.float64, (length,), address_space="private")
  mapped = p.for_(p.range_("i", 0, length), mapped.args[1:])
  copied = p.for_(p.range_("i", 0, length), copied.args[1:])
  extra = p.buffer("extra", dtypes.float64, (length,), address_space="private")
  i = p.var("i")
  read_x = p.load(p.view(x, [p.div(i, p.const_int(4)) if case in {"few_indices", "stage_indices"} else i]))
  if case == "fold_trip_consumer":
    read_x = p.load(p.view(x, [p.mod(i, p.const_int(3))]))
  if case == "shared_trip_consumer":
    read_x = p.load(p.view(x, [p.div(i, p.const_int(length // 2))]))
  read_extra = p.load(p.view(extra, [i]))
  declarations, upstream = [extra], []
  rhs = read_x
  if case == "fold_store":
    zeros = p.const_buffer("zeros", dtypes.float64, (length,), [0.0] * length)
    declarations.append(zeros)
    rhs = p.mul(rhs, p.load(p.view(zeros, [i])))
  if case == "signed_zero":
    zeros = p.const_buffer("zeros", dtypes.float64, (length,), [0.0] * (length // 2) + [-0.0] * (length // 2))
    declarations.append(zeros)
    rhs = p.div(p.const_float(1), p.load(p.view(zeros, [i])))
  if case == "fold_aliases":
    for offset in range(length):
      rhs = p.mul(rhs, p.sub(read_x, p.load(p.view(x, [p.const_int(offset)]))))
  if case == "mixed_trips":
    initialized = p.for_(p.range_("i", 0, length), [p.store(p.view(extra, [i]), p.const_float(0))])
    partial = p.for_(p.range_("i", 0, length // 2), [p.store(p.view(extra, [i]), p.load(p.view(tmp, [i])))])
    prepared = p.buffer("prepared", dtypes.float64, (length,), address_space="private")
    declarations.append(prepared)
    upstream = [initialized, mapped, partial, p.for_(p.range_("i", 0, length), [p.store(p.view(prepared, [i]), p.mul(read_extra, p.const_float(1)))])]
    read_extra = p.load(p.view(prepared, [i]))
  else:
    upstream = [p.for_(p.range_("i", 0, length), [p.store(p.view(extra, [i]), rhs)]), mapped]
  if case == "fold_consumer":
    read_extra = p.mul(read_extra, p.const_float(0))
  if case == "fold_trip_consumer":
    read_extra = p.mul(p.sub(read_extra, read_x), p.load(p.view(x, [p.const_int(length - 1)])))
  if case == "shared_trip_consumer":
    read_extra = p.add(read_extra, read_x)
  if case == "stage_indices":
    read_extra = p.mul(read_extra, p.sub(p.load(p.view(tmp, [i])), p.load(p.view(x, [p.div(i, p.const_int(6))]))))
  copied = p.for_(copied.args[0], [p.store(p.view(y, [i]), p.add(p.load(p.view(tmp, [i])), read_extra))])
  root = ProgramNode(root.op, (x, y, tmp, *declarations, *upstream, copied), root.attrs, dtype=root.dtype)
  return p.program([callee, root])


@pytest.mark.parametrize(
  "case",
  [
    "few_indices",
    "fold_store",
    "fold_aliases",
    "fold_consumer",
    "fold_trip_consumer",
    "shared_trip_consumer",
    "mixed_trips",
    "signed_zero",
    "stage_indices",
  ],
)
def test_position_dependent_cells_fold_and_fuse(tmp_path, case):
  import ctypes
  import subprocess

  from scaly.codegen.c import render_program_c
  from scaly.passes.program import optimize_program
  from scaly.utils.env import shared_lib_ext, shared_lib_flag
  from tests.compilers import c_compiler

  length = 12
  original = _position_dependent_program(case, length)
  fused = fuse_ranges(original)
  assert fused is not original
  assert not any(n.op == ProgramOp.CALL for n in walk_program(fused.args[-1]))
  verify_program(fused)
  optimized = optimize_program(original)
  _assert_fused(optimized)
  arg = sc.sym("x", length)
  signature = ConcreteFunction._from_exprs("root", [arg], [arg], ["x"], ["y"])
  source = tmp_path / "cells.c"
  library = tmp_path / ("cells" + shared_lib_ext())
  source.write_text(render_program_c(optimized, signature))
  subprocess.run([*c_compiler(), "-O2", "-fPIC", shared_lib_flag(), str(source), "-lm", "-o", str(library)], check=True)
  entry = ctypes.CDLL(str(library)).root
  ptr = ctypes.POINTER(ctypes.c_double)
  entry.argtypes = [ctypes.POINTER(ptr), ctypes.POINTER(ptr), ctypes.POINTER(ctypes.c_int), ptr, ctypes.c_int]
  entry.restype = ctypes.c_int
  data, out = np.linspace(-0.7, 0.8, length), np.empty(length)
  assert entry((ptr * 1)(data.ctypes.data_as(ptr)), (ptr * 1)(out.ctypes.data_as(ptr)), None, None, 0) == 0
  expected = data.copy()
  if case == "few_indices":
    expected += data[np.arange(length) // 4]
  elif case == "mixed_trips":
    expected[: length // 2] += data[: length // 2]
  elif case == "signed_zero":
    with np.errstate(divide="ignore"):
      expected += 1.0 / np.r_[np.zeros(length // 2), -np.zeros(length // 2)]
  elif case == "stage_indices":
    expected += data[np.arange(length) // 4] * (data - data[np.arange(length) // 6])
  elif case == "shared_trip_consumer":
    expected += 2 * data[np.arange(length) // (length // 2)]
  np.testing.assert_allclose(out, expected)


@pytest.mark.parametrize("case", ["fold_trip_consumer", "shared_trip_consumer"])
def test_deferred_load_matches_stage_load_without_position_rewrites(monkeypatch, case):
  fusion = importlib.import_module("scaly.passes.program.fuse_ranges")
  rewrite = fusion._rewrite
  counts = []
  for length in (64, 4096):
    calls = []

    def counted(*args):
      calls.append(None)
      return rewrite(*args)

    monkeypatch.setattr(fusion, "_rewrite", counted)
    original = _position_dependent_program(case, length)
    assert fuse_ranges(original) is not original
    counts.append(len(calls))
  assert counts[0] == counts[1]


def test_signed_zero_constants_preserve_atan2_quadrants():
  a = sc.sym("a", 1)
  stage = _function("zero_stage", [a], [a])
  x = sc.sym("x", 5)
  zeros = np.array([0.0, -0.0, 0.0, -0.0, 0.0])
  result = sc.const(zeros).atan2(sc.vmap(stage, 5)(x).vec())
  fn = _function("signed_zero", [x], [result])
  data = -np.ones(5)
  np.testing.assert_allclose(fn(data), np.arctan2(zeros, data))
