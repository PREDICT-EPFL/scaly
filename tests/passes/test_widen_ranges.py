"""Lane splitting preserves independent range provenance and bounds live scalar values."""

import pytest

from scaly.ir import program as p
from scaly.ir.program import ProgramNode, ProgramOp, RangeKind
from scaly.ir.program_spec import verify_program
from scaly.ir.types import dtypes
from scaly.ir.program import walk_program
from scaly.passes.program.widen_ranges import _peak_live, widen_ranges


def _program(count=9):
  x, y = (p.buffer(name, dtypes.float64, (count,)) for name in ("x", "y"))
  i = p.var("stage")
  rng = p.range_("stage", 0, count)
  rng = ProgramNode(rng.op, rng.args, {**rng.attrs, "mapped": True}, rng.dtype)
  return p.program([p.proc("mapped", [x, y], [p.for_(rng, [p.store(p.view(y, [i]), p.load(p.view(x, [i])))])])])


@pytest.mark.parametrize("lanes", ["auto", 1, 2, 4, 8])
def test_explicit_lane_range_and_clamped_stage(lanes):
  result = widen_ranges(_program(17), lanes=lanes)
  verify_program(result)
  ranges = [n for n in walk_program(result) if n.op == ProgramOp.RANGE]
  assert sum(n.attrs["kind"] == RangeKind.VECTOR for n in ranges) == 1
  assert any(n.op == ProgramOp.MINIMUM for n in walk_program(result))
  assert any(n.attrs.get("vector_count") == 17 for n in walk_program(result))


@pytest.mark.parametrize(("count", "cap"), [(2, 2), (3, 4), (5, 8)])
def test_lane_cap_does_not_exceed_rounded_trip_count(count, cap):
  result = widen_ranges(_program(count))
  verify_program(result)
  vector = next(n for n in walk_program(result) if n.op == ProgramOp.RANGE and n.attrs["kind"] == RangeKind.VECTOR)
  assert vector.attrs["lane_caps"] == (cap,) * 4
  assert vector.attrs["lane_cap"] == cap


def test_empty_and_unit_ranges_stay_scalar():
  for count in (0, 1):
    prog = _program(count)
    assert widen_ranges(prog) is prog


def test_peak_liveness_uses_last_use_not_total_definitions():
  body = [p.assign("v0", p.const_float(1), declare=True)]
  for i in range(1, 100):
    body.append(p.assign(f"v{i}", p.add(p.var(f"v{i - 1}", dtypes.float64), p.const_float(1)), declare=True))
  assert _peak_live(body) == 2


def test_invalid_lane_option_rejected():
  with pytest.raises(ValueError, match="lanes"):
    widen_ranges(_program(), lanes=3)  # ty: ignore[invalid-argument-type]


def test_overlapping_contiguous_stores_are_not_independent():
  x = p.buffer("x", dtypes.float64, (10,))
  i = p.var("i")
  loop = p.for_(p.range_("i", 0, 9), [p.store(p.view(x, [i]), p.const_float(1)), p.store(p.view(x, [p.add(i, p.const_int(1))]), p.const_float(2))])
  prog = p.program([p.proc("overlap", [x], [loop])])
  assert widen_ranges(prog) is prog


@pytest.mark.parametrize("count", [2, 3, 5, 9])
def test_register_cap_uses_peak_live_values(count):
  proc = _program(count).args[0]
  x, y, loop = proc.args
  declarations = [p.assign(f"v{i}", p.load(p.view(x, [p.const_int(0)])), declare=True) for i in range(300)]
  value = p.var("v0", dtypes.float64)
  for i in range(1, 300):
    value = p.add(value, p.var(f"v{i}", dtypes.float64))
  body = [*declarations, p.store(p.view(y, [p.var("stage")]), value)]
  prog = p.program([p.proc("pressure", [x, y], [p.for_(loop.args[0], body)])])
  result = widen_ranges(prog)
  vector = next(n for n in walk_program(result) if n.op == ProgramOp.RANGE and n.attrs["kind"] == RangeKind.VECTOR)
  assert vector.attrs["peak_live"] == 300
  assert vector.attrs["lane_caps"] == (1, 1, 1, 2)


def test_integer_value_loops_stay_scalar():
  x, y = (p.buffer(name, dtypes.int64, (9,)) for name in ("x", "y"))
  i = p.var("i")
  loop = p.for_(p.range_("i", 0, 9), [p.store(p.view(y, [i]), p.add(p.load(p.view(x, [i])), p.const_int(1)))])
  prog = p.program([p.proc("integer_values", [x, y], [loop])])
  assert widen_ranges(prog) is prog


def test_reduction_does_not_stage_an_aliased_accumulator():
  y = p.buffer("y", dtypes.float64, (1,))
  alias = ProgramNode(ProgramOp.BUFFER, (), {**y.attrs, "name": "alias", "alias_of": "y", "alias_offset": 0}, y.dtype)
  target = p.view(y, [p.const_int(0)])
  rhs = p.add(p.load(target), p.load(p.view(alias, [p.const_int(0)])))
  loop = p.for_(p.range_("i", 0, 9, kind=RangeKind.REDUCE), [p.store(target, rhs)])
  prog = p.program([p.proc("aliased_sum", [y], [alias, loop])])
  assert widen_ranges(prog) is prog


def test_store_schedule_reduces_live_values_without_reordering_memory():
  from scaly.passes.program.widen_ranges import _interleave_stores

  x, y = p.buffer("x", dtypes.float64, (1,)), p.buffer("y", dtypes.float64, (20,))
  base = p.var("base", dtypes.float64)
  body = [p.assign("base", p.load(p.view(x, [p.const_int(0)])), declare=True)]
  body += [p.assign(f"v{i}", p.add(base, p.const_float(i)), declare=True) for i in range(20)]
  body += [p.store(p.view(y, [p.const_int(i)]), p.var(f"v{i}", dtypes.float64)) for i in range(20)]
  scheduled = _interleave_stores(body, {})
  assert _peak_live(body) == 21
  assert _peak_live(scheduled) == 2
  assert [n for n in scheduled if n.op == ProgramOp.STORE] == body[-20:]
  assert scheduled[0] is body[0]


def test_store_schedule_refuses_alias_reads_and_mutable_scalars():
  from scaly.passes.program.widen_ranges import _interleave_stores

  x = p.buffer("x", dtypes.float64, (1,))
  y = p.buffer("alias", dtypes.float64, (1,))
  assignment = p.assign("v", p.load(p.view(x, [p.const_int(0)])), declare=True)
  store = p.store(p.view(y, [p.const_int(0)]), p.var("v", dtypes.float64))
  body = [assignment, store]
  assert _interleave_stores(body, {"alias": "x"}) is body
  mutable = [p.assign("v", p.const_float(1)), store]
  assert _interleave_stores(mutable, {}) is mutable


def test_scheduled_index_is_not_an_independent_contiguous_output():
  y = p.buffer("y", dtypes.float64, (2,))
  i = p.var("i")
  offset = p.assign("offset", p.mul(p.div(i, p.const_int(2)), p.const_int(-2)), declare=True)
  store = p.store(p.view(y, [p.add(p.var("offset"), i)]), p.const_float(1))
  prog = p.program([p.proc("noninjective", [y], [p.for_(p.range_("i", 0, 8), [offset, store])])])
  assert widen_ranges(prog) is prog


def test_local_alias_of_external_buffer_stays_scalar():
  x, y = p.buffer("x", dtypes.float64, (9,)), p.buffer("y", dtypes.float64, (9,))
  alias = ProgramNode(ProgramOp.BUFFER, (), {**x.attrs, "name": "local_alias", "alias_of": "x", "alias_offset": 0}, x.dtype)
  i = p.var("stage")
  rng = p.range_("stage", 0, 9)
  rng = ProgramNode(rng.op, rng.args, {**rng.attrs, "mapped": True}, rng.dtype)
  prog = p.program([p.proc("external_alias", [x, y], [p.for_(rng, [alias, p.store(p.view(y, [i]), p.load(p.view(alias, [i])))])])])
  assert widen_ranges(prog, lanes=4) is prog
