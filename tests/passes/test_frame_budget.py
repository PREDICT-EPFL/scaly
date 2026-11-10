"""Procedure frame estimates and largest-first workspace spilling."""

import pytest

from scaly.ir import program as p
from scaly.ir.program import ProgramNode, ProgramOp, walk_program
from scaly.ir.types import dtypes
from scaly.passes.program.pack_workspace import pack_workspace


def _procedure(name, sizes, calls=(), *, noinline=False, dtype=dtypes.float64):
  out = p.buffer("out", dtype, (1,))
  buffers = [p.buffer(f"t{i}", dtype, (size,), address_space="private") for i, size in enumerate(sizes)]
  zero = ProgramNode(ProgramOp.CONST_FLOAT if dtype.is_floating else ProgramOp.CONST_INT, (), {"value": 0.0 if dtype.is_floating else 0}, dtype)
  body = [*buffers, *(p.store(p.view(b, [p.const_int(0)]), zero) for b in buffers)]
  body += [p.store(p.view(out, [p.const_int(0)]), p.load(p.view(b, [p.const_int(0)]))) for b in buffers]
  body += [p.call(callee, [out]) for callee in calls]
  proc = p.proc(name, [out], body)
  return ProgramNode(proc.op, proc.args, {**proc.attrs, "noinline": noinline}, proc.dtype)


def _slots(proc):
  return [n for n in walk_program(proc) if n.op == ProgramOp.BUFFER and n.attrs.get("address_space") == "private"]


def test_many_small_buffers_spill_largest_first():
  proc = pack_workspace(p.program([_procedure("f", [900] * 10 + [400])])).args[0]
  slots = _slots(proc)
  assert sum("workspace_offset" in n.attrs for n in slots) == 2
  assert all(n.attrs["shape"] == (900,) for n in slots if "workspace_offset" in n.attrs)
  assert proc.attrs["sz_w"] == 1800
  assert proc.attrs["frame_bytes"] == 4096 + (8 * 900 + 400) * 8
  assert proc.attrs["frame_bytes"] <= 65536


def test_frame_counts_each_inline_call_and_spills_callee_buffers():
  leaf = _procedure("leaf", [900])
  middle = _procedure("middle", [], ["leaf"] * 3)
  root = _procedure("root", [500], ["middle"] * 3)
  packed = pack_workspace(p.program([leaf, middle, root]))
  assert packed.args[0].attrs["sz_w"] == 900
  assert packed.args[-1].attrs["sz_w"] == 900
  assert packed.args[-1].attrs["frame_bytes"] == 4096 + 500 * 8
  assert "workspace_offset" not in _slots(packed.args[-1])[0].attrs


def test_noinline_callee_has_its_own_frame_budget():
  packed = pack_workspace(p.program([_procedure("leaf", [900], noinline=True), _procedure("root", [500], ["leaf"] * 10)]))
  assert packed.args[0].attrs["frame_bytes"] == 4096 + 900 * 8
  assert packed.args[-1].attrs["frame_bytes"] == 4096 + 500 * 8
  assert packed.args[-1].attrs["sz_w"] == 0


def test_repeated_calls_inside_loop_are_counted_separately():
  leaf = _procedure("leaf", [900])
  root = _procedure("root", [], ["leaf"] * 10)
  pc = root.attrs["param_count"]
  root = p.proc("root", root.args[:pc], [p.for_(p.range_("i", 0, 2), root.args[pc:])])
  packed = pack_workspace(p.program([leaf, root]))
  assert packed.args[0].attrs["sz_w"] == 900
  assert packed.args[-1].attrs["frame_bytes"] == 4096


def test_zero_budget_spills_all_float64_buffers():
  proc = pack_workspace(p.program([_procedure("f", [900, 1])]), frame_budget=0).args[0]
  assert proc.attrs["sz_w"] == 901
  assert all("workspace_offset" in n.attrs for n in _slots(proc))
  assert proc.attrs["frame_bytes"] == 4096


@pytest.mark.parametrize("dtype", [dtypes.float32, dtypes.int64, dtypes.int32, dtypes.bool_])
def test_non_float64_buffers_stay_local_and_count_toward_frame(dtype):
  proc = pack_workspace(p.program([_procedure("f", [900, 1], dtype=dtype)]), frame_budget=0).args[0]
  assert proc.attrs["sz_w"] == 0
  assert all("workspace_offset" not in n.attrs for n in _slots(proc))
  assert proc.attrs["frame_bytes"] == 4096 + 901 * 8
