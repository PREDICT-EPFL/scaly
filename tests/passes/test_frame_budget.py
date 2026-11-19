"""Procedure frame estimates and largest-first workspace spilling."""

import pytest

from scaly.ir import program as p
from scaly.ir.program import ProgramNode, ProgramOp, walk_program
from scaly.ir.types import dtypes
from scaly.passes.program.pack_workspace import pack_workspace


def _procedure(name, sizes, calls=(), *, noinline=False, dtype=dtypes.float64):
  out = p.buffer("out", dtype, (1,))
  buffers = [p.buffer(f"t{i}", dtype, (size,), address_space="private") for i, size in enumerate(sizes)]
  zero = ProgramNode(ProgramOp.CONST_FLOAT if dtype.is_floating else ProgramOp.CONST_INT, (), {"value": 0.0 if dtype.is_floating else 0}, dtype=dtype)
  body = [*buffers, *(p.store(p.view(b, [p.const_int(0)]), zero) for b in buffers)]
  body += [p.store(p.view(out, [p.const_int(0)]), p.load(p.view(b, [p.const_int(0)]))) for b in buffers]
  body += [p.call(callee, [out]) for callee in calls]
  proc = p.proc(name, [out], body)
  return ProgramNode(proc.op, proc.args, {**proc.attrs, "noinline": noinline}, dtype=proc.dtype)


def _slots(proc):
  return [n for n in walk_program(proc) if n.op == ProgramOp.BUFFER and n.attrs.get("address_space") == "private"]


def test_many_small_buffers_spill_largest_first():
  proc = pack_workspace(p.program([_procedure("f", [900] * 10 + [400])])).args[0]
  slots = _slots(proc)
  assert sum("workspace_offset" in n.attrs for n in slots) == 2
  assert all(n.attrs["shape"] == (900,) for n in slots if "workspace_offset" in n.attrs)
  assert proc.attrs["workspace_layout"] == ((dtypes.float64, 2 * 904),)
  assert proc.attrs["frame_bytes"] == 4096 + (8 * 900 + 400) * 8
  assert proc.attrs["frame_bytes"] <= 65536


def test_frame_counts_each_inline_call_and_spills_callee_buffers():
  leaf = _procedure("leaf", [900])
  middle = _procedure("middle", [], ["leaf"] * 3)
  root = _procedure("root", [500], ["middle"] * 3)
  packed = pack_workspace(p.program([leaf, middle, root]))
  assert packed.args[0].attrs["workspace_layout"] == ((dtypes.float64, 904),)
  assert packed.args[-1].attrs["workspace_layout"] == ((dtypes.float64, 904),)
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
  assert packed.args[0].attrs["workspace_layout"] == ((dtypes.float64, 904),)
  assert packed.args[-1].attrs["frame_bytes"] == 4096


@pytest.mark.parametrize("dtype", [dtypes.float64, dtypes.float32, dtypes.int64, dtypes.bool_])
def test_zero_budget_spills_every_buffer(dtype):
  proc = pack_workspace(p.program([_procedure("f", [900, 1], dtype=dtype)]), frame_budget=0).args[0]
  slots = sorted(_slots(proc), key=lambda n: n.attrs["workspace_offset"])
  first, second = slots
  assert first.attrs["workspace_offset"] == 0
  assert second.attrs["workspace_offset"] * dtype.itemsize % 64 == 0
  assert second.attrs["workspace_offset"] >= 900
  assert proc.attrs["frame_bytes"] == 0


def test_workspace_spills_the_largest_slot_in_bytes_first():
  out = p.buffer("out", dtypes.float64, (1,))
  buffers = [
    p.buffer(f"t{i}", dtype, (size,), address_space="private")
    for i, (dtype, size) in enumerate(((dtypes.float64, 900), (dtypes.float32, 1000), (dtypes.int32, 3)))
  ]
  zero = [
    ProgramNode(ProgramOp.CONST_FLOAT, (), {"value": 0.0}, dtype=dtypes.float64),
    ProgramNode(ProgramOp.CONST_FLOAT, (), {"value": 0.0}, dtype=dtypes.float32),
    p.const_int(0),
  ]
  body = [*buffers, *(p.store(p.view(b, [p.const_int(0)]), z) for b, z in zip(buffers, zero, strict=True))]
  body.append(p.store(p.view(out, [p.const_int(0)]), p.load(p.view(buffers[0], [p.const_int(0)]))))
  prog = p.program([p.proc("f", [out], body)])
  proc = pack_workspace(prog, frame_budget=8196).args[0]
  assert [n.dtype for n in _slots(proc) if "workspace_offset" in n.attrs] == [dtypes.float64]
  assert proc.attrs["sz_w"] == 7232 and proc.attrs["frame_bytes"] == 4096 + 4000 + 12
