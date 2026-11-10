"""Carried recurrences and early-exit trip counts agree with a NumPy loop after every pass."""

import ctypes
import subprocess

import numpy as np
import pytest

from scaly.codegen.c import _includes, _render_raw_callee
from scaly.ir import program as p
from scaly.ir.program import ProgramNode, ProgramOp, walk_program
from scaly.ir.program_spec import verify_program
from scaly.ir.types import dtypes
from scaly.passes.program import optimize_program
from tests.compilers import c_compiler


@pytest.mark.parametrize("dialect", ["c", "gnu"])
@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("length,stop_at", [(0, 0), (1, 0), (1, 1), (6, 0), (6, 3), (6, 6)])
def test_carried_recurrence_and_trip_count_match_numpy(tmp_path, dialect, dynamic, length, stop_at):
  flags = p.buffer("flags", dtypes.bool_, (max(1, length),))
  trip = p.buffer("trip", dtypes.int64, (1,))
  out = p.buffer("out", dtypes.float64, (1,))
  count = p.buffer("count", dtypes.int64, (1,))
  at = lambda buf: p.view(buf, [p.const_int(0)])
  loop = p.for_(
    p.range_("i", 0, p.load(at(trip)) if dynamic else length, carried=True),
    [p.block(p.break_if(p.load(p.view(flags, [p.var("i")])))), p.store(at(out), p.add(p.mul(p.load(at(out)), p.const_float(0.5)), p.const_float(1)))],
    exit_var=True,
  )
  raw = p.proc("recurrence", [flags, trip, out, count], [p.store(at(out), p.const_float(0)), loop, p.store(at(count), p.var("i"))])
  raw = ProgramNode(raw.op, raw.args, {**raw.attrs, "input_count": 2, "lowering": "scalar", "scalarize_mode": "procedure"}, raw.dtype)

  def observe(_name, stage):
    verify_program(stage)
    kept = [n for n in walk_program(stage) if n.op == ProgramOp.FOR and n.args[0].attrs.get("carried")]
    assert len(kept) == 1 and kept[0].attrs["exit_var"] is True
    assert any(n.op == ProgramOp.BREAK_IF for n in walk_program(kept[0]))

  prog = optimize_program(p.program([raw]), observe=observe, lanes=2)
  source = "\n".join([*_includes(dialect=dialect, prog=prog), *_render_raw_callee(prog.args[0], dialect=dialect)])
  assert "long long i = 0;\n  for (i = 0;" in source
  assert "if (flags[i]) break;" in source
  path, library = tmp_path / "recurrence.c", tmp_path / "recurrence.so"
  path.write_text(
    source + "\nvoid run(bool *flags, long long *trip, double *out, long long *count) { recurrence_raw(flags, trip, out, count, NULL); }\n"
  )
  subprocess.run([*c_compiler(), "-shared", "-fPIC", "-std=c99", "-O2", str(path), "-lm", "-o", str(library)], check=True, capture_output=True)
  run = ctypes.CDLL(str(library)).run
  run.argtypes = [
    ctypes.POINTER(ctypes.c_bool),
    ctypes.POINTER(ctypes.c_longlong),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_longlong),
  ]
  values = np.arange(max(1, length)) >= stop_at
  result, trips = ctypes.c_double(-1), ctypes.c_longlong(-1)
  run(values.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)), ctypes.byref(ctypes.c_longlong(length)), ctypes.byref(result), ctypes.byref(trips))
  expected, n = np.float64(0), 0
  for i in range(length):
    if values[i]:
      break
    expected = expected * 0.5 + 1
    n += 1
  assert result.value == expected and trips.value == n
