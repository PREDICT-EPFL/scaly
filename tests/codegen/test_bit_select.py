"""BIT_SELECT in lane code: a word that differs from lane to lane is read per lane, not broadcast."""

from __future__ import annotations

import ctypes
import subprocess

import numpy as np
import pytest

from scaly.codegen.c import _includes, _render_raw_callee
from scaly.ir import program as p
from scaly.ir.program import ProgramOp, walk_program
from scaly.ir.types import dtypes
from scaly.passes.program import optimize_program

COUNT = 37  # not a multiple of any lane width, so the last chunk is partial


@pytest.mark.parametrize("lanes", [1, 2, 4, 8])
def test_lane_varying_word_matches_numpy(tmp_path, lanes):
  words = np.random.default_rng(lanes).integers(0, 256, size=COUNT)
  x = p.buffer("x", dtypes.float64, (COUNT,))
  y = p.buffer("y", dtypes.float64, (COUNT,))
  i = p.var("i")
  table = p.const_buffer("words", dtypes.uint8, (COUNT,), [int(w) for w in words])
  value = p.bit_select(p.load(p.view(table, [i])), p.mod(i, p.const_int(5)), p.load(p.view(x, [i])))
  prog = p.program([p.proc("kernel", [x, y], [table, p.for_(p.range_("i", 0, COUNT), [p.store(p.view(y, [i]), value)])])])
  prog = optimize_program(prog, lanes="auto")
  assert any(
    n.op == ProgramOp.FOR and "vector_helper" in n.attrs and any(m.op == ProgramOp.BIT_SELECT for m in walk_program(n)) for n in walk_program(prog)
  )

  source = "\n".join([*_includes(dialect="gnu", prog=prog), *_render_raw_callee(prog.args[0], dialect="gnu")])
  source += "\nvoid run(double *x, double *y) { double w[1]; kernel_raw(x, y, w); }\n"
  path, library = tmp_path / "kernel.c", tmp_path / "kernel.so"
  path.write_text(source)
  subprocess.run(
    ["cc", "-shared", "-fPIC", "-O3", "-march=native", "-fno-math-errno", "-std=c99", f"-DSCALY_LANES={lanes}", str(path), "-lm", "-o", str(library)],
    check=True,
    capture_output=True,
  )
  run = ctypes.CDLL(str(library)).run
  run.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
  xs = np.random.default_rng(10 + lanes).normal(size=COUNT)
  out = np.full(COUNT, 8765.0)
  run(xs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
  expected = ((words >> (np.arange(COUNT) % 5)) & 1) * xs
  np.testing.assert_array_equal(out, expected)
  np.testing.assert_array_equal(np.signbit(out), np.signbit(expected))
