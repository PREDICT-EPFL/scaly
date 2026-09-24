"""Periodic constant folding preserves exact values and pointer-visible storage."""

import numpy as np
import pytest

from scaly.ir import program as p
from scaly.ir.program import ProgramNode, ProgramOp
from scaly.ir.types import dtypes
from scaly.ir.program import walk_program
from scaly.passes.program.fold_arith import fold_arith
from scaly.passes.program.fold_tiles import fold_tiles


def _program(values, dtype=dtypes.float64):
  table = p.const_buffer("tile", dtype, (len(values),), values)
  out = p.buffer("out", dtype, (len(values),))
  loop = p.for_(p.range_("i", 0, len(values)), [p.store(p.view(out, [p.var("i")]), p.load(p.view(table, [p.var("i")])))])
  return p.program([p.proc("tile_test", [out], [table, loop])])


@pytest.mark.parametrize("values,period", [([2.0, 3.0] * 6, 2), ([1.0, 0.0, 0.0] * 5, 3), ([1.0, 2.0, 1.0], 3), ([1.0] * 9, 0)])
def test_periodic_tables(values, period):
  result = fold_tiles(_program(values))
  tables = [n for n in walk_program(result) if n.op == ProgramOp.BUFFER and "values" in n.attrs]
  assert [len(n.attrs["values"]) for n in tables] == ([period] if period else [])
  if tables:
    actual = np.asarray(tables[0].attrs["values"])[np.arange(len(values)) % period]
    np.testing.assert_array_equal(actual, values)
  assert any(n.op == ProgramOp.MOD for n in walk_program(result)) == (0 < period < len(values))


def test_integer_tile_preserves_large_values():
  values = [2**60 + 1, 2**60 + 2] * 5
  result = fold_tiles(_program(values, dtypes.int64))
  table = next(n for n in walk_program(result) if n.op == ProgramOp.BUFFER and "values" in n.attrs)
  assert table.attrs["values"] == tuple(values[:2])


@pytest.mark.parametrize("escape", ["alias", "call", "store"])
def test_storage_escapes_keep_original_table(escape):
  prog = _program([1.0, 2.0] * 4)
  proc = prog.args[0]
  out, table, loop = proc.args
  if escape == "alias":
    extra = ProgramNode(ProgramOp.BUFFER, (), {**table.attrs, "name": "alias", "alias_of": "tile"}, table.dtype)
  elif escape == "call":
    extra = p.call("other", [table])
  else:
    extra = p.store(p.view(table, [p.const_int(0)]), p.const_float(3))
  modified = p.program([p.proc("escape", [out], [table, extra, loop])])
  assert fold_tiles(modified) is modified


def test_signed_zero_tiles_and_uniform_folding_preserve_bits():
  values = [0.0, -0.0] * 4
  result = fold_arith(fold_tiles(_program(values)))
  table = next(n for n in walk_program(result) if n.op == ProgramOp.BUFFER and "values" in n.attrs)
  assert np.asarray(table.attrs["values"]).view(np.uint64).tolist() == [0, 1 << 63]
  negative = fold_arith(fold_tiles(_program([-0.0] * 7)))
  assert any(n.op == ProgramOp.LOAD for n in walk_program(negative))


def test_nan_payloads_compare_by_bits():
  values = np.asarray([0x7FF8000000000001, 0x7FF8000000000002] * 4, dtype=np.uint64).view(np.float64).tolist()
  result = fold_tiles(_program(values))
  table = next(n for n in walk_program(result) if n.op == ProgramOp.BUFFER and "values" in n.attrs)
  assert np.asarray(table.attrs["values"]).view(np.uint64).tolist() == [0x7FF8000000000001, 0x7FF8000000000002]
