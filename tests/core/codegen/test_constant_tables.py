from __future__ import annotations

import re

import numpy as np
import pytest

import scaly as sc
from scaly.codegen import render_c_source


def test_a_constant_table_reaches_the_c_bit_for_bit() -> None:
  """Tables are printed with 17 significant digits, which round-trips every double: interpolation
  parity at 1e-14 rests on the table in the C being the table in NumPy."""
  rng = np.random.default_rng(0)
  table = np.concatenate(
    [
      rng.normal(size=200) * 10.0 ** rng.integers(-300, 300, 200),
      [np.pi, -np.e, 5e-324, -2.2250738585072014e-308, np.finfo(np.float64).max, 0.1, 1.0 / 3.0, -0.0, 0.0],
    ]
  )
  index = sc.sym("i", table.size, dtype="int64")
  fn = sc.Function.from_exprs("table_bits", [index], [sc.take(sc.const(table), index, in_range=True)], ["i"], ["v"])
  got = fn(np.arange(table.size))
  assert got.tobytes() == table.tobytes()


def _declared(source: str, c_type: str, entries: int) -> list[str]:
  """What stands between each ``entries``-long constant table's size and its initializer."""
  return re.findall(rf"static const {c_type} \w+\[{entries}\]([^=]*)=", source)


@pytest.mark.parametrize(
  ("target", "entries", "aligned"),
  [
    ("apple-m3", 16384, False),  # 128 KiB, the M3's level-1 data cache: it fits
    ("apple-m3", 16385, True),
    ("generic", 4096, False),  # 32 KiB
    ("generic", 4097, True),
    ("x86-64-v3", 4097, True),
    (sc.Target.preset("generic", rounding="portable"), 4097, False),  # the reference machine's cache
    (sc.Target.preset("generic", rounding="portable"), 16385, True),
  ],
  ids=["apple-m3-fits", "apple-m3", "generic-fits", "generic", "x86-64-v3", "portable-fits", "portable"],
)
def test_a_table_of_values_too_large_for_the_level_1_cache_starts_on_a_multiple_of_64(target, entries: int, aligned: bool) -> None:
  """The C compiler aligns a table to its element, so the vector loads of a constant matrix
  straddled two cache lines every few loads, and a product reading the matrix from the level-2
  cache ran at half the rate of the same product with the matrix an input. A table of
  floating-point values larger than the target's level-1 data cache is declared aligned to 64
  bytes. One that fits, and a table of indices of any length, is left where the compiler puts it:
  aligned, those measured no faster and two kernels slower."""
  rng = np.random.default_rng(entries)
  index, x = sc.sym("i", 3, dtype="int64"), sc.sym("x", entries)
  values = sc.Function.from_exprs(
    f"aligned_values_{entries}", [index], [sc.take(sc.const(rng.standard_normal(entries)), index, in_range=True)], ["i"], ["v"]
  )
  indices = sc.Function.from_exprs(f"aligned_indices_{entries}", [x], [sc.take(x, rng.permutation(entries)).block()], ["x"], ["v"])
  assert _declared(render_c_source(values, target=target), "double", entries) == [" __attribute__((aligned(64))) " if aligned else " "]
  assert _declared(render_c_source(indices, target=target), "int64_t", entries) == [" "]


def test_a_large_constant_operand_keeps_its_alignment_through_a_call_and_computes_the_product() -> None:
  """The mark is made where the constant is lowered and has to reach the C through every pass,
  in a callee as in the entry: a product with a constant ``b`` of 320 KB, called from a map."""
  rng = np.random.default_rng(5)
  bv = rng.standard_normal((200, 200))
  a = sc.sym("a", 200)
  row = sc.Function.from_exprs("aligned_row", [a], [a @ sc.const(bv)], ["a"], ["y"])
  rows = sc.sym("rows", 3 * 200)
  fn = sc.Function.from_exprs("aligned_rows", [rows], [sc.vmap(row, 3, [(rows, 0, 200)])], ["rows"], ["y"])
  for target in ("apple-m3", "generic"):
    assert _declared(render_c_source(fn, target=target), "double", 200 * 200) == [" __attribute__((aligned(64))) "], target
  av = rng.standard_normal((3, 200))
  np.testing.assert_allclose(fn(av.reshape(-1)).reshape(3, 200), av @ bv, rtol=1e-12, atol=1e-12)
