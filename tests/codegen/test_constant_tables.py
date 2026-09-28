from __future__ import annotations

import numpy as np

import scaly as sc


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
  fn = sc.Function._from_exprs("table_bits", [index], [sc.take(sc.const(table), index, in_range=True)], ["i"], ["v"])
  got = fn(np.arange(table.size))
  assert got.tobytes() == table.tobytes()
