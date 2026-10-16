"""pack_masks stores 0/1 multiplier tables of a direction x position loop as bits and splits the loop."""

from __future__ import annotations

import ctypes
import subprocess

import numpy as np
import pytest

from scaly.ir import program as p
from scaly.ir.program import ProgramOp, walk_program
from scaly.ir.types import dtypes
from scaly.codegen.c import _includes, _render_raw_callee
from scaly.passes.program import optimize_program
from scaly.passes.program.pack_masks import pack_masks

DIRECTIONS, STAGES, ENTRIES = 7, 9, 5  # positions = 45, not a multiple of 8, so the split loop has a tail


def _masks(seed: int, count: int = 2, stages: int = STAGES) -> list[np.ndarray]:
  """Irregular 0/1 masks: each entry belongs to at most one direction, as a coloring seed's do."""
  rng = np.random.default_rng(seed)
  owner = rng.integers(-1, DIRECTIONS, size=(count, stages * ENTRIES))
  return [(np.arange(DIRECTIONS)[:, None] == owner[c][None, :]).astype(np.float64) for c in range(count)]


def _program(masks: list[np.ndarray], *, index=None, extra_read=False, bare_use=False, second_loop=False):
  """``y[i] = sum_c mask_c[i] * x[c * positions + f(i)]`` with the selector's own ``((i / E) % S) * E + i % E`` index."""
  stages = masks[0].shape[1] // ENTRIES
  positions = stages * ENTRIES
  length = DIRECTIONS * positions
  x = p.buffer("x", dtypes.float64, (len(masks) * positions,))
  y = p.buffer("y", dtypes.float64, (length + (1 if extra_read else 0),))
  i = p.var("i")
  if index is None:
    index = p.add(p.mul(p.mod(p.div(i, p.const_int(ENTRIES)), p.const_int(stages)), p.const_int(ENTRIES)), p.mod(i, p.const_int(ENTRIES)))
  tables, total = [], None
  for c, mask in enumerate(masks):
    table = p.const_buffer(f"mask{c}", dtypes.float64, (DIRECTIONS, stages, ENTRIES), mask.reshape(-1).tolist())
    tables.append(table)
    term = p.mul(p.load(p.view(table, [i])), p.load(p.view(x, [p.add(p.const_int(c * positions), index)])))
    total = term if total is None else p.add(total, term)
  assert total is not None
  if bare_use:
    total = p.add(total, p.load(p.view(tables[0], [i])))  # the same mask[i] once more, outside a product
  body = [*tables, p.for_(p.range_("i", 0, length), [p.store(p.view(y, [i]), total)])]
  if extra_read:
    body.append(p.store(p.view(y, [p.const_int(length)]), p.load(p.view(tables[0], [p.const_int(3)]))))
  if second_loop:
    # The same interned mask[i] view, read by another loop over the same variable name.
    z = p.buffer("z", dtypes.float64, (length,))
    body.append(p.for_(p.range_("i", 0, length), [p.store(p.view(z, [i]), p.load(p.view(tables[0], [i])))]))
    return p.program([p.proc("kernel", [x, y, z], body)])
  return p.program([p.proc("kernel", [x, y], body)])


def _assert_identical(result: np.ndarray, expected: np.ndarray) -> None:
  """Equal as IEEE values, signed zeros and NaNs included, as the multiply by the table gave them.

  A NaN's sign is left out: it depends on which operand of an addition the CPU propagates.
  """
  np.testing.assert_array_equal(result, expected)
  number = ~np.isnan(expected)
  np.testing.assert_array_equal(np.signbit(result[number]), np.signbit(expected[number]))


def _reference(masks: list[np.ndarray], x: np.ndarray) -> np.ndarray:
  positions = masks[0].shape[1]
  terms = [mask * x[c * positions : (c + 1) * positions][None, :] for c, mask in enumerate(masks)]
  return sum(terms[1:], terms[0]).reshape(-1)  # starting from 0 would turn -0.0 into +0.0


def _run(prog, tmp_path, lanes: int, x: np.ndarray, out_size: int) -> np.ndarray:
  source = "\n".join([*_includes(dialect="gnu", prog=prog), *_render_raw_callee(prog.args[0], dialect="gnu")])
  workspace = int(prog.args[0].attrs.get("sz_w", 0))
  source += f"\nvoid run(double *x, double *y) {{ double w[{max(workspace, 1)}]; kernel_raw(x, y, w); }}\n"
  path, library = tmp_path / f"kernel{lanes}.c", tmp_path / f"kernel{lanes}.so"
  path.write_text(source)
  subprocess.run(
    ["cc", "-shared", "-fPIC", "-O3", "-march=native", "-fno-math-errno", "-std=c99", f"-DSCALY_LANES={lanes}", str(path), "-lm", "-o", str(library)],
    check=True,
    capture_output=True,
  )
  run = ctypes.CDLL(str(library)).run
  run.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
  x = np.ascontiguousarray(x, dtype=np.float64)
  y = np.full(out_size, 8765.0)
  run(x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
  return y


def test_masks_become_bytes_and_the_loop_splits_into_directions_and_lane_blocks():
  masks = _masks(0)
  packed = pack_masks(_program(masks))
  tables = {n.attrs["name"]: n for n in walk_program(packed) if n.op == ProgramOp.BUFFER and "values" in n.attrs}
  row = -(-STAGES * ENTRIES // 8)
  assert {name: (t.dtype, t.attrs["shape"]) for name, t in tables.items()} == {f"mask{c}": (dtypes.uint8, (DIRECTIONS * row,)) for c in range(2)}
  for c, mask in enumerate(masks):
    stored = np.unpackbits(np.asarray(tables[f"mask{c}"].attrs["values"], dtype=np.uint8).reshape(DIRECTIONS, row), axis=1, bitorder="little")
    np.testing.assert_array_equal(stored[:, : STAGES * ENTRIES], mask)
    assert not stored[:, STAGES * ENTRIES :].any()
  assert sum(n.op == ProgramOp.BIT_SELECT for n in walk_program(packed)) == 4  # two masks, in the block loop and the tail
  assert not any(
    n.op == ProgramOp.MUL and any(a.op == ProgramOp.LOAD and a.args[0].attrs["buffer"].startswith("mask") for a in n.args)
    for n in walk_program(packed)
  )


@pytest.mark.parametrize("lanes", [1, 2, 4, 8])
def test_split_loop_matches_numpy_at_every_lane_width(tmp_path, lanes):
  masks = _masks(lanes)
  prog = optimize_program(_program(masks), lanes="auto")
  assert any(n.op == ProgramOp.BIT_SELECT for n in walk_program(prog))
  assert any(
    n.op == ProgramOp.FOR and "vector_helper" in n.attrs and any(m.op == ProgramOp.BIT_SELECT for m in walk_program(n)) for n in walk_program(prog)
  )
  x = np.random.default_rng(10 + lanes).normal(size=2 * STAGES * ENTRIES)
  expected = _reference(masks, x)
  result = _run(prog, tmp_path, lanes, x, expected.size)
  _assert_identical(result, expected)


def test_masks_with_the_same_lane_bits_share_one_bit_vector():
  prog = optimize_program(_program(_masks(5)), lanes="auto")
  source = "\n".join(_render_raw_callee(prog.args[0], dialect="gnu"))
  # Both masks test the same lane bits, so each helper builds the bit vector once.
  assert source.count("_vec_i64 kb;") == 2 and "kb_2" not in source


@pytest.mark.parametrize(
  "case", ["not_zero_one", "read_elsewhere", "read_by_another_loop", "used_outside_product", "index_not_periodic", "negative_dividend"]
)
def test_ineligible_tables_and_loops_are_left_alone(case):
  masks = _masks(3)
  if case == "not_zero_one":
    # The loop multiplies by both masks, and one of them is not 0/1, so the loop is not split.
    masks[0][0, 0] = 2.0
    prog = _program(masks)
  elif case == "read_elsewhere":
    prog = _program(masks, extra_read=True)
  elif case == "read_by_another_loop":
    prog = _program(masks, second_loop=True)
  elif case == "used_outside_product":
    prog = _program(masks, bare_use=True)
  elif case == "negative_dividend":
    # (i - 45) % 45 repeats every 45 entries under Python's floor, but C reads x[i - 45] for i < 45.
    i = p.var("i")
    prog = _program(masks, index=p.mod(p.sub(i, p.const_int(STAGES * ENTRIES)), p.const_int(STAGES * ENTRIES)))
  else:
    prog = _program(masks, index=p.var("i"))
  assert pack_masks(prog) is prog


def test_scalar_rendering_matches_numpy(tmp_path):
  masks = _masks(21)
  prog = optimize_program(_program(masks), lanes=1)
  assert any(n.op == ProgramOp.BIT_SELECT for n in walk_program(prog))
  assert not any(n.op == ProgramOp.FOR and "vector_helper" in n.attrs for n in walk_program(prog))
  x = np.random.default_rng(22).normal(size=2 * STAGES * ENTRIES)
  expected = _reference(masks, x)
  _assert_identical(_run(prog, tmp_path, 1, x, expected.size), expected)


@pytest.mark.parametrize("stages", [1, 8], ids=["tail_only", "no_tail"])
@pytest.mark.parametrize("lanes", [1, 4])
def test_small_tables_are_packed_and_match_numpy(tmp_path, stages, lanes):
  # 5 positions leave no full block of eight; 40 positions leave no tail.
  masks = _masks(30 + stages, stages=stages)
  prog = optimize_program(_program(masks), lanes="auto")
  assert any(n.op == ProgramOp.BIT_SELECT for n in walk_program(prog))
  x = np.random.default_rng(40 + stages).normal(size=2 * stages * ENTRIES)
  expected = _reference(masks, x)
  _assert_identical(_run(prog, tmp_path, lanes, x, expected.size), expected)


@pytest.mark.parametrize("lanes", ["scalar", 1, 2, 4, 8])
def test_cleared_bits_keep_ieee_products_of_nonfinite_and_negative_values(tmp_path, lanes):
  masks = _masks(50)
  prog = optimize_program(_program(masks), lanes=1 if lanes == "scalar" else "auto")
  lanes = 1 if lanes == "scalar" else lanes
  assert any(n.op == ProgramOp.BIT_SELECT for n in walk_program(prog))
  x = -np.abs(np.random.default_rng(51).normal(size=2 * STAGES * ENTRIES))  # 0 * -x is -0.0
  x[::7], x[3::11] = np.inf, np.nan
  with np.errstate(invalid="ignore"):
    expected = _reference(masks, x)
  assert np.isnan(expected).any() and np.signbit(expected[expected == 0]).any()
  _assert_identical(_run(prog, tmp_path, lanes, x, expected.size), expected)
