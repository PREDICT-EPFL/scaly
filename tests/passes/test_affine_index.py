"""Affine gathers and scatters lower to arithmetic on the trip index, not a ``static const`` table (C-9)."""

from __future__ import annotations

import numpy as np
import pytest

import scaly as sc
from scaly.codegen.aot import render_c_source
from scaly.ir.expr import gather, scatter
from scaly.ir.program import ProgramNode, ProgramOp
from scaly.ir.types import dtypes
from scaly.passes.affine import affine_index_map
from scaly.passes.lowering import lower_function

# (indices, expected dims, expected coefficients, expected residual length)
CASES = [
  (np.arange(100), (100,), (1,), 1),
  (7 + 3 * np.arange(50), (50,), (3,), 1),
  (np.full(40, 9), (40,), (0,), 1),
  # ``start + it * stride + j``: the shape the AD rules build.
  ((4 + 11 * np.arange(20)[:, None] + np.arange(3)[None, :]).reshape(-1), (20, 3), (11, 1), 1),
  # Three levels, as the multi-seed forward rule emits (colour, iteration, slice).
  ((np.arange(5)[:, None, None] * 60 + 7 * np.arange(4)[None, :, None] + np.arange(3)[None, None, :]).reshape(-1), (5, 4, 3), (60, 7, 1), 1),
  # A repeated block: the innermost range contributes nothing and the term is dropped.
  (np.repeat(2 * np.arange(30), 4), (30, 4), (2, 0), 1),
  # A reversed window is affine too, with a negative coefficient.
  (99 - np.arange(60), (60,), (-1,), 1),
  # An arbitrary but small inner tile stays a table; the two outer ranges still fold.
  ((np.arange(8)[:, None] * 50 + np.array([0, 3, 1, 7, 2])[None, :]).reshape(-1), (8,), (50,), 5),
  # No structure at all: the whole array survives, which is what every gather used to emit.
  (np.array([5, 0, 9, 2, 7, 1, 8]), (), (), 7),
]


def _const_int_buffers(prog: ProgramNode) -> list[ProgramNode]:
  """Every materialized integer table in the program, at any nesting depth."""
  found, stack = [], [prog]
  while stack:
    node = stack.pop()
    if node.op == ProgramOp.BUFFER and node.dtype == dtypes.int64 and "values" in node.attrs:
      found.append(node)
    stack.extend(node.args)
  return found


def _index_table_bytes(source: str) -> int:
  return sum(len(line) for line in source.splitlines() if "static const int64_t" in line)


@pytest.mark.parametrize(("indices", "dims", "coeffs", "residual"), CASES)
def test_affine_index_map_recovers_the_ranges_and_round_trips(indices, dims, coeffs, residual) -> None:
  amap = affine_index_map(indices)
  assert (amap.dims, amap.coeffs, len(amap.residual)) == (dims, coeffs, residual)
  np.testing.assert_array_equal(amap.evaluate(len(indices)), indices)


def test_a_perturbed_affine_index_stops_folding() -> None:
  """The check has teeth: one changed entry in an otherwise strided window keeps the whole table."""
  indices = 3 * np.arange(64)
  assert len(affine_index_map(indices).residual) == 1
  perturbed = indices.copy()
  perturbed[17] += 1
  assert len(affine_index_map(perturbed).residual) == len(perturbed)


@pytest.mark.parametrize("build", [lambda x, idx: gather(x, idx), lambda x, idx: scatter(x[: len(idx)], idx, (256,))])
def test_affine_gather_and_scatter_emit_no_index_table(build) -> None:
  x = sc.sym("x", 256)
  affine = 1 + 3 * np.arange(80, dtype=np.int64)
  perturbed = affine.copy()
  perturbed[9] += 1

  def tables(indices: np.ndarray) -> list[ProgramNode]:
    fun = sc.Function._from_exprs("affine_probe", [x], [build(x, indices)], ["x"], ["y"])
    return _const_int_buffers(lower_function(fun))

  assert tables(affine) == []
  # The same gather with one index moved is not affine, so the table comes back: the assertion
  # above is not vacuous.
  assert [int(np.prod(t.attrs["shape"])) for t in tables(perturbed)] == [80]


def test_an_empty_index_lowers_and_declares_nothing() -> None:
  """An empty gather or scatter has no index to be affine in; its loop runs zero times."""
  x = sc.sym("x", 8)
  empty = np.zeros(0, dtype=np.int64)
  for name, expr in (("gather", gather(x, empty)), ("scatter", scatter(x[:0], empty, (8,)))):
    fun = sc.Function._from_exprs(f"affine_empty_{name}", [x], [expr], ["x"], ["y"])
    assert _const_int_buffers(lower_function(fun)) == []
  values = np.arange(8.0)
  scattered = sc.Function._from_exprs("affine_empty_scatter_values", [x], [scatter(x[:0], empty, (8,))], ["x"], ["y"])
  np.testing.assert_array_equal(np.asarray(scattered(values)).reshape(-1), np.zeros(8))


@pytest.mark.parametrize("case", range(len(CASES)), ids=[str(len(c[0])) for c in CASES])
def test_a_gather_reads_exactly_the_elements_numpy_would(case: int) -> None:
  """The emitted arithmetic, not just the factoring: ``index_at`` sums plain quotients, so this is
  what checks that the telescoped form agrees with ``(k // stride) % dim`` on every case above."""
  indices = np.asarray(CASES[case][0], dtype=np.int64)
  size = int(indices.max()) + 1
  x = sc.sym("x", size)
  fun = sc.Function._from_exprs(f"affine_gather_values_{case}", [x], [gather(x, indices)], ["x"], ["y"])
  values = np.random.default_rng(case).normal(size=size)
  np.testing.assert_array_equal(np.asarray(fun(values)).reshape(-1), values[indices])


# --- the VMAP derivative rules, which are where the tables came from -------------


@sc.function(sc.L("x", 2), sc.L("y", ...), name="affine_stage")
def _stage(x):
  return sc.stack([x[0].sin() * x[1], x[0] * x[1] * x[1]])


def _stage_jac_np(x: np.ndarray) -> np.ndarray:
  return np.array([[np.cos(x[0]) * x[1], np.sin(x[0])], [x[1] ** 2, 2 * x[0] * x[1]]])


@pytest.mark.parametrize("length", [3, 17])
def test_vmap_forward_jacobian_matches_numpy(length: int) -> None:
  z = sc.sym("z", 2 * length)
  mapped = sc.vmap(_stage, length, [(z, 0, 2)])
  fun = sc.Function._from_exprs(f"affine_jac_{length}", [z], [sc.jacobian(mapped, z)], ["z"], ["jac"])
  values = np.random.default_rng(1).normal(size=2 * length)
  expected = np.zeros((2 * length, 2 * length))
  for it in range(length):
    expected[2 * it : 2 * it + 2, 2 * it : 2 * it + 2] = _stage_jac_np(values[2 * it : 2 * it + 2])
  np.testing.assert_allclose(np.asarray(fun(values)), expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("length", [3, 17])
def test_vmap_reverse_gradient_matches_numpy(length: int) -> None:
  z = sc.sym("z", 2 * length)
  mapped = sc.vmap(_stage, length, [(z, 0, 2)])
  fun = sc.Function._from_exprs(f"affine_grad_{length}", [z], [sc.gradient(mapped.sum(), z)], ["z"], ["g"])
  values = np.random.default_rng(2).normal(size=2 * length)
  expected = np.concatenate([_stage_jac_np(values[2 * it : 2 * it + 2]).sum(axis=0) for it in range(length)])
  np.testing.assert_allclose(np.asarray(fun(values)).reshape(-1), expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("length", [3, 17])
def test_vmap_hessian_matches_the_unrolled_function(length: int) -> None:
  """Forward-over-reverse over a VMAP, against the same problem written out stage by stage."""
  z = sc.sym("z", 2 * length)
  mapped = sc.vmap(_stage, length, [(z, 0, 2)])
  unrolled = sc.concat([_stage(z[2 * it : 2 * it + 2]) for it in range(length)])
  values = np.random.default_rng(3).normal(size=2 * length)
  results = []
  for name, expr in (("mapped", mapped), ("unrolled", unrolled)):
    cost = (expr * expr).sum()
    fun = sc.Function._from_exprs(f"affine_hess_{name}_{length}", [z], [sc.hessian(cost, z)], ["z"], ["h"])
    results.append(np.asarray(fun(values)))
  np.testing.assert_allclose(results[0], results[1], rtol=1e-10, atol=1e-10)


def test_vmap_hessian_index_tables_do_not_grow_with_the_trip_count() -> None:
  """C-9's gate, at the scale of one test: the mapped index tables are no longer O(length)."""

  def rendered(length: int) -> str:
    z = sc.sym("z", 2 * length)
    mapped = sc.vmap(_stage, length, [(z, 0, 2)])
    cost = (mapped * mapped).sum()
    return render_c_source(sc.Function._from_exprs(f"affine_hess_size_{length}", [z], [sc.hessian(cost, z)], ["z"], ["h"]))

  small, large = rendered(8), rendered(64)
  assert _index_table_bytes(small) == _index_table_bytes(large)
  assert small.count("\n") == large.count("\n")
