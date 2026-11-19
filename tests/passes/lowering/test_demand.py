"""Demanded entries: a stored node is computed only at the entries its consumers read, against NumPy."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

import scaly as sc
from scaly.ir.expr import _scatter_zero
from scaly.ir.program import ProgramNode, ProgramOp, walk_program
from scaly.passes.program._common import trip_count
from tests.passes.lowering.test_fusion import _function, _lowered, needs_cc


def _flops(proc: ProgramNode, op: ProgramOp) -> int:
  """How many floating-point ``op`` the procedure runs, each counted once per trip of its loops."""
  total, stack = 0, [(stmt, 1) for stmt in proc.args]
  while stack:
    node, trips = stack.pop()
    if node.op == ProgramOp.FOR:
      count = trip_count(node.args[0])
      assert count is not None
      stack.extend((arg, trips * count) for arg in node.args[1:])
    else:
      total += trips * sum(n.op == op and n.dtype.is_floating for n in walk_program(node))
  return total


def _check(fn: sc.Function, shapes: tuple[tuple[int, ...], ...], reference: Callable[..., np.ndarray]) -> None:
  rng = np.random.default_rng(75)
  values = tuple(rng.uniform(-1.0, 1.0, s) for s in shapes)
  np.testing.assert_allclose(fn(values), reference(*values), rtol=1e-13, atol=1e-14)


@needs_cc
def test_a_product_read_through_a_gather_computes_only_the_gathered_entries() -> None:
  """A sparse derivative's recovery: a seed-major product, transposed, gathered at its nonzeros. The
  product is computed at the 527 gathered entries of its 984, as in devrush's race-car Hessian."""
  picks = np.sort(np.random.default_rng(984).choice(984, 527, replace=False))
  shapes = ((4, 3), (3, 246))
  fn = _function("recover", shapes, lambda a, b: sc.gather((a @ b).T.reshape((984,)), picks))
  _check(fn, shapes, lambda a, b: (a @ b).T.reshape(-1)[picks])
  assert _flops(_lowered(fn), ProgramOp.MUL) == 527 * 3


_PLACED = np.array([1, 4, 5, 8, 9, 11])
# Each family reads a product of 4 x 3 and 3 x 6 matrices, 24 entries of three products each, through
# one movement and a gather, and must compute only the entries the gather reaches.
_FAMILIES = {
  "transpose": (
    lambda p, q: sc.gather(p.reshape((2, 2, 6)).transpose((2, 0, 1)), np.array([0, 5, 7, 23, 11])),
    lambda p, q: p.reshape(2, 2, 6).transpose(2, 0, 1).reshape(-1)[[0, 5, 7, 23, 11]],
    5,
  ),
  "slice": (lambda p, q: sc.gather(p[1:4:2, 1:], np.array([[0, 7], [5, 3]])), lambda p, q: p[1:4:2, 1:].reshape(-1)[[[0, 7], [5, 3]]], 4),
  "sum": (lambda p, q: sc.gather(p + q, np.array([2, 13, 21])), lambda p, q: (p + q).reshape(-1)[[2, 13, 21]], 6),
  "concatenation": (
    lambda p, q: sc.gather(sc.concat([p.reshape((24,)), q.T.reshape((24,))]), np.array([3, 30, 47])),
    lambda p, q: np.concatenate([p.reshape(-1), q.T.reshape(-1)])[[3, 30, 47]],
    3,
  ),
  "scatter": (
    lambda p, q: sc.gather(_scatter_zero(p.reshape((24,))[:6], _PLACED, 12), np.array([4, 0, 9])),
    lambda p, q: np.bincount(_PLACED, p.reshape(-1)[:6], 12)[[4, 0, 9]],
    2,
  ),
  "mostly-zero constant": (
    lambda p, q: sc.gather(sc.const(np.eye(4, 6)) * p, np.array([0, 7, 15])),
    lambda p, q: (np.eye(4, 6) * p).reshape(-1)[[0, 7, 15]],
    3,
  ),
}


@needs_cc
@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_a_gather_reaches_a_stored_product_through(family: str) -> None:
  """Devrush's rewrite families as reads: a gather through a transpose, a slice, a sum, a
  concatenation, a scatter or a product with a mostly-zero constant demands of the product only the
  entries it reaches, three products each."""
  body, reference, demanded = _FAMILIES[family]
  shapes = ((4, 3), (3, 6), (4, 3), (3, 6))
  fn = _function(f"family_{family.replace(' ', '_').replace('-', '_')}", shapes, lambda a, b, c, d: body(a @ b, c @ d))
  _check(fn, shapes, lambda a, b, c, d: reference(a @ b, c @ d))
  assert _flops(_lowered(fn), ProgramOp.MUL) - (3 if family == "mostly-zero constant" else 0) == 3 * demanded


@needs_cc
def test_a_gather_that_repeats_entries_reads_a_sum_computed_once_per_entry() -> None:
  """Devrush's first guard: a gather that repeats entries does not distribute over a sum. The sum
  is stored and computed once at each entry read, which the gather then repeats."""
  picks = np.array([1, 1, 2, 3, 3, 3, 17])
  fn = _function("repeats", ((24,),), lambda x: sc.gather(x.sin() + x.cos(), picks))
  _check(fn, ((24,),), lambda x: (np.sin(x) + np.cos(x))[picks])
  proc = _lowered(fn)
  assert (_flops(proc, ProgramOp.SIN), _flops(proc, ProgramOp.COS), _flops(proc, ProgramOp.ADD)) == (4, 4, 4)


@needs_cc
def test_a_computed_part_of_a_concatenation_stays_whole() -> None:
  """Devrush's second guard: a gather that reads most of a computed part of a concatenation computes
  the part in its own contiguous loop and reads it from there, instead of through the gather's table
  at every gathered entry."""
  picks = np.array([0, 1, 3, 4, 5, 9, 13])
  fn = _function("joined", ((6,), (12,)), lambda a, b: sc.gather(sc.concat([(a * a).sin(), b]), picks))
  _check(fn, ((6,), (12,)), lambda a, b: np.concatenate([np.sin(a * a), b])[picks])
  proc = _lowered(fn)
  loops = [stmt for stmt in proc.args if stmt.op == ProgramOp.FOR and any(n.op == ProgramOp.SIN for n in walk_program(stmt))]
  assert [trip_count(loop.args[0]) for loop in loops] == [6] and _flops(proc, ProgramOp.SIN) == 6
  assert not any(n.op == ProgramOp.LOAD and n.args[0].attrs["buffer"].startswith("k") for n in walk_program(loops[0]))


@needs_cc
def test_a_gathered_dense_product_is_the_sampled_adjoint() -> None:
  """The sampled adjoint of #46 as a special case: a dense product gathered at a pattern's entries
  computes one dot per stored entry, as the sampled kernel does for a sparse operand."""
  rows, cols = np.array([0, 0, 2, 3, 5]), np.array([1, 4, 4, 0, 2])
  shapes = ((6, 7), (5, 7))
  fn = _function("sampled", shapes, lambda ybar, x: sc.gather(ybar @ x.T, rows * 5 + cols))
  _check(fn, shapes, lambda ybar, x: np.einsum("pk,pk->p", ybar[rows], x[cols]))
  assert _flops(_lowered(fn), ProgramOp.MUL) == 5 * 7


@needs_cc
@pytest.mark.parametrize("shapes", [((2, 1, 3, 4), (3, 4, 5)), ((6, 4), (4,)), ((4,), (4, 6))])
def test_a_gathered_batched_or_vector_product_reads_only_its_entries(shapes) -> None:
  """A product with broadcast batch axes or a vector operand is computed at the gathered entries."""
  size = int(np.prod(np.broadcast_shapes(shapes[0][:-2], shapes[1][:-2]))) * (shapes[0][-2] if len(shapes[0]) > 1 else 1)
  size *= shapes[1][-1] if len(shapes[1]) > 1 else 1
  picks = np.array([size - 1, 0, size // 2])
  fn = _function(f"batched_{len(shapes[0])}_{len(shapes[1])}", shapes, lambda a, b: sc.gather((a @ b).reshape((size,)), picks))
  _check(fn, shapes, lambda a, b: (a @ b).reshape(-1)[picks])
  assert _flops(_lowered(fn), ProgramOp.MUL) == 3 * 4
