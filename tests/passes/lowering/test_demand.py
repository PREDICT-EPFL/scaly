"""Demanded entries: a stored node is computed only at the entries its consumers read, against NumPy."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest
from scipy import sparse

import scaly as sc
from scaly.ad.reverse import vjp
from scaly.ir.expr import _scatter_zero
from scaly.ir.program import ProgramNode, ProgramOp, walk_program
from scaly.ir.text import format_program
from scaly.ir.types import CanonicalPattern
from scaly.passes.lowering import ctx, lower_function
from scaly.passes.program._common import trip_count
from tests.passes.lowering.test_fusion import _function, _lowered, needs_cc

_INDEX = {
  ProgramOp.ADD: lambda a, b: a + b,
  ProgramOp.SUB: lambda a, b: a - b,
  ProgramOp.MUL: lambda a, b: a * b,
  ProgramOp.DIV: lambda a, b: a // b,
  ProgramOp.MOD: lambda a, b: a % b,
  ProgramOp.LT: lambda a, b: a < b,
  ProgramOp.LE: lambda a, b: a <= b,
  ProgramOp.AND: lambda a, b: a and b,
}


def _reads_only_written_entries(proc: ProgramNode) -> None:
  """Run the procedure's index arithmetic and assert that every load from a buffer it declares hits
  an entry an earlier store wrote, including loads a select discards."""
  inputs, written, tables = {b.attrs["name"] for b in proc.args[: proc.attrs["input_count"]]}, {}, {}

  def index(node: ProgramNode, env: dict[str, int]) -> int:
    if node.op == ProgramOp.CONST_INT:
      return node.attrs["value"]
    if node.op == ProgramOp.VAR:
      return env[node.attrs["name"]]
    if node.op == ProgramOp.LOAD:
      return tables[node.args[0].attrs["buffer"]][index(node.args[0].args[0], env)]
    if node.op == ProgramOp.SELECT:
      return index(node.args[1] if index(node.args[0], env) else node.args[2], env)
    return _INDEX[node.op](*(index(arg, env) for arg in node.args))

  def loads(node: ProgramNode, env: dict[str, int]) -> None:
    for n in walk_program(node):
      if n.op == ProgramOp.LOAD and (name := n.args[0].attrs["buffer"]) in written:
        assert written[name][index(n.args[0].args[0], env)], f"{name}[{index(n.args[0].args[0], env)}] is read before it is written"

  def run(statements, env: dict[str, int]) -> None:
    for stmt in statements:
      if stmt.op == ProgramOp.BUFFER and "values" in stmt.attrs:
        tables[stmt.attrs["name"]] = list(stmt.attrs["values"])
      elif stmt.op == ProgramOp.BUFFER and stmt.attrs["name"] not in inputs:
        written[stmt.attrs["name"]] = [False] * math.prod(stmt.attrs["shape"])
      elif stmt.op == ProgramOp.FOR:
        bounds = stmt.args[0].args
        for value in range(*(index(bound, env) for bound in bounds)):
          run(stmt.args[1:], {**env, stmt.args[0].attrs["name"]: value})
      elif stmt.op == ProgramOp.STORE:
        loads(stmt.args[1], env)
        written[stmt.args[0].attrs["buffer"]][index(stmt.args[0].args[0], env)] = True

  run(proc.args, {})


def _lowered_checked(fn: sc.Function) -> ProgramNode:
  proc = _lowered(fn)
  _reads_only_written_entries(proc)
  return proc


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
  """A sparse derivative's recovery: products of seed vectors, transposed, gathered at their
  nonzeros. The products are computed at the 527 gathered entries of 984, as in devrush's race-car
  Hessian, and the sines they read only in the rows those entries use."""
  picks = np.sort(np.random.default_rng(984).choice(984, 527, replace=False))
  shapes = ((4, 246, 3), (3,))
  fn = _function("recover", shapes, lambda a, v: sc.gather((a.sin() @ v).T.reshape((984,)), picks))
  _check(fn, shapes, lambda a, v: (np.sin(a) @ v).T.reshape(-1)[picks])
  proc = _lowered_checked(fn)
  assert (_flops(proc, ProgramOp.MUL), _flops(proc, ProgramOp.SIN)) == (527 * 3, 527 * 3)


_SCATTERED = np.array([1, 4, 5, 8, 9, 11])
_SEEDS = np.zeros((4, 6))
_SEEDS[0, 0], _SEEDS[1, 1], _SEEDS[3, 5] = 1.0, -2.5, 1.0
# Each family reads products of a 4 x 6 x 3 tensor and a 3-vector, 24 entries of three products each,
# through one movement and a gather, and must compute only the entries the gather reaches.
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
    lambda p, q: sc.gather(_scatter_zero(p.reshape((24,))[:6], _SCATTERED, 12), np.array([4, 0, 9])),
    lambda p, q: np.bincount(_SCATTERED, p.reshape(-1)[:6], 12)[[4, 0, 9]],
    2,
  ),
  "mostly-zero constant": (
    lambda p, q: sc.gather(sc.const(_SEEDS) * p, np.array([0, 7, 14])),
    lambda p, q: (_SEEDS * p).reshape(-1)[[0, 7, 14]],
    3,
  ),
}


@needs_cc
@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_a_gather_reaches_a_stored_product_through(family: str) -> None:
  """Devrush's rewrite families as reads: a gather through a transpose, a slice, a sum, a
  concatenation, a scatter or a product with a mostly-zero constant demands of the products only the
  entries it reaches, three multiplies each. The constant keeps one entry in eight, and the gather
  also picks a zero of it, which is still multiplied."""
  body, reference, demanded = _FAMILIES[family]
  shapes = ((4, 6, 3), (3,), (4, 6, 3), (3,))
  fn = _function(f"family_{family.replace(' ', '_').replace('-', '_')}", shapes, lambda a, v, b, w: body(a @ v, b @ w))
  _check(fn, shapes, lambda a, v, b, w: reference(a @ v, b @ w))
  assert _flops(_lowered_checked(fn), ProgramOp.MUL) - (3 if family == "mostly-zero constant" else 0) == 3 * demanded


@needs_cc
def test_a_gather_that_repeats_entries_reads_a_sum_computed_once_per_entry() -> None:
  """Devrush's first guard: a gather that repeats entries does not distribute over a sum. The sum
  is stored and computed once at each entry read, which the gather then repeats."""
  picks = np.array([1, 1, 2, 3, 3, 3, 17])
  fn = _function("repeats", ((24,),), lambda x: sc.gather(x.sin() + x.cos(), picks))
  _check(fn, ((24,),), lambda x: (np.sin(x) + np.cos(x))[picks])
  proc = _lowered_checked(fn)
  assert (_flops(proc, ProgramOp.SIN), _flops(proc, ProgramOp.COS), _flops(proc, ProgramOp.ADD)) == (4, 4, 4)


@needs_cc
def test_a_computed_part_of_a_concatenation_stays_whole() -> None:
  """Devrush's second guard: a gather that reads most of a computed part of a concatenation computes
  the part in its own contiguous loop and reads it from there, instead of through the gather's table
  at every gathered entry."""
  picks = np.array([0, 1, 3, 4, 5, 9, 13])
  fn = _function("joined", ((6,), (12,)), lambda a, b: sc.gather(sc.concat([(a * a).sin(), b]), picks))
  _check(fn, ((6,), (12,)), lambda a, b: np.concatenate([np.sin(a * a), b])[picks])
  proc = _lowered_checked(fn)
  loops = [stmt for stmt in proc.args if stmt.op == ProgramOp.FOR and any(n.op == ProgramOp.SIN for n in walk_program(stmt))]
  assert [trip_count(loop.args[0]) for loop in loops] == [6] and _flops(proc, ProgramOp.SIN) == 6
  assert not any(n.op == ProgramOp.LOAD and n.args[0].attrs["buffer"].startswith("k") for n in walk_program(loops[0]))


def _sampled_adjoint() -> tuple[sc.Function, CanonicalPattern]:
  pattern = CanonicalPattern((6, 5), (0, 0, 2, 3, 5), (1, 4, 4, 0, 2))

  @sc.function(sc.arg("a", sc.TensorType(pattern.shape, pattern=pattern)), sc.arg("x", (5, 7)), sc.arg("ybar", (6, 7)), name="sampled_adjoint")
  def fn(a, x, ybar):
    return sc.values(vjp((a @ x,), (a,), (ybar,))[0]).block()

  return fn, pattern


@needs_cc
def test_the_sampled_adjoint_is_the_product_at_its_demanded_entries(monkeypatch) -> None:
  """#46's sampled adjoint computes a product only at a pattern's entries, which demand now does for
  any product: a dense product gathered along one column computes that column, and of its stored
  right operand only that column. The sampled kernel is lowered as before, demand or not."""
  shapes = ((4, 3), (3, 5))
  column = _function("column", shapes, lambda a, b: sc.gather(a.sin() @ b.cos(), np.array([2, 7, 12, 17])))
  _check(column, shapes, lambda a, b: (np.sin(a) @ np.cos(b))[:, 2])
  proc = _lowered_checked(column)
  assert (_flops(proc, ProgramOp.MUL), _flops(proc, ProgramOp.SIN), _flops(proc, ProgramOp.COS)) == (4 * 3, 12, 3)
  fn, pattern = _sampled_adjoint()
  rng = np.random.default_rng(46)
  a, x, ybar = (
    sparse.coo_array((rng.normal(size=pattern.nnz), (pattern.rows, pattern.cols)), shape=pattern.shape),
    rng.normal(size=(5, 7)),
    rng.normal(size=(6, 7)),
  )
  np.testing.assert_allclose(fn(a, x, ybar), np.einsum("pk,pk->p", ybar[list(pattern.rows)], x[list(pattern.cols)]), rtol=1e-13)
  programs = []
  for demand in (True, False):
    if not demand:
      monkeypatch.setattr(ctx, "plan_demand", lambda *args: {})
    seen: dict[str, ProgramNode] = {}
    lower_function(fn, observe=seen.__setitem__)
    programs.append(format_program(seen["lowered"]))
  assert programs[0] == programs[1]


def _copies(x: sc.Expr) -> dict[str, sc.Expr]:
  s = x.sin()
  return {
    "transpose": s.T,
    "slice": s[1:, ::2],
    "gather": sc.gather(s.reshape((24,)), np.array([5, 3, 17, 0, 9, 2, 22, 11])),
    "concatenation": sc.concat([s.reshape((24,)), x.reshape((24,)) * 2.0]),
    "stack": sc.stack([s, x]),
  }


@needs_cc
@pytest.mark.parametrize("copy", ["transpose", "slice", "gather", "concatenation", "stack"])
def test_a_stored_copy_computes_only_the_entries_two_readers_use(copy: str) -> None:
  """A copy two gathers read is stored. Read along evenly spaced entries, it copies only those, and
  the sine under it is evaluated there alone."""
  first, second = np.array([0, 1, 2]), np.array([4, 5, 6])

  def body(x: sc.Expr) -> sc.Expr:
    stored = _copies(x)[copy]
    return sc.gather(stored, first) + sc.gather(stored, second)

  fn = _function(f"copy_{copy}", ((4, 6),), body)
  reached = {
    "transpose": np.arange(24).reshape(4, 6).T.reshape(-1),
    "slice": np.arange(24).reshape(4, 6)[1:, ::2].reshape(-1),
    "gather": np.array([5, 3, 17, 0, 9, 2, 22, 11]),
    "concatenation": np.arange(24),
    "stack": np.arange(24),
  }[copy][np.concatenate([first, second])]
  _check(fn, ((4, 6),), lambda x: (lambda s: s[first] + s[second])(_reference_copy(copy, x)))
  assert _flops(_lowered_checked(fn), ProgramOp.SIN) == len(np.unique(reached))


def _reference_copy(copy: str, x: np.ndarray) -> np.ndarray:
  s = np.sin(x)
  return {
    "transpose": s.T,
    "slice": s[1:, ::2],
    "gather": s.reshape(-1)[[5, 3, 17, 0, 9, 2, 22, 11]],
    "concatenation": np.concatenate([s.reshape(-1), x.reshape(-1) * 2.0]),
    "stack": np.stack([s, x]),
  }[copy].reshape(-1)


@needs_cc
def test_a_reduction_and_a_sum_of_scatters_compute_only_the_entries_read() -> None:
  """An axis reduction read at three of its rows sums only those rows, and a sum of two scatters
  read at three entries stores only the values landing there, zeroing the one entry the first
  scatter does not write."""
  rows, filled = np.array([1, 4, 6]), np.array([0, 1, 2])

  def body(x: sc.Expr, y: sc.Expr) -> sc.Expr:
    sums = sc.gather(x.sin().sum(axis=1), rows)
    scattered = _scatter_zero(y.cos(), np.arange(0, 12, 2), 12) + _scatter_zero(y.exp(), np.arange(1, 12, 2), 12)
    return sums + sc.gather(scattered, filled)

  fn = _function("reduce_scatter", ((8, 5), (6,)), body)
  _check(fn, ((8, 5), (6,)), lambda x, y: np.sin(x).sum(axis=1)[rows] + np.array([np.cos(y[0]), np.exp(y[0]), np.cos(y[1])]))
  proc = _lowered_checked(fn)
  assert (_flops(proc, ProgramOp.SIN), _flops(proc, ProgramOp.COS), _flops(proc, ProgramOp.EXP)) == (15, 2, 1)


@needs_cc
def test_a_scalar_product_no_entry_of_which_is_read_still_lowers() -> None:
  """A scalar product in a part that two selected concatenations never select is demanded nowhere
  and computed whole."""

  def body(a: sc.Expr, b: sc.Expr, c: sc.Expr) -> sc.Expr:
    joined = sc.concat([sc.stack([a @ b]), c])
    return sc.gather(joined, np.array([1, 2])) + sc.gather(joined, np.array([2, 3]))

  fn = _function("unread_scalar", ((3,), (3,), (4,)), body)
  _check(fn, ((3,), (3,), (4,)), lambda a, b, c: c[[0, 1]] + c[[1, 2]])
  _lowered_checked(fn)


@needs_cc
def test_a_narrowed_scatter_keeps_the_sign_of_zero() -> None:
  """A scatter that repeats an index nobody reads still adds into zeros where it stores one value,
  so a negative zero comes out positive, as from the whole scatter."""
  fn = _function("signed_zero", ((3,),), lambda v: sc.gather(_scatter_zero(v, np.array([0, 2, 2]), 4), np.array([0])))
  assert not np.signbit(fn((np.array([-0.0, 1.0, 2.0]),))).any()
  _lowered_checked(fn)


@pytest.mark.parametrize(
  ("shapes", "body"),
  [
    (((1 << 24, 1), (1, 1 << 24)), lambda a, b: sc.gather(a + b, np.array([0]))),
    (((2, 1), (1, 1 << 40)), lambda a, b: sc.gather((a + b).sum(axis=1), np.array([0]))),
    (((1 << 40,), (1 << 40,)), lambda a, b: sc.gather(sc.concat([a, b]), np.array([0, 1 << 40]))),
  ],
  ids=["broadcast", "reduction", "concatenation"],
)
def test_a_gather_from_a_huge_value_plans_no_work_in_proportion_to_it(shapes, body) -> None:
  """Demand of a value too large to track is the whole value, so a gather reading one of its entries
  lowers without allocating anything of its size."""
  _lowered(_function("huge", shapes, body))
