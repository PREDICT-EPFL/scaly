"""Fusion decided at lowering: each rule against NumPy, with the work and storage it must leave."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

import scaly as sc
from scaly.codegen.toolchain import find_c_compiler
from scaly.function.sugar import _mapped_call
from scaly.ir import program as p
from scaly.ir.expr import Expr, _scatter_zero
from scaly.ir.types import CanonicalPattern
from scaly.ir.program import ProgramNode, ProgramOp, RangeKind, walk_program
from scaly.passes.lowering import ctx, lower_function, main_proc
from scaly.passes.lowering.schedule import emit_reduction
from scaly.passes.program._common import trip_count
from scaly.utils.names import NameScope

needs_cc = pytest.mark.skipif(find_c_compiler() is None, reason="no C compiler available for JIT numeric check")


def _function(name: str, shapes: tuple[tuple[int, ...], ...], body: Callable[..., Expr]) -> sc.Function:
  @sc.function(sc.group(*(sc.arg(f"x{k}", s) for k, s in enumerate(shapes))), outputs=sc.arg("y"), name=name)
  def fn(inputs):
    return body(*inputs)

  return fn


def _lowered(fn: sc.Function) -> ProgramNode:
  seen: dict[str, ProgramNode] = {}
  lower_function(fn, observe=seen.__setitem__)
  return main_proc(seen["lowered"])


def _privates(proc: ProgramNode) -> list[ProgramNode]:
  return [s for s in proc.args if s.op == ProgramOp.BUFFER and s.attrs["address_space"] == "private" and "values" not in s.attrs]


def _work(proc: ProgramNode, op: ProgramOp) -> int:
  """How many times ``op`` runs, counting each occurrence once per trip of its enclosing loops."""
  total = 0
  stack = [(stmt, 1) for stmt in proc.args]
  while stack:
    node, trips = stack.pop()
    if not any(n.op == op for n in walk_program(node)):
      continue
    if node.op == ProgramOp.FOR:
      count = trip_count(node.args[0])
      assert count is not None
      stack.extend((arg, trips * count) for arg in node.args[1:])
      continue
    total += trips * sum(n.op == op for n in walk_program(node))
  return total


def _check(fn: sc.Function, shapes: tuple[tuple[int, ...], ...], reference: Callable[..., np.ndarray]) -> None:
  rng = np.random.default_rng(69)
  values = tuple(rng.uniform(-1.0, 1.0, s) for s in shapes)
  np.testing.assert_allclose(fn(values), reference(*values), rtol=1e-13, atol=1e-14)


_MOVEMENT = {
  "transpose": (((3, 4), (4, 3)), lambda a, b: a.T * b, lambda a, b: a.T * b),
  "reshape_transpose": (
    ((2, 3, 4), (4, 6)),
    lambda a, b: a.transpose((2, 0, 1)).reshape((4, 6)) - b,
    lambda a, b: a.transpose(2, 0, 1).reshape(4, 6) - b,
  ),
  "strided_slice": (((6, 4), (3, 2)), lambda a, b: a[::2, 1:3] * b, lambda a, b: a[::2, 1:3] * b),
  "contiguous_slice": (((5, 2), (4, 2)), lambda a, b: a[1:] + b, lambda a, b: a[1:] + b),
  "gather_transpose": (
    ((3, 4), (5,)),
    lambda a, b: sc.gather(a.T.reshape((12,)), np.array([0, 5, 7, 11, 2])) * b,
    lambda a, b: a.T.reshape(12)[[0, 5, 7, 11, 2]] * b,
  ),
}


@needs_cc
@pytest.mark.parametrize("case", sorted(_MOVEMENT))
def test_movement_is_an_index_remap(case: str) -> None:
  """A transpose, reshape, slice or gather under an elementwise operation is read in place: the
  lowered procedure holds no temporary, and the values match NumPy."""
  shapes, body, reference = _MOVEMENT[case]
  fn = _function(f"remap_{case}", shapes, lambda *xs: body(*xs).sin())
  assert not _privates(_lowered(fn))
  _check(fn, shapes, lambda *xs: np.sin(reference(*xs)))


@needs_cc
def test_empty_reshape_transpose_keeps_the_nonempty_concat_piece() -> None:
  shapes = ((0, 3, 4), (4, 2))
  fn = _function("empty_reshape_concat", shapes, lambda a, b: sc.concat((a.transpose((2, 0, 1)).reshape((4, 0)), b), axis=1))
  assert not _privates(_lowered(fn))
  _check(fn, shapes, lambda a, b: np.concatenate((a.transpose(2, 0, 1).reshape(4, 0), b), axis=1))


@needs_cc
def test_single_consumer_chain_is_one_loop() -> None:
  fn = _function("chain", ((16,), (16,)), lambda x, y: ((x.sin() + y) * y).exp())
  proc = _lowered(fn)
  assert not _privates(proc)
  assert sum(s.op == ProgramOp.FOR for s in proc.args) == 1
  assert _work(proc, ProgramOp.SIN) == 16
  _check(fn, ((16,), (16,)), lambda x, y: np.exp((np.sin(x) + y) * y))


@needs_cc
def test_value_read_by_two_consumers_is_stored_once() -> None:
  fn = _function("shared", ((16,),), lambda x: x.sin().exp() + x.sin().cos())
  proc = _lowered(fn)
  assert _work(proc, ProgramOp.SIN) == 16
  assert len(_privates(proc)) == 1
  _check(fn, ((16,),), lambda x: np.exp(np.sin(x)) + np.cos(np.sin(x)))


@needs_cc
@pytest.mark.parametrize("consumer", ["broadcast", "matmul_operand"])
def test_value_read_several_times_per_element_is_stored(consumer: str) -> None:
  """A broadcast operand or a matrix product operand is read more than once per element, so it is
  computed once into storage rather than at each read."""
  if consumer == "broadcast":
    fn = _function("reread_broadcast", ((4,), (3, 4)), lambda x, y: x.sin() * y)
    reference = lambda x, y: np.sin(x) * y
  else:
    fn = _function("reread_matmul", ((4,), (3, 4)), lambda x, y: y @ x.sin())
    reference = lambda x, y: y @ np.sin(x)
  proc = _lowered(fn)
  assert _work(proc, ProgramOp.SIN) == 4
  _check(fn, ((4,), (3, 4)), reference)


@needs_cc
def test_matmul_operand_read_once_is_inlined() -> None:
  """A product reads each entry of a matrix operand once when the other operand is a vector."""
  fn = _function("once_matmul", ((3, 4), (4,)), lambda a, x: a.sin() @ x)
  proc = _lowered(fn)
  assert not _privates(proc)
  assert _work(proc, ProgramOp.SIN) == 12
  _check(fn, ((3, 4), (4,)), lambda a, x: np.sin(a) @ x)


@needs_cc
@pytest.mark.parametrize("expensive", [False, True])
def test_a_libm_call_is_not_duplicated_by_two_reads(expensive: bool) -> None:
  op = (lambda v: v.sin()) if expensive else (lambda v: v * 3.0)
  fn = _function(f"twice_{expensive}", ((8,),), lambda x: op(x) * op(x))
  proc = _lowered(fn)
  if expensive:
    assert _work(proc, ProgramOp.SIN) == 8
    assert len(_privates(proc)) == 1
  else:
    assert not _privates(proc)
  _check(fn, ((8,),), (lambda x: np.sin(x) ** 2) if expensive else (lambda x: (3 * x) ** 2))


@needs_cc
def test_automatic_scalarization_preserves_known_zeros_with_infinity() -> None:
  stage = _function("known_zero_stage", ((2,), (2,)), lambda x, w: (x * sc.const([0.0, 1.0])) * sc.const([np.inf, 1.0]) + w)

  @sc.function(sc.group(sc.arg("z", 4), sc.arg("v", 4)), outputs=sc.arg("y", 4), name="known_zero_map")
  def fn(inputs):
    z, v = inputs
    return _mapped_call(stage, 2, [(z, 0, 2), (v, 0, 2)])

  np.testing.assert_array_equal(fn((np.array([1.0, 2.0, 3.0, 4.0]), np.array([5.0, 6.0, 7.0, 8.0]))), [5.0, 8.0, 7.0, 12.0])


def test_reused_selection_stores_only_the_selected_exp_values() -> None:
  def body(x):
    picked = sc.gather(x.exp(), [3, 7])
    return picked * picked

  proc = _lowered(_function("reused_exp_selection", ((1000,),), body))
  assert _work(proc, ProgramOp.EXP) == 2
  assert [buf.attrs["shape"] for buf in _privates(proc)] == [(2,)]


def test_reused_selection_of_an_input_needs_no_buffer() -> None:
  def body(x):
    picked = sc.gather(x, [3, 7, 9])
    return picked * 3.0 + picked.sin()

  assert not _privates(_lowered(_function("reused_input_selection", ((1000,),), body)))


def test_sampled_product_counts_its_select_condition_read() -> None:
  a = CanonicalPattern((3, 3), (0, 1, 1, 2), (0, 0, 2, 1))
  b = CanonicalPattern((3, 3), (0, 0, 1, 2), (0, 2, 1, 2))
  output = CanonicalPattern((3, 3), (0, 1, 2), (0, 2, 1))

  def body(v, w):
    masked = sc.pack(sc.where(v.sin() > 0, v, 0.0), a)
    return sc.values(sc.matmul(masked, sc.pack(w, b), sampled=output)).block()

  assert _work(_lowered(_function("sampled_condition", ((4,), (4,)), body)), ProgramOp.SIN) == 4


@needs_cc
@pytest.mark.parametrize("case", ["private_alias", "carry", "scatter_sum"])
def test_deferred_reads_keep_the_value_before_an_indexed_write(case: str) -> None:
  def update(carry, alias=False):
    old = sc.gather(carry, [0])
    base = carry.reshape((2, 2)) if alias else carry
    first = sc.scatter(base, [0], old + 1, combine="set")
    return sc.scatter(first.reshape((4,)), [1], old + 2, combine="set")

  @sc.function(sc.arg("carry", 4), name="shared_before_update")
  def stage(carry):
    return update(carry)

  @sc.function(sc.arg("x", 4), name=f"read_before_update_{case}")
  def fn(x):
    if case == "carry":
      return sc.scan(stage, x, length=1, in_place=(0,))[0].block()
    if case == "scatter_sum":
      base = x + 1
      first = _scatter_zero(base, [1, 0, 3, 2], 4)
      updated = sc.scatter(base, [0], sc.const([99.0]), combine="set")
      return first + _scatter_zero(updated.sum().reshape((1,)), [0], 4)
    return update(x + 1, alias=True)

  x = np.arange(4.0)
  before = x if case == "carry" else x + 1
  expected = before.copy()
  if case == "scatter_sum":
    expected = before[[1, 0, 3, 2]]
    expected[0] += 99 + before[1:].sum()
  else:
    expected[[0, 1]] = before[0] + np.array([1, 2])
  np.testing.assert_array_equal(fn(x), expected)
  np.testing.assert_array_equal(x, np.arange(4.0))


@needs_cc
@pytest.mark.parametrize("variant", ["static", "runtime", "index_set", "index_add", "nested", "fill_control"])
def test_shared_gather_from_earlier_carry_version(variant):
  def update(base, index, value):
    if variant in ("index_set", "index_add", "fill_control"):
      method = sc.index_add if variant == "index_add" else sc.index_set
      return method(base, index, value, mode="fill" if variant == "fill_control" else "promise_in_bounds")
    return sc.scatter(base, index, value, combine="set")

  def body(carry, j):
    first = update(carry, [3], sc.const([10.0]))
    second = update(first, [2], sc.const([20.0]))
    old = sc.gather(first, [0])
    third = update(second, j if variant == "runtime" else [0], old + 1)
    return update(third, [1], old + 2)

  @sc.function(sc.arg("carry", 4), sc.arg("j", sc.TensorType((), sc.dtypes.int64)), name="integration_step_" + variant)
  def step(carry, j):
    return body(carry, j)

  @sc.function(sc.arg("carry", 4), name="integration_nested_step_" + variant)
  def nested_step(carry):
    return sc.scan(step, carry, length=1, index=True, in_place=(0,))[0].block()

  @sc.function(sc.arg("x", 4), name="integration_run_" + variant)
  def run(x):
    if variant == "nested":
      return sc.scan(nested_step, x, length=2)[0].block()
    return sc.scan(step, x, length=1, index=True, in_place=() if variant == "fill_control" else (0,))[0].block()

  initial = np.arange(4.0)
  expected = initial.copy()
  for _ in range(2 if variant == "nested" else 1):
    if variant == "index_add":
      expected[3] += 10
      expected[2] += 20
      old = expected[0].copy()
      expected[0] += old + 1
      expected[1] += old + 2
    else:
      expected[3] = 10
      expected[2] = 20
      old = expected[0].copy()
      expected[0] = old + 1
      expected[1] = old + 2
  actual = run(initial)
  np.testing.assert_array_equal(initial, np.arange(4.0))
  np.testing.assert_array_equal(actual, expected)


@needs_cc
def test_reduction_evaluates_its_operand_in_the_reduction_loop() -> None:
  fn = _function("reduce_fused", ((5, 6),), lambda x: x.sin().sum(axis=1))
  proc = _lowered(fn)
  assert not _privates(proc)
  assert _work(proc, ProgramOp.SIN) == 30
  _check(fn, ((5, 6),), lambda x: np.sin(x).sum(axis=1))


@needs_cc
def test_broadcast_inputs_of_a_map_keep_their_work_apart() -> None:
  """In a mapped function, work on the arguments a map broadcasts stays a statement of its own, so
  ``hoist_invariant`` lifts it out of the map instead of repeating it on each trip."""
  stage = _function("hoist_apart_stage", ((3,), (9,)), lambda x, w: w.reshape((3, 3)).exp() @ x)

  @sc.function(sc.group(sc.arg("z", 12), sc.arg("w", 9)), outputs=sc.arg("y", 12), name="hoist_apart")
  def fn(inputs):
    z, w = inputs
    return _mapped_call(stage, 4, [(z, 0, 3), (w, 0, 0)])

  procs = lower_function(fn).args
  hoisted = [pr for pr in procs if "hoist" in pr.attrs["name"] and pr is not procs[-1]]
  assert any(ProgramOp.EXP in {n.op for n in walk_program(pr)} for pr in hoisted)
  rng = np.random.default_rng(3)
  z, w = rng.uniform(-1, 1, 12), rng.uniform(-1, 1, 9)
  np.testing.assert_allclose(fn((z, w)), (np.exp(w.reshape(3, 3)) @ z.reshape(4, 3).T).T.reshape(-1), rtol=1e-13)


@needs_cc
@pytest.mark.parametrize("overlap", [False, True])
def test_sum_of_scatters_accumulates_into_one_buffer(overlap: bool) -> None:
  second = [2, 2, 3, 5] if overlap else [1, 3, 5, 7]

  @sc.function(sc.arg("x", 4), outputs=sc.arg("y"), name=f"scatter_sum_{overlap}")
  def fn(x):
    return _scatter_zero(x.sin(), [0, 2, 4, 6], 8) + _scatter_zero(2 * x, second, 8) + _scatter_zero(-x, [6, 7, 0, 2], 8)

  proc = _lowered(fn)
  zero_fills = [s for s in proc.args if s.op == ProgramOp.FOR and s.args[1].op == ProgramOp.STORE and s.args[1].args[1].op == ProgramOp.CONST_FLOAT]
  # A repeated index that an earlier scatter also writes ends the left-associated sum, for the order of its terms.
  assert len(zero_fills) == (3 if overlap else 1)
  data = np.array([0.3, -1.2, 2.5, 0.7])
  expected = np.zeros(8)
  for ids, values in (([0, 2, 4, 6], np.sin(data)), (second, 2 * data), ([6, 7, 0, 2], -data)):
    term = np.zeros(8)
    np.add.at(term, ids, values)
    expected += term
  np.testing.assert_allclose(np.asarray(fn(data)), expected, rtol=1e-14)


@needs_cc
def test_scatter_of_a_mapped_transpose_reads_by_affine_indices() -> None:
  """A scatter whose source is a transposed mapped result splits its flat loop at the divisors of
  the transpose's coordinates, so every index is affine and no quotient remains."""
  stage = _function("delinearize_stage", ((2,),), lambda x: sc.stack([x.sin(), x * x]).reshape((4,)))

  @sc.function(sc.arg("z", 6), outputs=sc.arg("y"), name="delinearize")
  def fn(z):
    mapped = _mapped_call(stage, 3, [(z, 0, 2)]).reshape((3, 2, 2)).transpose((1, 0, 2)).reshape((12,))
    return _scatter_zero(mapped, np.arange(12), 12) + _scatter_zero(z * 0.5, np.arange(0, 12, 2), 12)

  proc = _lowered(fn)
  divides = [n for n in walk_program(proc) if n.op in (ProgramOp.DIV, ProgramOp.MOD)]
  assert not divides
  z = np.linspace(-1, 1, 6)
  stages = np.stack([np.concatenate([np.sin(z[2 * i : 2 * i + 2]), z[2 * i : 2 * i + 2] ** 2]) for i in range(3)]).reshape(3, 2, 2)
  expected = stages.transpose(1, 0, 2).reshape(12)
  expected[::2] += 0.5 * z
  np.testing.assert_allclose(np.asarray(fn(z)), expected, rtol=1e-14)


def _bare_ctx() -> ctx.LowerCtx:
  lctx = object.__new__(ctx.LowerCtx)
  lctx.names = NameScope()
  lctx.statements = []
  return lctx


def _carried(name: str, stop: int) -> ProgramNode:
  rng = p.range_(name, 0, stop)
  return ProgramNode(rng.op, rng.args, {**rng.attrs, "carried": True}, dtype=rng.dtype)


def test_a_carried_range_is_never_merged_split_or_reordered() -> None:
  """Lowering never merges, delinearizes, splits into lanes or moves a reduction across a carried
  range, which runs its trips in order."""
  out = p.buffer("out", p.dtypes.float64, (8, 4))
  x = p.buffer("x", p.dtypes.float64, (32,))
  i, j = p.var("i"), p.var("j")
  lctx = _bare_ctx()
  # Contiguous across both loops, which would merge if neither were carried.
  (loop,) = lctx.nest_body(
    [_carried("i", 8), p.range_("j", 0, 4)],
    [p.store(p.view(out, [p.add(p.mul(i, p.const_int(4)), j)]), p.load(p.view(x, [p.add(p.mul(i, p.const_int(4)), j)])))],
  )
  assert loop.args[0].attrs.get("carried") and loop.args[1].op == ProgramOp.FOR
  ranges, _ = lctx.delinearize(
    [_carried("i", 8)],
    [p.store(p.view(out, [i]), p.load(p.view(x, [p.add(p.mul(p.div(i, p.const_int(2)), p.const_int(4)), p.mod(i, p.const_int(2)))])))],
  )
  assert [r.attrs["name"] for r in ranges] == ["i"]
  lctx = _bare_ctx()
  k = p.var("k")
  emit_reduction(
    lctx, [_carried("i", 8)], [p.range_("k", 0, 4, kind=RangeKind.REDUCE)], p.view(out, [i]), p.load(p.view(x, [p.add(p.mul(k, p.const_int(8)), i)]))
  )
  (stmt,) = lctx.statements
  assert stmt.args[0].attrs.get("carried") and trip_count(stmt.args[0]) == 8


def _carried_loop(proc: ProgramNode) -> ProgramNode:
  (loop,) = [s for s in proc.args if s.op == ProgramOp.FOR and s.args[0].attrs.get("carried")]
  return loop


@needs_cc
def test_loop_operands_are_computed_once_outside_the_carried_range() -> None:
  """A loop reads its operands from buffers, so a value computed for it, whether shared by every step
  or sliced per step, runs once before the carried range and never inside it."""

  @sc.function(sc.arg("c", 3), sc.arg("w", 3), sc.arg("v", 3), outputs=sc.group(sc.arg("cn", 3), sc.arg("y", ())))
  def step(c, w, v):
    return c * w + v, c.sum()

  @sc.function(sc.arg("c0", 3), sc.arg("w", 3), sc.arg("v", (4, 3)), name="loop_operands")
  def fn(c0, w, v):
    # One used result, so the loop is each operand's only reader.
    return sc.scan(step, c0, (sc.broadcast(w.sin()), v.exp()), length=4)[0].block()

  proc = _lowered(fn)
  loop = _carried_loop(proc)
  assert not any(n.op in (ProgramOp.SIN, ProgramOp.EXP) for n in walk_program(loop))
  assert _work(proc, ProgramOp.SIN) == 3 and _work(proc, ProgramOp.EXP) == 12
  rng = np.random.default_rng(31)
  c0, w, v = rng.uniform(-1, 1, 3), rng.uniform(-1, 1, 3), rng.uniform(-1, 1, (4, 3))
  c = c0
  for k in range(4):
    c = c * np.sin(w) + np.exp(v[k])
  np.testing.assert_allclose(np.asarray(fn(c0, w, v)), c, rtol=1e-14)


@needs_cc
def test_loop_body_fuses_within_one_trip_and_keeps_shared_work_apart() -> None:
  """A loop body is its own procedure, where every rule applies within one step. Work on an argument
  every step shares stays a statement of its own, so ``hoist_invariant`` lifts it out of the loop,
  while the per-step chain fuses into one loop."""

  @sc.function(sc.arg("c", 3), sc.arg("w", 3), sc.arg("v", 3), outputs=sc.group(sc.arg("cn", 3), sc.arg("y", ())))
  def step(c, w, v):
    return (c * w.exp() + v).sin() * 0.5, c.sum()

  @sc.function(sc.arg("c0", 3), sc.arg("w", 3), sc.arg("v", (4, 3)), name="loop_body")
  def fn(c0, w, v):
    c, (ys,) = sc.scan(step, c0, (sc.broadcast(w), v), length=4)
    return c.block(), ys

  seen: dict[str, ProgramNode] = {}
  lower_function(fn, observe=seen.__setitem__)
  lowered = seen["lowered"]
  (body,) = [pr for pr in lowered.args[:-1] if any(n.op == ProgramOp.SIN for n in walk_program(pr))]
  # exp(w) is stored apart from the per-step product, and the product, sin and scale are one loop.
  assert len(_privates(body)) == 1
  (fused,) = [s for s in body.args if s.op == ProgramOp.FOR and any(n.op == ProgramOp.SIN for n in walk_program(s))]
  assert any(n.op == ProgramOp.MUL for n in walk_program(fused)) and not any(n.op == ProgramOp.EXP for n in walk_program(fused))
  final = lower_function(fn)
  root = final.args[-1]
  (shared,) = [stmt for stmt in root.args if stmt.op == ProgramOp.CALL]
  shared_proc = next(proc for proc in final.args if proc.attrs["name"] == shared.attrs["callee"])
  assert any(node.op == ProgramOp.EXP for node in walk_program(shared_proc))
  assert root.args.index(shared) < root.args.index(_carried_loop(root)), "the shared work runs once before the loop"
  rng = np.random.default_rng(32)
  c0, w, v = rng.uniform(-1, 1, 3), rng.uniform(-1, 1, 3), rng.uniform(-1, 1, (4, 3))
  c, ys = c0, []
  for k in range(4):
    ys.append(c.sum())
    c = np.sin(c * np.exp(w) + v[k]) * 0.5
  got = fn(c0, w, v)
  np.testing.assert_allclose(got[0], c, rtol=1e-14)
  np.testing.assert_allclose(got[1], ys, rtol=1e-14)
