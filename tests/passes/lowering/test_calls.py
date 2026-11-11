"""Call invocations keep sibling outputs and custom-rule residuals together."""

import numpy as np
import pytest
from scipy import sparse

import scaly as sc
from scaly.function.concrete import ConcreteFunction
from scaly.function.model import as_concrete
from scaly.function.sugar import _mapped_call
from scaly.ir.program import ProgramOp, walk_program
from scaly.ir.types import CanonicalPattern, SparsityPattern
from scaly.passes.lowering import lower_function, main_proc


def _lowered_main(fun):
  stages = {}
  lower_function(fun, observe=lambda phase, program: stages.setdefault(phase, program))
  return main_proc(stages["lowered"])


@pytest.mark.parametrize("length", [0, 1, 5])
def test_multi_output_map_uses_one_loop_and_matches_numpy(length) -> None:
  @sc.function(sc.arg("x", 2), sc.arg("p", ()))
  def pair(x, p):
    return x.sin() + p, x * x, x.sum() * p

  @sc.function(sc.arg("xs", 2 * length + 1), sc.arg("p", ()))
  def outer(xs, p):
    a, _, b = sc.vmap(pair, length)(sc.window(xs, 1, 2), sc.broadcast(p))
    return b, a, b

  xs, p = np.linspace(-0.8, 0.9, 2 * length + 1), np.array(1.3)
  b, a, again = outer(xs, p)
  rows = xs[1:].reshape(length, 2)
  np.testing.assert_allclose(a, np.sin(rows) + p)
  np.testing.assert_allclose(b, rows.sum(axis=1) * p)
  np.testing.assert_array_equal(again, b)
  nodes = walk_program(_lowered_main(outer))
  calls = [node for node in nodes if node.op == ProgramOp.CALL]
  assert len(calls) == int(length > 0)
  if calls:
    assert calls[0].attrs["n_out"] == 3
    assert [arg.op for arg in calls[0].args[2:]] == [ProgramOp.VIEW, ProgramOp.BUFFER, ProgramOp.VIEW]


@pytest.mark.parametrize("mapped", [False, True])
def test_custom_rule_residual_shares_its_primal_invocation(mapped) -> None:
  @sc.function(sc.arg("x", ()))
  def square(x):
    return x * x

  @sc.function()
  def fwd(x):
    return x * x, sc.stack([x, 2.0 * x])

  @sc.function()
  def bwd(residual, cotangent):
    return residual[1] * cotangent

  callee = as_concrete(sc.custom_derivative(square, fwd=fwd, bwd=bwd))
  length = 5 if mapped else 1
  x = sc.sym("x", length if mapped else ())
  if mapped:
    primal, residual = (_mapped_call(callee, length, [(x, 0, 1)], i) for i in range(2))
  else:
    primal, residual = callee._call((x,))
  fn = ConcreteFunction._from_exprs("residual_siblings", [x], [primal, residual], ["x"], ["value", "residual"])
  values = np.linspace(-0.7, 1.1, length) if mapped else np.array(0.3)
  y, res = fn(values)
  np.testing.assert_allclose(y, values * values)
  expected = np.stack([values, 2.0 * values], axis=-1)
  np.testing.assert_allclose(res, expected.reshape(res.shape))
  calls = [node for node in walk_program(_lowered_main(fn)) if node.op == ProgramOp.CALL]
  assert len(calls) == 1
  assert calls[0].attrs["n_out"] == 2


def test_different_map_layouts_are_distinct_invocations() -> None:
  @sc.function(sc.arg("x", 2))
  def pair(x):
    return x.sin(), x.sum()

  @sc.function(sc.arg("xs", 9))
  def outer(xs):
    a, b = sc.vmap(pair, 3)(sc.window(xs, 0, 2))
    c, d = sc.vmap(pair, 3)(sc.window(xs, 1, 2))
    e, f = sc.vmap(pair, 3)(sc.window(xs, 0, 3))
    g, h = sc.vmap(pair, 2)(sc.window(xs, 0, 2))
    return a, b, c, d, e, f, g, h

  values = np.linspace(-0.8, 1.3, 9)
  got = outer(values)
  for i, (start, stride, length) in enumerate(((0, 2, 3), (1, 2, 3), (0, 3, 3), (0, 2, 2))):
    rows = np.stack([values[start + k * stride : start + k * stride + 2] for k in range(length)])
    np.testing.assert_allclose(got[2 * i], np.sin(rows))
    np.testing.assert_allclose(got[2 * i + 1], rows.sum(axis=1))
  assert sum(node.op == ProgramOp.CALL for node in walk_program(_lowered_main(outer))) == 4


@pytest.mark.parametrize("length", [None, 0, 1, 5])
def test_sparse_siblings_preserve_each_declared_layout(length) -> None:
  pattern = CanonicalPattern((3, 4), (0, 0, 1, 2, 2), (0, 3, 1, 0, 2))
  short = CanonicalPattern((2, 4), (0, 0, 1), (0, 3, 1))
  short_layout = SparsityPattern(short.shape, (0, 1, 0), (3, 1, 0))
  wide_type, short_type = sc.TensorType(pattern.shape, pattern=pattern), sc.TensorType(short.shape, pattern=short)

  @sc.function(
    sc.arg("x", wide_type, layout="csc"),
    outputs=sc.group(sc.arg("wide", wide_type, layout="csc"), sc.arg("unused", pattern.shape), sc.arg("short", short_type, layout=short_layout)),
  )
  def stage(x):
    dense = sc.to_dense(x)
    return (2.0 * x).block(), dense, sc.from_dense(dense[:2, :].sin(), short).block()

  shape = pattern.shape if length is None else (length, *pattern.shape)

  @sc.function(sc.arg("xs", sc.TensorType(shape, pattern=pattern), layout="csr"))
  def outer(xs):
    a, _, b = stage(xs) if length is None else sc.vmap(stage, length)(xs)
    return b, a

  count = 1 if length is None else length
  data = np.linspace(-0.9, 1.2, count * pattern.nnz)
  coords = (
    (pattern.rows, pattern.cols)
    if length is None
    else (np.repeat(np.arange(count), pattern.nnz), np.tile(pattern.rows, count), np.tile(pattern.cols, count))
  )
  source = sparse.coo_array((data, coords), shape=shape)
  b, a = outer(source)
  expected = source.toarray()
  np.testing.assert_allclose(a.toarray(), 2.0 * expected)
  np.testing.assert_allclose(b.toarray(), np.sin(expected[..., :2, :]))
  calls = [node for node in walk_program(_lowered_main(outer)) if node.op == ProgramOp.CALL]
  assert len(calls) == int(length != 0)
  if calls:
    assert calls[0].attrs["n_out"] == 3


def test_small_static_loop_unrolls_and_block_hint_keeps_it_rolled() -> None:
  @sc.function(sc.arg("c", ()))
  def step(c):
    return c * 1.25 + 0.1

  def build(hint):
    x = sc.sym("x")
    c, _ = sc.scan(step, x, length=3)
    return ConcreteFunction._from_exprs("static_loop", [x], [c.with_lowering(hint)], ["x"], ["y"])

  expanded, rolled = build("auto"), build("block")
  assert not any(n.op == ProgramOp.FOR and n.args[0].attrs.get("carried") for n in walk_program(_lowered_main(expanded)))
  assert any(n.op == ProgramOp.FOR and n.args[0].attrs.get("carried") for n in walk_program(_lowered_main(rolled)))
  assert expanded(np.array(0.3)).tobytes() == rolled(np.array(0.3)).tobytes()


def _static_scan(step, length, hint="auto", *, reverse=False, trip=None):
  x = sc.sym("x", as_concrete(step).inputs[0].shape)
  c, _ = sc.scan(step, x, length=length, reverse=reverse, trip=trip)
  return ConcreteFunction._from_exprs("static_scan", [x], [c.with_lowering(hint)], ["x"], ["y"])


def _has_carried_loop(fun):
  return any(n.op == ProgramOp.FOR and n.args[0].attrs.get("carried") for n in walk_program(_lowered_main(fun)))


def test_static_loop_budget_boundary_and_scalar_override(monkeypatch) -> None:
  from scaly.passes.lowering import calls

  @sc.function(sc.arg("c", ()))
  def step(c):
    return c * 1.25 + 0.1

  raw = _lowered_main(step)
  cost = calls._loop_cost(raw, {})
  assert cost is not None and cost > 0
  monkeypatch.setattr(calls, "STATIC_LOOP_BUDGET", 3 * cost)
  assert not _has_carried_loop(_static_scan(step, 3))
  assert _has_carried_loop(_static_scan(step, 4))
  assert not _has_carried_loop(_static_scan(step, 4, "scalar"))
  assert _has_carried_loop(_static_scan(step, 1, "block"))
  assert _has_carried_loop(_static_scan(step, 1, "opaque"))
  assert _has_carried_loop(_static_scan(step, 1, "scalar", trip=sc.const(np.int64(1))))


def test_budget_counts_all_solve_rhs_and_nested_loop_iterations() -> None:
  from scaly.ir import program as p
  from scaly.passes.lowering.calls import _loop_cost

  rhs = p.buffer("rhs", sc.dtypes.float64, (100, 8))
  solve = p.proc("solve", [rhs], [p.for_(p.range_("row", 0, 8), [p.store(p.view(rhs, [p.var("row")]), p.const_float(1.0))])])
  many_rhs = p.proc(
    "many_rhs",
    [rhs],
    [p.for_(p.range_("seed", 0, 100), [p.call("solve", [rhs])])],
  )
  assert _loop_cost(solve, {}) == 8
  assert _loop_cost(many_rhs, {"solve": solve}) == 800


@pytest.mark.parametrize("reverse", [False, True])
def test_unrolling_preserves_reduction_order_bitwise(reverse) -> None:
  @sc.function(sc.arg("c", 3))
  def step(c):
    return sc.stack([c.sum(), c[1] * 0.5, c[2] + 0.125])

  expanded = _static_scan(step, 5, "scalar", reverse=reverse)
  rolled = _static_scan(step, 5, "block", reverse=reverse)
  for x in (np.array([1e16, 1.0, -1e16]), np.array([0.1, -0.2, 0.3]), np.array([-0.0, 0.0, -0.0])):
    assert expanded(x).tobytes() == rolled(x).tobytes()


def test_small_static_triangle_folds_masked_terms() -> None:
  from scaly.codegen import render_c_source

  @sc.function(sc.arg("c", 4), sc.arg("j", sc.TensorType((), sc.dtypes.int64)))
  def step(c, j):
    return c + sc.where(sc.const(np.arange(4, dtype=np.int64)) < j, c.sin(), 0.0)

  x = sc.sym("x", 4)
  y, _ = sc.scan(step, x, length=3, index=True)
  expanded = ConcreteFunction._from_exprs("triangle", [x], [y], ["x"], ["y"])
  rolled = ConcreteFunction._from_exprs("triangle", [x], [y.block()], ["x"], ["y"])
  assert not _has_carried_loop(expanded)
  source = render_c_source(expanded)
  assert " ? " not in source and "i_loop" not in source
  assert source.count("sin(") == 3
  v = np.array([0.1, 0.2, 0.3, 0.4])
  assert expanded(v).tobytes() == rolled(v).tobytes()


def test_nested_scan_work_is_included_in_outer_budget() -> None:
  @sc.function(sc.arg("c", 8))
  def rhs(c):
    return c.sin() + c

  @sc.function(sc.arg("c", 8))
  def solve_many(c):
    value, _ = sc.scan(rhs, c, length=100)
    return value

  auto = _static_scan(solve_many, 2)
  assert _has_carried_loop(auto)
  x = np.linspace(-0.1, 0.1, 8)
  np.testing.assert_array_equal(auto(x), _static_scan(solve_many, 2, "block")(x))


def test_condition_and_narrow_store_prevent_expansion() -> None:
  @sc.function(sc.arg("c", ()))
  def step(c):
    return (c.cast(sc.dtypes.float32) + sc.const(np.float32(0.1))).cast(sc.dtypes.float64)

  assert _has_carried_loop(_static_scan(step, 2, "scalar"))

  @sc.function(sc.arg("c", ()))
  def cond(c):
    return c < 2.0

  x = sc.sym("x")
  y, _ = sc.while_loop(cond, step, x, max_iter=2)
  fn = ConcreteFunction._from_exprs("conditional", [x], [y.scalar()], ["x"], ["y"])
  assert _has_carried_loop(fn)


@pytest.mark.parametrize("hints", [("scalar", "block"), ("auto", "scalar")])
def test_duplicate_loop_results_share_storage_and_hints(hints) -> None:
  @sc.function(sc.arg("c", ()))
  def step(c):
    return c * 1.25 + 0.1

  x = sc.sym("x")
  y, _ = sc.scan(step, x, length=3)
  fn = ConcreteFunction._from_exprs("duplicate_loop", [x], [y.with_lowering(h) for h in hints], ["x"], ["a", "b"])
  assert _has_carried_loop(fn) == ("block" in hints)
  a, b = fn(np.array(0.3))
  assert a.tobytes() == b.tobytes()


def test_static_loop_preserves_a_block_callee_boundary() -> None:
  @sc.function(sc.arg("c", ()))
  def blocked(c):
    return (c + 1.0).block()

  assert _has_carried_loop(_static_scan(blocked, 2))
  assert _has_carried_loop(_static_scan(blocked, 2, "scalar"))
