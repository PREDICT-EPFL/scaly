"""Scalar expansion preserves tensor indexing, call boundaries, and derivative values."""

from __future__ import annotations

import math

import numpy as np
import pytest

import alloy as al
from alloy.codegen.c import render_program_c_source
from alloy.ir import program as p
from alloy.ir.program import ProgramNode, ProgramOp
from alloy.ir.types import Lowering, dtypes
from alloy.passes.lowering import lower_function, main_proc
from alloy.passes.program._common import _walk
from alloy.passes.program.scalarize import (
  AUTO_EXPANSION_WORK_PER_PROC,
  AUTO_SCALAR_GROWTH_PER_PROGRAM,
  AUTO_SCALAR_OPS_PER_PROC,
  _fold,
  scalarize_program,
)


def _function(name: str, inputs: list[al.Expr], outputs: list[al.Expr]) -> al.Function:
  return al.Function._from_exprs(name, inputs, outputs, [str(x.name) for x in inputs], [f"out{i}" for i in range(len(outputs))])


def _body(proc: ProgramNode) -> tuple[ProgramNode, ...]:
  return proc.args[proc.attrs["param_count"] :]


def _assert_scalar(proc: ProgramNode) -> None:
  assert proc.attrs.get("scalarized")
  assert all(n.op not in {ProgramOp.FOR, ProgramOp.CALL, ProgramOp.BUFFER} for stmt in _body(proc) for n in _walk(stmt))


@pytest.mark.parametrize("hint", ["auto", "scalar", "block", "opaque"])
def test_hint_selects_callee_without_expanding_mapped_horizon(hint: Lowering) -> None:
  x = al.sym("x", 3)
  stage = _function("hint_stage", [x], [(x.sin() + x * x).with_lowering(hint)])
  procs = []
  for length in (3, 41):
    z = al.sym("z", 3 * length)
    fn = _function("hint_map", [z], [al.vmap(stage, length, [(z, 0, 3)])])
    prog = lower_function(fn)
    callee = prog.args[0]
    if hint in {"auto", "scalar"}:
      _assert_scalar(callee)
    else:
      assert not callee.attrs.get("scalarized")
      assert any(n.op == ProgramOp.FOR for n in _body(callee))
    root = main_proc(prog)
    assert not root.attrs.get("scalarized")
    loops = [s for s in _body(root) if s.op == ProgramOp.FOR]
    assert len(loops) == 1 and loops[0].args[0].args[1].attrs["value"] == length
    data = np.linspace(-0.7, 1.2, 3 * length)
    np.testing.assert_allclose(fn(data), np.sin(data) + data * data, rtol=1e-14, atol=1e-14)
    procs.append(callee)
  assert procs[0] is procs[1]


def test_explicit_scalar_root_and_block_precedence() -> None:
  x = al.sym("x", 4)
  scalar = _function("explicit", [x], [(x.sin() + x * x).scalar()])
  _assert_scalar(main_proc(lower_function(scalar)))
  blocked = _function("blocked", [x], [(x.sin().block() + x * x).scalar()])
  assert not main_proc(lower_function(blocked)).attrs.get("scalarized")
  data = np.arange(4, dtype=float) / 3
  np.testing.assert_array_equal(scalar(data), blocked(data))


def test_scalarization_can_be_reapplied_to_its_result() -> None:
  x = al.sym("x", 3)
  fn = _function("repeated_pass", [x], [(x.sin() * x.sin()).scalar()])
  prog = lower_function(fn)
  _assert_scalar(main_proc(prog))
  assert scalarize_program(prog) is prog


def _policy_proc(name: str, count: int, *, arithmetic: bool = True, lowering: Lowering = "auto") -> ProgramNode:
  x, y = (p.buffer(label, dtypes.float64, (count,)) for label in (f"{name}_x", f"{name}_y"))
  i = p.var(f"{name}_i", dtypes.int64)
  source = p.load(p.view(x, [i]))
  value = p.add(source, p.const_float(1.0)) if arithmetic else source
  loop = p.for_(p.range_(i.attrs["name"], 0, count), [p.store(p.view(y, [i]), value)])
  raw = p.proc(name, [x, y], [loop])
  return ProgramNode(raw.op, raw.args, {**raw.attrs, "input_count": 1, "scalarize": True, "lowering": lowering}, raw.dtype)


def test_auto_uses_folded_scalar_operations_instead_of_tensor_shape() -> None:
  accepted = scalarize_program(p.program([_policy_proc("accepted", AUTO_SCALAR_OPS_PER_PROC)])).args[0]
  rejected = scalarize_program(p.program([_policy_proc("rejected", AUTO_SCALAR_OPS_PER_PROC + 1)])).args[0]
  explicit = scalarize_program(p.program([_policy_proc("explicit_large", AUTO_SCALAR_OPS_PER_PROC + 1, lowering="scalar")])).args[0]
  _assert_scalar(accepted)
  assert not rejected.attrs.get("scalarized")
  _assert_scalar(explicit)

  x = al.sym("x", 64)
  value = x
  for _ in range(AUTO_SCALAR_OPS_PER_PROC // 64 + 1):
    value = value.sin()
  stage = _function("costly_stage", [x], [value])
  outer = _function("costly_outer", [x], [stage(x)])
  assert not lower_function(outer).args[0].attrs.get("scalarized")


def test_auto_code_growth_counts_output_stores_and_aggregates_across_procs() -> None:
  copy = _policy_proc("large_copy", AUTO_SCALAR_GROWTH_PER_PROGRAM + 1, arithmetic=False)
  assert scalarize_program(p.program([copy])).args[0] is copy
  procs = [_policy_proc(f"aggregate_{i}", 2000) for i in range(5)]
  scalarized = scalarize_program(p.program(procs)).args
  assert all(proc.attrs.get("scalarized") for proc in scalarized[:4])
  assert not scalarized[4].attrs.get("scalarized")

  explicit = _policy_proc("explicit_capacity", 6000, lowering="scalar")
  auto = _policy_proc("auto_after_explicit", 3000)
  scalarized = scalarize_program(p.program([explicit, auto])).args
  _assert_scalar(scalarized[0])
  assert scalarized[1] is auto


def test_auto_counts_shared_post_fold_computation_once() -> None:
  count = 5000
  x, y = (p.buffer(name, dtypes.float64, (size,)) for name, size in (("shared_x", 1), ("shared_y", count)))
  i = p.var("shared_i", dtypes.int64)
  shared = ProgramNode(ProgramOp.SIN, (p.load(p.view(x, [p.const_int(0)])),), dtype=dtypes.float64)
  loop = p.for_(p.range_(i.attrs["name"], 0, count), [p.store(p.view(y, [i]), p.add(shared, p.const_float(0)))])
  raw = p.proc("shared_computation", [x, y], [loop])
  proc = ProgramNode(raw.op, raw.args, {**raw.attrs, "input_count": 1, "scalarize": True, "lowering": "auto"}, raw.dtype)
  scalarized = scalarize_program(p.program([proc])).args[0]
  _assert_scalar(scalarized)
  nodes = {node for stmt in _body(scalarized) for node in _walk(stmt)}
  assert sum(node.op == ProgramOp.SIN for node in nodes) == 1
  assert not any(node.op == ProgramOp.ADD for node in nodes)


def test_auto_preflight_bounds_buffer_materialization() -> None:
  scratch = p.buffer("huge_scratch", dtypes.float64, (AUTO_EXPANSION_WORK_PER_PROC + 1,), address_space="private")
  raw = p.proc("huge_allocation", [], [scratch])
  proc = ProgramNode(raw.op, raw.args, {**raw.attrs, "input_count": 0, "scalarize": True, "lowering": "auto"}, raw.dtype)
  assert scalarize_program(p.program([proc])).args[0] is proc


def test_auto_rejects_call_when_callee_exceeds_budget() -> None:
  callee = _policy_proc("expensive_callee", AUTO_SCALAR_OPS_PER_PROC + 1)
  x, y = (p.buffer(name, dtypes.float64, (AUTO_SCALAR_OPS_PER_PROC + 1,)) for name in ("caller_x", "caller_y"))
  raw = p.proc("caller", [x, y], [p.call(callee.attrs["name"], [x, y])])
  caller = ProgramNode(raw.op, raw.args, {**raw.attrs, "input_count": 1, "scalarize": True, "lowering": "auto"}, raw.dtype)
  result = scalarize_program(p.program([callee, caller]))
  assert result.args == (callee, caller)


@pytest.mark.parametrize("dtype", [dtypes.float32, dtypes.int32, dtypes.int64])
def test_other_dtypes_keep_their_store_boundaries(dtype) -> None:
  x = al.sym("x", 4, dtype=dtype)
  y = (x + al.const([1, 2, 3, 4], dtype=dtype)).scalar()
  fn = _function("typed_stage", [x], [y])
  assert not main_proc(lower_function(fn)).attrs.get("scalarized")


def test_views_broadcast_gather_and_scatter() -> None:
  x = al.sym("x", (3, 4))
  sliced = x[::-1, 1::2]
  broadcast = sliced + x[1:2, 1::2]
  moved = al.concat([broadcast.T, sliced.T], axis=1).reshape((12,))
  gathered = al.gather(moved, np.array([[11, 0, 4], [3, 9, 1]]))
  scattered = al.scatter(gathered.reshape((6,)), np.array([3, 1, 4, 0, 5, 2]), (7,))
  fn = _function("views", [x], [al.stack([scattered, scattered * 2], axis=1).scalar(), x[1:, :][1:, 1:3]])
  _assert_scalar(main_proc(lower_function(fn)))
  data = np.arange(12, dtype=float).reshape(3, 4) / 7
  s = data[::-1, 1::2]
  moved_np = np.concatenate([(s + data[1:2, 1::2]).T, s.T], axis=1).reshape(-1)
  result = np.zeros(7)
  result[[3, 1, 4, 0, 5, 2]] = moved_np[[11, 0, 4, 3, 9, 1]]
  actual, alias = fn(data)
  np.testing.assert_array_equal(actual, np.stack([result, result * 2], axis=1))
  np.testing.assert_array_equal(alias, data[2:, 1:3])


def test_scalar_constants_cse_and_expression_trees() -> None:
  x = al.sym("v0", 3)
  a = al.gather(x, np.array([2, 0, 2])).sin()
  b = al.gather(x, np.array([0, 2, 0])).sin()
  masked = a * al.const([1.0, 0.0, 2.0]) + b * al.const([0.0, 1.0, 0.0])
  constants = al.const([0.0, 1.0, 2.0]).sin()
  fn = _function("folded", [x], [(masked + constants).scalar()])
  proc = main_proc(lower_function(fn))
  _assert_scalar(proc)
  nodes = {n for s in _body(proc) for n in _walk(s)}
  assert sum(n.op == ProgramOp.SIN for n in nodes) == 1
  assert all(n.attrs["target"] != "v0" for n in _body(proc) if n.op == ProgramOp.ASSIGN)
  source = render_program_c_source(fn)
  assert "static const" not in source
  assert source.count("sin(") == 1
  assert any(s.op == ProgramOp.STORE and s.args[1].op == ProgramOp.ADD and s.args[1].args[0].op == ProgramOp.MUL for s in _body(proc))
  data = np.array([0.2, -0.4, 0.9])
  np.testing.assert_allclose(fn(data), np.sin(data[2]) * np.array([1, 1, 2]) + np.sin([0, 1, 2]), rtol=1e-14, atol=1e-14)


def test_nested_calls_offsets_multiple_outputs_and_repeated_invocations() -> None:
  x = al.sym("x", 3)
  inner = _function("nested_inner", [x], [x * x, x.sum()])
  z = al.sym("z", 7)
  a, b = inner(z[1:4])
  c, d = inner(z[3:6])
  again, _ = inner(z[1:4])
  fn = _function("nested_outer", [z], [(a + c + again).scalar(), b + d, a, al.const([2.0, 3.0])])
  _assert_scalar(main_proc(lower_function(fn)))
  data = np.arange(7, dtype=float) / 4
  got = fn(data)
  expected = [2 * data[1:4] ** 2 + data[3:6] ** 2, data[1:4].sum() + data[3:6].sum(), data[1:4] ** 2, [2, 3]]
  for actual, ref in zip(got, expected, strict=True):
    np.testing.assert_array_equal(actual, ref)


def test_block_callee_keeps_its_call_boundary() -> None:
  x = al.sym("x", 4)
  inner = _function("block_inner", [x], [x.sin().block()])
  fn = _function("block_outer", [x], [(inner(x) * x).scalar()])
  prog = lower_function(fn)
  assert not any(pr.attrs.get("scalarized") for pr in prog.args)
  assert any(n.op == ProgramOp.CALL for n in _walk(main_proc(prog)))
  data = np.linspace(0, 1, 4)
  np.testing.assert_allclose(fn(data), np.sin(data) * data, rtol=1e-14, atol=1e-14)


def test_reductions_keep_left_to_right_order_and_handle_empty_inputs() -> None:
  x = al.sym("x", 4)
  empty = al.sym("empty", 0)
  fn = _function("reduce", [x, empty], [x.sum().scalar(), x @ al.const(np.ones(4)), empty.sum(), empty])
  _assert_scalar(main_proc(lower_function(fn)))
  data = np.array([1e16, 1.0, -1e16, 1.0])
  total, dot, zero, blank = fn((data, np.empty(0)))
  assert total == dot == 1.0
  assert zero == 0.0 and blank.size == 0


def test_repeated_stores_and_aliases_read_the_latest_value() -> None:
  out = p.buffer("out", dtypes.float64, (2,))
  temp = p.buffer("temp", dtypes.float64, (3,), address_space="private")
  alias = p.buffer("alias", dtypes.float64, (1,), address_space="private")
  alias = ProgramNode(alias.op, alias.args, {**alias.attrs, "alias_of": "temp", "alias_offset": 2}, alias.dtype)
  v = p.view(alias, [p.const_int(0)])
  read = p.load(v)
  assert p.load(v) is read
  raw = p.proc(
    "stores",
    [out],
    [
      temp,
      alias,
      p.store(p.view(temp, [p.const_int(2)]), p.const_float(2)),
      p.for_(p.range_("i", 1, 4), [p.store(v, p.mul(p.load(v), p.const_float(2)))]),
      p.store(p.view(out, [p.const_int(0)]), read),
      p.store(v, p.const_float(99)),
      p.store(p.view(out, [p.const_int(1)]), read),
    ],
  )
  proc = ProgramNode(raw.op, raw.args, {**raw.attrs, "input_count": 0, "scalarize": True, "lowering": "scalar"}, raw.dtype)
  scalar = scalarize_program(p.program([proc])).args[0]
  _assert_scalar(scalar)
  assert len(_body(scalar)) == 2
  assert [stmt.args[1].attrs["value"] for stmt in _body(scalar)] == [16, 99]


def test_small_nested_maps_expand_inside_a_stage() -> None:
  x = al.sym("x", 2)
  inner = _function("small_map_inner", [x], [x * x, x.sum()])
  z = al.sym("z", 8)
  mapped = al.vmap(inner, 3, [(z, 1, 2)], output=1)
  stage = _function("small_map_stage", [z], [mapped])
  outer = _function("small_map_outer", [z], [stage(z)])
  proc = next(pr for pr in lower_function(outer).args if pr.attrs["name"] == stage.name)
  _assert_scalar(proc)
  data = np.arange(8, dtype=float) / 3
  np.testing.assert_array_equal(outer(data), data[1:7].reshape(3, 2).sum(axis=1))


def test_forward_over_reverse_folds_constant_seeds_and_matches_analytic_hessian() -> None:
  dist = al.sym("dist", 3)
  link = _function("spring_link", [dist], [(1 - 0.3 / al.norm_2(dist)) * dist])
  x, lam = al.sym("x", 6), al.sym("lam", 6)
  accel = link(x[3:] - x[:3])
  residual = al.concat([x[:3] + 0.1 * accel, x[3:] - 0.1 * accel])
  grad = al.vjp((residual,), (x,), (lam,))[0]
  seeds = al.sym("seeds", (6, 6))
  symbolic = _function("spring_symbolic", [x, lam, seeds], [al.jvp_many(grad, x, seeds).scalar()])
  baked = _function("spring_baked", [x, lam], [al.jvp_many(grad, x, al.const(np.eye(6))).scalar()])
  procs = [main_proc(lower_function(fn)) for fn in (symbolic, baked)]
  for proc in procs:
    _assert_scalar(proc)
  arithmetic = lambda pr: sum(n.op in {ProgramOp.ADD, ProgramOp.SUB, ProgramOp.MUL, ProgramOp.DIV} for s in _body(pr) for n in _walk(s))
  assert arithmetic(procs[1]) < arithmetic(procs[0]) * 2 / 3
  rng = np.random.default_rng(92)
  for _ in range(3):
    xv, lv = rng.normal(size=6), rng.normal(size=6)
    d, w = xv[3:] - xv[:3], 0.1 * (lv[:3] - lv[3:])
    r = np.linalg.norm(d)
    h = 0.3 * ((np.outer(w, d) + np.outer(d, w) + (w @ d) * np.eye(3)) / r**3 - 3 * (w @ d) * np.outer(d, d) / r**5)
    ref = np.block([[h, -h], [-h, h]])
    np.testing.assert_allclose(baked((xv, lv)), ref, rtol=1e-12, atol=1e-14)
    seed_values = rng.normal(size=(6, 6))
    np.testing.assert_allclose(symbolic((xv, lv, seed_values)), seed_values @ ref, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("left,right", [((3,), (3,)), ((5, 3), (3,)), ((3,), (3, 5)), ((5, 3), (3, 4))])
def test_scalar_matmul_matches_numpy(left: tuple[int, ...], right: tuple[int, ...]) -> None:
  x, y = al.sym("x", left), al.sym("y", right)
  fn = _function("scalar_mm", [x, y], [(x @ y).scalar()])
  _assert_scalar(main_proc(lower_function(fn)))
  rng = np.random.default_rng(8)
  xv, yv = rng.normal(size=left), rng.normal(size=right)
  np.testing.assert_allclose(fn((xv, yv)), xv @ yv, rtol=1e-14, atol=1e-14)


def test_deep_scalar_reduction_has_bounded_expression_trees() -> None:
  x = al.sym("x", 1500)
  fn = _function("deep_reduce", [x], [x.sum().scalar()])
  proc = main_proc(lower_function(fn))
  _assert_scalar(proc)
  assert any(n.op == ProgramOp.ASSIGN for n in _body(proc))
  data = np.arange(1500, dtype=float) / 2
  np.testing.assert_array_equal(fn(data), data.sum())


def test_constant_index_division_uses_c_truncation_without_float_rounding() -> None:
  for x, y in [(-7, 3), (7, -3), (-7, -3), (2**62 + 3, 7)]:
    args = p.const_int(x), p.const_int(y)
    q = _fold(ProgramOp.DIV, args, dtypes.int64).attrs["value"]
    r = _fold(ProgramOp.MOD, args, dtypes.int64).attrs["value"]
    assert x == q * y + r
    assert abs(r) < abs(y)
    assert not r or (r < 0) == (x < 0)


@pytest.mark.parametrize("op,values", [(ProgramOp.DIV, (0.0, 0.0)), (ProgramOp.SQRT, (-1.0,)), (ProgramOp.EXP, (1000.0,)), (ProgramOp.LOG, (0.0,))])
def test_invalid_constant_math_remains_a_runtime_operation(op: ProgramOp, values: tuple[float, ...]) -> None:
  assert _fold(op, tuple(p.const_float(v) for v in values), dtypes.float64).op == op


def test_symbolic_zero_identities_and_known_invalid_constants_have_distinct_semantics() -> None:
  x = al.sym("x", 4)
  fn = _function("symbolic_zero", [x], [((x * 0) + (0 / x)).scalar()])
  values = np.array([np.nan, np.inf, -np.inf, 0.0])
  np.testing.assert_array_equal(fn(values), np.zeros(4))
  zero_div_zero = _fold(ProgramOp.DIV, (p.const_float(0), p.const_float(0)), dtypes.float64)
  infinity_times_zero = _fold(ProgramOp.MUL, (p.const_float(math.inf), p.const_float(0)), dtypes.float64)
  assert zero_div_zero.op == ProgramOp.DIV
  assert infinity_times_zero.op == ProgramOp.CONST_FLOAT and math.isnan(infinity_times_zero.attrs["value"])


def test_unfolded_constant_division_uses_floating_point_in_c() -> None:
  x = al.sym("x", 1)
  fn = _function("invalid_division", [x], [(al.const([0.0, 1.0]) / al.const([0.0, 0.0])).scalar()])
  source = render_program_c_source(fn)
  assert "(0.0 / 0.0)" in source and "(1.0 / 0.0)" in source
  result = fn(np.ones(1))
  assert np.isnan(result[0]) and np.isposinf(result[1])


def test_mixed_constant_math_folds_per_element() -> None:
  x = al.sym("x", 3)
  y = al.const([0.0, 1.0, 4.0]).sqrt() + al.const([2.0, 3.0, 4.0]) ** al.const([0.0, 1.0, 2.0])
  fn = _function("constant_math", [x], [(x + y).scalar()])
  proc = main_proc(lower_function(fn))
  assert not any(n.op in {ProgramOp.SQRT, ProgramOp.POW} for n in _walk(proc))
  np.testing.assert_array_equal(fn(np.array([2.0, 3.0, 4.0])), [3, 7, 22])
  assert _fold(ProgramOp.SIN, (p.const_float(1.0),), dtypes.float64).attrs["value"] == math.sin(1.0)
