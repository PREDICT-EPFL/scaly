"""Unit tests for the Program IR optimization passes (``alloy.passes.program``).

These exercise the passes directly on lowered Program IR (structural assertions) plus a couple of
end-to-end numeric checks under the Program IR renderer. See ``docs/how_it_works/lowering.md``
(Step 5b): fusion + workspace packing/spilling + contiguous-slice aliasing.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

import alloy as al
from alloy.codegen.c import render_program_c_source
from alloy.codegen.jit import _find_compiler
from alloy.passes.lowering import lower_function, main_proc
from alloy.passes.program.combine_scatter_sums import combine_scatter_sums
from alloy.passes.program.hoist_invariant import hoist_invariant
from alloy.passes.program._common import _walk
from alloy.passes.program.pack_workspace import WORKSPACE_SPILL_THRESHOLD
from alloy.passes.program.unroll_unit_loops import unroll_unit_loops
from alloy.ir.program import (
  ProgramNode,
  ProgramOp,
  add,
  buffer,
  const_float,
  const_int,
  for_,
  load,
  mul,
  proc as proc_,
  program,
  range_,
  store,
  var,
  view,
)
from alloy.ir.types import dtypes

_HAVE_CC = _find_compiler() is not None


def _main_body(fn: al.Function) -> list:
  proc = main_proc(lower_function(fn))
  pc = int(proc.attrs["param_count"])
  return list(proc.args[pc:])


def _classify(body: list) -> tuple[list, list, list]:
  """(compute_buffers, alias_buffers, for_loops) among the proc's top-level statements."""
  compute = [s for s in body if s.op == ProgramOp.BUFFER and s.attrs.get("address_space") == "private" and "alias_of" not in s.attrs]
  aliases = [s for s in body if s.op == ProgramOp.BUFFER and "alias_of" in s.attrs]
  loops = [s for s in body if s.op == ProgramOp.FOR]
  return compute, aliases, loops


def _sz_w(fn: al.Function) -> int:
  return int(main_proc(lower_function(fn)).attrs.get("sz_w", 0))


# --- fusion ---------------------------------------------------------------------


def test_fusion_collapses_elementwise_chain() -> None:
  """A same-shape elementwise chain fuses into a single loop with no intermediate buffers."""

  @al.function(al.G(al.L("x", 8), al.L("y", 8)), al.L("out0", ...), name="chain")
  def f(inputs):
    x, y = inputs
    return ((x.sin() + y) * y - x).tanh()

  compute, aliases, loops = _classify(_main_body(f))
  assert compute == []  # every intermediate inlined into the output loop
  assert len(loops) == 1


def _long_scalar_chain(length: int, *, scalar: bool = False) -> al.Function:
  """``x[0]*1 + x[1]*2 + ...`` accumulated as a left fold, so the expression is ``length`` deep."""
  x = al.sym("x", 4)
  acc = al.const(0.0)
  for i in range(length):
    acc = acc + x[i % 4] * float(i + 1)
  out = acc.reshape((1,))
  return al.Function._from_exprs(f"fold{length}", [x], [out.scalar() if scalar else out], ["x"], ["y"])


def _long_scalar_chain_numpy(length: int, x: np.ndarray) -> float:
  return sum(x[i % 4] * float(i + 1) for i in range(length))


def _flat_stage_fold(stages: int, n: int = 4) -> al.Function:
  """Per-stage ``slice * weights`` products summed flat, one term at a time, as a left fold."""
  x = al.sym("x", stages * n)
  w = al.sym("w", n)
  terms = [(x[i * n : (i + 1) * n] * w)[k] for i in range(stages) for k in range(n)]
  total = sum(terms, al.const(0.0))
  return al.Function._from_exprs(f"stages{stages}", [x, w], [total.reshape((1,))], ["x", "w"], ["y"])


@pytest.mark.skipif(not _HAVE_CC, reason="no C compiler for the deep-fold witnesses")
@pytest.mark.parametrize("length", [400, 3000])
def test_deep_scalar_fold_lowers_renders_and_runs(length: int) -> None:
  """Depth in the expression must not become depth on the Python stack or in the C source: the
  passes and renderer are iterative, and the renderer splits trees at ``MAX_SCALAR_DEPTH``."""
  f = _long_scalar_chain(length)
  assert "_h0 = " in render_program_c_source(f)
  x = np.arange(1.0, 5.0)
  np.testing.assert_allclose(f(x), _long_scalar_chain_numpy(length, x), rtol=1e-12)


@pytest.mark.skipif(not _HAVE_CC, reason="no C compiler for the deep-fold witnesses")
def test_deep_scalar_fold_through_scalarize_runs() -> None:
  f = _long_scalar_chain(3000, scalar=True)
  assert main_proc(lower_function(f)).attrs.get("scalarized")
  x = np.arange(1.0, 5.0)
  np.testing.assert_allclose(f(x), _long_scalar_chain_numpy(3000, x), rtol=1e-12)


@pytest.mark.skipif(not _HAVE_CC, reason="no C compiler for the deep-fold witnesses")
def test_flat_per_stage_fold_at_a_hundred_stages_runs() -> None:
  """The NPMPC-shaped objective written as a flat Python fold over per-stage products."""
  f = _flat_stage_fold(100)
  x = np.linspace(-1.0, 1.0, 400)
  w = np.arange(1.0, 5.0)
  np.testing.assert_allclose(f((x, w)), (x.reshape(100, 4) * w).sum(), rtol=1e-12)


def test_fusion_skips_matmul_operand() -> None:
  """A matmul operand must NOT be inlined: a contraction reads each operand element m*k times,
  so inlining the producer's expression there multiplies compute. Regression guard for the
  blow-up that an unguarded fusion introduces."""

  @al.function(al.G(al.L("A", (4, 4)), al.L("x", 4)), al.L("out0", ...), name="mm_operand")
  def f(inputs):
    A, x = inputs
    return A @ x.sin()  # sin(x) is the matvec operand — must stay materialized

  compute, _aliases, _loops = _classify(_main_body(f))
  assert len(compute) == 1  # exactly the materialized sin(x); the A@x result aliases the output
  assert compute[0].attrs["shape"] == (4,)


def test_fusion_into_reduction() -> None:
  """An elementwise producer feeding a SUM fuses into the reduce loop (each element read once)."""

  @al.function(al.L("x", 8), al.L("out0", ...), name="sumf")
  def f(x):
    return (x.sin() + x).sum()

  compute, _aliases, loops = _classify(_main_body(f))
  # Only the scalar accumulator survives; the elementwise buffer is gone (inlined into the reduce).
  assert all(s.attrs["shape"] == (1,) for s in compute)
  assert len(loops) == 1  # the single reduce loop


# --- unit-loop unrolling ---------------------------------------------------------


def test_unit_loop_unrolls_scalar_elementwise_output() -> None:
  @al.function(al.G(al.L("x", 1), al.L("y", 1)), al.L("out0", ...), name="scalar_add")
  def f(inputs):
    x, y = inputs
    return x + y

  _compute, _aliases, loops = _classify(_main_body(f))
  assert loops == []


def test_unit_loop_pass_removes_empty_and_substitutes_only_value() -> None:
  x = buffer("x", dtypes.float64, (1,))
  y = buffer("y", dtypes.float64, (5,))
  i = var("i")
  j = var("j")
  empty = for_(range_("j", 4, 4), [store(view(y, [j]), const_float(2.0))])
  one = for_(range_("i", 3, 4), [store(view(y, [i]), const_float(1.0)), empty])
  prog = program([proc_("unit", [x, y], [one])])

  lowered_proc = unroll_unit_loops(prog).args[0]
  body = list(lowered_proc.args[int(lowered_proc.attrs["param_count"]) :])
  assert len(body) == 1
  assert body[0].op == ProgramOp.STORE
  assert body[0].args[0].args[0].op == ProgramOp.CONST_INT
  assert body[0].args[0].args[0].attrs["value"] == 3


# --- contiguous-slice aliasing --------------------------------------------------


def test_contiguous_slice_aliases_source() -> None:
  """A contiguous slice becomes a zero-copy pointer alias (no copy loop), unlike a strided one."""

  @al.function(al.L("x", 8), al.L("out0", ...), name="slc")
  def f(x):
    s = x[2:6]
    return (s * s).sin()  # use s twice so it stays materialized (not inlined) -> visible alias

  _compute, aliases, _loops = _classify(_main_body(f))
  assert len(aliases) == 1
  assert aliases[0].attrs["alias_of"] == "x"
  assert aliases[0].attrs["alias_offset"] == 2


def test_strided_slice_is_not_aliased() -> None:
  @al.function(al.L("x", 8), al.L("out0", ...), name="strided")
  def f(x):
    return x[::2] + x[1::2]  # strided -> no contiguous offset -> no alias

  _compute, aliases, _loops = _classify(_main_body(f))
  assert aliases == []


# --- workspace packing / spilling -----------------------------------------------


def test_no_spill_when_temps_small() -> None:
  @al.function(al.G(al.L("x", 8), al.L("y", 8)), al.L("out0", ...), name="small")
  def f(inputs):
    x, y = inputs
    return (x.sin() + y).tanh()

  assert _sz_w(f) == 0  # everything stays on the stack


def test_packing_reuses_slots_and_spills() -> None:
  """Two disjoint-lifetime 1600-element matmul temps share ONE spilled slot, so sz_w is 1600
  (not 3200) — proving both lifetime slot-reuse and the >= 1024 spill-to-w[] threshold."""

  @al.function(al.G(al.L("A", (40, 40)), al.L("B", (40, 40))), al.L("out0", ...), name="spill")
  def f(inputs):
    A, B = inputs
    c = (A @ B).sum()  # C (1600) lives only until this reduce
    d = (B @ A).sum()  # D (1600) is born after C is dead -> reuses C's slot
    return c + d

  assert 1600 >= WORKSPACE_SPILL_THRESHOLD
  assert _sz_w(f) == 1600


@pytest.mark.skipif(not _HAVE_CC, reason="no C compiler for the numeric workspace collision check")
def test_workspace_slots_do_not_collide_with_output_names() -> None:
  x = al.sym("x", 4)
  matrix = x.reshape((2, 2))
  square = matrix @ matrix
  cube = square @ matrix
  f = al.Function._from_exprs("slot_collision", [x], [square + cube, square - cube], ["x"], ["s1", "s2"])

  data = np.arange(1.0, 5.0)
  square_ref = data.reshape((2, 2)) @ data.reshape((2, 2))
  cube_ref = square_ref @ data.reshape((2, 2))
  got_sum, got_difference = f(data)
  np.testing.assert_allclose(got_sum, square_ref + cube_ref)
  np.testing.assert_allclose(got_difference, square_ref - cube_ref)


def test_call_output_does_not_reuse_slot_that_produced_input() -> None:
  """Regression for the macOS Apple-clang inline-callee miscompile.

  The bad packed shape was ``inner_fwd2_y_x_raw(s0, s2, s1)``: ``s2`` was computed from
  ``s1``, then the call output reused ``s1``. That lifetime reuse is legal IR, but an older
  Apple clang rematerialized through the clobbered slot after inlining. The packer now keeps
  call-input producer closures live through the call, so the output lands in a distinct slot
  while raw callees can remain inlineable.
  """
  x = al.sym("x", 2)
  inner = al.Function._from_exprs("inner", [x], [x.sin() + x * x], ["x"], ["y"])
  z = al.sym("z", 2)
  inner_z = inner(z * z)
  outer = al.Function._from_exprs("outer", [z], [inner_z], ["z"], ["y"])
  jf = outer.factory("J", ["z"], [al.factory.Jac("y", "z")])

  source = render_program_c_source(jf)
  declaration = re.search(r"static __attribute__\(\(noinline\)\) void (inner_fwd2\w+_raw)\(", source)
  assert declaration is not None
  match = re.search(re.escape(declaration.group(1)) + r"\(([^)]*)\);", source)
  assert match is not None
  args = [a.strip() for a in match.group(1).split(",")]
  assert args[:3] == ["s0", "s3", "s4"]


def test_regular_raw_callees_stay_inline() -> None:
  x = al.sym("x", 2)
  inner = al.Function._from_exprs("inner", [x], [x.sin()], ["x"], ["y"])
  z = al.sym("z", 2)
  inner_z = inner(z)
  outer = al.Function._from_exprs("outer", [z], [inner_z + 1.0], ["z"], ["out"])

  source = render_program_c_source(outer)
  assert "static inline void inner_raw" in source
  assert "noinline" not in source


@pytest.mark.skipif(not _HAVE_CC, reason="no C compiler for the numeric spill check")
def test_spilled_function_matches_numpy() -> None:
  """End-to-end: a function whose temporaries spill to w[] still computes correctly (the JIT
  allocates w from the rendered sz_w and passes it through)."""

  @al.function(al.G(al.L("A", (40, 40)), al.L("B", (40, 40))), al.L("out0", ...), name="spill_num")
  def f(inputs):
    A, B = inputs
    return ((A @ B) + (B @ A)).sum()

  assert _sz_w(f) >= WORKSPACE_SPILL_THRESHOLD  # the (40,40) matmul temp spills
  render_program_c_source(f)  # loud coverage gate
  f.recompile()
  rng = np.random.default_rng(0)
  a, b = rng.standard_normal((40, 40)), rng.standard_normal((40, 40))
  got = f((a, b))
  ref = ((a @ b) + (b @ a)).sum()
  np.testing.assert_allclose(np.asarray(got).reshape(-1), np.asarray(ref).reshape(-1), rtol=1e-9, atol=1e-10)


def test_optimized_program_still_verifies() -> None:
  """The pass pipeline output must pass the Program IR verifier (lower_function asserts this)."""

  @al.function(al.L("x", 6), al.L("out0", ...), name="verif")
  def f(x):
    m = x.reshape((2, 3))
    return (m @ x[:3]).sin() + x[3:5]  # (2,3)@(3,) -> (2,), + x[3:5] (2,)

  prog = lower_function(f)  # raises VerifyError on any malformed node
  assert prog.op == ProgramOp.PROGRAM


@pytest.mark.parametrize("count", [2, 8])
def test_slice_gradient_combines_pads(count: int) -> None:
  x = al.sym("x", 4 * count)
  cost = sum(((x[2 * i : 2 * i + 3] ** 2).sum() for i in range(count)), start=al.const(0.0))
  f = al.Function._from_exprs("slice_cost", [x], [cost], ["x"], ["cost"])
  grad = f.factory("slice_grad", ["x"], [al.factory.Grad("cost", "x")])
  stages = {}
  lower_function(grad, observe=lambda name, prog: stages.__setitem__(name, prog))
  combined = main_proc(stages["pass:combine_scatter_sums"])
  buffers, _, _ = _classify(list(combined.args[int(combined.attrs["param_count"]) :]))
  assert not [b for b in buffers if b.attrs["shape"] == (4 * count,)]
  compute, _, loops = _classify(_main_body(grad))
  assert not [b for b in compute if b.attrs["shape"] == (4 * count,)]
  output_loops = [loop for loop in loops if loop.args[1].op == ProgramOp.STORE and loop.args[1].args[0].attrs["buffer"] == "grad_cost_x"]
  assert len(output_loops) == count + 1
  assert sum(loop.args[1].args[1].op == ProgramOp.CONST_FLOAT for loop in loops if len(loop.args) == 2 and loop.args[1].op == ProgramOp.STORE) == 1
  data = np.linspace(-1.0, 2.0, 4 * count)
  expected = np.zeros_like(data)
  for i in range(count):
    expected[2 * i : 2 * i + 3] += 2 * data[2 * i : 2 * i + 3]
  np.testing.assert_allclose(grad(data), expected)


@pytest.mark.parametrize("shared", [False, True])
def test_scatter_sum_preserves_overlaps_and_shared_outputs(shared: bool) -> None:
  from alloy.ir.expr import scatter

  x = al.sym("x", 3)
  a = scatter(x, [1, 3, 5], 8)
  b = scatter(2 * x, [3, 5, 7], 8)
  c = scatter(-x, [0, 3, 7], 8)
  outputs = [a + (b + c), a] if shared else [a + (b + c)]
  f = al.Function._from_exprs("scatter_sum", [x], outputs, ["x"], ["sum", "a"] if shared else ["sum"])
  data = np.array([-1.5, 2.0, 0.25])
  expected = np.zeros(8)
  expected[[1, 3, 5]] += data
  expected[[3, 5, 7]] += 2 * data
  expected[[0, 3, 7]] -= data
  result = f(data)
  np.testing.assert_allclose(result[0] if shared else result, expected)
  if shared:
    original = np.zeros(8)
    original[[1, 3, 5]] = data
    np.testing.assert_array_equal(result[1], original)


@pytest.mark.parametrize("alias", [False, True])
def test_scatter_sum_does_not_move_source_reads_past_writes(alias: bool) -> None:
  from alloy.ir.expr import scatter
  from alloy.ir.program import ProgramNode, const_int

  x = al.sym("x", 4)
  a = scatter(x[:2] if alias else x, [0, 2] if alias else [0, 2, 4, 6], 8)
  b = scatter(x[2:] if alias else 2 * x, [1, 3] if alias else [1, 3, 5, 7], 8)
  f = al.Function._from_exprs("mutated_scatter", [x], [a + b], ["x"], ["sum"])
  stages = {}
  lower_function(f, observe=lambda name, prog: stages.__setitem__(name, prog))
  lowered = stages["lowered"]
  original = main_proc(lowered)
  params = list(original.args[: int(original.attrs["param_count"])])
  body = list(original.args[len(params) :])
  body.insert(-1, store(view(params[0], [const_int(0)]), const_float(42.0)))
  mutated = ProgramNode(ProgramOp.PROC, (*params, *body), original.attrs, original.dtype)
  prog = program([mutated])
  assert combine_scatter_sums(prog) is prog


@pytest.mark.parametrize("left_associated", [False, True])
def test_scatter_sum_preserves_addition_grouping(left_associated: bool) -> None:
  from alloy.ir.expr import scatter

  x = al.sym("x", 3)
  a, b, c = (scatter(x[i : i + 1], [1], 4) for i in range(3))
  f = al.Function._from_exprs("grouped_scatter", [x], [(a + b) + c if left_associated else a + (b + c)], ["x"], ["sum"])
  np.testing.assert_array_equal(f(np.array([1e16, -1e16, 1.0])), [0, 1 if left_associated else 0, 0, 0])


def test_scatter_sum_combines_reshaped_scatters() -> None:
  """The chain Hessian sums flat scatters through a reshape: same size, different declared shape."""
  from alloy.ir.expr import scatter

  x = al.sym("x", 3)
  index_sets = [[0, 3, 6], [1, 4, 7], [2, 5, 6], [0, 1, 2]]
  terms = [scatter((i + 1) * x, idx, 8).reshape((2, 4)) for i, idx in enumerate(index_sets)]
  f = al.Function._from_exprs("reshaped_scatter_sum", [x], [sum(terms[1:], start=terms[0])], ["x"], ["sum"])
  _, _, loops = _classify(_main_body(f))
  zero_fills = [
    loop for loop in loops if len(loop.args) == 2 and loop.args[1].op == ProgramOp.STORE and loop.args[1].args[1].op == ProgramOp.CONST_FLOAT
  ]
  assert len(zero_fills) == 1
  data = np.array([-1.5, 2.0, 0.25])
  expected = np.zeros(8)
  for i, idx in enumerate(index_sets):
    expected[idx] += (i + 1) * data
  np.testing.assert_allclose(f(data), expected.reshape(2, 4))


# --- loop-invariant hoisting ---------------------------------------------------------


def _mapped_mlp(stages: int, *, broadcast: bool) -> tuple[al.Function, np.ndarray, np.ndarray]:
  """``sin(exp(W) @ x_i)`` per stage; ``W`` is one broadcast matrix or a fresh one per stage."""
  x, w = al.sym("x", 3), al.sym("w", 9)
  stage = al.Function._from_exprs("hoist_stage", [x, w], [(w.reshape((3, 3)).exp() @ x).sin()], ["x", "w"], ["y"])
  z, weights = al.sym("z", 3 * stages), al.sym("weights", 9 if broadcast else 9 * stages)
  out = al.vmap(stage, stages, {"x": (z, 0, 3), "w": (weights, 0, 0 if broadcast else 9)})
  fn = al.Function._from_exprs(f"hoist_map_{broadcast}", [z, weights], [out], ["z", "weights"], ["y"])
  rng = np.random.default_rng(3)
  return fn, rng.normal(size=3 * stages), rng.normal(size=weights.shape[0])


def _proc_names(fn: al.Function) -> list[str]:
  prog = lower_function(fn)
  return [pr.attrs["name"] for pr in prog.args[: int(prog.attrs["proc_count"])]]


@pytest.mark.skipif(not _HAVE_CC, reason="no C compiler")
def test_hoist_moves_broadcast_argument_work_before_the_mapped_loop() -> None:
  fn, z, w = _mapped_mlp(5, broadcast=True)
  assert _proc_names(fn) == ["hoist_stage_hoist1", "hoist_stage_hoisted1", "hoist_map_True"]
  body = [s for s in _main_body(fn) if s.op != ProgramOp.BUFFER]
  calls = [s for s in body if s.op == ProgramOp.CALL]
  loops = [s for s in body if s.op == ProgramOp.FOR and s.args[1].op == ProgramOp.CALL]
  assert [c.attrs["callee"] for c in calls] == ["hoist_stage_hoist1"] and body.index(calls[0]) < body.index(loops[0])
  assert loops[0].args[1].attrs["callee"] == "hoist_stage_hoisted1"
  # The prologue owns the exp of the broadcast matrix; the body keeps the per-stage product and
  # its zero-filled accumulator, which is written per trip and so must not move.
  prologue, hoisted = lower_function(fn).args[:2]
  ops = lambda proc: {n.op for stmt in proc.args[int(proc.attrs["param_count"]) :] for n in _walk(stmt)}
  assert ProgramOp.EXP in ops(prologue) and ProgramOp.EXP not in ops(hoisted)
  assert ProgramOp.SIN in ops(hoisted) and ProgramOp.SIN not in ops(prologue)
  expected = np.sin(np.exp(w.reshape(3, 3)) @ z.reshape(5, 3).T).T.reshape(-1)
  np.testing.assert_allclose(fn((z, w)), expected, rtol=1e-14, atol=1e-14)


@pytest.mark.skipif(not _HAVE_CC, reason="no C compiler")
def test_hoist_leaves_per_trip_arguments_in_the_loop() -> None:
  fn, z, w = _mapped_mlp(5, broadcast=False)
  assert _proc_names(fn) == ["hoist_stage", "hoist_map_False"]
  expected = np.sin(np.einsum("sij,sj->si", np.exp(w.reshape(5, 3, 3)), z.reshape(5, 3))).reshape(-1)
  np.testing.assert_allclose(fn((z, w)), expected, rtol=1e-14, atol=1e-14)


@pytest.mark.skipif(not _HAVE_CC, reason="no C compiler")
def test_hoist_names_each_invariant_position_set_of_one_callee() -> None:
  """The same callee mapped once with ``w`` broadcast and once with ``x`` broadcast gets two distinct splits."""
  x, w = al.sym("x", 3), al.sym("w", 9)
  stage = al.Function._from_exprs("hoist_two", [x, w], [(w.reshape((3, 3)).exp() @ x.exp()).sin()], ["x", "w"], ["y"])
  z, weights = al.sym("z", 12), al.sym("weights", 36)
  first = al.vmap(stage, 4, {"x": (z, 0, 3), "w": (weights, 0, 0)})
  second = al.vmap(stage, 4, {"x": (z, 0, 0), "w": (weights, 0, 9)})
  fn = al.Function._from_exprs("hoist_two_map", [z, weights], [first + second], ["z", "weights"], ["y"])
  assert _proc_names(fn) == ["hoist_two_hoist1", "hoist_two_hoisted1", "hoist_two_hoist0", "hoist_two_hoisted0", "hoist_two_map"]
  zv, wv = np.random.default_rng(5).normal(size=12), np.random.default_rng(6).normal(size=36)
  ew, ex = np.exp(wv.reshape(4, 3, 3)), np.exp(zv.reshape(4, 3))
  expected = np.sin(ew[0] @ ex.T).T + np.sin(np.einsum("sij,j->si", ew, ex[0]))
  np.testing.assert_allclose(fn((zv, wv)), expected.reshape(-1), rtol=1e-14, atol=1e-14)


def test_hoist_refuses_a_buffer_read_between_two_invariant_writes() -> None:
  """``t = a; y = t * x; t = 2a; y += t``: both writes of ``t`` are invariant, but the first read must see the first."""
  a, x, y = (buffer(n, dtypes.float64, (1,)) for n in ("a", "x", "y"))
  t = buffer("t", dtypes.float64, (1,), address_space="private")
  at = lambda b: view(b, [const_int(0)])
  body = [
    t,
    store(at(t), load(at(a))),
    store(at(y), mul(load(at(t)), load(at(x)))),
    store(at(t), mul(const_float(2.0), load(at(a)))),
    store(at(y), add(load(at(y)), load(at(t)))),
  ]
  callee = ProgramNode(
    ProgramOp.PROC, (a, x, y, *body), {**proc_("twice", [a, x, y], body).attrs, "input_count": 2, "lowering": "auto", "scalarize": True}
  )
  za, zx, zy = (buffer(n, dtypes.float64, (4,)) for n in ("za", "zx", "zy"))
  rng = range_("it", 0, 4)
  call = ProgramNode(
    ProgramOp.CALL,
    (view(za, [const_int(0)]), view(zx, [var("it")]), view(zy, [var("it")])),
    {"callee": "twice", "n_in": 2, "n_out": 1, "returns": ()},
  )
  root = ProgramNode(ProgramOp.PROC, (za, zx, zy, for_(rng, [call])), {**proc_("root", [za, zx, zy], []).attrs, "input_count": 2})
  prog = program([callee, root])
  assert hoist_invariant(prog) is prog
