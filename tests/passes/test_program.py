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
from alloy.passes.program import WORKSPACE_SPILL_THRESHOLD, unroll_unit_loops
from alloy.ir.program import ProgramOp, buffer, const_float, for_, proc as proc_, program, range_, store, var, view
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


def _long_scalar_chain(length: int) -> al.Function:
  """``x[0]*1 + x[1]*2 + ...`` accumulated as a left fold, so the expression is ``length`` deep."""
  x = al.sym("x", 4)
  acc = al.const(0.0)
  for i in range(length):
    acc = acc + x[i % 4] * float(i + 1)
  return al.Function._from_exprs(f"fold{length}", [x], [acc.reshape((1,))], ["x"], ["y"])


def test_fusion_survives_a_moderately_deep_scalar_fold() -> None:
  """Guards the headroom below the recursion limit documented in `docs/how_it_works/lowering.md`."""
  render_program_c_source(_long_scalar_chain(150))


@pytest.mark.xfail(raises=RecursionError, strict=True, reason="fuse_elementwise recurses per chain level; see docs/how_it_works/lowering.md")
def test_fusion_of_a_very_deep_scalar_fold_exhausts_the_python_stack() -> None:
  """`_expand_inlinables` re-enters itself for every inlined producer and `_transform` recurses
  per argument, so depth in the *expression* becomes depth on the *Python stack* — about five
  frames per level, which exhausts the default 1000-frame limit somewhere just under 200 chained
  scalar ops. Anything that accumulates a long left fold (a per-stage NMPC cost written as
  `cost = cost + ...`) hits this as a RecursionError during codegen rather than a clean error.

  Fixing the passes to iterate instead of recurse turns this into an XPASS.
  """
  render_program_c_source(_long_scalar_chain(400))


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
  (inner_z,) = inner.call([z * z])
  outer = al.Function._from_exprs("outer", [z], [inner_z], ["z"], ["y"])
  jf = outer.factory("J", ["z"], [al.factory.Jac("y", "z")])

  source = render_program_c_source(jf)
  assert "static __attribute__((noinline)) void inner_fwd2_y_x_raw" in source
  match = re.search(r"inner_fwd2_y_x_raw\(([^)]*)\);", source)
  assert match is not None
  args = [a.strip() for a in match.group(1).split(",")]
  assert args[:3] == ["s0", "s2", "s3"]


def test_regular_raw_callees_stay_inline() -> None:
  x = al.sym("x", 2)
  inner = al.Function._from_exprs("inner", [x], [x.sin()], ["x"], ["y"])
  z = al.sym("z", 2)
  (inner_z,) = inner.call([z])
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
  got = f(a, b)
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
