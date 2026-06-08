"""Unit tests for the Program IR optimization passes (``alloy.passes``).

These exercise the passes directly on lowered Program IR (structural assertions) plus a couple of
end-to-end numeric checks under the Program IR renderer. See ``docs/program_ir_migration.md``
(Step 5b): fusion + workspace packing/spilling + contiguous-slice aliasing.
"""

from __future__ import annotations

import numpy as np
import pytest

import alloy as al
from alloy.codegen.program_c import render_program_c_source
from alloy.jit import _find_compiler
from alloy.lowering import lower_function, main_proc
from alloy.passes import WORKSPACE_SPILL_THRESHOLD
from alloy.program import POps

_HAVE_CC = _find_compiler() is not None


def _main_body(fn: al.Function) -> list:
  proc = main_proc(lower_function(fn))
  pc = int(proc.attrs["param_count"])
  return list(proc.args[pc:])


def _classify(body: list) -> tuple[list, list, list]:
  """(compute_buffers, alias_buffers, for_loops) among the proc's top-level statements."""
  compute = [s for s in body if s.op == POps.BUFFER and s.attrs.get("address_space") == "private" and "alias_of" not in s.attrs]
  aliases = [s for s in body if s.op == POps.BUFFER and "alias_of" in s.attrs]
  loops = [s for s in body if s.op == POps.FOR]
  return compute, aliases, loops


def _sz_w(fn: al.Function) -> int:
  return int(main_proc(lower_function(fn)).attrs.get("sz_w", 0))


# --- fusion ---------------------------------------------------------------------


def test_fusion_collapses_elementwise_chain() -> None:
  """A same-shape elementwise chain fuses into a single loop with no intermediate buffers."""

  @al.function("chain", {"x": 8, "y": 8})
  def f(x, y):
    return ((x.sin() + y) * y - x).tanh()

  compute, aliases, loops = _classify(_main_body(f))
  assert compute == []  # every intermediate inlined into the output loop
  assert len(loops) == 1


def test_fusion_skips_matmul_operand() -> None:
  """A matmul operand must NOT be inlined: a contraction reads each operand element m*k times,
  so inlining the producer's expression there multiplies compute. Regression guard for the
  blow-up that an unguarded fusion introduces."""

  @al.function("mm_operand", {"A": (4, 4), "x": 4})
  def f(A, x):
    return A @ x.sin()  # sin(x) is the matvec operand — must stay materialized

  compute, _aliases, _loops = _classify(_main_body(f))
  assert len(compute) == 1  # exactly the materialized sin(x); the A@x result aliases the output
  assert compute[0].attrs["shape"] == (4,)


def test_fusion_into_reduction() -> None:
  """An elementwise producer feeding a SUM fuses into the reduce loop (each element read once)."""

  @al.function("sumf", {"x": 8})
  def f(x):
    return (x.sin() + x).sum()

  compute, _aliases, loops = _classify(_main_body(f))
  # Only the scalar accumulator survives; the elementwise buffer is gone (inlined into the reduce).
  assert all(s.attrs["shape"] == (1,) for s in compute)
  assert len(loops) == 1  # the single reduce loop


# --- contiguous-slice aliasing --------------------------------------------------


def test_contiguous_slice_aliases_source() -> None:
  """A contiguous slice becomes a zero-copy pointer alias (no copy loop), unlike a strided one."""

  @al.function("slc", {"x": 8})
  def f(x):
    s = x[2:6]
    return (s * s).sin()  # use s twice so it stays materialized (not inlined) -> visible alias

  _compute, aliases, _loops = _classify(_main_body(f))
  assert len(aliases) == 1
  assert aliases[0].attrs["alias_of"] == "x"
  assert aliases[0].attrs["alias_offset"] == 2


def test_strided_slice_is_not_aliased() -> None:
  @al.function("strided", {"x": 8})
  def f(x):
    return x[::2] + x[1::2]  # strided -> no contiguous offset -> no alias

  _compute, aliases, _loops = _classify(_main_body(f))
  assert aliases == []


# --- workspace packing / spilling -----------------------------------------------


def test_no_spill_when_temps_small() -> None:
  @al.function("small", {"x": 8, "y": 8})
  def f(x, y):
    return (x.sin() + y).tanh()

  assert _sz_w(f) == 0  # everything stays on the stack


def test_packing_reuses_slots_and_spills() -> None:
  """Two disjoint-lifetime 1600-element matmul temps share ONE spilled slot, so sz_w is 1600
  (not 3200) — proving both lifetime slot-reuse and the >= 1024 spill-to-w[] threshold."""

  @al.function("spill", {"A": (40, 40), "B": (40, 40)})
  def f(A, B):
    c = (A @ B).sum()  # C (1600) lives only until this reduce
    d = (B @ A).sum()  # D (1600) is born after C is dead -> reuses C's slot
    return c + d

  assert 1600 >= WORKSPACE_SPILL_THRESHOLD
  assert _sz_w(f) == 1600


@pytest.mark.skipif(not _HAVE_CC, reason="no C compiler for the numeric spill check")
def test_spilled_function_matches_interpreter() -> None:
  """End-to-end: a function whose temporaries spill to w[] still computes correctly (the JIT
  allocates w from the rendered sz_w and passes it through)."""

  @al.function("spill_num", {"A": (40, 40), "B": (40, 40)})
  def f(A, B):
    return ((A @ B) + (B @ A)).sum()

  assert _sz_w(f) >= WORKSPACE_SPILL_THRESHOLD  # the (40,40) matmul temp spills
  render_program_c_source(f)  # loud coverage gate
  f.recompile()
  rng = np.random.default_rng(0)
  a, b = rng.standard_normal((40, 40)), rng.standard_normal((40, 40))
  got = f(a, b)
  ref = f.eval_interpreter(a, b)
  np.testing.assert_allclose(np.asarray(got).reshape(-1), ref[0].reshape(-1), rtol=1e-9, atol=1e-10)


def test_optimized_program_still_verifies() -> None:
  """The pass pipeline output must pass the Program IR verifier (lower_function asserts this)."""

  @al.function("verif", {"x": 6})
  def f(x):
    m = x.reshape((2, 3))
    return (m @ x[:3]).sin() + x[3:5]  # (2,3)@(3,) -> (2,), + x[3:5] (2,)

  prog = lower_function(f)  # raises ProgramVerifyError on any malformed node
  assert prog.op == POps.PROGRAM
