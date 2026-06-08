"""Program IR migration harness (see docs/program_ir_migration.md).

The discipline is *self-certifying*: a covered function must render through the
Program IR path as the SOLE renderer and match the interpreter oracle. To avoid
false greens (jit silently falling back to the interpreter on an uncovered op),
each check renders ``render_program_c_source`` directly — which raises loudly on
any coverage gap — and confirms ``render_c_source`` selects that exact source
under the flag before trusting the compiled numerics.

Step 1 coverage: elementwise unary/binary (identical shapes), RESHAPE, small CONST.
"""

from __future__ import annotations

import numpy as np
import pytest

import alloy as al
from alloy.codegen.c import render_c_source
from alloy.codegen.program_c import can_render_program_c, render_program_c_source
from alloy.jit import _find_compiler
from alloy.lowering import LoweringError, lower_function, main_proc
from alloy.program import POps, verify_program

_HAVE_CC = _find_compiler() is not None


# --- covered corpus: (name, builder, inputs) -------------------------------------


def _neg() -> al.Function:
  @al.function("pm_neg", {"x": 4})
  def f(x):
    return -x

  return f


def _trig_chain() -> al.Function:
  @al.function("pm_trig", {"x": 4})
  def f(x):
    return x.sin().cos() + x.tan()

  return f


def _exp_log() -> al.Function:
  @al.function("pm_exp_log", {"x": 3, "y": 3})
  def f(x, y):
    return x.exp() + y.log()

  return f


def _arith() -> al.Function:
  @al.function("pm_arith", {"x": 3, "y": 3})
  def f(x, y):
    return x * y - x / y

  return f


def _tanh_sqrt() -> al.Function:
  @al.function("pm_tanh_sqrt", {"x": 3, "y": 3})
  def f(x, y):
    return x.tanh() * y.sqrt() + x.abs()

  return f


def _pow_same_shape() -> al.Function:
  @al.function("pm_pow", {"x": 2, "y": 2})
  def f(x, y):
    return x**y

  return f


def _const_add() -> al.Function:
  @al.function("pm_const", {"x": 4})
  def f(x):
    return x + al.const(np.array([1.0, 2.0, 3.0, 4.0]))

  return f


def _reshape() -> al.Function:
  @al.function("pm_reshape", {"x": 4})
  def f(x):
    r = x.reshape((2, 2))
    return r * r

  return f


def _large_const() -> al.Function:
  @al.function("pm_large_const", {"x": 24})
  def f(x):
    return x + al.const(np.arange(24, dtype=np.float64))  # > the old size-16 inline cap

  return f


def _slice_contiguous() -> al.Function:
  @al.function("pm_slice_contig", {"x": 5})
  def f(x):
    return x[1:4].sin()  # rank-1 contiguous slice feeding an elementwise op

  return f


def _slice_scalar() -> al.Function:
  @al.function("pm_slice_scalar", {"x": 5})
  def f(x):
    return x[2] * x[2]  # integer index -> scalar (drops the dim)

  return f


def _slice_strided() -> al.Function:
  @al.function("pm_slice_strided", {"x": 6})
  def f(x):
    return x[::2] + x[1::2]  # strided slices, same output length

  return f


def _slice_multidim_row() -> al.Function:
  @al.function("pm_slice_row", {"x": 12})
  def f(x):
    m = x.reshape((3, 4))
    return m[1, :] * m[2, :]  # integer index on dim 0, full slice on dim 1

  return f


def _dot() -> al.Function:
  @al.function("pm_dot", {"x": 4, "y": 4})
  def f(x, y):
    return x @ y

  return f


def _matvec() -> al.Function:
  @al.function("pm_matvec", {"A": (3, 4), "x": 4})
  def f(A, x):
    return A @ x

  return f


def _vecmat() -> al.Function:
  @al.function("pm_vecmat", {"x": 3, "A": (3, 4)})
  def f(x, A):
    return x @ A

  return f


def _matmat() -> al.Function:
  @al.function("pm_matmat", {"A": (2, 3), "B": (3, 2)})
  def f(A, B):
    return A @ B

  return f


def _sum() -> al.Function:
  @al.function("pm_sum", {"x": 5})
  def f(x):
    return (x.sin() + x).sum()

  return f


def _transpose() -> al.Function:
  @al.function("pm_transpose", {"x": 6})
  def f(x):
    return x.reshape((2, 3)).transpose()

  return f


def _call() -> al.Function:
  @al.function("pm_call_inner", {"a": 3})
  def inner(a):
    return a.sin() + a

  @al.function("pm_call_outer", {"x": 3})
  def f(x):
    (y,) = inner.call([x])
    return y * x

  return f


def _map() -> al.Function:
  @al.function("pm_map_cell", {"s": 2})
  def cell(s):
    return s.tanh() + s

  @al.function("pm_map_outer", {"z": 6})
  def f(z):
    return al.map_(cell, 3, [(z, 0, 2)])  # 3 independent calls over z[2i:2i+2]

  return f


_CORPUS = [
  (_neg, [np.array([0.5, -1.0, 2.0, -3.0])]),
  (_trig_chain, [np.array([0.1, 0.2, -0.3, 0.4])]),
  (_exp_log, [np.array([0.5, -1.0, 0.25]), np.array([1.0, 2.0, 3.0])]),
  (_arith, [np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 0.5])]),
  (_tanh_sqrt, [np.array([0.5, -1.0, 2.0]), np.array([1.0, 4.0, 9.0])]),
  (_pow_same_shape, [np.array([2.0, 3.0]), np.array([3.0, 2.0])]),
  (_const_add, [np.array([10.0, 20.0, 30.0, 40.0])]),
  (_reshape, [np.array([1.0, 2.0, 3.0, 4.0])]),
  (_large_const, [np.arange(100.0, 124.0)]),
  (_slice_contiguous, [np.array([0.1, 0.2, 0.3, 0.4, 0.5])]),
  (_slice_scalar, [np.array([1.0, 2.0, 3.0, 4.0, 5.0])]),
  (_slice_strided, [np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])]),
  (_slice_multidim_row, [np.arange(1.0, 13.0)]),
  (_dot, [np.array([1.0, 2.0, 3.0, 4.0]), np.array([0.5, 0.25, 2.0, -1.0])]),
  (_matvec, [np.arange(1.0, 13.0).reshape(3, 4), np.array([1.0, 0.5, -1.0, 2.0])]),
  (_vecmat, [np.array([1.0, 2.0, 3.0]), np.arange(1.0, 13.0).reshape(3, 4)]),
  (_matmat, [np.arange(1.0, 7.0).reshape(2, 3), np.arange(1.0, 7.0).reshape(3, 2)]),
  (_sum, [np.array([0.1, 0.2, 0.3, 0.4, 0.5])]),
  (_transpose, [np.arange(1.0, 7.0)]),
  (_call, [np.array([0.3, -0.5, 1.2])]),
  (_map, [np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])]),
]


@pytest.mark.parametrize("builder, inputs", _CORPUS, ids=[b.__name__ for b, _ in _CORPUS])
def test_covered_function_renders_through_program_ir(builder, inputs, monkeypatch) -> None:
  fn = builder()
  # Loud coverage gate: raises LoweringError if any op/case is not covered yet.
  src = render_program_c_source(fn)
  assert can_render_program_c(fn)
  # Under the flag, the public renderer must select exactly this Program IR source
  # (proves no silent fallback to the legacy renderer).
  monkeypatch.setenv("ALLOY_USE_PROGRAM_IR_C", "1")
  assert render_c_source(fn) == src


@pytest.mark.skipif(not _HAVE_CC, reason="no C compiler available for JIT numeric check")
@pytest.mark.parametrize("builder, inputs", _CORPUS, ids=[b.__name__ for b, _ in _CORPUS])
def test_program_ir_matches_interpreter(builder, inputs, monkeypatch) -> None:
  fn = builder()
  # Gate first so a coverage gap errors here rather than silently falling back.
  render_program_c_source(fn)
  monkeypatch.setenv("ALLOY_USE_PROGRAM_IR_C", "1")
  fn.recompile()
  got = fn(*inputs)
  ref = fn.eval_interpreter(*inputs)
  np.testing.assert_allclose(np.asarray(got).reshape(-1), ref[0].reshape(-1), rtol=1e-10, atol=1e-12)


def test_lowered_program_verifies_and_has_single_proc() -> None:
  fn = _arith()
  prog = lower_function(fn)
  verify_program(prog)  # also called inside lower_function; assert it stays clean
  assert int(prog.attrs["proc_count"]) == 1
  assert main_proc(prog).op == POps.PROC


def test_uncovered_case_raises_loudly_and_fallback_is_opt_in(monkeypatch) -> None:
  # A host function calling a device-placed callee: mixed-device lowering is deferred
  # for the whole CPU-parity migration, so this stays a coverage gap across steps.
  @al.function("pm_inner_dev", {"a": 3})
  def inner(a):
    return a.sin()

  inner_gpu = inner.with_device("cuda:0")

  @al.function("pm_outer_mix", {"x": 3})
  def fn(x):
    (y,) = inner_gpu.call([x])
    return y + x

  with pytest.raises(LoweringError):
    render_program_c_source(fn)

  # Selected + strict (default): the LoweringError propagates, no silent fallback.
  monkeypatch.setenv("ALLOY_USE_PROGRAM_IR_C", "1")
  with pytest.raises(LoweringError):
    render_c_source(fn)

  # Explicit escape hatch: fall back to the legacy renderer instead of raising.
  monkeypatch.setenv("ALLOY_PROGRAM_IR_FALLBACK", "1")
  assert "pm_outer_mix" in render_c_source(fn)
