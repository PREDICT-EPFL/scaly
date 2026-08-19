"""Program IR migration harness (see internal/notes/program_ir_migration.md).

The discipline is *self-certifying*: a covered function must render through the
Program IR path as the SOLE renderer. To avoid false greens, each check renders
``render_program_c_source`` directly — which raises loudly on any coverage gap —
and confirms ``render_c_source`` selects that exact source before trusting the
compiled execution path.

Step 1 coverage: elementwise unary/binary (identical shapes), RESHAPE, small CONST.
"""

from __future__ import annotations

import numpy as np
import pytest

import alloy as al
from alloy.codegen.aot import render_c_source
from alloy.codegen.c import can_render_program_c, render_program_c_source
from alloy.codegen.jit import _find_compiler
from alloy.passes.lowering import LoweringError, lower_function, main_proc
from alloy.ir.program import ProgramOp
from alloy.ir.program_spec import verify_program

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


def _gather() -> al.Function:
  @al.function("pm_gather", {"x": 6})
  def f(x):
    return x.gather(np.array([5, 0, 3, 3, 1]))  # repeats + reorder, via const index table

  return f


def _scatter() -> al.Function:
  @al.function("pm_scatter", {"x": 3})
  def f(x):
    return al.scatter(x, np.array([4, 1, 2]), 6)  # zero-filled length-6 output

  return f


def _broadcast_matrix() -> al.Function:
  @al.function("pm_bcast_mat", {"x": (3, 4), "b": 4})
  def f(x, b):
    return x + b  # (3,4) + (4,) row broadcast

  return f


def _broadcast_scalar() -> al.Function:
  @al.function("pm_bcast_scalar", {"x": 4})
  def f(x):
    return x * 2.0 + 1.0  # scalar-const broadcast

  return f


def _concat() -> al.Function:
  @al.function("pm_concat", {"x": 3, "y": 2})
  def f(x, y):
    return al.concat([x.sin(), y])

  return f


def _mlp_layer() -> al.Function:
  @al.function("pm_mlp_layer", {"W": (4, 3), "x": 3, "b": 4})
  def f(W, x, b):
    return (W @ x + b).tanh()  # matmul + bias broadcast + activation

  return f


def _concat_axis1() -> al.Function:
  @al.function("pm_concat_ax1", {"a": (2, 3), "b": (2, 2)})
  def f(a, b):
    return al.concat([a, b], axis=1)  # (2,3) ++ (2,2) -> (2,5) along axis 1

  return f


def _stack_axis1() -> al.Function:
  @al.function("pm_stack_ax1", {"x": 3, "y": 3})
  def f(x, y):
    return al.stack([x.sin(), y], axis=1)  # two (3,) -> (3,2) along a new axis 1

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
  (_gather, [np.arange(10.0, 16.0)]),
  (_scatter, [np.array([10.0, 20.0, 30.0])]),
  (_broadcast_matrix, [np.arange(1.0, 13.0).reshape(3, 4), np.array([0.1, 0.2, 0.3, 0.4])]),
  (_broadcast_scalar, [np.array([1.0, 2.0, 3.0, 4.0])]),
  (_concat, [np.array([0.1, 0.2, 0.3]), np.array([4.0, 5.0])]),
  (_mlp_layer, [np.arange(1.0, 13.0).reshape(4, 3) * 0.1, np.array([0.5, -0.5, 1.0]), np.array([0.1, 0.2, 0.3, 0.4])]),
  (_concat_axis1, [np.arange(1.0, 7.0).reshape(2, 3), np.arange(7.0, 11.0).reshape(2, 2)]),
  (_stack_axis1, [np.array([0.1, 0.2, 0.3]), np.array([4.0, 5.0, 6.0])]),
]


@pytest.mark.parametrize("builder, inputs", _CORPUS, ids=[b.__name__ for b, _ in _CORPUS])
def test_covered_function_renders_through_program_ir(builder, inputs) -> None:
  fn = builder()
  # Loud coverage gate: raises LoweringError if any op/case is not covered yet.
  src = render_program_c_source(fn)
  assert can_render_program_c(fn)
  # The public renderer is Program IR (the sole CPU path), so it must select exactly this source.
  assert render_c_source(fn) == src


@pytest.mark.skipif(not _HAVE_CC, reason="no C compiler available for JIT numeric check")
@pytest.mark.parametrize("builder, inputs", _CORPUS, ids=[b.__name__ for b, _ in _CORPUS])
def test_program_ir_jit_executes(builder, inputs) -> None:
  fn = builder()
  # Gate first so a coverage gap errors here loudly (there is no fallback).
  render_program_c_source(fn)
  fn.recompile()
  got = np.asarray(fn(*inputs)).reshape(-1)
  assert got.size == fn.outputs[0].size
  assert np.all(np.isfinite(got))


def test_lowered_program_verifies_and_has_single_proc() -> None:
  fn = _arith()
  prog = lower_function(fn)
  verify_program(prog)  # also called inside lower_function; assert it stays clean
  assert int(prog.attrs["proc_count"]) == 1
  assert main_proc(prog).op == ProgramOp.PROC


def test_uncovered_case_raises_loudly() -> None:
  # A host function calling a device-placed callee: mixed-device lowering is deferred
  # (a host->GPU call is meaningless on a CPU build). With the legacy renderer deleted there is
  # no fallback — both the Program-IR renderer and the public entry raise loudly.
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
  with pytest.raises(LoweringError):
    render_c_source(fn)


def _import_sibling(name):
  """Import a sibling test-fixture module (e.g. test_stage_transcription)."""
  import sys
  from pathlib import Path

  here = str(Path(__file__).parent)
  if here not in sys.path:
    sys.path.insert(0, here)
  return pytest.importorskip(name)


@pytest.mark.skipif(not _HAVE_CC, reason="no C compiler available for JIT numeric check")
@pytest.mark.parametrize("kind", ["forward", "jacobian", "sparse_jacobian"])
def test_stage_transcription_renders_and_matches(kind) -> None:
  pytest.importorskip("casadi")  # the fixture module needs CasADi at import
  tw = _import_sibling("test_stage_transcription")
  base = tw.bicycle_eq_function(3)
  fn = {
    "forward": base,
    "jacobian": base.factory("bicycle_program_jac", ["z", "p"], [al.jac("eq", "z")]),
    "sparse_jacobian": base.factory("bicycle_program_spjac", ["z", "p"], [al.spjac("eq", "z")]),
  }[kind]
  render_program_c_source(fn)  # loud: must render through Program IR
  assert can_render_program_c(fn)
  inputs = [np.random.default_rng(0).standard_normal(e.size).reshape(e.shape) for e in fn.inputs]
  fn.recompile()
  outs = fn.eval_list(*inputs)
  assert len(outs) == len(fn.outputs)
  for got, out_expr in zip(outs, fn.outputs, strict=True):
    got_arr = np.asarray(got).reshape(-1)
    assert got_arr.size == out_expr.size
    assert np.all(np.isfinite(got_arr))


def test_gather_fed_chained_maps_render_through_program_ir() -> None:
  # Pairwise-barrier shape: MAP -> gather -> MAP, concatenated with a per-body MAP.
  fn = _import_sibling("test_map")._build_pairs_fn(True)
  assert can_render_program_c(fn)
  assert can_render_program_c(fn.factory("pairs_program_spjac", ["u", "p"], [al.spjac("h", "u")]))
