"""Program IR migration harness (see internal/notes/program_ir_migration.md).

The discipline is *self-certifying*: a covered function must render through the
Program IR path as the SOLE renderer. To avoid false greens, each check renders
``render_program_c_source`` directly — which raises loudly on any coverage gap —
and confirms ``render_c_source`` selects that exact source before trusting the
compiled execution path.

Step 1 coverage: elementwise unary/binary (identical shapes), RESHAPE, small CONST.
"""

from __future__ import annotations

import dataclasses
import re
from unittest import mock

import numpy as np
import pytest

import scaly as sc
from scaly.codegen.aot import render_c_source
from scaly.codegen.c import can_render_program_c, render_program_c_source
from scaly.codegen.jit import _find_compiler
from scaly.passes.lowering import LoweringError, lower_function, main_proc
from scaly.ir.expr import topo
from scaly.ir.program import ProgramOp
from scaly.ir.program_spec import verify_program
from scaly.ir.types import Lowering

_HAVE_CC = _find_compiler() is not None


# --- covered corpus: (name, builder, inputs) -------------------------------------


def _neg() -> sc.Function:
  @sc.function(sc.L("x", 4), output=sc.L("out0", ...), name="pm_neg")
  def f(x):
    return -x

  return f


def _trig_chain() -> sc.Function:
  @sc.function(sc.L("x", 4), output=sc.L("out0", ...), name="pm_trig")
  def f(x):
    return x.sin().cos() + x.tan()

  return f


def _exp_log() -> sc.Function:
  @sc.function(sc.G(sc.L("x", 3), sc.L("y", 3)), output=sc.L("out0", ...), name="pm_exp_log")
  def f(inputs):
    x, y = inputs
    return x.exp() + y.log()

  return f


def _arith() -> sc.Function:
  @sc.function(sc.G(sc.L("x", 3), sc.L("y", 3)), output=sc.L("out0", ...), name="pm_arith")
  def f(inputs):
    x, y = inputs
    return x * y - x / y

  return f


def _tanh_sqrt() -> sc.Function:
  @sc.function(sc.G(sc.L("x", 3), sc.L("y", 3)), output=sc.L("out0", ...), name="pm_tanh_sqrt")
  def f(inputs):
    x, y = inputs
    return x.tanh() * y.sqrt() + x.abs()

  return f


def _pow_same_shape() -> sc.Function:
  @sc.function(sc.G(sc.L("x", 2), sc.L("y", 2)), output=sc.L("out0", ...), name="pm_pow")
  def f(inputs):
    x, y = inputs
    return x**y

  return f


def _const_add() -> sc.Function:
  @sc.function(sc.L("x", 4), output=sc.L("out0", ...), name="pm_const")
  def f(x):
    return x + sc.const(np.array([1.0, 2.0, 3.0, 4.0]))

  return f


def _reshape() -> sc.Function:
  @sc.function(sc.L("x", 4), output=sc.L("out0", ...), name="pm_reshape")
  def f(x):
    r = x.reshape((2, 2))
    return r * r

  return f


def _large_const() -> sc.Function:
  @sc.function(sc.L("x", 24), output=sc.L("out0", ...), name="pm_large_const")
  def f(x):
    return x + sc.const(np.arange(24, dtype=np.float64))  # > the old size-16 inline cap

  return f


def _slice_contiguous() -> sc.Function:
  @sc.function(sc.L("x", 5), output=sc.L("out0", ...), name="pm_slice_contig")
  def f(x):
    return x[1:4].sin()  # rank-1 contiguous slice feeding an elementwise op

  return f


def _slice_scalar() -> sc.Function:
  @sc.function(sc.L("x", 5), output=sc.L("out0", ...), name="pm_slice_scalar")
  def f(x):
    return x[2] * x[2]  # integer index -> scalar (drops the dim)

  return f


def _slice_strided() -> sc.Function:
  @sc.function(sc.L("x", 6), output=sc.L("out0", ...), name="pm_slice_strided")
  def f(x):
    return x[::2] + x[1::2]  # strided slices, same output length

  return f


def _slice_multidim_row() -> sc.Function:
  @sc.function(sc.L("x", 12), output=sc.L("out0", ...), name="pm_slice_row")
  def f(x):
    m = x.reshape((3, 4))
    return m[1, :] * m[2, :]  # integer index on dim 0, full slice on dim 1

  return f


def _dot() -> sc.Function:
  @sc.function(sc.G(sc.L("x", 4), sc.L("y", 4)), output=sc.L("out0", ...), name="pm_dot")
  def f(inputs):
    x, y = inputs
    return x @ y

  return f


def _matvec() -> sc.Function:
  @sc.function(sc.G(sc.L("A", (3, 4)), sc.L("x", 4)), output=sc.L("out0", ...), name="pm_matvec")
  def f(inputs):
    A, x = inputs
    return A @ x

  return f


def _vecmat() -> sc.Function:
  @sc.function(sc.G(sc.L("x", 3), sc.L("A", (3, 4))), output=sc.L("out0", ...), name="pm_vecmat")
  def f(inputs):
    x, A = inputs
    return x @ A

  return f


def _matmat() -> sc.Function:
  @sc.function(sc.G(sc.L("A", (2, 3)), sc.L("B", (3, 2))), output=sc.L("out0", ...), name="pm_matmat")
  def f(inputs):
    A, B = inputs
    return A @ B

  return f


def _sum() -> sc.Function:
  @sc.function(sc.L("x", 5), output=sc.L("out0", ...), name="pm_sum")
  def f(x):
    return (x.sin() + x).sum()

  return f


def _transpose() -> sc.Function:
  @sc.function(sc.L("x", 6), output=sc.L("out0", ...), name="pm_transpose")
  def f(x):
    return x.reshape((2, 3)).transpose()

  return f


def _call() -> sc.Function:
  @sc.function(sc.L("a", 3), output=sc.L("out0", ...), name="pm_call_inner")
  def inner(a):
    return a.sin() + a

  @sc.function(sc.L("x", 3), output=sc.L("out0", ...), name="pm_call_outer")
  def f(x):
    y = inner(x)
    return y * x

  return f


def _vmap() -> sc.Function:
  @sc.function(sc.L("s", 2), output=sc.L("out0", ...), name="pm_vmap_cell")
  def cell(s):
    return s.tanh() + s

  @sc.function(sc.L("z", 6), output=sc.L("out0", ...), name="pm_vmap_outer")
  def f(z):
    return sc.vmap(cell, 3, [(z, 0, 2)])  # 3 independent calls over z[2i:2i+2]

  return f


def _gather() -> sc.Function:
  @sc.function(sc.L("x", 6), output=sc.L("out0", ...), name="pm_gather")
  def f(x):
    return x.gather(np.array([5, 0, 3, 3, 1]))  # repeats + reorder, via const index table

  return f


def _scatter() -> sc.Function:
  @sc.function(sc.L("x", 3), output=sc.L("out0", ...), name="pm_scatter")
  def f(x):
    return sc.scatter(x, np.array([4, 1, 2]), 6)  # zero-filled length-6 output

  return f


def _broadcast_matrix() -> sc.Function:
  @sc.function(sc.G(sc.L("x", (3, 4)), sc.L("b", 4)), output=sc.L("out0", ...), name="pm_bcast_mat")
  def f(inputs):
    x, b = inputs
    return x + b  # (3,4) + (4,) row broadcast

  return f


def _broadcast_scalar() -> sc.Function:
  @sc.function(sc.L("x", 4), output=sc.L("out0", ...), name="pm_bcast_scalar")
  def f(x):
    return x * 2.0 + 1.0  # scalar-const broadcast

  return f


def _concat() -> sc.Function:
  @sc.function(sc.G(sc.L("x", 3), sc.L("y", 2)), output=sc.L("out0", ...), name="pm_concat")
  def f(inputs):
    x, y = inputs
    return sc.concat([x.sin(), y])

  return f


def _mlp_layer() -> sc.Function:
  @sc.function(sc.G(sc.L("W", (4, 3)), sc.L("x", 3), sc.L("b", 4)), output=sc.L("out0", ...), name="pm_mlp_layer")
  def f(inputs):
    W, x, b = inputs
    return (W @ x + b).tanh()  # matmul + bias broadcast + activation

  return f


def _concat_axis1() -> sc.Function:
  @sc.function(sc.G(sc.L("a", (2, 3)), sc.L("b", (2, 2))), output=sc.L("out0", ...), name="pm_concat_ax1")
  def f(inputs):
    a, b = inputs
    return sc.concat([a, b], axis=1)  # (2,3) ++ (2,2) -> (2,5) along axis 1

  return f


def _stack_axis1() -> sc.Function:
  @sc.function(sc.G(sc.L("x", 3), sc.L("y", 3)), output=sc.L("out0", ...), name="pm_stack_ax1")
  def f(inputs):
    x, y = inputs
    return sc.stack([x.sin(), y], axis=1)  # two (3,) -> (3,2) along a new axis 1

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
  (_vmap, [np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])]),
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
  got = np.asarray(fn(*fn.input_tree.unflatten(tuple(inputs)))).reshape(-1)
  assert got.size == fn.outputs[0].size
  assert np.all(np.isfinite(got))


# One entry per matmul loop shape: the serial dot, a blocked mat@vec (two four-row blocks, a block plus
# a tail of rows, a tail only), the reduction-outermost vec@mat and mat@mat, and the two transpose folds.
_MATMUL_CASES = {
  "dot": ((5,), (5,), lambda a, b: a @ b),
  "matvec_two_blocks": ((8, 5), (5,), lambda a, b: a @ b),
  "matvec_block_and_tail": ((6, 5), (5,), lambda a, b: a @ b),
  "matvec_tail_only": ((3, 5), (5,), lambda a, b: a @ b),
  "vecmat": ((6,), (6, 5), lambda a, b: a @ b),
  "matmat": ((6, 5), (5, 7), lambda a, b: a @ b),
  "transposed_mat_vec": ((5, 6), (5,), lambda a, b: a.T @ b),
  "vec_transposed_mat": ((6,), (5, 6), lambda a, b: a @ b.T),
}


@pytest.mark.skipif(not _HAVE_CC, reason="no C compiler available for JIT numeric check")
@pytest.mark.parametrize("name", _MATMUL_CASES)
def test_matmul_lowering_matches_numpy_exactly(name) -> None:
  """Integer-valued inputs keep every product and partial sum exact, so the check is exact in any summation order."""
  sa, sb, product = _MATMUL_CASES[name]
  a, b = sc.sym("a", sa), sc.sym("b", sb)
  fn = sc.Function.from_exprs(f"mm_{name}", [a, b], [sc.simplify(product(a, b))], ["a", "b"], ["y"])
  assert all(e.op != sc.ExprOp.TRANSPOSE for e in topo(fn.outputs)), "the transpose was not folded into the product"
  rng = np.random.default_rng(0)
  av, bv = (rng.integers(-8, 9, shape).astype(np.float64) for shape in (sa, sb))
  fn.recompile()
  np.testing.assert_array_equal(fn((av, bv)), product(av, bv))


@pytest.mark.skipif(not _HAVE_CC, reason="no C compiler available for JIT numeric check")
def test_transpose_fold_is_bit_identical_to_the_transposed_product() -> None:
  """Both forms sum each output's terms in the same k order, so folding the transpose changes no bit."""
  a, v, w = sc.sym("a", (32, 12)), sc.sym("v", 32), sc.sym("w", 12)
  rng = np.random.default_rng(1)
  av, vv, wv = rng.standard_normal((32, 12)), rng.standard_normal(32), rng.standard_normal(12)
  for tag, x, xv, product in (("v", v, vv, lambda m, x: m.T @ x), ("w", w, wv, lambda m, x: x @ m.T)):
    transposed = sc.Function.from_exprs(f"mm_transposed_{tag}", [a, x], [product(a, x)], ["a", tag], ["y"])
    folded = sc.Function.from_exprs(f"mm_folded_{tag}", [a, x], [sc.simplify(product(a, x))], ["a", tag], ["y"])
    for fn in (transposed, folded):
      fn.recompile()
    np.testing.assert_array_equal(folded((av, xv)), transposed((av, xv)))


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
  @sc.function(sc.L("a", 3), output=sc.L("out0", ...), name="pm_inner_dev")
  def inner(a):
    return a.sin()

  inner_gpu = inner.with_device("cuda:0")

  @sc.function(sc.L("x", 3), output=sc.L("out0", ...), name="pm_outer_mix")
  def fn(x):
    y = inner_gpu(x)
    return y + x

  with pytest.raises(LoweringError):
    render_program_c_source(fn)
  with pytest.raises(LoweringError):
    render_c_source(fn)


def _import_sibling(name):
  """Import a sibling test-fixture module (e.g. test_stage_transcription)."""
  import sys
  from pathlib import Path

  here = str(Path(__file__).parents[1] / "integration")
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
    "jacobian": base.factory("bicycle_program_jac", ["z", "p"], [sc.factory.Jac("eq", "z")]),
    "sparse_jacobian": base.factory("bicycle_program_spjac", ["z", "p"], [sc.factory.SpJac("eq", "z")]),
  }[kind]
  render_program_c_source(fn)  # loud: must render through Program IR
  assert can_render_program_c(fn)
  inputs = [np.random.default_rng(0).standard_normal(e.size).reshape(e.shape) for e in fn.inputs]
  fn.recompile()
  assert len(fn.outputs) == 1
  got = np.asarray(fn(*fn.input_tree.unflatten(tuple(inputs)))).reshape(-1)
  assert got.size == fn.outputs[0].size
  assert np.all(np.isfinite(got))


def test_gather_fed_chained_vmaps_render_through_program_ir() -> None:
  # Pairwise-barrier shape: VMAP -> gather -> VMAP, concatenated with a per-body VMAP.
  fn = _import_sibling("test_vmap")._build_pairs_fn(True)
  assert can_render_program_c(fn)
  assert can_render_program_c(fn.factory("pairs_program_spjac", ["u", "p"], [sc.factory.SpJac("h", "u")]))


@pytest.mark.skipif(not _HAVE_CC, reason="no C compiler available for JIT numeric check")
def test_empty_reduction_and_output_leave_adjacent_memory_untouched() -> None:
  import ctypes

  from scaly.codegen.jit import get_compiled

  @sc.function(sc.L("x", 3), output=sc.G(sc.L("cost", ...), sc.L("empty", ...)))
  def fn(x):
    return x[:2].sum() + 1000.0 * x[2:2].sum(), sc.const(np.zeros(0))

  compiled = get_compiled(fn.concrete)
  values = np.array([2.0, 3.0, 17.0])
  cost = np.array([-1.0])
  untouched = np.array([23.0])
  pointer = ctypes.POINTER(ctypes.c_double)
  args = (pointer * 1)(values.ctypes.data_as(pointer))
  outputs = (pointer * 2)(cost.ctypes.data_as(pointer), untouched.ctypes.data_as(pointer))
  work = (ctypes.c_double * compiled._sz_w)()
  assert compiled._entry(args, outputs, None, work, 0) == 0
  np.testing.assert_array_equal(cost, [values[:2].sum() + 1000.0 * values[2:2].sum()])
  np.testing.assert_array_equal(untouched, [23.0])


def test_lowering_normalizes_a_private_function_and_preserves_metadata() -> None:
  matrix = sc.sym("matrix", (3, 2))
  vector = sc.sym("vector", 3)
  output = (matrix.T @ vector).block()
  sparsity = sc.SparsityType((2, 1), (0, 1), (0, 0))
  fn = sc.Function.from_exprs(
    "normalized_metadata",
    [matrix, vector],
    [output],
    ["matrix", "vector"],
    ["product"],
    [sparsity],
    output_coloring_widths=[2],
  )
  observed: list[tuple[str, sc.Function]] = []

  lower_function(fn, observe_expr=lambda name, normalized: observed.append((name, normalized)))

  assert fn.outputs == (output,)
  assert fn.outputs[0].args[0].op == sc.ExprOp.TRANSPOSE
  assert len(observed) == 1
  name, normalized = observed[0]
  assert name == "normalized"
  assert normalized is not fn
  assert normalized.inputs == fn.inputs
  assert normalized.input_tree is fn.input_tree
  assert normalized.output_tree is fn.output_tree
  assert normalized.output_sparsities == (sparsity,)
  assert normalized.output_coloring_widths == (2,)
  assert normalized.outputs[0].op == sc.ExprOp.MATMUL
  assert normalized._effective_lowering() == "block"


@pytest.mark.parametrize("identity", ["none", "compile", "simplify"])
def test_normalization_preserves_shared_work_across_hinted_outputs(identity: str) -> None:
  x = sc.sym("x", 64)
  a = x.sin()
  outputs = [a.cos().block(), a] if identity == "none" else [a.cos(), (a * 1.0).block()]
  if identity == "simplify":
    outputs = [sc.simplify(output) for output in outputs]
  fn = sc.Function.from_exprs(f"shared_outputs_{identity}", [x], outputs, ["x"], ["cos", "sin"])

  assert render_c_source(fn).count("sin(") == 1
  values = np.linspace(-2.0, 2.0, 64)
  cosine, sine = fn(values)
  np.testing.assert_allclose(cosine, np.cos(np.sin(values)))
  np.testing.assert_allclose(sine, np.sin(values))


def test_normalization_keeps_constant_and_conflicting_function_hints() -> None:
  x = sc.sym("x", 2, lowering="block")
  fn = sc.Function.from_exprs("normalized_hints", [x], [(x * 1.0).scalar(), sc.const([2.0, 3.0]).scalar()], ["x"], ["identity", "constant"])
  observed: list[sc.Function] = []

  lower_function(fn, observe_expr=lambda _name, normalized: observed.append(normalized))

  normalized = observed[0]
  assert normalized._effective_lowering() == "block"
  assert normalized.outputs[0] is x
  assert normalized.outputs[1].op == sc.ExprOp.CONST
  assert normalized.outputs[1] is fn.outputs[1]
  assert main_proc(lower_function(fn)).attrs["lowering"] == "block"


@pytest.mark.parametrize(("dtype", "value"), [("float32", np.float32(1.0)), ("int64", np.int64(1))])
def test_normalization_preserves_typed_identity_boundaries(dtype: str, value: object) -> None:
  x = sc.sym("x", 2, dtype=dtype)
  one = sc.const(np.full(2, value), dtype=dtype)
  fn = sc.Function.from_exprs(f"normalized_{dtype}", [x], [(x * one).scalar()], ["x"], ["y"])
  observed: list[sc.Function] = []

  proc = main_proc(lower_function(fn, observe_expr=lambda _name, normalized: observed.append(normalized)))

  assert observed[0].outputs[0].type.dtype == x.type.dtype
  # float32 arithmetic keeps its store boundaries; int64 values come only from explicit conversions.
  assert proc.attrs["scalarize_mode"] == ("procedure" if dtype == "int64" else "disabled")


@pytest.mark.skipif(not _HAVE_CC, reason="no C compiler available for JIT numeric check")
def test_automatic_transpose_normalization_preserves_cancellation_order_and_empty_reduction() -> None:
  matrix = sc.sym("matrix", (3, 2))
  vector = sc.sym("vector", 3)
  product = sc.Function.from_exprs("normalized_cancel", [matrix, vector], [matrix.T @ vector], ["matrix", "vector"], ["y"])
  matrix_value = np.array([[1e16, -1e16], [1.0, 1.0], [-1e16, 1e16]])
  vector_value = np.ones(3)
  np.testing.assert_array_equal(product((matrix_value, vector_value)), vector_value @ matrix_value)

  empty_matrix = sc.sym("empty_matrix", (2, 0))
  empty_vector = sc.sym("empty_vector", 0)
  empty = sc.Function.from_exprs("normalized_empty_product", [empty_matrix, empty_vector], [empty_matrix @ empty_vector], ["matrix", "vector"], ["y"])
  np.testing.assert_array_equal(empty((np.empty((2, 0)), np.empty(0))), np.zeros(2))


def test_two_different_functions_with_one_name_are_refused() -> None:
  """Generated code has one procedure per name, so a second, different Function with the same name
  would silently run the first one's body. Two Functions built from the same graph are the same
  procedure and are accepted."""
  c, u = sc.sym("c", 2), sc.sym("u", 1)
  first = sc.Function.from_exprs("dup_step", [c, u], [c * u[0] + c[::-1]], ["c", "u"], ["cn"])
  second = sc.Function.from_exprs("dup_step", [c, u], [c.sin() * u[0]], ["c", "u"], ["cn"])
  twin = sc.Function.from_exprs("dup_step", [c, u], [c * u[0] + c[::-1]], ["c", "u"], ["cn"])
  c0, us = sc.sym("c0", 2), sc.sym("us", 3)
  (a,) = sc.scan(first, c0, [(us, 0, 1)], length=3)
  (b,) = sc.scan(second, c0, [(us, 0, 1)], length=3)
  (t,) = sc.scan(twin, c0, [(us, 0, 1)], length=3)
  with pytest.raises(LoweringError, match="dup_step"):
    lower_function(sc.Function.from_exprs("dup_host", [c0, us], [a + b], ["c0", "us"], ["y"]))
  host = sc.Function.from_exprs("dup_twins", [c0, us], [a, t], ["c0", "us"], ["y", "z"])
  y, z = host((np.array([0.3, -0.4]), np.array([0.5, 1.1, -0.7])))
  np.testing.assert_array_equal(y, z)


@pytest.mark.parametrize("name", ["k0", "k2", "k3", "t0", "t8", "t9"])
def test_input_names_that_look_generated(name: str) -> None:
  """Generated buffers (``t<n>``, constant tables ``k<n>``) share a namespace with the inputs: a
  generated name never takes an input's. The gather's index is a table, and the second update's
  base a temporary; the names are the ones this graph generates."""
  x, idx = sc.sym("x", 5), sc.sym(name, 1, dtype="int64")
  y = sc.put_add(sc.gather(x, [4, 0, 3, 1, 2]) * 2.0, idx, sc.const([10.0]))
  fn = sc.Function.from_exprs(f"gen_name_{name}", [x, idx], [sc.put_add(y, idx, y[::-1][:1])], ["x", name], ["o"])
  (got,) = fn._flat_numerical_call(np.arange(5.0), np.array([0]))
  np.testing.assert_array_equal(got, [22.0, 0.0, 6.0, 2.0, 4.0])


def test_hoisting_keeps_a_loop_count_with_its_loop() -> None:
  """A mapped callee whose while loop reports its step count next to work on a broadcast input:
  the count's read of the loop variable is not hoisted out with the invariant work."""
  c, p = sc.sym("c", 2), sc.sym("p", 2)
  body = sc.Function.from_exprs("hk_body", [c], [c * 0.5], ["c"], ["cn"])
  cond = sc.Function.from_exprs("hk_cond", [c], [c[0] > 1e-3], ["c"], ["go"])
  x = sc.sym("x", 2)
  out, count = sc.while_loop(cond, body, x, max_iter=60)
  inner = sc.Function.from_exprs("hk_inner", [x, p], [out[:1] + count.reshape((1,)) + (p * p).sum().reshape((1,))], ["x", "p"], ["y"])
  xs, pp = sc.sym("xs", 6), sc.sym("pp", 2)
  fn = sc.Function.from_exprs("hk_outer", [xs, pp], [sc.vmap(inner, 3, [(xs, 0, 2), (pp, 0, 0)])], ["xs", "pp"], ["y"])
  (got,) = fn._flat_numerical_call(np.array([1.0, 0.0, 2.0, 0.0, 4.0, 0.0]), np.array([1.0, 2.0]))
  expected = []
  for start in (1.0, 2.0, 4.0):
    v, k = start, 0
    while v > 1e-3:
      v, k = v * 0.5, k + 1
    expected.append(v + k + 5.0)
  np.testing.assert_allclose(got, expected)


@pytest.mark.parametrize("n", [1, 5, 7, 8, 9, 12, 13, 64, 1001])
def test_max_and_min_reductions_in_four_lanes(n: int) -> None:
  """Past eight elements the extremum keeps four accumulators: the same value as NumPy's, whichever
  lane or the tail holds it, and NaN in any lane propagates."""
  from scaly.ir.expr import reduce_max, reduce_min

  x = sc.sym("x", n)
  fn = sc.Function.from_exprs(f"four_lanes_{n}", [x], [reduce_max(x).block(), reduce_min(x)], ["x"], ["mx", "mn"])
  rng = np.random.default_rng(n)
  for where in range(n):
    v = rng.standard_normal(n)
    v[where] = 10.0  # the maximum in each position in turn
    v[(where + 1) % n] = -10.0 if n > 1 else 10.0
    mx, mn = fn(v)
    assert (mx, mn) == (np.max(v), np.min(v))
    v[where] = np.nan
    mx, mn = fn(v)
    assert np.isnan(mx) and np.isnan(mn)
  if n >= 64:  # smaller functions become straight-line code
    assert "kb_" in render_c_source(fn)


@pytest.mark.parametrize("n", [7, 64, 203])
def test_elementwise_producers_fuse_into_extremum_reductions(n: int) -> None:
  """Every read of the reduced vector sits in one statement, so an elementwise producer is inlined
  into the reduction rather than stored first: ``norm_inf(x - 2 y)`` and a guarded ratio's minimum
  keep no vector of their own, and give NumPy's values, NaN included."""
  from scaly.ir.expr import norm_inf, reduce_min

  x, y = sc.sym("x", n), sc.sym("y", n)
  fn = sc.Function.from_exprs(f"fused_ext_{n}", [x, y], [norm_inf(x - 2.0 * y), reduce_min(sc.where(y < 0.0, -x / y, 1e30))], ["x", "y"], ["a", "b"])
  body = render_c_source(fn).split(f"int fused_ext_{n}(")[1]
  assert not re.search(rf"double \w+\[{n}\]", body) and "w + " not in body, "a vector temporary survived"
  rng = np.random.default_rng(n)
  xv, yv = rng.standard_normal(n), rng.standard_normal(n)
  a, b = fn((xv, yv))
  assert a == np.abs(xv - 2 * yv).max() and b == np.min(np.where(yv < 0, -xv / yv, 1e30))
  xv[n // 2] = np.nan
  a, _ = fn((xv, yv))
  assert np.isnan(a)


@pytest.mark.parametrize("sort", [True, False])
def test_segment_extrema_in_runs(sort: bool) -> None:
  """Bins in runs (a sorted table, as CSC column ids are) reduce run by run in a register; other
  tables keep one loop over the entries. Both give the entry-by-entry result, NaN included."""
  from scaly.ir.expr import segment_max, segment_min

  rng = np.random.default_rng(3)
  idx = rng.integers(0, 40, 400)
  idx = np.sort(idx) if sort else idx
  x = sc.sym("x", 400)
  fn = sc.Function.from_exprs(
    f"seg_runs_{int(sort)}", [x], [segment_max(x, idx, 41, fill=0.0), segment_min(x, idx, 41, fill=-1.0)], ["x"], ["a", "b"]
  )
  v = rng.standard_normal(400)
  v[[7, 200]] = np.nan
  want_a, want_b = np.zeros(41), np.full(41, -1.0)
  for k, val in zip(idx, v, strict=True):
    want_a[k] = val if (want_a[k] < val or val != val) else want_a[k]
    want_b[k] = val if (val < want_b[k] or val != val) else want_b[k]
  a, b = fn(v)
  np.testing.assert_array_equal(a, want_a)
  np.testing.assert_array_equal(b, want_b)
  assert ("for (long long r_" in render_c_source(fn)) == sort


def test_two_names_with_one_c_spelling_are_refused() -> None:
  """``f:_3`` and ``f__3`` are two names but one C symbol, so they clash like one name."""
  c, u = sc.sym("c", 2), sc.sym("u", 1)
  first = sc.Function.from_exprs("step:_3", [c, u], [c * u[0]], ["c", "u"], ["cn"])
  second = sc.Function.from_exprs("step__3", [c, u], [c.sin() * u[0]], ["c", "u"], ["cn"])
  c0, us = sc.sym("c0", 2), sc.sym("us", 3)
  (a,) = sc.scan(first, c0, [(us, 0, 1)], length=3)
  (b,) = sc.scan(second, c0, [(us, 0, 1)], length=3)
  with pytest.raises(LoweringError, match="two different Functions are named 'step.*' and 'step.*', both 'step__3' in C"):
    lower_function(sc.Function.from_exprs("c_spelling_host", [c0, us], [a + b], ["c0", "us"], ["y"]))


@pytest.mark.parametrize("loop", ["scan", "while"])
@pytest.mark.parametrize("name", ["clash_step_inplace", "clash:step_inplace"])
def test_a_function_named_like_a_loop_bodys_in_place_form_is_refused(loop: str, name: str) -> None:
  """A loop body that may overwrite its carry is lowered a second time as ``<name>_inplace``. A
  Function of that name, or of its C spelling, called in the same graph would take that
  procedure's place, or give its own up to it, whichever is lowered first."""
  x = sc.sym("x", 3)
  step = sc.Function.from_exprs(name.removesuffix("_inplace"), [x], [sc.index_set(x, [0], x[2:3] * 2.0)], ["x"], ["n"])
  other = sc.Function.from_exprs("clash_step_inplace", [x], [x[::-1] + 1.0], ["x"], ["r"])
  cond = sc.Function.from_exprs("clash_go", [x], [x[0] < 100.0], ["x"], ["go"])
  x0 = sc.sym("x0", 3)
  looped = sc.scan(step, x0, length=2)[0] if loop == "scan" else sc.while_loop(cond, step, x0, max_iter=2)[0]
  for outputs in ([other(x0), looped], [looped, other(x0)]):
    with pytest.raises(LoweringError, match="'clash_step_inplace' .* is named like the in-place form of the loop body 'clash.step'"):
      lower_function(sc.Function.from_exprs("clash_host", [x0], outputs, ["x0"], ["a", "b"]))
  alone = sc.Function.from_exprs("clash_alone", [x0], [looped], ["x0"], ["y"])  # the body's own in-place form is no clash
  assert "clash_step_inplace_raw" in render_c_source(alone)


# --- reductions in partial sums (C-196) ------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 7, 8, 9, 15, 16, 17, 64, 1001])
@pytest.mark.parametrize("hint", ["auto", "block"])
def test_sums_and_dots_in_partial_sums_match_numpy(n: int, hint: Lowering) -> None:
  """From eight elements on a sum or a dot runs in four partial sums with the tail in the first:
  NumPy's value at every length around the thresholds, in loop form and scalarized."""
  x, y = sc.sym("x", n), sc.sym("y", n)
  outs = [(x.sin() * y).sum(), x @ y, (x * x).sum() + 1.0]
  fn = sc.Function.from_exprs(f"partial_{n}_{hint}", [x, y], [o.with_lowering(hint) for o in outs], ["x", "y"], ["s", "d", "q"])
  rng = np.random.default_rng(n)
  xv, yv = rng.standard_normal(n), rng.standard_normal(n)
  s, d, q = fn((xv, yv))
  np.testing.assert_allclose(s, np.sum(np.sin(xv) * yv), rtol=1e-13, atol=1e-13)
  np.testing.assert_allclose(d, xv @ yv, rtol=1e-13, atol=1e-13)
  np.testing.assert_allclose(q, np.sum(xv * xv) + 1.0, rtol=1e-13)


@pytest.mark.parametrize(("m", "k"), [(1, 8), (3, 7), (4, 8), (5, 9), (9, 23), (12, 12), (37, 64)])
def test_blocked_matvec_matches_numpy(m: int, k: int) -> None:
  """Past seven columns ``a @ x`` runs four rows per pass in four partial sums each, the rows left
  over one at a time: NumPy's product for every row and column remainder, the output stored once."""
  a, x = sc.sym("a", (m, k)), sc.sym("x", k)
  fn = sc.Function.from_exprs(f"matvec_{m}_{k}", [a, x], [(a @ x.sin()).block()], ["a", "x"], ["y"])
  rng = np.random.default_rng(m * 100 + k)
  av, xv = rng.standard_normal((m, k)), rng.standard_normal(k)
  np.testing.assert_allclose(fn((av, xv)), av @ np.sin(xv), rtol=1e-13, atol=1e-13)
  if k >= 8:
    stores = [n for stmt in _stmts(fn) for n in _nodes(stmt) if n.op == ProgramOp.STORE and n.args[0].attrs["buffer"] == "y"]
    assert all(not _inside_reduce(fn, st) for st in stores)  # the output is stored once per row, after its sums


def _stmts(fn: sc.Function) -> list:
  proc = main_proc(lower_function(fn))
  return list(proc.args[int(proc.attrs["param_count"]) :])


def _nodes(node):
  stack = [node]
  while stack:
    n = stack.pop()
    yield n
    stack.extend(n.args)


def _inside_reduce(fn: sc.Function, target) -> bool:
  """Whether ``target`` sits inside a REDUCE loop of ``fn``'s entry."""
  from scaly.ir.program import RangeKind

  def walk(node, inside: bool) -> bool:
    if node is target:
      return inside
    if node.op == ProgramOp.FOR:
      inside = inside or node.args[0].attrs["kind"] == RangeKind.REDUCE
      return any(walk(a, inside) for a in node.args[1:])
    return False

  return any(walk(stmt, False) for stmt in _stmts(fn))


def test_an_expensive_producer_fuses_into_the_partial_sums() -> None:
  """Each element of ``sin(x) * y`` is read once across the four partial sums and the tail, so it
  is computed inside the reduction, never stored first."""
  x, y = sc.sym("x", 67), sc.sym("y", 67)
  fn = sc.Function.from_exprs("sin_sum", [x, y], [(x.sin() * y).sum().block()], ["x", "y"], ["s"])
  body = _stmts(fn)
  assert not [s for s in body if s.op == ProgramOp.BUFFER and int(np.prod(s.attrs["shape"])) == 67]
  assert sum(n.op == ProgramOp.SIN for stmt in body for n in _nodes(stmt)) == 5  # four lanes and the tail
  xv, yv = np.linspace(-2, 2, 67), np.cos(np.arange(67))
  np.testing.assert_allclose(fn((xv, yv)), np.sum(np.sin(xv) * yv), rtol=1e-13)


def test_partial_sums_have_a_fixed_order_and_never_accumulate_in_an_output() -> None:
  """The order is the generated code's own: lane ``q`` sums the ``k`` with ``k % 4 == q`` in turn,
  the tail goes into lane 0, and the lanes combine as ``(l0 + l1) + (l2 + l3)``; bit for bit, so a
  rounding change cannot slip in unnoticed. An output is stored once, after the sums, never used as
  an accumulator the C compiler would have to keep in memory."""
  n = 37
  x = sc.sym("x", n)
  fn = sc.Function.from_exprs("order_sum", [x], [x.sum().block()], ["x"], ["s"])
  rng = np.random.default_rng(24)  # a seed whose data the four orders below round differently
  xv = rng.standard_normal(n) * 10.0 ** rng.integers(-3, 4, n)

  def blocked(tail_lane: int, tree: bool = True) -> float:
    lanes = [0.0] * 4
    for k in range(n - n % 4):
      lanes[k % 4] += xv[k]
    for k in range(n - n % 4, n):
      lanes[tail_lane] += xv[k]
    return (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]) if tree else ((lanes[0] + lanes[1]) + lanes[2]) + lanes[3]

  # The data tell the orders apart: the tail in another lane, a sequential combine, one chain.
  assert len({blocked(0), blocked(3), blocked(0, tree=False), float(np.cumsum(xv)[-1])}) == 4
  assert fn(xv) == blocked(0)
  stores = [node for stmt in _stmts(fn) for node in _nodes(stmt) if node.op == ProgramOp.STORE and node.args[0].attrs["buffer"] == "s"]
  assert len(stores) == 1 and not _inside_reduce(fn, stores[0])


# --- tiles (C-200) ------------------------------------------------------------------------------


def _loops_storing(fn: sc.Function, prefix: str) -> list:
  """The top-level loops of ``fn``'s entry whose loop variable was named for ``prefix``'s buffer."""
  return [s for s in _stmts(fn) if s.op == ProgramOp.FOR and s.args[0].attrs["name"].startswith(prefix)]


def test_a_tile_is_one_loop_that_fuses_into_its_consumer() -> None:
  """A ``stack`` or ``concat`` along axis 0 of one input repeated, as forward mode builds for each
  seed, lowers to ``out[i] = src[i % size]``: fused into an elementwise consumer it leaves no copy
  and no buffer, and in front of a matrix product it is one loop, not one per copy."""
  x, y = sc.sym("x", 5), sc.sym("y", (3, 5))
  stacked = sc.stack([x.sin()] * 3)
  fused = sc.Function.from_exprs("tile_fused", [x, y], [(stacked * y).block()], ["x", "y"], ["out"])
  body = _stmts(fused)
  assert not [s for s in body if s.op == ProgramOp.BUFFER and s.attrs.get("shape") == (3, 5)]
  assert not _loops_storing(fused, "j_")
  a = sc.sym("a", (4, 15))
  tiled = sc.concat([x.sin()] * 3)
  # Two consumers keep the tile materialized; one would take it in (fusion into the product's operand).
  product = sc.Function.from_exprs("tile_product", [x, a], [(a @ tiled).block(), tiled * 2.0], ["x", "a"], ["out", "twice"])
  assert len(_loops_storing(product, "j_")) == 1
  xv, yv = np.linspace(-1.0, 1.0, 5), np.arange(15.0).reshape(3, 5)
  av = np.random.default_rng(5).standard_normal((4, 15))
  np.testing.assert_allclose(fused((xv, yv)), np.sin(xv) * yv, rtol=1e-15)
  out, twice = product((xv, av))
  np.testing.assert_allclose(out, av @ np.tile(np.sin(xv), 3), rtol=1e-13)
  np.testing.assert_allclose(twice, 2.0 * np.tile(np.sin(xv), 3), rtol=1e-15)


def test_stacks_that_are_not_tiles_keep_their_own_loops() -> None:
  """A repeated stack along another axis, and a stack or concat of different inputs, are not
  tiles: their values are NumPy's, element for element."""
  x, z = sc.sym("x", (2, 3)), sc.sym("z", (2, 3))
  outs = [sc.stack([x] * 3, axis=1).block(), sc.stack([x, z, x]), sc.concat([x, z, x], axis=1), sc.concat([x] * 2, axis=1)]
  fn = sc.Function.from_exprs("not_tiles", [x, z], outs, ["x", "z"], ["a", "b", "c", "d"])
  xv, zv = np.arange(6.0).reshape(2, 3), -np.arange(6.0).reshape(2, 3) - 1.0
  a, b, c, d = fn((xv, zv))
  np.testing.assert_array_equal(a, np.stack([xv] * 3, axis=1))
  np.testing.assert_array_equal(b, np.stack([xv, zv, xv]))
  np.testing.assert_array_equal(c, np.concatenate([xv, zv, xv], axis=1))
  np.testing.assert_array_equal(d, np.concatenate([xv] * 2, axis=1))


# --- column blocks (C-201) --------------------------------------------------------------------------


@pytest.mark.parametrize(
  ("m", "k", "n"),
  [
    (None, 32, 32),
    (None, 32, 9),
    (None, 2, 3),
    (None, 7, 21),
    (None, 256, 6),
    (None, 3, 65),
    (4, 32, 32),
    (48, 48, 48),
    (3, 5, 1),
    (7, 3, 29),
    (12, 12, 12),
    (5, 9, 6),
    (2, 3, 64),
    (3, 4, 65),
    (4, 4, 5),  # one register tile of rows: a tile of 4 columns, one streamed
    (9, 20, 13),  # two tiles of rows, of five and four; tiles of 8 and 4 columns, one streamed
    (11, 7, 9),  # tiles of six and of five rows
    (13, 6, 16),  # three tiles: one of five rows, then two of four in a loop
    (32, 5, 24),  # two tiles of six rows in a loop, then four of five in another
    (70, 300, 64),  # the same over packed panels: ten tiles of six rows and two of five
    (6, 96, 96),  # tiles at any width, reading b in place
    (67, 300, 123),  # past the L1 cache from 64 rows: tiles over packed panels, rows and a column left over
    (64, 1100, 24),  # panels in two chunks of k: the second resumes from the outputs
  ],
)
@pytest.mark.parametrize("target", ["apple-m3", "x86-64-v3", "x86-64-v4", "armv8-a", "generic"])
def test_column_blocked_products_match_numpy(m: int | None, k: int, n: int, target: str) -> None:
  """``x @ b`` and ``a @ b`` in column blocks (16, 8 and 4 on the M3, 64, 32 and 16 with AVX-512, 8,
  4 and 2 in scalar C) with the rest, and a row wider than the target's limit as before: NumPy's
  product for every width, remainder, row count and target."""
  rng = np.random.default_rng(1000 * (m or 0) + 10 * k + n)
  b, bv = sc.sym("b", (k, n)), rng.standard_normal((k, n))
  a, av = (sc.sym("a", k), rng.standard_normal(k)) if m is None else (sc.sym("a", (m, k)), rng.standard_normal((m, k)))
  fn = sc.Function.from_exprs(f"columns_{m}_{k}_{n}", [a, b], [(a @ b).block()], ["a", "b"], ["y"])
  with sc.target(target):
    np.testing.assert_allclose(fn((av, bv)), av @ bv, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize(
  ("m", "n", "blocked"),
  [
    (None, 12, True),
    (3, 28, True),
    (None, 64, True),
    (None, 80, False),
    (1, 96, False),
    (3, 96, False),
    (4, 96, True),
    (6, 96, True),
    (4, 10, True),
    (4, 5, True),
  ],
)
def test_column_blocks_store_each_output_once(m: int | None, n: int, blocked: bool) -> None:
  """On the M3, a row of up to 64 columns in whole blocks (12 is one of 8 and one of 4, 28 one of
  16, 8 and 4) keeps its sums in registers and stores each output once, after its reduction; a
  wider row of a vector or of up to three rows, even one of whole blocks (80, 96), keeps the
  reduction outermost and accumulates in the output, and four rows or more fill register tiles at
  any width, storing each output once again (10 columns: a tile of 8 and one of 2; 5 columns: a
  tile of 4 and one of scalar sums)."""
  a, b = sc.sym("a", 5) if m is None else sc.sym("a", (m, 5)), sc.sym("b", (5, n))
  fn = sc.Function.from_exprs(f"stores_{m}_{n}", [a, b], [(a @ b).block()], ["a", "b"], ["y"])
  with sc.target("apple-m3"):
    stores = [node for stmt in _stmts(fn) for node in _nodes(stmt) if node.op == ProgramOp.STORE and node.args[0].attrs["buffer"] == "y"]
    inside = [st for st in stores if _inside_reduce(fn, st)]
  assert (not inside) if blocked else inside


def _tile_heights(fn: sc.Function, vectors: int) -> list[int]:
  """The rows of each pass a lowered product makes over ``b``, in order: a reduction over ``k``
  holds one statement for each of a row's ``vectors`` vectors of sums, and one inside a loop over
  tiles runs once a trip."""
  from scaly.ir.program import RangeKind

  heights: list[int] = []

  def walk(node, trips: int) -> None:
    if node.op != ProgramOp.FOR:
      return
    rng = node.args[0]
    if rng.attrs["kind"] == RangeKind.REDUCE:
      rows, rest = divmod(len(node.args) - 1, vectors)
      assert rest == 0
      heights.extend([rows] * trips)
      return
    start, stop = rng.args[0], rng.args[1]
    for stmt in node.args[1:]:
      walk(stmt, trips * (stop.attrs["value"] - start.attrs["value"]))

  for stmt in _stmts(fn):
    walk(stmt, 1)
  return heights


@pytest.mark.parametrize(
  ("m", "heights", "starved"),
  [
    (4, [4], [4]),
    (5, [5], [3, 2]),
    (6, [6], [3, 3]),
    (7, [4, 3], [4, 3]),
    (9, [5, 4], [3, 3, 3]),
    (10, [5, 5], [4, 3, 3]),  # tiles of four would leave two rows over and read b a third time
    (11, [6, 5], [4, 4, 3]),
    (13, [5, 4, 4], [4, 3, 3, 3]),
    (32, [6, 6, 5, 5, 5, 5], [4] * 8),
  ],
  ids=[f"{m}-rows" for m in (4, 5, 6, 7, 9, 10, 11, 13, 32)],
)
def test_a_products_rows_go_in_the_fewest_tiles_the_registers_hold(m: int, heights: list[int], starved: list[int]) -> None:
  """Every tile of rows reads all of ``b``, so the rows go in the fewest tiles of at most
  ``Target.tile_rows_max`` rows, six on the M3, their heights within one of each other, and no row
  passes over ``b`` alone. With registers for four rows and no more, the tiles are of four or fewer."""
  from scaly.ir.target import PRESETS

  a, b = sc.sym("a", (m, 5)), sc.sym("b", (5, 8))
  fn = sc.Function.from_exprs(f"heights_{m}", [a, b], [(a @ b).block()], ["a", "b"], ["y"])
  with sc.target("apple-m3"):
    assert _tile_heights(fn, 4) == heights
  four = dataclasses.replace(PRESETS["apple-m3"], vector_registers=24)
  assert (four.product_tile, four.tile_rows_max) == ((4, 8), 4)
  with sc.target(four):
    assert _tile_heights(fn, 4) == starved
  assert sum(heights) == sum(starved) == m


def _map_loops(fn: sc.Function, prefix: str) -> list:
  """The calls of procedures named ``prefix``... that ``fn``'s entry makes from inside a loop."""
  calls = []

  def walk(node, inside: bool) -> None:
    if node.op == ProgramOp.CALL and inside and node.attrs["callee"].startswith(prefix):
      calls.append(node)
    for arg in node.args:
      walk(arg, inside or node.op == ProgramOp.FOR)

  for stmt in _stmts(fn):
    walk(stmt, False)
  return calls


def test_the_outputs_of_one_map_share_one_loop() -> None:
  """A callee of three outputs mapped over the same windows, two of them read: one loop calls it
  once a trip and writes both, each at its own stride (the second is a scalar a trip), and the
  third goes to a scratch buffer. A loop for each output read would call the callee twice a trip.
  The same callee over other windows is another map, with a loop of its own."""
  x, w = sc.sym("x", 3), sc.sym("w", 3)
  three = sc.Function.from_exprs("share_three", [x, w], [(x * w).sin(), (x @ w).reshape((1,)), (x + w).exp()], ["x", "w"], ["a", "b", "c"])
  xs, ws, us = sc.sym("xs", 12), sc.sym("ws", 12), sc.sym("us", 12)
  specs = [(xs, 0, 3), (ws, 0, 3)]
  a, b = sc.vmap(three, 4, specs, output=0), sc.vmap(three, 4, specs, output=1)
  other = sc.vmap(three, 4, [(us, 0, 3), (ws, 0, 3)], output=2)
  fn = sc.Function.from_exprs("share_map", [xs, ws, us], [a, b * 2.0, a.sum() + other.sum()], ["xs", "ws", "us"], ["a", "b", "s"])
  loops = _map_loops(fn, "share_three")
  assert len(loops) == 2
  first = loops[0]
  outs = first.args[int(first.attrs["n_in"]) :]
  written = [arg.attrs.get("buffer", arg.attrs.get("name")) for arg in outs]
  assert written[0] == "a" and len(set(written)) == 3  # the Function's own output, a temporary, a scratch
  assert [arg.op for arg in outs] == [ProgramOp.VIEW, ProgramOp.VIEW, ProgramOp.BUFFER]  # the unread output is not indexed by the trip
  rng = np.random.default_rng(12)
  xv, wv, uv = rng.normal(size=12), rng.normal(size=12), rng.normal(size=12)
  av, bv, sv = fn((xv, wv, uv))
  X, W, U = xv.reshape(4, 3), wv.reshape(4, 3), uv.reshape(4, 3)
  np.testing.assert_allclose(av, np.sin(X * W).reshape(-1), rtol=1e-14)
  np.testing.assert_allclose(bv, 2.0 * (X * W).sum(axis=1), rtol=1e-14)
  np.testing.assert_allclose(sv, np.sin(X * W).sum() + np.exp(U + W).sum(), rtol=1e-14)


def test_the_outputs_of_every_loop_are_found_in_one_walk_of_the_graph() -> None:
  """A map, a scan and a while loop each lower once for all their outputs the graph reads, which
  takes knowing them. They are found for every loop in one walk: a walk for each loop made
  lowering quadratic in their number (2 000 maps in one Function, 9 s). The walks lowering makes
  do not grow with the number of maps."""
  from scaly.passes import lowering

  u, v = sc.sym("u", 2), sc.sym("v", 2)
  two = sc.Function.from_exprs("walk_two", [u, v], [u * v, (u + v).sin()], ["u", "v"], ["a", "b"])

  def walks(maps: int) -> tuple[sc.Function, int]:
    xs = sc.sym("xs", 8)
    acc = xs
    for k in range(maps):  # each map reads the one before, so each is a loop of its own
      acc = sc.vmap(two, 4, [(acc, 0, 2), (xs, 0, 2)], output=k % 2) * 0.5 + xs
    fn = sc.Function.from_exprs(f"walk_{maps}", [xs], [acc], ["xs"], ["y"])
    with mock.patch.object(lowering, "topo", wraps=lowering.topo) as walked:
      lower_function(fn)
    return fn, walked.call_count

  fn, few = walks(3)
  _, many = walks(30)
  assert many == few, (few, many)
  xv = np.random.default_rng(2).normal(size=8)
  want = xv
  for k in range(3):
    pairs = want.reshape(4, 2), xv.reshape(4, 2)
    want = ((pairs[0] * pairs[1]) if k % 2 == 0 else np.sin(pairs[0] + pairs[1])).reshape(-1) * 0.5 + xv
  np.testing.assert_allclose(fn(xv), want, rtol=1e-14)


# --- products past the level-1 cache (C-204) ---------------------------------------------------------


def _panel_product(m: int | None, k: int, n: int) -> sc.Function:
  a, b = sc.sym("a", k) if m is None else sc.sym("a", (m, k)), sc.sym("b", (k, n))
  return sc.Function.from_exprs(f"panels_{m}_{k}_{n}", [a, b], [(a @ b).block()], ["a", "b"], ["y"])


def _packed_buffers(fn: sc.Function) -> list[int]:
  """The sizes of the private buffers a lowered product copies its panels of ``b`` into (one slot
  when the workspace packer lets two panels share it); a block's running sums are buffers of one
  vector's lanes, eight at most."""
  sizes = {}
  for stmt in _stmts(fn):
    for node in _nodes(stmt):
      if (
        node.op == ProgramOp.BUFFER and node.attrs.get("address_space") == "private" and len(node.attrs["shape"]) == 1 and node.attrs["shape"][0] > 8
      ):
        sizes[node.attrs["name"]] = node.attrs["shape"][0]
  return sorted(sizes.values())


def _chunk_lengths(fn: sc.Function) -> set[int]:
  """The trip counts of the reductions over ``k``: one per chunk length."""
  trips = set()
  for stmt in _stmts(fn):
    for node in _nodes(stmt):
      if node.op == ProgramOp.RANGE and node.attrs["kind"] == "reduce":
        start, stop = node.args[0], node.args[1]
        if start.op == stop.op == ProgramOp.CONST_INT:
          trips.add(stop.attrs["value"] - start.attrs["value"])
  return trips


@pytest.mark.parametrize(
  ("m", "k", "n"),
  [
    (96, 96, 96),  # wider than 64 columns
    (65, 300, 64),  # tiles over packed panels: ten of six rows in a loop, then one of five by itself
    (20, 12, 100),  # wider, whole blocks and a tail of four
    (16, 1100, 40),  # b past L1 at 40 columns: two whole chunks of 512 and a tail for the 16-wide blocks
    (16, 1025, 64),  # a chunk of one row of k after two whole ones
    (6, 700, 7),  # a narrow block and three streamed columns
    (8, 5, 130),  # a streamed tail of two past eight 16-wide blocks
    (5, 300, 256),  # too few rows to pay for the copy: streamed
    (1, 300, 256),  # one row: the vector path, streamed
    (None, 128, 256),  # a vector: streamed, as unbumpercars' products measured fastest
    (8, 0, 100),  # an empty reduction: zeros
    (3, 0, 70),
  ],
)
@pytest.mark.parametrize("target", ["apple-m3", "x86-64-v3", "generic"])
def test_products_past_the_cache_match_numpy_exactly(m: int | None, k: int, n: int, target: str) -> None:
  """Integer-valued inputs keep every partial sum exact, so the chunks resumed from the outputs must
  give NumPy's product exactly, on every target's blocks and budget."""
  rng = np.random.default_rng(k + n)
  av = rng.integers(-8, 9, k if m is None else (m, k)).astype(np.float64)
  bv = rng.integers(-8, 9, (k, n)).astype(np.float64)
  with sc.target(target):
    np.testing.assert_array_equal(_panel_product(m, k, n)((av, bv)), av @ bv)


def test_the_tiles_over_packed_panels_take_the_same_heights() -> None:
  """With the column blocks outermost, every row of ``a`` passes over each packed panel in the
  tiles a product read in place takes: 70 rows in ten tiles of six and two of five, for each of
  the eight panels of eight columns."""
  fn = _panel_product(70, 300, 64)
  with sc.target("apple-m3"):
    assert _packed_buffers(fn) == [300 * 8]
    heights = _tile_heights(fn, 4)
  assert sorted(set(heights)) == [5, 6] and (heights.count(6), heights.count(5)) == (8 * 10, 8 * 2)


def test_panels_are_packed_in_chunks_that_fit_the_budget() -> None:
  """On the M3 (64 KiB of panel, 8-column tiles): a product in register tiles copies panels only for
  a ``b`` larger than the L1 data cache and from 64 rows, in chunks of 1 024 rows of ``k`` for an
  8-wide panel. Without tiles, the rules of single rows hold (``generic``, 32 KiB of L1: 8 columns,
  chunks of 256): a ``b`` past the cache from sixteen rows, a wide one from six."""
  with sc.target("apple-m3"):
    wide_k = _panel_product(64, 1100, 24)  # 211 KiB of b
    assert _packed_buffers(wide_k) == [8 * 1024]
    assert _chunk_lengths(wide_k) == {1024, 1100 - 1024}
    assert _packed_buffers(_panel_product(63, 1100, 24)) == []
    assert _packed_buffers(_panel_product(64, 256, 256)) == [8 * 256]  # one chunk: all of k
    assert _packed_buffers(_panel_product(16, 256, 256)) == []  # a tile of rows reads b in place
    assert _packed_buffers(_panel_product(96, 96, 96)) == []  # 72 KiB: within the L1 cache
    assert _packed_buffers(_panel_product(3, 96, 96)) == []  # fewer rows than a tile, fewer than six
    assert _packed_buffers(_panel_product(1, 300, 256)) == []
    # From sixteen rows, a b past half the 16 MiB L2 cache, which every tile of rows would read from
    # memory again: 8 MiB and a column more packs, 8 MiB does not, nor fifteen rows.
    assert _packed_buffers(_panel_product(16, 1024, 1025)) == [8 * 1024]
    assert _packed_buffers(_panel_product(16, 1024, 1024)) == []
    assert _packed_buffers(_panel_product(15, 1024, 1025)) == []
  with sc.target("generic"):
    assert _packed_buffers(_panel_product(16, 256, 24)) == [8 * 256]  # 48 KiB, within 32 columns: sixteen rows
    assert _packed_buffers(_panel_product(15, 256, 24)) == []
    assert _packed_buffers(_panel_product(6, 96, 96)) == [8 * 96]  # wider than 32 columns, six rows
    assert _packed_buffers(_panel_product(5, 96, 96)) == []


# --- small products stay loops (C-206) ---------------------------------------------------------------


def _loops(fn: sc.Function, target: str | sc.Target) -> bool:
  return "for (" in render_c_source(fn, target=target).split(f"int {fn.name}(")[1]


@pytest.mark.parametrize(
  ("m", "k", "n", "loops"),
  [
    (8, 8, 8, True),  # a row fills the M3's middle block (8 columns), over four terms or more
    (2, 8, 8, True),
    (4, 4, 16, True),
    (8, 2, 8, False),  # a reduction of two: the scalar code is as fast
    (-8, 8, 8, False),  # a constant operand: its zeros and ones fold away only in scalar code
    (6, 6, 6, True),  # a register tile's rows (4 on the M3) and more than half its columns (8)
    (3, 6, 6, False),  # narrower, with fewer rows than a tile: its procedure's other work gains more expanded
    (8, 8, 4, False),
    (4, 8, 5, True),  # a tile's rows and five columns
    (4, 8, 4, False),  # four columns: its procedure's other work gains more expanded (a 4-state Riccati step)
  ],
)
def test_block_wide_products_keep_their_procedure_in_loops(m: int, k: int, n: int, loops: bool) -> None:
  """Expanded to scalars, a matrix product's outputs are chains the C compiler does not vectorize;
  its loops run vectorized across a column block."""
  constant = m < 0
  m = abs(m)
  av, bv = np.arange(m * k, dtype=float).reshape(m, k), np.arange(k * n, dtype=float).reshape(k, n) - 3.0
  b = sc.sym("b", (k, n))
  if constant:  # a selection matrix, as a seed or a kinematic matrix is
    sel = np.eye(m, k)
    fn = sc.Function.from_exprs(f"small_product_const_{m}_{k}_{n}", [b], [sc.const(sel) @ b], ["b"], ["c"])
    av, args = sel, (bv,)
  else:
    a = sc.sym("a", (m, k))
    fn = sc.Function.from_exprs(f"small_product_{m}_{k}_{n}", [a, b], [a @ b], ["a", "b"], ["c"])
    args = ((av, bv),)
  assert _loops(fn, "apple-m3") == loops
  with sc.target("apple-m3"):
    np.testing.assert_array_equal(fn(*args), av @ bv)


def test_the_product_rule_follows_the_target_and_yields_to_a_scalar_hint() -> None:
  a, b = sc.sym("a", (8, 8)), sc.sym("b", (8, 8))
  fn = sc.Function.from_exprs("small_product_hint", [a, b], [a @ b], ["a", "b"], ["c"])
  assert _loops(fn, "apple-m3") and not _loops(fn, "x86-64-v4")  # AVX-512's tile is 16 columns, its middle block 32
  # Tiles of 4 columns (the two-unit Arm cores) do not bring a 4-column product into loops, and
  # a target whose registers hold no tile of rows keeps the rule of the middle block.
  x, y = sc.sym("x", (4, 8)), sc.sym("y", (8, 4))
  four = sc.Function.from_exprs("small_product_four", [x, y], [x @ y], ["x", "y"], ["c"])
  assert not _loops(four, "armv8-a") and not _loops(four, "apple-m3")
  u, v = sc.sym("u", (6, 6)), sc.sym("v", (6, 6))
  six = sc.Function.from_exprs("small_product_six", [u, v], [u @ v], ["u", "v"], ["c"])
  assert _loops(six, "apple-m3") and not _loops(six, sc.Target("starved", vector_bytes=16, vector_registers=3))
  hinted = sc.Function.from_exprs("small_product_scalar", [a, b], [(a @ b).scalar()], ["a", "b"], ["c"])
  assert not _loops(hinted, "apple-m3")  # the user asked for scalar code


# --- transposes (C-202) -----------------------------------------------------------------------------


@pytest.mark.parametrize(
  ("shape", "axes"), [((3, 4), (1, 0)), ((2, 3, 4), (2, 0, 1)), ((2, 3, 4), (1, 2, 0)), ((2, 2, 3, 2), (3, 1, 0, 2)), ((1, 7), (1, 0))]
)
def test_transposes_match_numpy_and_lose_their_divisions(shape: tuple[int, ...], axes: tuple[int, ...]) -> None:
  """A transpose is one flat loop over the output; standing alone it is split back into a loop per
  axis, so the C divides nothing; values are NumPy's, plain and through an elementwise producer."""
  x = sc.sym("x", shape)
  name = "transpose_" + "_".join(map(str, axes)) + f"_{len(shape)}"
  fn = sc.Function.from_exprs(name, [x], [x.transpose(axes).block(), x.transpose(axes).sin()], ["x"], ["a", "b"])
  xv = np.random.default_rng(len(shape)).standard_normal(shape)
  a, b = fn(xv)
  np.testing.assert_array_equal(a, xv.transpose(axes))
  np.testing.assert_allclose(b, np.sin(xv.transpose(axes)), rtol=1e-15)
  body = render_c_source(fn).split(f"int {name}(")[1]
  assert " / " not in body and " % " not in body


def test_a_gather_reads_through_a_transposed_computation() -> None:
  """The recovery of a sparse derivative gathers a few entries of a transposed product: the
  transpose fuses into the gather, so only the gathered entries are computed. A transpose that only
  moves data fuses too: the gather's table and the transpose compose into one table when the code
  is generated, so nothing is copied and no index is divided at run time."""
  x = sc.sym("x", (4, 6))
  picks = np.array([0, 5, 7, 23, 11])
  computed = sc.Function.from_exprs("gather_computed", [x], [sc.gather(x.T.sin(), picks).block()], ["x"], ["g"])
  assert sum(n.op == ProgramOp.SIN for stmt in _stmts(computed) for n in _nodes(stmt)) == 1
  moved = sc.Function.from_exprs("gather_moved", [x], [sc.gather(x.T, picks).block()], ["x"], ["g"])
  for fn in (computed, moved):
    body = _stmts(fn)
    assert not [s for s in body if s.op == ProgramOp.BUFFER and int(np.prod(s.attrs["shape"])) == 24]
    assert not [n for stmt in body for n in _nodes(stmt) if n.op in (ProgramOp.DIV, ProgramOp.MOD) and n.dtype.is_integer]
    (table,) = [s for s in body if s.op == ProgramOp.BUFFER and "values" in s.attrs]
    assert list(table.attrs["values"]) == [int(q // 4 + (q % 4) * 6) for q in picks]
  xv = np.random.default_rng(6).standard_normal((4, 6))
  np.testing.assert_allclose(computed(xv), np.sin(xv.T).ravel()[picks], rtol=1e-15)
  np.testing.assert_array_equal(moved(xv), xv.T.ravel()[picks])
  # At run-time indices there is no table to compose with: the transpose is copied, then read.
  at = sc.sym("at", 5, dtype="int64")
  taken = sc.Function.from_exprs("take_moved", [x, at], [sc.take(x.T.reshape((24,)), at).block()], ["x", "at"], ["g"])
  assert [s for s in _stmts(taken) if s.op == ProgramOp.BUFFER and int(np.prod(s.attrs["shape"])) == 24]
  np.testing.assert_array_equal(taken((xv, picks)), xv.T.ravel()[picks])
