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

from scaly.function.model import as_concrete
from scaly.function.concrete import ConcreteFunction
from scaly.function.sugar import _mapped_call
import scaly as sc
from scaly.codegen.aot import render_c_source
from scaly.codegen.c import can_render_program_c, render_program_c_source
from scaly.codegen.toolchain import BuildRecipe, find_c_compiler
from scaly.passes.lowering import lower_function, main_proc
from scaly.ir.expr import topo
from scaly.ir.program import ProgramOp
from scaly.ir.program_spec import verify_program

_HAVE_CC = find_c_compiler() is not None


# --- covered corpus: (name, builder, inputs) -------------------------------------


def _neg() -> sc.Function:
  @sc.function(sc.arg("x", 4), outputs=sc.arg("out0"), name="pm_neg")
  def f(x: sc.Expr) -> sc.Expr:
    return -x

  return f


def _trig_chain() -> sc.Function:
  @sc.function(sc.arg("x", 4), outputs=sc.arg("out0"), name="pm_trig")
  def f(x: sc.Expr) -> sc.Expr:
    return x.sin().cos() + x.tan()

  return f


def _exp_log() -> sc.Function:
  @sc.function(sc.group(sc.arg("x", 3), sc.arg("y", 3)), outputs=sc.arg("out0"), name="pm_exp_log")
  def f(inputs: tuple[sc.Expr, sc.Expr]) -> sc.Expr:
    x, y = inputs
    return x.exp() + y.log()

  return f


def _arith() -> sc.Function:
  @sc.function(sc.group(sc.arg("x", 3), sc.arg("y", 3)), outputs=sc.arg("out0"), name="pm_arith")
  def f(inputs: tuple[sc.Expr, sc.Expr]) -> sc.Expr:
    x, y = inputs
    return x * y - x / y

  return f


def _tanh_sqrt() -> sc.Function:
  @sc.function(sc.group(sc.arg("x", 3), sc.arg("y", 3)), outputs=sc.arg("out0"), name="pm_tanh_sqrt")
  def f(inputs: tuple[sc.Expr, sc.Expr]) -> sc.Expr:
    x, y = inputs
    return x.tanh() * y.sqrt() + x.abs()

  return f


def _pow_same_shape() -> sc.Function:
  @sc.function(sc.group(sc.arg("x", 2), sc.arg("y", 2)), outputs=sc.arg("out0"), name="pm_pow")
  def f(inputs: tuple[sc.Expr, sc.Expr]) -> sc.Expr:
    x, y = inputs
    return x**y

  return f


def _const_add() -> sc.Function:
  @sc.function(sc.arg("x", 4), outputs=sc.arg("out0"), name="pm_const")
  def f(x: sc.Expr) -> sc.Expr:
    return x + sc.const(np.array([1.0, 2.0, 3.0, 4.0]))

  return f


def _reshape() -> sc.Function:
  @sc.function(sc.arg("x", 4), outputs=sc.arg("out0"), name="pm_reshape")
  def f(x: sc.Expr) -> sc.Expr:
    r = x.reshape((2, 2))
    return r * r

  return f


def _large_const() -> sc.Function:
  @sc.function(sc.arg("x", 24), outputs=sc.arg("out0"), name="pm_large_const")
  def f(x: sc.Expr) -> sc.Expr:
    return x + sc.const(np.arange(24, dtype=np.float64))  # > the old size-16 inline cap

  return f


def _slice_contiguous() -> sc.Function:
  @sc.function(sc.arg("x", 5), outputs=sc.arg("out0"), name="pm_slice_contig")
  def f(x: sc.Expr) -> sc.Expr:
    return x[1:4].sin()  # rank-1 contiguous slice feeding an elementwise op

  return f


def _slice_scalar() -> sc.Function:
  @sc.function(sc.arg("x", 5), outputs=sc.arg("out0"), name="pm_slice_scalar")
  def f(x: sc.Expr) -> sc.Expr:
    return x[2] * x[2]  # integer index -> scalar (drops the dim)

  return f


def _slice_strided() -> sc.Function:
  @sc.function(sc.arg("x", 6), outputs=sc.arg("out0"), name="pm_slice_strided")
  def f(x: sc.Expr) -> sc.Expr:
    return x[::2] + x[1::2]  # strided slices, same output length

  return f


def _slice_multidim_row() -> sc.Function:
  @sc.function(sc.arg("x", 12), outputs=sc.arg("out0"), name="pm_slice_row")
  def f(x: sc.Expr) -> sc.Expr:
    m = x.reshape((3, 4))
    return m[1, :] * m[2, :]  # integer index on dim 0, full slice on dim 1

  return f


def _dot() -> sc.Function:
  @sc.function(sc.group(sc.arg("x", 4), sc.arg("y", 4)), outputs=sc.arg("out0"), name="pm_dot")
  def f(inputs: tuple[sc.Expr, sc.Expr]) -> sc.Expr:
    x, y = inputs
    return x @ y

  return f


def _matvec() -> sc.Function:
  @sc.function(sc.group(sc.arg("A", (3, 4)), sc.arg("x", 4)), outputs=sc.arg("out0"), name="pm_matvec")
  def f(inputs: tuple[sc.Expr, sc.Expr]) -> sc.Expr:
    A, x = inputs
    return A @ x

  return f


def _vecmat() -> sc.Function:
  @sc.function(sc.group(sc.arg("x", 3), sc.arg("A", (3, 4))), outputs=sc.arg("out0"), name="pm_vecmat")
  def f(inputs: tuple[sc.Expr, sc.Expr]) -> sc.Expr:
    x, A = inputs
    return x @ A

  return f


def _matmat() -> sc.Function:
  @sc.function(sc.group(sc.arg("A", (2, 3)), sc.arg("B", (3, 2))), outputs=sc.arg("out0"), name="pm_matmat")
  def f(inputs: tuple[sc.Expr, sc.Expr]) -> sc.Expr:
    A, B = inputs
    return A @ B

  return f


def _sum() -> sc.Function:
  @sc.function(sc.arg("x", 5), outputs=sc.arg("out0"), name="pm_sum")
  def f(x: sc.Expr) -> sc.Expr:
    return (x.sin() + x).sum()

  return f


def _transpose() -> sc.Function:
  @sc.function(sc.arg("x", 6), outputs=sc.arg("out0"), name="pm_transpose")
  def f(x: sc.Expr) -> sc.Expr:
    return x.reshape((2, 3)).transpose()

  return f


def _call() -> sc.Function:
  @sc.function(sc.arg("a", 3), outputs=sc.arg("out0"), name="pm_call_inner")
  def inner(a: sc.Expr) -> sc.Expr:
    return a.sin() + a

  @sc.function(sc.arg("x", 3), outputs=sc.arg("out0"), name="pm_call_outer")
  def f(x: sc.Expr) -> sc.Expr:
    y = inner(x)
    return y * x

  return f


def _vmap() -> sc.Function:
  @sc.function(sc.arg("s", 2), outputs=sc.arg("out0"), name="pm_vmap_cell")
  def cell(s: sc.Expr) -> sc.Expr:
    return s.tanh() + s

  @sc.function(sc.arg("z", 6), outputs=sc.arg("out0"), name="pm_vmap_outer")
  def f(z: sc.Expr) -> sc.Expr:
    return _mapped_call(cell, 3, [(z, 0, 2)])  # 3 independent calls over z[2i:2i+2]

  return f


def _gather() -> sc.Function:
  @sc.function(sc.arg("x", 6), outputs=sc.arg("out0"), name="pm_gather")
  def f(x: sc.Expr) -> sc.Expr:
    return x.gather(np.array([5, 0, 3, 3, 1]))  # repeats + reorder, via const index table

  return f


def _scatter() -> sc.Function:
  @sc.function(sc.arg("x", 3), outputs=sc.arg("out0"), name="pm_scatter")
  def f(x: sc.Expr) -> sc.Expr:
    return sc.scatter(x, np.array([4, 1, 2]), 6)  # zero-filled length-6 output

  return f


def _broadcast_matrix() -> sc.Function:
  @sc.function(sc.group(sc.arg("x", (3, 4)), sc.arg("b", 4)), outputs=sc.arg("out0"), name="pm_bcast_mat")
  def f(inputs: tuple[sc.Expr, sc.Expr]) -> sc.Expr:
    x, b = inputs
    return x + b  # (3,4) + (4,) row broadcast

  return f


def _broadcast_scalar() -> sc.Function:
  @sc.function(sc.arg("x", 4), outputs=sc.arg("out0"), name="pm_bcast_scalar")
  def f(x: sc.Expr) -> sc.Expr:
    return x * 2.0 + 1.0  # scalar-const broadcast

  return f


def _concat() -> sc.Function:
  @sc.function(sc.group(sc.arg("x", 3), sc.arg("y", 2)), outputs=sc.arg("out0"), name="pm_concat")
  def f(inputs: tuple[sc.Expr, sc.Expr]) -> sc.Expr:
    x, y = inputs
    return sc.concat([x.sin(), y])

  return f


def _mlp_layer() -> sc.Function:
  @sc.function(sc.group(sc.arg("W", (4, 3)), sc.arg("x", 3), sc.arg("b", 4)), outputs=sc.arg("out0"), name="pm_mlp_layer")
  def f(inputs: tuple[sc.Expr, sc.Expr, sc.Expr]) -> sc.Expr:
    W, x, b = inputs
    return (W @ x + b).tanh()  # matmul + bias broadcast + activation

  return f


def _concat_axis1() -> sc.Function:
  @sc.function(sc.group(sc.arg("a", (2, 3)), sc.arg("b", (2, 2))), outputs=sc.arg("out0"), name="pm_concat_ax1")
  def f(inputs: tuple[sc.Expr, sc.Expr]) -> sc.Expr:
    a, b = inputs
    return sc.concat([a, b], axis=1)  # (2,3) ++ (2,2) -> (2,5) along axis 1

  return f


def _stack_axis1() -> sc.Function:
  @sc.function(sc.group(sc.arg("x", 3), sc.arg("y", 3)), outputs=sc.arg("out0"), name="pm_stack_ax1")
  def f(inputs: tuple[sc.Expr, sc.Expr]) -> sc.Expr:
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
  assert render_c_source(fn, lanes=1) == BuildRecipe(lanes=1).comment(f"{fn.name}.c") + src


@pytest.mark.skipif(not _HAVE_CC, reason="no C compiler available for JIT numeric check")
@pytest.mark.parametrize("builder, inputs", _CORPUS, ids=[b.__name__ for b, _ in _CORPUS])
def test_program_ir_jit_executes(builder, inputs) -> None:
  fn = builder()
  # Gate first so a coverage gap errors here loudly (there is no fallback).
  render_program_c_source(fn)
  fn.recompile()
  got = np.asarray(fn(*as_concrete(fn).input_tree.unflatten(tuple(inputs)))).reshape(-1)
  assert got.size == as_concrete(fn).outputs[0].size
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

  @sc.function(sc.group(sc.arg("a", sa), sc.arg("b", sb)), outputs=sc.arg("y"), name=f"mm_{name}")
  def fn(inputs):
    a, b = inputs
    return sc.simplify(product(a, b))

  assert all(e.op != sc.ExprOp.TRANSPOSE for e in topo(as_concrete(fn).outputs)), "the transpose was not folded into the product"
  rng = np.random.default_rng(0)
  av, bv = (rng.integers(-8, 9, shape).astype(np.float64) for shape in (sa, sb))
  fn.recompile()
  np.testing.assert_array_equal(fn((av, bv)), product(av, bv))


@pytest.mark.skipif(not _HAVE_CC, reason="no C compiler available for JIT numeric check")
def test_transpose_fold_is_bit_identical_to_the_transposed_product() -> None:
  """Both forms sum each output's terms in the same k order, so folding the transpose changes no bit."""
  rng = np.random.default_rng(1)
  av, vv, wv = rng.standard_normal((32, 12)), rng.standard_normal(32), rng.standard_normal(12)
  for tag, shape, xv, product in (("v", 32, vv, lambda m, x: m.T @ x), ("w", 12, wv, lambda m, x: x @ m.T)):

    @sc.function(sc.group(sc.arg("a", (32, 12)), sc.arg(tag, shape)), outputs=sc.arg("y"), name=f"mm_transposed_{tag}")
    def transposed(inputs):
      a, x = inputs
      return product(a, x)

    @sc.function(sc.group(sc.arg("a", (32, 12)), sc.arg(tag, shape)), outputs=sc.arg("y"), name=f"mm_folded_{tag}")
    def folded(inputs):
      a, x = inputs
      return sc.simplify(product(a, x))

    for fn in (transposed, folded):
      fn.recompile()
    np.testing.assert_array_equal(folded((av, xv)), transposed((av, xv)))


def test_lowered_program_verifies_and_has_single_proc() -> None:
  fn = _arith()
  prog = lower_function(fn)
  verify_program(prog)  # also called inside lower_function; assert it stays clean
  assert int(prog.attrs["proc_count"]) == 1
  assert main_proc(prog).op == ProgramOp.PROC


def _import_sibling(name):
  """Import a sibling test-fixture module (e.g. test_stage_transcription)."""
  import sys
  from pathlib import Path

  here = str(Path(__file__).parents[2] / "integration")
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
  inputs = [np.random.default_rng(0).standard_normal(e.size).reshape(e.shape) for e in as_concrete(fn).inputs]
  fn.recompile()
  assert len(as_concrete(fn).outputs) == 1
  got = np.asarray(fn(*as_concrete(fn).input_tree.unflatten(tuple(inputs)))).reshape(-1)
  assert got.size == as_concrete(fn).outputs[0].size
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

  @sc.function(sc.arg("x", 3), outputs=sc.group(sc.arg("cost"), sc.arg("empty")))
  def fn(x: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
    return x[:2].sum() + 1000.0 * x[2:2].sum(), sc.const(np.zeros(0))

  compiled = get_compiled(fn)
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
  sparsity = sc.SparsityPattern((2, 1), (0, 1), (0, 0))
  fn = ConcreteFunction._from_exprs(
    "normalized_metadata",
    [matrix, vector],
    [output],
    ["matrix", "vector"],
    ["product"],
    [sparsity],
    output_coloring_widths=[2],
  )
  observed: list[tuple[str, ConcreteFunction]] = []

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
  @sc.function(sc.arg("x", 64), outputs=sc.group(sc.arg("cos"), sc.arg("sin")), name=f"shared_outputs_{identity}")
  def fn(x):
    a = x.sin()
    outputs = [a.cos().block(), a] if identity == "none" else [a.cos(), (a * 1.0).block()]
    return tuple(sc.simplify(output) for output in outputs) if identity == "simplify" else tuple(outputs)

  assert render_c_source(fn, lanes=1).count("sin(") == 1
  values = np.linspace(-2.0, 2.0, 64)
  cosine, sine = fn(values)
  np.testing.assert_allclose(cosine, np.cos(np.sin(values)))
  np.testing.assert_allclose(sine, np.sin(values))


def test_normalization_keeps_constant_and_conflicting_function_hints() -> None:
  x = sc.sym("x", 2, lowering="block")
  fn = ConcreteFunction._from_exprs("normalized_hints", [x], [(x * 1.0).scalar(), sc.const([2.0, 3.0]).scalar()], ["x"], ["identity", "constant"])
  observed: list[ConcreteFunction] = []

  lower_function(fn, observe_expr=lambda _name, normalized: observed.append(normalized))

  normalized = observed[0]
  assert normalized._effective_lowering() == "block"
  assert normalized.outputs[0] is x
  assert normalized.outputs[1].op == sc.ExprOp.CONST
  assert normalized.outputs[1] is fn.outputs[1]
  assert main_proc(lower_function(fn)).attrs["lowering"] == "block"


@pytest.mark.parametrize(("dtype", "value"), [("float32", np.float32(1.0)), ("int64", np.int64(1))])
def test_normalization_preserves_typed_identity_boundaries(dtype: str, value: object) -> None:
  one = sc.const(np.full(2, value), dtype=dtype)

  @sc.function(sc.arg("x", sc.TensorType((2,), dtype=sc.as_dtype(dtype))), outputs=sc.arg("y"), name=f"normalized_{dtype}")
  def fn(x):
    return (x * one).scalar()

  x = as_concrete(fn).inputs[0]
  observed: list[ConcreteFunction] = []

  proc = main_proc(lower_function(fn, observe_expr=lambda _name, normalized: observed.append(normalized)))

  assert observed[0].outputs[0].type.dtype == x.type.dtype
  assert proc.attrs["scalarize_mode"] == "disabled"


@pytest.mark.skipif(not _HAVE_CC, reason="no C compiler available for JIT numeric check")
def test_automatic_transpose_normalization_preserves_cancellation_order_and_empty_reduction() -> None:
  @sc.function(sc.group(sc.arg("matrix", (3, 2)), sc.arg("vector", 3)), outputs=sc.arg("y"), name="normalized_cancel")
  def product(inputs):
    matrix, vector = inputs
    return matrix.T @ vector

  matrix_value = np.array([[1e16, -1e16], [1.0, 1.0], [-1e16, 1e16]])
  vector_value = np.ones(3)
  # summed row by row, (1e16 + 1) - 1e16 rounds to 0; NumPy's BLAS product may sum in another order
  np.testing.assert_array_equal(product((matrix_value, vector_value)), np.zeros(2))

  @sc.function(sc.group(sc.arg("matrix", (2, 0)), sc.arg("vector", 0)), outputs=sc.arg("y"), name="normalized_empty_product")
  def empty(inputs):
    matrix, vector = inputs
    return matrix @ vector

  np.testing.assert_array_equal(empty((np.empty((2, 0)), np.empty(0))), np.zeros(2))
