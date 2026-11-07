"""Lowering to Program IR: exact matrix products and reductions, and normalization of hinted functions."""

from __future__ import annotations

import numpy as np
import pytest

from scaly.function.model import as_concrete
from scaly.function.concrete import ConcreteFunction
import scaly as sc
from scaly.codegen.aot import render_c_source
from scaly.codegen.toolchain import find_c_compiler
from scaly.passes.lowering import lower_function, main_proc
from scaly.ir.expr import topo
from scaly.ir.program import ProgramOp

_HAVE_CC = find_c_compiler() is not None


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
def test_matmul_lowering_matches_numpy_exactly(name, isolated_cache) -> None:
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
def test_transpose_fold_is_bit_identical_to_the_transposed_product(isolated_cache) -> None:
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
  assert normalized._effective_lowering == "block"


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
  assert normalized._effective_lowering == "block"
  assert normalized.outputs[0] is x
  assert normalized.outputs[1].op == sc.ExprOp.CONST
  assert normalized.outputs[1] is fn.outputs[1]
  assert main_proc(lower_function(fn)).attrs["lowering"] == "block"


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


@pytest.mark.parametrize("shape", [(5, 9, 4), (2, 32, 1), (3, 0, 4), (5, 9, 7), (32, 12, 8), (5, 9, 2), (4, 9, 2), (2, 9, 9), (2, 9, 17)])
def test_matrix_panels_keep_ordered_reductions_private(shape, monkeypatch):
  from scaly.ir.program import walk_program
  from scaly.codegen import jit

  flags = jit.compile_flags
  monkeypatch.setattr(jit, "compile_flags", lambda recipe=None: (*flags(recipe), "-ffp-contract=off"))

  m, k, n = shape

  @sc.function(sc.group(sc.arg("a", (m, k)), sc.arg("b", (k, n))), outputs=sc.arg("out"), name="private_product")
  def product(inputs):
    a, b = inputs
    return a @ b

  observed = {}
  lower_function(product, observe=lambda phase, program: observed.__setitem__(phase, program))
  reductions = [node for node in walk_program(observed["lowered"]) if node.op == ProgramOp.FOR and node.args[0].attrs["kind"].value == "reduce"]
  assert bool(reductions) == bool(k)
  if k:
    expected_accumulators = 4 * n if n <= 4 and m >= 4 else min(n, 8)
    assert max(len(loop.args) - 1 for loop in reductions) == expected_accumulators
    assert not any(node.op == ProgramOp.STORE for loop in reductions for stmt in loop.args[1:] for node in walk_program(stmt))
  rng = np.random.default_rng(184)
  a, b = rng.normal(size=(m, k)), rng.normal(size=(k, n))
  expected = np.zeros((m, n))
  for index in range(k):
    expected += a[:, index, None] * b[index, None, :]
  np.testing.assert_array_equal(product((a, b)), expected)


def test_concat_seed_mass_component_axes_match_numpy_after_index_folding():
  from scaly.ir.program import walk_program

  @sc.function(sc.group(sc.arg("a", (5, 31, 3)), sc.arg("b", (5, 1, 3))), outputs=sc.arg("out"), name="concat_coordinates")
  def joined(inputs):
    a, b = inputs
    return sc.concat([b.reshape((5, 3)), (a.sin() * b).reshape((5, 93)), (b * b).reshape((5, 3))], axis=1)

  observed = {}
  lower_function(joined, observe=lambda phase, program: observed.__setitem__(phase, program))
  assert not any(node.op in (ProgramOp.DIV, ProgramOp.MOD) for node in walk_program(observed["pass:fold_arith"]))
  rng = np.random.default_rng(19)
  a, b = rng.normal(size=(5, 31, 3)), rng.normal(size=(5, 1, 3))
  np.testing.assert_allclose(
    joined((a, b)), np.concatenate([b.reshape(5, 3), (np.sin(a) * b).reshape(5, 93), (b * b).reshape(5, 3)], axis=1), rtol=1e-14, atol=1e-14
  )
