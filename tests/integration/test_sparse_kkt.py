"""The Maros-Meszaros problems exercise sparse Karush-Kuhn-Tucker assembly, products and derivatives."""

from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

import scaly as sc
from scaly.ad.sparsity import jacobian_sparsity
from scaly.function.model import as_concrete
from scaly.ir.program import ProgramOp, walk_program
from scaly.passes.lowering import lower_function

DATA = Path(__file__).with_name("data") / "maros_meszaros"
PROBLEMS = sorted(p.stem for p in DATA.glob("*.npz") if not p.stem.endswith("_patterns"))
assert len(PROBLEMS) == 48


def _problem(name):
  with np.load(DATA / f"{name}.npz") as raw:

    def matrix(key):
      return sparse.csc_array((raw[f"{key}_data"], raw[f"{key}_indices"], raw[f"{key}_indptr"]), shape=tuple(raw[f"{key}_shape"]))

    P, constraints = matrix("P"), matrix("A").tocsr()
    single = np.diff(constraints.indptr) == 1
    single_rows = np.flatnonzero(single)
    single[single_rows] &= constraints.data[constraints.indptr[single_rows]] == 1.0
    equality = raw["l"] == raw["u"]
    boxes = single & ~equality
    return P, constraints[equality].tocsc(), constraints[~equality & ~boxes].tocsc()


def _coordinates(matrix):
  csc = sparse.csc_array(matrix)
  return tuple(map(int, csc.indices)), tuple(map(int, np.repeat(np.arange(csc.shape[1]), np.diff(csc.indptr))))


def _leaf(matrix, values):
  return sc.pack(values, sc.SparsityPattern(matrix.shape, *_coordinates(matrix)))


def _same_pattern(actual, reference, prefix):
  shape = tuple(reference[f"{prefix}_shape"])
  assert actual.shape == shape
  actual_pairs = sorted(zip(actual.rows, actual.cols))
  expected_pairs = sorted(zip(reference[f"{prefix}_rows"], reference[f"{prefix}_cols"]))
  assert actual_pairs == expected_pairs


def _check_buffers(program, budget, full_sizes):
  for node in walk_program(program):
    if node.op == ProgramOp.BUFFER and node.attrs["address_space"] == "private":
      size = np.prod(node.attrs["shape"])
      assert size <= budget
      assert size not in full_sizes


@pytest.mark.parametrize("name", PROBLEMS)
def test_maros_meszaros_kkt_and_products_without_dense_matrix_buffers(name):
  P, A, G = _problem(name)
  n, p, m = P.shape[0], A.shape[0], G.shape[0]
  sizes = [P.nnz, A.nnz, G.nnz, n, m, 1]
  offsets = np.cumsum([0, *sizes])

  def assemble(v):
    parts = [v[int(a) : int(b)] for a, b in zip(offsets[:-1], offsets[1:])]
    pv, av, gv = [_leaf(pattern, values) for pattern, values in zip((P, A, G), parts[:3])]
    diagonal = sc.SparsityPattern((n, n), tuple(range(n)), tuple(range(n)))
    top_left = pv + sc.pack(parts[3], diagonal)
    bottom = sc.pack(-parts[4], sc.SparsityPattern((m, m), tuple(range(m)), tuple(range(m))))
    zero = sc.const(sparse.csc_array((p, m)))
    return sc.concat(
      (
        sc.concat((top_left, av.T, gv.T), axis=1),
        sc.concat((av, -parts[5][0] * sc.const(sparse.eye_array(p)), zero), axis=1),
        sc.concat((gv, zero.T, bottom), axis=1),
      ),
      axis=0,
    )

  kkt = sc.function(sc.arg("v", int(offsets[-1])), name=f"kkt_{name}")(lambda v: assemble(v).block())
  output = as_concrete(kkt).outputs[0]
  pattern = output.type.pattern
  assert pattern is not None
  full_sizes = {int(np.prod(mat.shape)) for mat in (P, A, G) if np.prod(mat.shape) >= 4096 and mat.nnz < np.prod(mat.shape)}
  if (n + p + m) ** 2 >= 4096 and pattern.nnz < (n + p + m) ** 2:
    full_sizes.add((n + p + m) ** 2)
  permutation = np.lexsort((pattern.rows, pattern.cols))
  values = sc.function(sc.arg("v", int(offsets[-1])), name=f"kkt_values_{name}")(lambda v: sc.gather(sc.values(assemble(v)), permutation).block())

  def product(matrix, label, transposed):
    cp = sc.SparsityPattern(matrix.shape, *_coordinates(matrix)).canonicalize()[0]
    input_type = sc.TensorType(matrix.shape, pattern=cp)
    width = matrix.shape[0] if transposed else matrix.shape[1]
    fn = sc.function(sc.arg("a", input_type, layout="csc"), sc.arg("x", width), name=f"product_{label}_{name}")(
      lambda a, x: ((a.T @ x) if transposed else (a @ x)).block()
    )
    return fn

  numeric = np.concatenate((P.data, A.data, G.data, np.linspace(0.1, 0.3, n), np.linspace(0.2, 0.4, m), [0.03]))
  reference = sparse.block_array(
    [
      [P + sparse.diags_array(numeric[offsets[3] : offsets[4]]), A.T, G.T],
      [A, -0.03 * sparse.eye_array(p), None],
      [G, None, -sparse.diags_array(numeric[offsets[4] : offsets[5]])],
    ],
    format="csc",
  )
  with np.load(DATA / f"{name}_patterns.npz") as expected:
    _same_pattern(pattern, expected, "kkt")
    actual = kkt(numeric)
    np.testing.assert_allclose(sparse.coo_array(actual).toarray(), reference.toarray(), atol=1e-11)
    concrete = as_concrete(values)
    incidence = jacobian_sparsity(concrete.outputs[0], concrete.inputs[0])
    _same_pattern(incidence, expected, "kkt_jac")
    derivative = sc.sparse_jacobian(values)
    dp = derivative.sparsity()
    assert dp is not None
    jac = sparse.coo_array((derivative(numeric), (dp.rows, dp.cols)), shape=dp.shape).tocsc()
    for column in np.unique(np.linspace(0, len(numeric) - 1, min(len(numeric), 5), dtype=int)):
      change = np.zeros_like(numeric)
      rows = jac.indices[jac.indptr[column] : jac.indptr[column + 1]]
      scale = max(1.0, abs(numeric[column]), np.max(np.abs(values(numeric)[rows]), initial=0))
      step = 1e-5 * scale
      change[column] = step
      difference = (values(numeric + change) - values(numeric - change)) / (2 * step)
      np.testing.assert_allclose(difference, jac[:, [column]].toarray().reshape(-1), rtol=1e-7, atol=1e-7)
    for label, matrix, transposed in (("P", P, False), ("A", A, False), ("G", G, False), ("AT", A, True), ("GT", G, True)):
      fn = product(matrix, label, transposed)
      concrete = as_concrete(fn)
      incidence = jacobian_sparsity(concrete.outputs[0], concrete.inputs[1])
      _same_pattern(incidence, expected, f"{label}_jac")
      x = np.random.default_rng(46).normal(size=matrix.shape[0] if transposed else matrix.shape[1])
      np.testing.assert_allclose(fn(matrix, x), (matrix.T if transposed else matrix) @ x, rtol=1e-11, atol=1e-10)
      seen = {}
      lower_function(fn, observe=seen.__setitem__)
      _check_buffers(seen["lowered"], 4 * (matrix.nnz + sum(matrix.shape)), full_sizes)
    seen = {}
    lower_function(derivative, observe=seen.__setitem__)
    lower_function(kkt, observe=lambda stage, program: seen.__setitem__("kkt_" + stage, program))
    for key in ("lowered", "kkt_lowered"):
      _check_buffers(seen[key], 4 * (pattern.nnz + len(numeric)), full_sizes)
