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
# Each of these repeats the KKT structure of a kept problem in its family; S268 is HS268 byte for byte.
PROBLEMS = [name for name in PROBLEMS if name not in {"CVXQP1_S", "DUAL2", "DUALC2", "DUALC5", "HS52", "HS53", "PRIMALC1", "S268"}]


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
    inequalities = ~equality & ~boxes
    G = constraints[inequalities].copy()
    free = (raw["l"][inequalities] <= -1e19) & (raw["u"][inequalities] >= 1e19)
    G.data[np.repeat(free, np.diff(G.indptr))] = 0
    return P, constraints[equality].tocsc(), G.tocsc()


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
  upper_P = sparse.triu(P, format="csc")
  sizes = [upper_P.nnz, A.nnz, G.nnz, n, m, 1]
  offsets = np.cumsum([0, *sizes])
  rows, cols = map(np.asarray, _coordinates(upper_P))
  off = np.flatnonzero(rows != cols)
  mirrored = sc.SparsityPattern(P.shape, tuple(map(int, np.concatenate((rows, cols[off])))), tuple(map(int, np.concatenate((cols, rows[off])))))
  sources = np.concatenate((np.arange(upper_P.nnz), off))

  def quantities(v):
    parts = [v[int(a) : int(b)] for a, b in zip(offsets[:-1], offsets[1:])]
    return sc.pack(sc.gather(parts[0], sources), mirrored), _leaf(upper_P, parts[0]), _leaf(A, parts[1]), _leaf(G, parts[2]), parts

  def assemble(v, upper):
    pv, pu, av, gv, parts = quantities(v)
    diagonal = sc.SparsityPattern((n, n), tuple(range(n)), tuple(range(n)))
    top_left = (pu if upper else pv) + sc.pack(parts[3], diagonal)
    bottom = sc.pack(-parts[4], sc.SparsityPattern((m, m), tuple(range(m)), tuple(range(m))))
    zero = sc.const(sparse.csc_array((p, m)))
    return sc.concat(
      (
        sc.concat((top_left, av.T, gv.T), axis=1),
        sc.concat((sc.const(sparse.csc_array((p, n))) if upper else av, -parts[5][0] * sc.const(sparse.eye_array(p)), zero), axis=1),
        sc.concat((sc.const(sparse.csc_array((m, n))) if upper else gv, zero.T, bottom), axis=1),
      ),
      axis=0,
    )

  def product(matrix, label, transposed):
    cp = sc.SparsityPattern(matrix.shape, *_coordinates(matrix)).canonicalize()[0]
    input_type = sc.TensorType(matrix.shape, pattern=cp)
    width = matrix.shape[0] if transposed else matrix.shape[1]
    fn = sc.function(sc.arg("a", input_type, layout="csc"), sc.arg("x", width), name=f"product_{label}_{name}")(
      lambda a, x: ((a.T @ x) if transposed else (a @ x)).block()
    )
    return fn

  numeric = np.concatenate((upper_P.data, A.data, G.data, np.linspace(0.1, 0.3, n), np.linspace(0.2, 0.4, m), [0.03]))
  reference = sparse.block_array(
    [
      [P + sparse.diags_array(numeric[offsets[3] : offsets[4]]), A.T, G.T],
      [A, -0.03 * sparse.eye_array(p), None],
      [G, None, -sparse.diags_array(numeric[offsets[4] : offsets[5]])],
    ],
    format="csc",
  )
  with np.load(DATA / f"{name}_patterns.npz") as expected:
    full_sizes = {int(np.prod(mat.shape)) for mat in (P, A, G) if np.prod(mat.shape) >= 4096 and mat.nnz < np.prod(mat.shape)}
    budget = 4 * (len(expected["kkt_rows"]) + len(numeric))
    if (n + p + m) ** 2 >= 4096 and len(expected["kkt_rows"]) < (n + p + m) ** 2:
      full_sizes.add((n + p + m) ** 2)
    for upper, prefix in ((False, "kkt"), (True, "upper")):
      kkt = sc.function(sc.arg("v", int(offsets[-1])), name=f"{prefix}_{name}")(lambda v: assemble(v, upper).block())
      pattern = as_concrete(kkt).outputs[0].type.pattern
      assert pattern is not None
      _same_pattern(pattern, expected, prefix)
      permutation = np.lexsort((pattern.rows, pattern.cols))
      values = sc.function(sc.arg("v", int(offsets[-1])), name=f"{prefix}_values_{name}")(
        lambda v: sc.gather(sc.values(assemble(v, upper)), permutation).block()
      )
      actual = kkt(numeric)
      np.testing.assert_allclose(sparse.coo_array(actual).toarray(), sparse.triu(reference).toarray() if upper else reference.toarray(), atol=1e-11)
      concrete = as_concrete(values)
      incidence = jacobian_sparsity(concrete.outputs[0], concrete.inputs[0])
      _same_pattern(incidence, expected, f"{prefix}_jac")
      derivative = sc.sparse_jacobian(values)
      jac = derivative(numeric)
      assert isinstance(jac, sparse.csc_array)
      numeric_values = values(numeric)
      assert isinstance(numeric_values, np.ndarray)
      for column in np.unique(np.linspace(0, len(numeric) - 1, min(len(numeric), 5), dtype=int)):
        change = np.zeros_like(numeric)
        selected = jac.indices[jac.indptr[column] : jac.indptr[column + 1]]
        scale = max(1.0, abs(numeric[column]), np.max(np.abs(numeric_values[selected]), initial=0))
        step = 1e-5 * scale
        change[column] = step
        difference = (values(numeric + change) - values(numeric - change)) / (2 * step)
        np.testing.assert_allclose(difference, jac[:, [column]].toarray().reshape(-1), rtol=1e-7, atol=1e-7)
      for fn in (kkt, derivative):
        seen = {}
        lower_function(fn, observe=seen.__setitem__)
        _check_buffers(seen["lowered"], budget, full_sizes)
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
      index = 0 if label == "P" else 2 if label in ("A", "AT") else 3
      data_fn = sc.function(sc.arg("v", int(offsets[-1])), sc.arg("x", len(x)), name=f"data_product_{label}_{name}")(
        lambda v, x: ((quantities(v)[index].T @ x) if transposed else (quantities(v)[index] @ x)).block()
      )
      concrete = as_concrete(data_fn)
      incidence = jacobian_sparsity(concrete.outputs[0], concrete.inputs[0])
      _same_pattern(incidence, expected, f"{label}_data")
      np.testing.assert_allclose(data_fn(numeric, x), (matrix.T if transposed else matrix) @ x, rtol=1e-11, atol=1e-10)
      seen = {}
      lower_function(data_fn, observe=seen.__setitem__)
      _check_buffers(seen["lowered"], budget, full_sizes)
