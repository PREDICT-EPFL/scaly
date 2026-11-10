"""Sparse iteration plans visit stored coordinates and preserve ordered reductions."""

import numpy as np
import pytest
from scipy import sparse

import scaly as sc
from scaly.ir.program import ProgramOp, walk_program
from scaly.ir.types import CanonicalPattern
from scaly.passes.lowering import lower_function
from scaly.passes.lowering.plans import map_regions
from scaly.ir import program as p
from scaly.ir.program import RangeKind
from scaly.passes.program.fuse_ranges import fuse_ranges
from scaly.passes.program.widen_ranges import widen_ranges


def _pattern(array):
  coo = sparse.coo_array(array)
  return CanonicalPattern(coo.shape, tuple(map(int, coo.row)), tuple(map(int, coo.col)))


def _lowered(fn):
  seen = {}
  lower_function(fn, observe=seen.__setitem__)
  return seen["lowered"]


@pytest.mark.parametrize("operation", ["transpose", "slice", "reshape", "concat", "stack", "gather", "scatter"])
@pytest.mark.parametrize("lowering", ["block", "scalar"])
def test_sparse_movement_maps_match_numpy(operation, lowering):
  source = sparse.coo_array(([2.0, 0.0, -3.0, 4.0], ([0, 1, 2, 2], [3, 0, 1, 3])), shape=(3, 4))
  pattern = _pattern(source)
  indices = np.array([[11, 0, 3], [8, 9, 4]])
  scatter_indices = np.array([0, 1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 8])

  def body(x):
    if operation == "transpose":
      return x.T
    if operation == "slice":
      return x[::-1, ::2]
    if operation == "reshape":
      return x.reshape((2, 6))
    if operation == "concat":
      return sc.concat((x, -x), axis=1)
    if operation == "stack":
      return sc.stack((x, -x))
    if operation == "gather":
      return sc.gather(x, indices)
    return sc.scatter(x, scatter_indices, (3, 3))

  fn = sc.function(sc.arg("x", sc.TensorType(pattern.shape, pattern=pattern), layout="csc"), name=f"sparse_{operation}_{lowering}")(
    lambda x: body(x).with_lowering(lowering)
  )
  dense = source.toarray()
  if operation == "scatter":
    reference = np.zeros(9)
    np.add.at(reference, scatter_indices, dense.reshape(-1))
    reference = reference.reshape(3, 3)
  else:
    reference = {
      "transpose": lambda: dense.T,
      "slice": lambda: dense[::-1, ::2],
      "reshape": lambda: dense.reshape(2, 6),
      "concat": lambda: np.concatenate((dense, -dense), axis=1),
      "stack": lambda: np.stack((dense, -dense)),
      "gather": lambda: dense.reshape(-1)[indices],
    }[operation]()
  result = fn(source)
  assert isinstance(result, sparse.sparray)
  np.testing.assert_array_equal(result.toarray(), reference)
  buffers = [n for n in walk_program(_lowered(fn)) if n.op == ProgramOp.BUFFER and n.attrs["address_space"] == "private"]
  assert all(np.prod(n.attrs["shape"]) <= 2 * source.nnz for n in buffers)


@pytest.mark.parametrize("left", [True, False])
@pytest.mark.parametrize("vector", [True, False])
def test_mixed_product_uses_loaded_bounds_and_linear_tables(left, vector):
  n = 41
  source = sparse.diags_array([np.arange(1.0, n), np.arange(2.0, n + 2)], offsets=[-1, 0], shape=(n, n), format="csc")
  pattern = _pattern(source)
  dense_shape = (n,) if vector else ((n, 7) if left else (7, n))
  fn = sc.function(sc.arg("a", sc.TensorType(pattern.shape, pattern=pattern), layout="csc"), sc.arg("b", dense_shape), name=f"mixed_{left}_{vector}")(
    lambda a, b: (a @ b if left else b @ a).block()
  )
  values = np.random.default_rng(46).normal(size=dense_shape)
  result = fn(source, values)
  reference = source @ values if left else values @ source
  np.testing.assert_allclose(result.toarray() if sparse.issparse(result) else result, reference, atol=1e-12)
  program = _lowered(fn)
  loaded = [n for n in walk_program(program) if n.op == ProgramOp.RANGE and any(a.op == ProgramOp.LOAD for a in n.args[:2])]
  assert loaded
  tables = [n for n in walk_program(program) if n.op == ProgramOp.BUFFER and "values" in n.attrs]
  assert sum(np.prod(n.attrs["shape"]) for n in tables) <= 8 * (source.nnz + n)


def test_compact_output_writes_its_declared_layout_without_permutation_copy():
  pattern = CanonicalPattern((4, 5), (0, 1, 3, 3), (4, 0, 1, 4))
  leaf = sc.arg("x", sc.TensorType(pattern.shape, pattern=pattern), layout="csc")
  fn = sc.function(leaf, outputs=leaf.relabel("y"), name="direct_sparse_layout")(lambda x: x.sin().block())
  program = _lowered(fn)
  assert not any(n.op == ProgramOp.BUFFER and n.attrs["address_space"] == "private" for n in walk_program(program))
  actual = sparse.coo_array(([0.2, -0.5, 1.2, 2.3], (pattern.rows, pattern.cols)), shape=pattern.shape)
  result = fn(actual)
  assert isinstance(result, sparse.sparray)
  np.testing.assert_allclose(sparse.coo_array(result).toarray(), np.sin(actual.toarray()))


@pytest.mark.parametrize("left", [True, False])
def test_structural_zero_products_skip_infinite_factors(left):
  pattern = CanonicalPattern((2, 2), (0, 1), (0, 1))
  fn = sc.function(sc.arg("a", sc.TensorType(pattern.shape, pattern=pattern)), sc.arg("x", 2), name=f"structural_inf_{left}")(
    lambda a, x: (a @ x if left else x @ a).block()
  )
  a = sparse.eye_array(2, format="csc")
  np.testing.assert_array_equal(fn(a, np.array([np.inf, 3.0])), [np.inf, 3.0])


@pytest.mark.parametrize("form", ["straight", "dense", "affine", "table"])
def test_each_map_region_form_matches_numpy(form):
  indices = {
    "straight": np.array([9, 2, 4]),
    "dense": np.arange(13),
    "affine": np.arange(12, -1, -1),
    "table": np.array([4, 7, 0, 1, 11, 6, 10, 8, 2, 12, 3, 9, 5]),
  }[form]
  (region,) = map_regions(np.arange(len(indices)), (indices,))
  assert region.form == form
  pattern = CanonicalPattern((13, 17), tuple(range(13)), tuple(i * 5 % 17 for i in range(13)))
  logical = np.asarray(pattern.rows) * 17 + pattern.cols
  fn = sc.function(sc.arg("x", sc.TensorType(pattern.shape, pattern=pattern), layout="csr"), name=f"region_{form}")(
    lambda x: sc.gather(x, logical[indices]).block()
  )
  values = np.random.default_rng(46).normal(size=13)
  np.testing.assert_array_equal(fn(values), values[indices])
  program = _lowered(fn)
  tables = [n for n in walk_program(program) if n.op == ProgramOp.BUFFER and "values" in n.attrs]
  assert bool(tables) == (form == "table")
  loops = [n for n in walk_program(program) if n.op == ProgramOp.FOR]
  assert bool(loops) == (form != "straight")


def test_accumulating_scatter_range_is_never_widened():
  x, out = p.buffer("x", sc.dtypes.float64, (19,)), p.buffer("out", sc.dtypes.float64, (5,))
  i = p.var("i")
  dst = p.view(out, [p.mod(i, p.const_int(5))])
  loop = p.for_(p.range_("i", 0, 19, kind=RangeKind.REDUCE), [p.store(dst, p.add(p.load(dst), p.load(p.view(x, [i]))))])
  program = p.program([p.proc("scatter_contract", [x, out], [loop])])
  assert widen_ranges(program, lanes=4) is program


def test_carried_range_is_never_split_into_lanes():
  out = p.buffer("out", sc.dtypes.float64, (19,))
  carry = p.var("carry", sc.dtypes.float64)
  loop = p.for_(
    p.range_("i", 0, 19, kind=RangeKind.SERIAL), [p.assign("carry", p.mul(carry, p.const_float(2))), p.store(p.view(out, [p.var("i")]), carry)]
  )
  program = p.program([p.proc("carried_contract", [out], [p.assign("carry", p.const_float(1), declare=True), loop])])
  assert widen_ranges(program, lanes=4) is program


def test_loaded_bound_reduce_is_never_fused_or_widened():
  x, out = p.buffer("x", sc.dtypes.float64, (19,)), p.buffer("out", sc.dtypes.float64, (9,))
  bounds = p.const_buffer("bounds", sc.dtypes.int64, (10,), [0, 2, 3, 7, 8, 10, 12, 15, 18, 19])
  row, k = p.var("row"), p.var("k")
  reduced = p.range_("k", p.load(p.view(bounds, [row])), p.load(p.view(bounds, [p.add(row, p.const_int(1))])), kind=RangeKind.REDUCE)
  acc = p.var("acc", sc.dtypes.float64)
  body = [
    p.assign("acc", p.const_float(0), declare=True),
    p.for_(reduced, [p.assign("acc", p.add(acc, p.load(p.view(x, [k]))))]),
    p.store(p.view(out, [row]), acc),
  ]
  loop = p.for_(p.range_("row", 0, 9), body)
  program = p.program([p.proc("loaded_contract", [x, out], [bounds, loop])])
  assert fuse_ranges(program) is program
  assert widen_ranges(program, lanes=4) is program
