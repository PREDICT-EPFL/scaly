"""Sparse iteration plans visit stored coordinates and preserve ordered reductions."""

from scaly.ir.expr import _scatter_zero

import numpy as np
import pytest
from scipy import sparse

import scaly as sc
from scaly.ir.program import ProgramOp, walk_program
from scaly.ir.types import CanonicalPattern
from scaly.passes.lowering import lower_function
from scaly.ir import program as p
from scaly.ir.program import RangeKind
from scaly.passes.program.fuse_ranges import fuse_ranges
from scaly.passes.program.widen_ranges import widen_ranges
from scaly.function.model import as_concrete


@pytest.mark.parametrize("condition_kind", ["broadcast", "matrix_true"])
def test_sparse_select_constant_regions_read_only_the_chosen_branch(monkeypatch, condition_kind):
  from scaly.passes.lowering import plans
  from scaly.ir.expr import ExprOp
  from scaly.ir.types import stored_indices

  pattern = CanonicalPattern((1000, 1000), (0, 4, 99, 999), (9, 3, 71, 998))
  condition = (
    np.arange(1000)[None, :] % 2 == 0
    if condition_kind == "broadcast"
    else np.broadcast_to(np.asarray(condition_kind == "matrix_true"), pattern.shape)
  )
  captured = []
  emit = plans.emit_map

  def observe(ctx, node, reads, body, **kwargs):
    if node.op == ExprOp.SELECT:
      captured.append(reads)
    return emit(ctx, node, reads, body, **kwargs)

  monkeypatch.setattr(plans, "emit_map", observe)

  @sc.function(sc.arg("v", pattern.nnz), name="sparse_select_constant_regions")
  def fn(v):
    x = sc.pack(v, pattern)
    return sc.where(sc.const(condition), x.sin(), x.tanh()).block()

  observed = {}
  lower_function(fn, observe=observed.__setitem__)
  for node in walk_program(observed["lowered"]):
    if node.op == ProgramOp.BUFFER and node.attrs["address_space"] == "private":
      assert np.prod(node.attrs["shape"]) <= 20 * pattern.nnz
  assert len(captured) == 1
  chosen = condition.reshape(-1)[np.asarray(pattern.cols)]
  slots = np.arange(pattern.nnz)
  np.testing.assert_array_equal(captured[0][1], np.where(chosen, slots, -1))
  np.testing.assert_array_equal(captured[0][2], np.where(chosen, -1, slots))
  values = np.linspace(0.1, 1.0, pattern.nnz)
  actual = fn(values)
  expected = np.where(chosen, np.sin(values), np.tanh(values))
  np.testing.assert_allclose(actual.toarray().reshape(-1)[stored_indices(sc.TensorType(pattern.shape, pattern=pattern))], expected)


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
    return _scatter_zero(x, scatter_indices, (3, 3))

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


@pytest.mark.parametrize("operation", ["gather", "scatter", "stack", "concat"])
def test_sparse_zero_movement_keeps_its_stored_type(operation):
  pattern = CanonicalPattern((2, 2), (0, 1), (0, 1))

  def body(x):
    zero = x * 0
    return {
      "gather": lambda: sc.gather(zero, [[0, 1], [2, 3]]),
      "scatter": lambda: _scatter_zero(zero, [0, 1, 2, 3], (2, 2)),
      "stack": lambda: sc.stack((zero, zero)),
      "concat": lambda: sc.concat((zero, zero)),
    }[operation]().block()

  fn = sc.function(sc.arg("x", sc.TensorType(pattern.shape, pattern=pattern)), name=f"sparse_zero_{operation}")(body)
  result = fn(sparse.eye_array(2))
  assert isinstance(result, sparse.sparray)
  np.testing.assert_array_equal(sparse.coo_array(result).data, 0)


def test_sparse_slice_of_stack_keeps_its_stored_type():
  pattern = CanonicalPattern((2, 2), (0, 1), (0, 1))
  fn = sc.function(sc.arg("x", sc.TensorType(pattern.shape, pattern=pattern)), name="sparse_slice_stack")(
    lambda x: sc.stack((x, -x))[:, 0, :].block()
  )
  result = fn(sparse.eye_array(2))
  assert isinstance(result, sparse.sparray)
  np.testing.assert_array_equal(sparse.coo_array(result).toarray(), [[1, 0], [-1, 0]])


@pytest.mark.parametrize("operation", ["reshape", "transpose"])
def test_sparse_identity_preserves_empty_batch_pattern(operation):
  pattern = CanonicalPattern((2, 2), (0, 1), (0, 1))
  fn = sc.function(sc.arg("x", sc.TensorType((0, 2, 2), pattern=pattern)), name=f"sparse_empty_{operation}")(
    lambda x: (x.reshape(x.shape) if operation == "reshape" else x.transpose((0, 1, 2))).block()
  )
  assert as_concrete(fn).outputs[0].type.pattern is pattern
  result = fn(np.empty((0, 2)))
  assert isinstance(result, sparse.sparray)
  assert result.shape == (0, 2, 2) and result.nnz == 0


@pytest.mark.parametrize("left", [True, False])
@pytest.mark.parametrize("vector", [True, False])
def test_sparse_scalar_hint_expands_mixed_products(left, vector):
  pattern = CanonicalPattern((6, 5), (0, 0, 1, 2, 3, 3, 5, 5), (0, 4, 2, 2, 1, 3, 0, 2))
  shape = ((5,) if left else (6,)) if vector else ((5, 3) if left else (3, 6))
  fn = sc.function(
    sc.arg("a", sc.TensorType(pattern.shape, pattern=pattern), layout="csc"), sc.arg("b", shape), name=f"sparse_scalar_{left}_{vector}"
  )(lambda a, b: ((a.sin() @ b if left else b @ a.sin()) * 2).with_lowering("scalar"))
  proc = lower_function(fn).args[0]
  assert proc.attrs.get("scalarized")
  assert not any(n.op == ProgramOp.FOR or n.op == ProgramOp.BUFFER and "values" in n.attrs for n in walk_program(proc))
  data = np.linspace(-0.5, 0.8, pattern.nnz)
  a = sparse.coo_array((data, (pattern.rows, pattern.cols)), shape=pattern.shape)
  b = np.random.default_rng(46).normal(size=shape)
  actual = fn(a, b)
  transformed = np.sin(a.toarray())
  np.testing.assert_allclose(
    sparse.coo_array(actual).toarray() if sparse.issparse(actual) else actual, 2 * (transformed @ b if left else b @ transformed), atol=1e-12
  )


@pytest.mark.parametrize("n", [4, 128])
def test_sparse_mapped_stage_expands_small_default_product(n):
  random = np.random.default_rng(46)
  pattern = (
    CanonicalPattern((4, 4), (0, 0, 1, 2, 3, 3), (0, 3, 1, 2, 0, 2))
    if n == 4
    else CanonicalPattern(
      (n, n), tuple(map(int, np.repeat(np.arange(n), 8))), tuple(map(int, np.concatenate([random.choice(n, 8, replace=False) for _ in range(n)])))
    )
  )
  leaf = sc.arg("a", sc.TensorType(pattern.shape, pattern=pattern), layout="csc")
  stage = sc.function(leaf, sc.arg("x", n), name="sparse_scalar_stage")(lambda a, x: (a @ x).sin() + x)
  fn = sc.function(leaf, sc.arg("xs", (10, n)), name="sparse_scalar_map")(lambda a, xs: sc.vmap(stage, 10)(sc.broadcast(a), xs))
  observed = {}
  lower_function(fn, observe=observed.__setitem__)
  callee = observed["pass:scalarize"].args[0]
  assert callee.attrs.get("scalarized")
  assert not any(n.op == ProgramOp.FOR or n.op == ProgramOp.BUFFER and "values" in n.attrs for n in walk_program(callee))
  a = sparse.coo_array((random.normal(size=pattern.nnz), (pattern.rows, pattern.cols)), shape=pattern.shape)
  xs = random.normal(size=(10, n))
  np.testing.assert_allclose(np.asarray(fn(a, xs)), np.sin((a @ xs.T).T) + xs, atol=1e-12)


@pytest.mark.parametrize("source_layout", ["csc", "csr"])
def test_sparse_map_orders_writes_to_avoid_duplicate_layout_tables(source_layout):
  rows, cols = np.nonzero(np.random.default_rng(3).random((7, 9)) < 0.3)
  pattern = CanonicalPattern((7, 9), tuple(map(int, rows)), tuple(map(int, cols)))
  type_ = sc.TensorType(pattern.shape, pattern=pattern)
  fn = sc.function(sc.arg("x", type_, layout=source_layout), outputs=sc.arg("y", type_, layout="csc"), name=f"sparse_contiguous_{source_layout}")(
    lambda x: x.sin().block()
  )
  tables = [n for n in walk_program(_lowered(fn)) if n.op == ProgramOp.BUFFER and "values" in n.attrs]
  assert len(tables) == (0 if source_layout == "csc" else 1)
  a = sparse.coo_array((np.linspace(-0.5, 0.8, pattern.nnz), (rows, cols)), shape=pattern.shape)
  np.testing.assert_allclose(sparse.coo_array(fn(a)).toarray(), np.sin(a.toarray()), atol=1e-12)


def test_sparse_scatter_folds_output_layout_into_destination_table():
  pattern = CanonicalPattern((4, 5), (0, 0, 1, 1, 2, 3), (0, 3, 0, 3, 4, 4))
  indices = np.arange(20)
  indices[[0, 3, 5, 8, 14, 19]] = [8, 0, 12, 3, 19, 8]
  fn = sc.function(sc.arg("x", sc.TensorType(pattern.shape, pattern=pattern), layout="csr"), name="sparse_scatter_layout")(
    lambda x: _scatter_zero(x, indices, pattern.shape).block()
  )
  tables = [n for n in walk_program(_lowered(fn)) if n.op == ProgramOp.BUFFER and "values" in n.attrs]
  assert len(tables) == 1
  a = sparse.coo_array((np.arange(1.0, 7.0), (pattern.rows, pattern.cols)), shape=pattern.shape)
  reference = np.zeros(20)
  np.add.at(reference, indices[np.asarray(pattern.rows) * 5 + pattern.cols], a.data)
  np.testing.assert_array_equal(sparse.coo_array(fn(a)).toarray(), reference.reshape(pattern.shape))


@pytest.mark.parametrize("lowering", ["scalar", "block"])
def test_elementwise_structural_zeros_skip_infinity_and_runtime_stored_zeros_do_not(lowering):
  pattern = CanonicalPattern((2, 2), (0, 1), (0, 1))
  fn = sc.function(
    sc.arg("a", sc.TensorType(pattern.shape, pattern=pattern), layout="csr"), sc.arg("b", pattern.shape), name=f"stored_zero_inf_{lowering}"
  )(lambda a, b: (a * b).with_lowering(lowering))
  a = sparse.coo_array(([1.0, 0.0], (pattern.rows, pattern.cols)), shape=pattern.shape)
  actual = sparse.coo_array(fn(a, np.full((2, 2), np.inf))).toarray()
  np.testing.assert_array_equal(actual, [[np.inf, 0], [0, np.nan]])


@pytest.mark.parametrize("operation", ["identity", "values"])
@pytest.mark.parametrize("layout", ["csr", "csc", "coordinates"])
def test_sparse_boundary_copy_writes_directly_without_private_values(operation, layout):
  pattern = CanonicalPattern((4, 5), (0, 1, 3, 3), (4, 0, 1, 4))
  source_layout = sc.SparsityPattern(pattern.shape, (3, 0, 3, 1), (4, 4, 1, 0)) if layout == "coordinates" else layout
  leaf = sc.arg("x", sc.TensorType(pattern.shape, pattern=pattern), layout=source_layout)
  output = sc.arg("y", pattern.nnz) if operation == "values" else sc.arg("y", sc.TensorType(pattern.shape, pattern=pattern), layout="csc")
  fn = sc.function(leaf, outputs=output, name=f"direct_{operation}_{layout}")(lambda x: sc.values(x) if operation == "values" else x)
  program = _lowered(fn)
  assert not any(n.op == ProgramOp.BUFFER and n.attrs["address_space"] == "private" for n in walk_program(program))
  data = np.array([0.2, -0.5, 1.2, 2.3])
  source = sparse.coo_array((data, (pattern.rows, pattern.cols)), shape=pattern.shape)
  actual = fn(source)
  if operation == "values":
    np.testing.assert_array_equal(actual, data)
  else:
    np.testing.assert_array_equal(sparse.coo_array(actual).toarray(), source.toarray())


def test_batched_shared_sparse_output_copies_between_declared_layouts():
  pattern = CanonicalPattern((4, 5), (0, 1, 3, 3), (4, 0, 1, 4))
  type_ = sc.TensorType((2, *pattern.shape), pattern=pattern)
  outputs = sc.group(sc.arg("y", type_, layout="csc"), sc.arg("z", type_, layout="csr"))

  def body(x):
    y = -x
    return y, y

  fn = sc.function(sc.arg("x", type_, layout="csc"), outputs=outputs, name="direct_shared_batched")(body)
  assert not any(n.op == ProgramOp.BUFFER and n.attrs["address_space"] == "private" for n in walk_program(_lowered(fn)))
  source = sparse.coo_array((np.arange(8.0), (np.repeat(np.arange(2), 4), np.tile(pattern.rows, 2), np.tile(pattern.cols, 2))), shape=type_.shape)
  for actual in fn(source):
    assert isinstance(actual, sparse.sparray)
    np.testing.assert_array_equal(sparse.coo_array(actual).toarray(), -source.toarray())
