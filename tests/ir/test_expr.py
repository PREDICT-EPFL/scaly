from __future__ import annotations

from scaly.ir.expr import _scatter_zero

import numpy as np
import pytest

from scaly.function.concrete import ConcreteFunction
from scaly.function.model import as_concrete
from scaly.function.sugar import _mapped_call
import scaly as sc
from scaly.ir.expr import Expr, ExprOp, substitute
from scaly.ir.types import TensorType, frozen
from scaly.passes.expr import cse_many


def test_scatter_accepts_tuple_values() -> None:
  result = sc.scatter(sc.const([10.0, 20.0]), [0, 1], (2.0, 3.0), combine="set")
  assert result.shape == (2,)
  from scaly.passes.expr import simplify_cse_fixpoint

  np.testing.assert_array_equal(simplify_cse_fixpoint(result).value, [2.0, 3.0])


def test_scatter_sparse_values_into_nonzero_base() -> None:
  from scaly.ir.expr import from_dense
  from scaly.ir.types import CanonicalPattern

  pattern = CanonicalPattern((2, 2), (0, 1), (0, 1))

  @sc.function(sc.arg("base", 6), sc.arg("v", (2, 2)), outputs=sc.arg("y"))
  def indexed(base, v):
    return sc.scatter(base, [[5, 4], [1, 0]], from_dense(v, pattern))

  base = np.arange(6.0) + 10
  expected = base.copy()
  np.add.at(expected, [5, 4, 1, 0], [1, 0, 0, 4])
  np.testing.assert_array_equal(indexed(base, np.array([[1.0, 2.0], [3.0, 4.0]])), expected)


def test_axis_scatter_rejects_transposed_update_shape() -> None:
  with pytest.raises(ValueError, match="shape"):
    sc.index_add(sc.const(np.zeros((2, 4))), [0, 1, 3], sc.sym("wrong_updates", (3, 2)))


def test_fill_accepts_constant_int64_extremes() -> None:
  indices = np.array([np.iinfo(np.int64).min, 0, np.iinfo(np.int64).max, 1])

  @sc.function(sc.arg("x", 3), outputs=sc.group(sc.arg("g"), sc.arg("s")))
  def indexed(x):
    return sc.gather(x, indices, mode="fill"), sc.scatter(x, indices, sc.const([99.0, 2.0, 99.0, 3.0]), mode="fill")

  g, s = indexed(np.array([5.0, 7.0, 11.0]))
  np.testing.assert_array_equal(g, [0, 5, 0, 7])
  np.testing.assert_array_equal(s, [7, 10, 11])


@pytest.mark.parametrize("kind", ["gather", "add", "set"])
def test_runtime_index_forward_graph_size_does_not_grow_with_seed_count(kind) -> None:
  from scaly.ir.expr import topo

  x, updates = sc.sym("batch_x", 8), sc.sym("batch_updates", 4)
  indices = sc.sym("batch_indices", 4, dtype=sc.dtypes.int64)
  y = sc.gather(x, indices, mode="fill") if kind == "gather" else sc.scatter(x, indices, updates, combine=kind, mode="fill")
  sizes = []
  for count in (2, 8):
    derivative = sc.jvp_many(y * y, x, sc.sym(f"batch_seeds_{count}", (count, 8)))
    sizes.append(sum(node.op in (ExprOp.GATHER, ExprOp.SCATTER) for node in topo((derivative,))))
  assert sizes[0] == sizes[1]


def test_runtime_gather_fill_matches_numpy() -> None:
  @sc.function(sc.group(sc.arg("x", 4), sc.arg("idx", (2, 3))), outputs=sc.arg("y"))
  def indexed(inputs):
    x, idx = inputs
    return sc.gather(x, sc.cast(idx, sc.dtypes.int64), mode="fill", fill_value=-7.0)

  x = np.array([2.0, 3.0, 5.0, 11.0])
  idx = np.array([[3, 1, -1], [4, 0, 1]], dtype=np.int64)
  expected = np.where((idx >= 0) & (idx < len(x)), x[np.clip(idx, 0, len(x) - 1)], -7.0)
  np.testing.assert_array_equal(indexed((x, idx)), expected)


@pytest.mark.parametrize("size", [0, 4])
def test_constant_gather_fill_matches_numpy(size) -> None:
  from scaly.passes.expr import simplify_cse_fixpoint

  x = np.arange(size, dtype=np.float64)
  idx = np.array([-2, 0, 3, 4])
  result = simplify_cse_fixpoint(sc.gather(sc.const(x), idx, mode="fill", fill_value=-7.0))
  expected = np.full(idx.shape, -7.0)
  valid = (idx >= 0) & (idx < size)
  expected[valid] = x[idx[valid]]
  np.testing.assert_array_equal(result.value, expected)


def test_runtime_gather_forward_derivative_matches_finite_difference() -> None:
  @sc.function(sc.group(sc.arg("x", 4), sc.arg("idx", 5), sc.arg("seed", 4)), outputs=sc.group(sc.arg("y"), sc.arg("dy"), sc.arg("adjoint")))
  def indexed(inputs):
    x, idx, seed = inputs
    y = sc.gather(x, idx.cast(sc.dtypes.int64), mode="fill", fill_value=-7.0)
    return y, sc.jvp(y, x, seed), sc.vjp((y,), (x,), (sc.const(np.ones(5)),))[0]

  x, seed = np.array([2.0, 3.0, 5.0, 11.0]), np.array([0.2, -0.3, 0.7, 0.9])
  idx = np.array([3.0, 1.0, -1.0, 4.0, 1.0])
  _, tangent, adjoint = indexed((x, idx, seed))
  plus, _, _ = indexed((x + 1e-5 * seed, idx, seed))
  minus, _, _ = indexed((x - 1e-5 * seed, idx, seed))
  np.testing.assert_allclose(np.asarray(tangent), (np.asarray(plus) - np.asarray(minus)) / 2e-5, rtol=1e-8, atol=1e-9)
  np.testing.assert_array_equal(adjoint, [0, 2, 0, 1])
  np.testing.assert_allclose(np.dot(np.asarray(adjoint), seed), ((np.asarray(plus) - np.asarray(minus)) / 2e-5).sum(), rtol=1e-8, atol=1e-9)


@pytest.mark.parametrize("combine", ["set", "add", "max", "min"])
def test_scatter_base_and_runtime_indices_match_numpy(combine) -> None:
  @sc.function(sc.group(sc.arg("base", 4), sc.arg("idx", 6), sc.arg("values", 6)), outputs=sc.arg("y"))
  def indexed(inputs):
    base, idx, values = inputs
    return sc.scatter(base, idx.cast(sc.dtypes.int64), values, combine=combine, mode="fill")

  base = np.array([3.0, 5.0, -1.0, 7.0])
  idx = np.array([1, -1, 1, 4, 3, 1])
  values = np.array([2.0, 99.0, 9.0, 99.0, np.nan, 4.0])
  expected = base.copy()
  for i, value in zip(idx, values, strict=True):
    if 0 <= i < base.size:
      expected[i] = value if combine == "set" else {"add": np.add, "max": np.maximum, "min": np.minimum}[combine](expected[i], value)
  np.testing.assert_array_equal(indexed((base, idx.astype(float), values)), expected)


@pytest.mark.parametrize("combine", ["set", "add", "max", "min"])
def test_runtime_scatter_derivatives_match_finite_differences(combine) -> None:
  @sc.function(
    sc.group(sc.arg("base", 4), sc.arg("idx", 6), sc.arg("values", 6)),
    outputs=sc.group(sc.arg("y"), sc.arg("db"), sc.arg("dv"), sc.arg("vb"), sc.arg("vv")),
  )
  def indexed(inputs):
    base, idx, values = inputs
    y = sc.scatter(base, idx.cast(sc.dtypes.int64), values, combine=combine, mode="fill").sum()
    db = sc.jvp(y, base, sc.const([0.2, -0.3, 0.7, 0.9]))
    dv = sc.jvp(y, values, sc.const([0.4, 0.2, -0.8, 0.6, 0.7, -0.5]))
    vb, vv = sc.vjp((y,), (base, values), (sc.const(1.0),))
    return y, db, dv, vb, vv

  base = np.array([3.0, 5.0, -1.0, 7.0])
  idx = np.array([1.0, -1.0, 1.0, 4.0, 3.0, 1.0])
  values = np.array([2.0, 99.0, 9.0, 99.0, 6.0, 4.0])
  sb, sv = np.array([0.2, -0.3, 0.7, 0.9]), np.array([0.4, 0.2, -0.8, 0.6, 0.7, -0.5])
  _, db, dv, vb, vv = indexed((base, idx, values))
  for seed, forward, reverse, is_base in [(sb, db, vb, True), (sv, dv, vv, False)]:
    plus = indexed((base + 1e-5 * seed if is_base else base, idx, values if is_base else values + 1e-5 * seed))[0]
    minus = indexed((base - 1e-5 * seed if is_base else base, idx, values if is_base else values - 1e-5 * seed))[0]
    finite = (np.asarray(plus) - np.asarray(minus)) / 2e-5
    np.testing.assert_allclose(np.asarray(forward), finite, rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(np.dot(np.asarray(reverse), seed), finite, rtol=1e-7, atol=1e-8)


@pytest.mark.parametrize(("axis", "runtime"), [(0, True), (1, False), (-1, True)])
def test_axis_gather_and_scatter_match_numpy(axis, runtime) -> None:
  shape = (2, 3)
  selected_axis = axis % 2
  ids = np.array([[1, -1], [0, 1]], dtype=np.int64)
  result_shape = (*shape[:selected_axis], *ids.shape, *shape[selected_axis + 1 :])

  @sc.function(sc.group(sc.arg("x", shape), sc.arg("idx", ids.shape), sc.arg("v", result_shape)), outputs=sc.group(sc.arg("g"), sc.arg("s")))
  def indexed(inputs):
    x, idx, v = inputs
    index = idx.cast(sc.dtypes.int64) if runtime else ids
    return sc.gather(x, index, axis=axis, mode="fill", fill_value=-7), sc.scatter(x, index, v, axis=axis, mode="fill")

  x = np.arange(6, dtype=float).reshape(shape) + 1
  values = np.arange(np.prod(result_shape), dtype=float).reshape(result_shape) / 10
  mask_shape = (1,) * selected_axis + ids.shape + (1,) * (1 - selected_axis)
  expected_gather = np.where((ids >= 0).reshape(mask_shape), np.take(x, np.clip(ids, 0, shape[selected_axis] - 1), axis=selected_axis), -7)
  expected_scatter = x.copy()
  for coord in np.ndindex(result_shape):
    index = ids[coord[selected_axis : selected_axis + ids.ndim]]
    if 0 <= index < shape[selected_axis]:
      dest = (*coord[:selected_axis], index, *coord[selected_axis + ids.ndim :])
      expected_scatter[dest] += values[coord]
  actual_gather, actual_scatter = indexed((x, ids.astype(float), values))
  np.testing.assert_array_equal(actual_gather, expected_gather)
  np.testing.assert_allclose(np.asarray(actual_scatter), expected_scatter)


def test_index_builders_match_numpy() -> None:
  @sc.function(sc.arg("x", (2, 4)), sc.arg("start"), outputs=sc.group(*(sc.arg(name) for name in ("take", "slice", "add", "set", "sum", "max"))))
  def indexed(x, start):
    index = start.cast(sc.dtypes.int64)
    ids = sc.stack([index, index, index + sc.const(2, dtype=sc.dtypes.int64)])
    updates = sc.const([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    return (
      sc.take(x, ids),
      sc.dynamic_slice(x, index, 3),
      sc.index_add(x, ids, updates),
      sc.index_set(x, ids, updates),
      sc.segment_sum(updates[0], ids, 4),
      sc.segment_max(updates[0], ids, 4),
    )

  x = np.arange(8, dtype=float).reshape(2, 4)
  take, sliced, added, assigned, summed, maxima = indexed(x, np.array(1.0))
  np.testing.assert_array_equal(take, x[:, [1, 1, 3]])
  np.testing.assert_array_equal(sliced, x[:, 1:4])
  expected_add, expected_set = x.copy(), x.copy()
  for lane, column in enumerate([1, 1, 3]):
    expected_add[:, column] += np.array([[1, 2, 3], [4, 5, 6]])[:, lane]
    expected_set[:, column] = np.array([[1, 2, 3], [4, 5, 6]])[:, lane]
  np.testing.assert_array_equal(added, expected_add)
  np.testing.assert_array_equal(assigned, expected_set)
  np.testing.assert_array_equal(summed, [0, 3, 0, 3])
  np.testing.assert_array_equal(maxima, [-np.inf, 2, -np.inf, 3])


@pytest.mark.parametrize(("combine", "sign", "runtime"), [("max", 1, True), ("min", -1, False)])
def test_scatter_extrema_split_ties_with_base(combine, sign, runtime) -> None:
  @sc.function(sc.arg("base", 2), sc.arg("updates", 3), sc.arg("idx", 3), outputs=sc.group(sc.arg("db"), sc.arg("dv")))
  def indexed(base, updates, idx):
    indices = idx.cast(sc.dtypes.int64) if runtime else [0, 0, 1]
    y = sc.scatter(base, indices, updates, combine=combine, mode="fill").sum()
    return sc.vjp((y,), (base, updates), (sc.const(1.0),))

  db, dv = indexed(sign * np.array([2.0, 1.0]), sign * np.array([2.0, 2.0, 0.0]), np.array([0.0, 0.0, 1.0]))
  np.testing.assert_allclose(np.asarray(db), [1 / 3, 1])
  np.testing.assert_allclose(np.asarray(dv), [1 / 3, 1 / 3, 0])


@pytest.mark.parametrize(("size", "hint"), [(0, "scalar"), (4, "block")])
def test_runtime_index_extremes_and_empty_sources(size, hint) -> None:
  @sc.function(sc.arg("x", size), sc.arg("idx", 3), outputs=sc.group(sc.arg("g"), sc.arg("s")))
  def indexed(x, idx):
    indices = idx.cast(sc.dtypes.int64)
    read = sc.gather(x, indices, mode="fill", fill_value=-7)
    written = sc.scatter(x, indices, sc.const([1.0, 2.0, 3.0]), mode="fill")
    return (read.scalar(), written.scalar()) if hint == "scalar" else (read.block(), written.block())

  x = np.arange(size, dtype=float)
  read, written = indexed(x, np.array([-(2.0**62), 2.0**62, -1.0]))
  np.testing.assert_array_equal(read, [-7, -7, -7])
  np.testing.assert_array_equal(written, x)


def test_set_sparsity_removes_overwritten_base_and_losing_updates() -> None:
  base, updates = sc.sym("indexed_base", 3), sc.sym("indexed_updates", 4)
  result = sc.scatter(base, [1, 1, -1, 2], updates, combine="set", mode="fill")
  db, dv = sc.jacobian_sparsity(result, base), sc.jacobian_sparsity(result, updates)
  assert set(zip(db.rows, db.cols, strict=True)) == {(0, 0)}
  assert set(zip(dv.rows, dv.cols, strict=True)) == {(1, 1), (2, 3)}
  known = sc.scatter(sc.const([2.0, 3.0, 4.0]), [1, 1, -1, 2], sc.const([8.0, 0.0, 9.0, 0.0]), combine="set", mode="fill")
  assert known.support == frozenset({0})


@pytest.mark.parametrize(
  ("first", "second"),
  [
    ([0.0, 13.0], [-0.0, 13.0]),
    (
      np.array([0x7FF8000000000001, 0x3FF0000000000000], dtype=np.uint64).view(np.float64),
      np.array([0x7FF8000000000002, 0x3FF0000000000000], dtype=np.uint64).view(np.float64),
    ),
  ],
)
def test_constant_keys_compare_bytes_when_digests_collide(monkeypatch, first, second) -> None:
  from scaly.ir.expr import hashlib

  blake2b = hashlib.blake2b
  monkeypatch.setattr(hashlib, "blake2b", lambda data, **kwargs: blake2b(b"", **kwargs))
  a, b = sc.const(first), sc.const(second)
  assert a is not b
  assert a.structural_key() != b.structural_key()
  assert cse_many([a, b]) == (a, b)
  assert sc.const(np.array(first)) is a
  assert sc.const(np.array(second)) is b
  for node, expected in ((a, first), (b, second)):
    assert node.value is not None
    np.testing.assert_array_equal(node.value.view(np.uint64), np.asarray(expected).view(np.uint64))


def test_colliding_array_attributes_and_constants_evaluate_separately(monkeypatch) -> None:
  from scaly.ir.expr import hashlib

  blake2b = hashlib.blake2b
  monkeypatch.setattr(hashlib, "blake2b", lambda data, **kwargs: blake2b(b"", **kwargs))
  x = sc.sym("collision_x", 3)
  forward, backward = x.gather([0, 2]), x.gather([2, 0])
  assert forward is not backward
  assert forward.structural_key() != backward.structural_key()
  assert x.gather([0, 2]) is forward
  a, b = sc.const([13.0, 17.0]), sc.const([19.0, 23.0])
  outputs = cse_many([a, b, forward, backward])
  assert outputs == (a, b, forward, backward)
  fn = ConcreteFunction._from_exprs("constant_digest_collision", [x], outputs, ["x"], ["a", "b", "forward", "backward"])
  values = np.array([2.0, 3.0, 5.0])
  for actual, expected in zip(fn(values), ([13.0, 17.0], [19.0, 23.0], values[[0, 2]], values[[2, 0]]), strict=True):
    np.testing.assert_array_equal(actual, expected)


def test_array_keys_serialize_only_compacted_bytes(monkeypatch) -> None:
  from scaly.ir.expr import _array_key, _attrs_key, hashlib

  blake2b = hashlib.blake2b
  serialized = []

  def digest(data, **kwargs):
    assert len(data) == 16
    serialized.append(data)
    return blake2b(data, **kwargs)

  monkeypatch.setattr(hashlib, "blake2b", digest)
  tile = np.array([13.0, 17.0])
  large = np.broadcast_to(tile, (100_000, 100_000, 2))
  constant = sc.const(large)
  constant.structural_key()
  assert cse_many([constant]) == (constant,)
  assert len(serialized) == 1
  attributed = Expr(ExprOp.NEG, (sc.sym("compact_attrs", 2),), TensorType((2,)), attrs={"tile": large})
  attributed.structural_key()
  assert cse_many([attributed]) == (attributed,)
  assert len(serialized) == 2
  assert _attrs_key({"tile": large}) == (("tile", ("ndarray", _array_key(large))),)
  repeated = np.broadcast_to(tile, (4, 3, 2))
  assert _array_key(repeated) == _array_key(repeated.copy())
  assert serialized and all(data == tile.tobytes() for data in serialized)


def test_uniform_constants_keep_compact_immutable_storage_and_bitwise_keys() -> None:
  from scaly.ir.expr import _uniform, zeros_like

  shape = (100_000, 100_000)
  zero = zeros_like(sc.sym("uniform_large", shape, dtype=sc.dtypes.float32))
  assert zero.type.dtype == sc.dtypes.float32
  assert zero.value is not None and zero.value.dtype == np.float32 and zero.value.strides == (0, 0)
  assert not zero.value.flags.writeable and zero.support == frozenset()
  assert zero.structural_key() is zero.structural_key()
  assert sc.cse(zero) is zero
  assert _uniform(-0.0, (3,)) is sc.const(np.full(3, -0.0))
  assert _uniform(0.0, (3,)) is not _uniform(-0.0, (3,))
  tiled = np.broadcast_to([1.0, -2.0], (4, 3, 2))
  assert sc.const(tiled) is sc.const(tiled.copy())
  source = np.array([7.0], dtype="float64")
  constant = sc.const(np.broadcast_to(source, shape))
  source[0] = 9
  assert constant.value is not None and constant.value[99_999, 99_999] == 7


@pytest.mark.parametrize("dense_first", [False, True])
def test_uniform_constant_storage_does_not_depend_on_creation_order(dense_first) -> None:
  values = np.full(2048 + dense_first, 0.1)
  view = np.broadcast_to(values[:1], values.shape)
  first, second = (values, view) if dense_first else (view, values)
  constant = sc.const(first)
  assert constant is sc.const(second)
  assert constant.value is not None and constant.value.strides == (0,)
  assert constant.support is None
  assert sc.simplify(constant.sum()).value == 0.1 * values.size


@pytest.mark.parametrize("index", [(3, slice(None)), (slice(None), 0), (slice(None, None, 0), slice(None))])
def test_slice_rejects_invalid_indices_without_a_shape_allocation(index) -> None:
  shape = (3, 0)
  with pytest.raises((IndexError, ValueError)):
    sc.sym("invalid_slice", shape)[index]


def test_binary_nonlinear_method_helpers_eval() -> None:
  @sc.function(
    sc.group(sc.arg("x", 3), sc.arg("y", 3)), outputs=sc.group(sc.arg("atan", 3), sc.arg("min", 3), sc.arg("max", 3)), name="binary_helpers"
  )
  def f(xy: tuple[sc.Expr, sc.Expr]) -> tuple[sc.Expr, sc.Expr, sc.Expr]:
    x, y = xy
    return x.atan2(y), x.minimum(y), x.maximum(y)

  atan, mn, mx = as_concrete(f).outputs
  xv = np.array([0.5, -1.0, 2.0])
  yv = np.array([1.5, 2.0, -0.25])

  atan_v, mn_v, mx_v = f((xv, yv))
  np.testing.assert_allclose(atan_v, np.arctan2(xv, yv))
  np.testing.assert_allclose(mn_v, np.minimum(xv, yv))
  np.testing.assert_allclose(mx_v, np.maximum(xv, yv))
  assert atan.type.diff
  assert mn.type.diff
  assert mx.type.diff


def test_dot_sumsqr_and_norm_2() -> None:
  @sc.function(sc.arg("x", (2, 2)), outputs=sc.group(sc.arg("dot", ()), sc.arg("sumsqr", ()), sc.arg("norm", ())))
  def f(x: sc.Expr) -> tuple[sc.Expr, sc.Expr, sc.Expr]:
    return sc.dot(x, x.T), x.sumsqr(), sc.norm_2(x)

  xv = np.array([[1.0, 2.0], [3.0, 4.0]])

  dot_val, sumsqr_val, norm_val = f(xv)
  np.testing.assert_allclose(dot_val, np.dot(xv.reshape(-1), xv.T.reshape(-1)))
  np.testing.assert_allclose(sumsqr_val, np.sum(xv * xv))
  np.testing.assert_allclose(norm_val, np.linalg.norm(xv.reshape(-1)))

  try:
    _ = sc.dot(sc.sym("a", 2), sc.sym("b", 3))
  except ValueError as e:
    assert "dot size mismatch" in str(e)
  else:  # pragma: no cover
    raise AssertionError("dot size mismatch should fail")


def test_shape_checks_for_structural_ops() -> None:
  x = sc.sym("x", (2, 3))
  y = sc.sym("y", (4, 2))

  try:
    _ = x @ y
  except ValueError as e:
    assert "cannot matmul shapes (2, 3) and (4, 2)" in str(e)
  else:  # pragma: no cover
    raise AssertionError("invalid matmul should fail")

  try:
    _ = x.transpose((0, 0))
  except ValueError as e:
    assert "not a permutation" in str(e)
  else:  # pragma: no cover
    raise AssertionError("invalid transpose should fail")

  try:
    _ = sc.concat([x, sc.sym("z", (2, 4))], axis=0)
  except ValueError as e:
    assert "cannot concat shapes" in str(e)
  else:  # pragma: no cover
    raise AssertionError("invalid concat should fail")


def test_debug_printing_uses_stable_topological_names() -> None:
  x = sc.sym("x", 2)
  text = ((x + 1.0) * x).debug()

  assert "%0 = input x : float64(2,)" in text
  assert "add(%0, %1)" in text
  assert text.endswith("outputs %3")


def test_structural_equality_collapses_to_identity() -> None:
  # Construction-time interning: two ``Expr``s with the same structural key are the same
  # Python object, so structural equality is the same as ``is`` equality.
  x0 = sc.sym("x", 2)
  x1 = sc.sym("x", 2)
  y = sc.sym("y", 2)

  assert x0 is x1
  assert x0.id == x1.id
  assert x0.structurally_equal(x1)
  assert x0.structural_hash() == x1.structural_hash()
  assert x0 is not y
  assert not x0.structurally_equal(y)


def test_interning_hit_returns_the_node_as_it_was_built() -> None:
  """The key matches values that are equal and not identical, so a hit must not assign them:
  Functions already hold the node."""
  x = sc.sym("intern_x", 2)
  node = Expr(ExprOp.NEG, (x,), TensorType((2,)), attrs={"axis": 0, "scale": np.float64(1.5)})
  attrs, key = node.attrs, node.structural_key()

  again = Expr(ExprOp.NEG, (x,), TensorType((2,)), attrs={"axis": np.int64(0), "scale": 1.5})

  assert again is node and node.attrs is attrs and node.structural_key() is key
  assert type(attrs["axis"]) is int and type(attrs["scale"]) is np.float64


def test_constants_and_attributes_are_frozen_copies() -> None:
  given = np.array([1.0, 2.0])
  c = sc.const(given)
  given[0] = 5.0
  assert c.value is not None and c.value.tolist() == [1.0, 2.0] and not c.value.flags.writeable
  assert sc.const(np.array([1.0, 2.0])) is c and sc.const(given) is not c

  indices = np.array([1, 0])
  g = sc.sym("intern_x", 2).gather(indices)
  indices[0] = 0
  assert g.args[1].value is not None
  assert g.args[1].value.tolist() == [1, 0] and not g.args[1].value.flags.writeable
  with pytest.raises(TypeError):
    g.attrs["indices"] = indices  # ty: ignore[invalid-assignment]


@pytest.mark.parametrize(
  ("first", "second"),
  [(0.0, -0.0), (1, True), (1, 1.0), (1.0, True), ((0.0, 1), (-0.0, 1)), ({"a": 1}, {"a": True})],
)
def test_interning_tells_apart_attributes_that_differ_in_bits_or_kind(first, second) -> None:
  """``==`` holds between each pair, and generated code can tell them apart."""
  assert first == second
  x = sc.sym("intern_x", 2)
  a = Expr(ExprOp.NEG, (x,), TensorType((2,)), attrs={"v": first})
  b = Expr(ExprOp.NEG, (x,), TensorType((2,)), attrs={"v": second})

  assert a is not b and not a.structurally_equal(b)
  assert repr(a.attrs["v"]) == repr(frozen(first)) and repr(b.attrs["v"]) == repr(frozen(second))
  assert len({n.id for n in cse_many([a, b])}) == 2
  assert Expr(ExprOp.NEG, (x,), TensorType((2,)), attrs={"v": first}) is a


def test_interning_shares_attributes_with_the_same_bits() -> None:
  x = sc.sym("intern_x", 2)

  def node(v):
    return Expr(ExprOp.NEG, (x,), TensorType((2,)), attrs={"v": v})

  nan = node(float("nan"))
  assert node(float("nan")) is nan
  assert node(-float("nan")) is not nan
  assert node(np.float64(1.5)) is node(1.5) and node(np.float32(1.5)) is node(1.5)
  assert node(np.int64(3)) is node(3) and node(np.True_) is node(True)


def test_differentiability_metadata_propagates_through_exprs() -> None:
  x = sc.sym("x", 3)
  p = sc.sym("p", 3, diff=False)
  c = sc.const([1.0, 2.0, 3.0])

  assert x.type.diff
  assert not p.type.diff
  assert not c.type.diff
  assert (x + c).type.diff
  assert not (p + c).type.diff
  assert x.reshape((3, 1)).T.type.diff
  assert x.gather([2, 0]).type.diff
  assert _scatter_zero(p.gather([1, 2]), [0, 2], 3).type.diff is False
  assert sc.stack([p, c]).type.diff is False
  assert sc.concat([x[:1], p[:1]]).type.diff
  assert x.floor().type.diff
  assert sc.minimum(x, p).type.diff

  @sc.function(sc.arg("u", 3), outputs=sc.arg("y"))
  def inner(u: sc.Expr) -> sc.Expr:
    return u * u

  diff_call = inner(x)
  const_call = inner(c)
  assert diff_call.type.diff
  assert not const_call.type.diff


def test_substitute_rebuilds_changed_ancestors_and_preserves_sharing() -> None:
  x = sc.sym("sub_x", 3)
  z = sc.sym("sub_z", 3)
  expr = (x * x + x).scalar()

  actual = substitute(expr, {x: z})
  expected = (z * z + z).scalar()

  assert actual is expected
  assert actual.lowering == expr.lowering
  assert substitute(expr, {}) is expr


def test_substitute_rejects_incompatible_shape_or_dtype() -> None:
  x = sc.sym("sub_bad_x", 2)
  with pytest.raises(ValueError, match="cannot substitute"):
    substitute(x + 1.0, {x: sc.sym("sub_bad_z", 3)})


def test_substitute_rebuilds_call_and_vmap_actuals_without_entering_callees() -> None:
  @sc.function(sc.arg("u", 2), outputs=sc.arg("y"), name="sub_callee")
  def callee(formal: sc.Expr) -> sc.Expr:
    return formal * formal

  x = sc.sym("sub_actual_x", 2)
  z = sc.sym("sub_actual_z", 2)
  called = callee(x)
  rewritten_call = substitute(called, {x: z})
  assert rewritten_call is callee(z)
  assert rewritten_call.attrs["callee"] is callee.instantiate()

  xs = sc.sym("sub_vmap_x", 4)
  zs = sc.sym("sub_vmap_z", 4)
  mapped = _mapped_call(callee, 2, [(xs, 0, 2)])
  rewritten_vmap = substitute(mapped, {xs: zs})
  assert rewritten_vmap is _mapped_call(callee, 2, [(zs, 0, 2)])
  assert rewritten_vmap.attrs["callee"] is callee.instantiate()


@pytest.mark.parametrize(
  ("fmt", "values", "error", "message"),
  [
    ("x", (), ValueError, "at least one value"),
    ("x={} y={}", (sc.sym("x"),), ValueError, "2 placeholders for 1 value"),
    ("x={0}", (sc.sym("x"),), ValueError, "only empty '{}' placeholders"),
    ("x={:.3f}", (sc.sym("x"),), ValueError, "only empty '{}' placeholders"),
    ("x=\0{}", (sc.sym("x"),), ValueError, "NUL"),
  ],
)
def test_print_rejects_a_format_it_cannot_render(fmt, values, error, message) -> None:
  with pytest.raises(error, match=message):
    sc.print(fmt, *values)


def test_reduction_axes_are_normalized_and_interned() -> None:
  x = sc.sym("axis_x", (2, 3, 4))
  assert x.sum((2, 0)) is x.sum((-1, -3)) is x.sum((0, 2))
  assert x.sum(axis=None).shape == ()
  assert x.sum(axis=()).shape == x.shape
  assert x.sum(axis=-1, keepdims=True).shape == (2, 3, 1)
  assert x.mean(axis=(0, 2), keepdims=True).shape == (1, 3, 1)
  for axis in ((0, -3), (3,), (-4,)):
    with pytest.raises(ValueError, match="axis"):
      x.sum(axis)
  with pytest.raises(TypeError, match="integers"):
    x.sum((0.5,))  # ty: ignore[invalid-argument-type]
  for shape in ((), (2, 3, 4)):
    with pytest.raises(ValueError, match="vector or a matrix"):
      sc.diag(sc.sym("bad_diag", shape))
  with pytest.raises(TypeError, match="offset"):
    sc.diag(x, k=0.5)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("axis", [True, [0]])
def test_reduction_rejects_boolean_and_list_axes(axis) -> None:
  with pytest.raises(TypeError, match="integer"):
    sc.sym("invalid_axis", (2, 3)).sum(axis)


@pytest.mark.parametrize("method", ["sum", "mean"])
def test_reduction_keepdims_is_keyword_only(method) -> None:
  with pytest.raises(TypeError):
    getattr(sc.sym("keepdims_keyword", (2, 3)), method)(1, True)


def test_sparse_batch_error():
  from scaly.ir.types import CanonicalPattern, TensorType

  pattern = CanonicalPattern((2, 3), (0, 1), (1, 2))
  x = sc.Expr(sc.ExprOp.INPUT, type=TensorType((4, 2, 3), pattern=pattern), name="x")
  with pytest.raises(NotImplementedError, match="#403"):
    x @ sc.sym("y", (3, 2))


@pytest.mark.parametrize("shapes", [((), (2, 3)), ((2, 3), ()), ((2, 3), (4, 2)), ((2, 2, 3), (3, 3, 2))])
def test_invalid_shapes(shapes):
  with pytest.raises(ValueError):
    sc.matmul(sc.sym("a", shapes[0]), sc.sym("b", shapes[1]))
