"""Sparse leaf layouts, copied numerical inputs and stored mapped windows."""

from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from scipy import sparse

import scaly as sc
from scaly.function.sugar import _view_window
from scaly.ir.layout import SparseLayout
from scaly.ir.types import CanonicalPattern, SparsityPattern


PATTERN = CanonicalPattern((3, 4), (0, 0, 1, 2), (0, 3, 1, 0))
COORDINATE_LAYOUT = SparsityPattern(PATTERN.shape, (0, 1, 2, 0), (3, 1, 0, 0))
LAYOUTS = ("csc", "csr", COORDINATE_LAYOUT)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_sparse_leaf_roundtrip_checks_copies_and_shares_indices(layout) -> None:
  leaf = sc.arg("x", sc.TensorType(PATTERN.shape, pattern=PATTERN), layout=layout)
  identity = sc.function(leaf, outputs=leaf.relabel("out_"), name="sparse_identity")(lambda x: x)
  source = sparse.coo_array(([3.0, 2.0, 0.0, 5.0, -1.0], ([2, 0, 1, 0, 2], [0, 0, 1, 3, 0])), shape=PATTERN.shape)
  original = (source.data.copy(), source.row.copy(), source.col.copy())
  first, second = identity(source), identity(source)
  assert isinstance(first, sparse.csc_array | sparse.csr_array | sparse.coo_array)
  assert isinstance(second, sparse.csc_array | sparse.csr_array | sparse.coo_array)
  assert first.nnz == PATTERN.nnz
  np.testing.assert_array_equal(first.toarray(), source.toarray())
  for before, after in zip(original, (source.data, source.row, source.col), strict=True):
    np.testing.assert_array_equal(before, after)
  if isinstance(first, sparse.coo_array):
    assert isinstance(second, sparse.coo_array)
    assert first.row is second.row and first.col is second.col
    assert not first.row.flags.writeable and not first.col.flags.writeable
  else:
    assert isinstance(second, sparse.csc_array | sparse.csr_array)
    assert first.indices is second.indices and first.indptr is second.indptr
    assert not first.indices.flags.writeable and not first.indptr.flags.writeable
  with pytest.raises(ValueError, match="pattern"):
    identity(sparse.coo_array(np.eye(3, 4)))
  with pytest.raises(ValueError, match="pattern"):
    identity.symbolic_call(sc.sym("dense", PATTERN.shape))
  other = CanonicalPattern(PATTERN.shape, (0, 1, 2, 2), (0, 1, 0, 3))
  with pytest.raises(ValueError, match="pattern"):
    identity.symbolic_call(sc.arg("other", replace(leaf.types[0], pattern=other)).symbols())


def test_sparse_leaf_type_and_layout_survive_reconstruction() -> None:
  leaf = sc.arg("x", sc.TensorType(PATTERN.shape, pattern=PATTERN), layout="csr")
  assert leaf.symbols(diff=False).type.pattern is PATTERN
  for tree in (leaf.relabel("p_"), leaf.with_types(leaf.types)):
    assert tree.layouts[0] is leaf.layouts[0]
    assert tree.types[0].pattern is PATTERN


def test_sparse_layout_permutations_and_numerical_copies() -> None:
  canonical = np.array([2.0, 5.0, 0.0, 2.0])
  source = sparse.coo_array((canonical, (PATTERN.rows, PATTERN.cols)), shape=PATTERN.shape)
  for requested in LAYOUTS:
    leaf = sc.arg("x", sc.TensorType(PATTERN.shape, pattern=PATTERN), layout=requested)
    layout = leaf.layouts[0]
    assert isinstance(layout, SparseLayout)
    compact = layout.numerical_values(source, PATTERN.shape, np.dtype(np.float64))
    np.testing.assert_array_equal(compact, canonical[layout.permutation])
    np.testing.assert_array_equal(compact[layout.inverse_permutation], canonical)
    restored = layout.sparse_array(compact, PATTERN.shape)
    assert isinstance(restored, sparse.coo_array | sparse.csc_array | sparse.csr_array)
    np.testing.assert_array_equal(restored.toarray(), source.toarray())
    assert sc.arg("same", leaf.types[0], layout=requested).layouts[0] is layout
    assert not layout.permutation.flags.writeable
    for indices in (layout.permutation, layout.inverse_permutation, layout.rows, layout.cols, layout.indices, layout.indptr):
      with pytest.raises(ValueError):
        indices.flags.writeable = True


@pytest.mark.parametrize("layout", ["csc", "csr"])
def test_empty_sparse_patterns_and_stored_values_fast_path(layout: str) -> None:
  for pattern in (PATTERN, CanonicalPattern((3, 4), (), ())):
    leaf = sc.arg("x", sc.TensorType(pattern.shape, pattern=pattern), layout=layout)
    identity = sc.function(leaf, outputs=leaf.relabel("out_"), name="sparse_values_identity")(lambda x: x)
    data = np.arange(pattern.nnz, dtype=float)
    result = identity(data)
    assert isinstance(result, sparse.csc_array | sparse.csr_array)
    assert result.nnz == pattern.nnz
    np.testing.assert_array_equal(result.data, data)
    with pytest.raises(ValueError, match="sparse|stored"):
      identity(np.zeros(pattern.shape))


def test_layout_is_part_of_instance_key_and_nested_calls_convert_it() -> None:
  type_ = sc.TensorType(PATTERN.shape, pattern=PATTERN)
  a = sc.arg("x", type_, layout="csc")
  b = sc.arg("y", type_, layout=COORDINATE_LAYOUT)
  inner = sc.function(a, outputs=b, name="sparse_reorder_inner")(lambda x: -x)
  outer = sc.function(sc.arg("z", type_, layout="csr"), outputs=a.relabel("out_"), name="sparse_reorder_outer")(lambda x: inner(x).sin())
  key = next(iter(inner.instances))
  assert a.layouts[0] in key[1] and b.layouts[0] in key[1]
  source = sparse.coo_array(([2.0, 5.0, 0.0, -3.0], (PATTERN.rows, PATTERN.cols)), shape=PATTERN.shape)
  result = outer(source)
  assert isinstance(result, sparse.csc_array)
  np.testing.assert_allclose(result.toarray(), np.sin(-source.toarray()))


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("length", [0, 1, 4])
def test_sparse_maps_use_stored_widths_and_return_batched_sparse_arrays(length: int, layout) -> None:
  leaf = sc.arg("x", sc.TensorType(PATTERN.shape, pattern=PATTERN), layout=layout)
  stage = sc.function(leaf, name="sparse_stage")(lambda x: -x)
  mapped = sc.vmap(stage, length)
  data = np.arange(length * PATTERN.nnz, dtype=float).reshape(length, PATTERN.nnz)
  rows = np.tile(PATTERN.rows, length)
  cols = np.tile(PATTERN.cols, length)
  batches = np.repeat(np.arange(length), PATTERN.nnz)
  source = sparse.coo_array((data.reshape(-1), (batches, rows, cols)), shape=(length, *PATTERN.shape))
  result = mapped(source)
  assert isinstance(result, sparse.sparray)
  assert result.shape == source.shape and result.nnz == length * PATTERN.nnz
  np.testing.assert_array_equal(result.toarray(), -source.toarray())
  again = mapped(source)
  assert isinstance(result, sparse.coo_array) and isinstance(again, sparse.coo_array)
  assert all(a is b for a, b in zip(result.coords, again.coords, strict=True))


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("length", [0, 1, 3])
def test_sparse_map_broadcast_and_raw_windows_use_declared_layout(layout, length: int) -> None:
  type_ = sc.TensorType(PATTERN.shape, pattern=PATTERN)
  leaf = sc.arg("x", type_, layout=layout)
  stage = sc.function(leaf, name="sparse_windows_stage")(lambda x: x.sin())
  mapped = sc.vmap(stage, length)
  source = sparse.coo_array(([2.0, 5.0, 0.0, -3.0], (PATTERN.rows, PATTERN.cols)), shape=PATTERN.shape)
  broadcast = mapped(sc.broadcast(source))
  assert isinstance(broadcast, sparse.coo_array)
  np.testing.assert_allclose(broadcast.toarray(), np.broadcast_to(np.sin(source.toarray()), (length, *PATTERN.shape)))
  data = np.arange(1 + length * 6, dtype=float)
  symbol = sc.sym("raw", data.size)
  result = mapped(sc.window(symbol, 1, 6))
  assert result.type.pattern is PATTERN
  wrapper = sc.function(sc.arg("raw", data.size), name="sparse_windows_outer")(lambda raw: mapped(sc.window(raw, 1, 6)))
  compiled = wrapper(data)
  direct = mapped(sc.window(data, 1, 6))
  assert isinstance(compiled, sparse.coo_array) and isinstance(direct, sparse.coo_array)
  expected = np.zeros((length, *PATTERN.shape))
  declared = leaf.layouts[0]
  assert isinstance(declared, SparseLayout)
  for i in range(length):
    compact = data[1 + 6 * i : 1 + 6 * i + PATTERN.nnz]
    expected[i, declared.rows, declared.cols] = np.sin(compact)
    np.testing.assert_allclose(stage(compact).toarray(), expected[i])
  np.testing.assert_allclose(compiled.toarray(), expected)
  np.testing.assert_allclose(direct.toarray(), expected)
  other = CanonicalPattern(PATTERN.shape, (0, 1, 2, 2), (0, 1, 0, 3))
  wrong = sc.arg("wrong", replace(type_, shape=(length, *PATTERN.shape), pattern=other)).symbols()
  with pytest.raises(ValueError, match="pattern"):
    mapped.symbolic_call(wrong)


@pytest.mark.parametrize("layout", ["csc", "csr", "coordinate"])
@pytest.mark.parametrize("batch", [(), (2, 3), (0,)])
@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("length", [0, 1, 3])
def test_sparse_window_layout_matches_direct_values_for_batches_and_empty_patterns(layout, batch, empty: bool, length: int) -> None:
  pattern = CanonicalPattern((3, 4), (), ()) if empty else CanonicalPattern((3, 4), (0, 0, 1, 2, 2), (0, 3, 1, 0, 2))
  order = SparsityPattern(pattern.shape, (2, 0, 2, 0, 1), (2, 0, 0, 3, 1)) if not empty else SparsityPattern(pattern.shape, (), ())
  leaf = sc.arg("x", sc.TensorType((*batch, *pattern.shape), pattern=pattern), layout=order if layout == "coordinate" else layout)
  stage = sc.function(leaf, name="sparse_batched_window_stage")(lambda x: x.sin())
  mapped = sc.vmap(stage, length)
  declared = leaf.layouts[0]
  assert isinstance(declared, SparseLayout)
  width = leaf.types[0].storage_size
  stride = width + 2
  data = np.arange(1 + length * stride, dtype=float) / 7
  raw = data[::-1]
  compact = np.asarray([raw[1 + i * stride : 1 + i * stride + width] for i in range(length)]).reshape((length, *leaf.types[0].storage_shape))
  expected = np.zeros((length, *batch, *pattern.shape))
  expected[..., declared.rows, declared.cols] = np.sin(compact)
  wrapper = sc.function(sc.arg("raw", data.size), name="sparse_batched_window_outer")(lambda raw: mapped(sc.window(raw[::-1], 1, stride)))
  for result in (mapped(compact), mapped(sc.window(raw, 1, stride)), wrapper(data)):
    assert isinstance(result, sparse.coo_array)
    assert result.nnz == length * width
    np.testing.assert_allclose(result.toarray(), expected)
  for i in range(length):
    np.testing.assert_allclose(stage(compact[i]).toarray(), expected[i])


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("stride", [0, 1, 6])
def test_sparse_window_layout_routes_overlapping_values_and_derivatives(layout, stride: int) -> None:
  leaf = sc.arg("x", sc.TensorType(PATTERN.shape, pattern=PATTERN), layout=layout)
  weights = np.arange(1.0, 13.0).reshape(PATTERN.shape)
  stage = sc.function(leaf, name="sparse_window_weighted_stage")(lambda x: (sc.to_dense(x.sin()) * weights).sum())
  mapped = sc.vmap(stage, 3)
  size = 1 + 2 * stride + PATTERN.nnz
  data = np.arange(size, dtype=float) / 7
  wrapper = sc.function(sc.arg("raw", size), name="sparse_window_weighted_outer")(lambda raw: mapped(sc.window(raw, 1, stride)))
  declared = leaf.layouts[0]
  assert isinstance(declared, SparseLayout)
  indices = 1 + np.arange(3)[:, None] * stride + np.arange(PATTERN.nnz)
  factors = weights[declared.rows, declared.cols]
  expected = (np.sin(data[indices]) * factors).sum(axis=1)
  for result in (mapped(data[indices]), mapped(sc.window(data, 1, stride)), wrapper(data)):
    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, expected)
  jac = np.zeros((3, size))
  jac[np.arange(3)[:, None], indices] = np.cos(data[indices]) * factors
  np.testing.assert_allclose(sc.jacobian(wrapper)(data), jac)
  objective = sc.function(sc.arg("raw", size), name="sparse_window_weighted_sum")(lambda raw: wrapper(raw).sum())
  np.testing.assert_allclose(sc.gradient(objective)(data), jac.sum(axis=0))


@pytest.mark.parametrize("layout", LAYOUTS)
def test_nested_sparse_maps_keep_shared_patterns_and_batch_offsets(layout) -> None:
  type_ = sc.TensorType(PATTERN.shape, pattern=PATTERN)
  stage = sc.function(sc.arg("x", type_, layout=layout), name="sparse_nested_stage")(lambda x: -x)
  mapped = sc.vmap(sc.vmap(stage, 3), 2)
  data = np.arange(24.0)
  batches = np.unravel_index(np.arange(6), (2, 3))
  source = sparse.coo_array(
    (data, (np.repeat(batches[0], PATTERN.nnz), np.repeat(batches[1], PATTERN.nnz), np.tile(PATTERN.rows, 6), np.tile(PATTERN.cols, 6))),
    shape=(2, 3, *PATTERN.shape),
  )
  result = mapped(source)
  assert isinstance(result, sparse.coo_array)
  np.testing.assert_array_equal(result.toarray(), -source.toarray())


@pytest.mark.parametrize("outputs", [None, sc.arg("out"), sc.arg("out", None)], ids=["inferred", "hole", "none_hole"])
def test_sparse_default_csc_layout_handles_a_non_involutive_permutation(outputs) -> None:
  pattern = CanonicalPattern((3, 4), (0, 0, 1, 2, 2), (0, 3, 1, 0, 2))
  leaf = sc.arg("x", sc.TensorType(pattern.shape, pattern=pattern))
  layout = leaf.layouts[0]
  assert isinstance(layout, SparseLayout) and not np.array_equal(layout.permutation, layout.inverse_permutation)
  decorator = sc.function(leaf, name="sparse_cycle") if outputs is None else sc.function(leaf, outputs=outputs, name="sparse_cycle")
  stage = decorator(lambda x: x.sin())
  source = sparse.coo_array(([2.0, 3.0, 5.0, 7.0, 11.0], (pattern.rows, pattern.cols)), shape=pattern.shape)
  direct = stage(source)
  assert isinstance(direct, sparse.csc_array)
  np.testing.assert_allclose(direct.toarray(), np.sin(source.toarray()))
  concrete = stage.instantiate()(source)
  assert isinstance(concrete, sparse.csc_array)
  np.testing.assert_array_equal(concrete.toarray(), direct.toarray())
  mapped = sc.vmap(stage, 2)
  result = mapped(sc.broadcast(source))
  assert isinstance(result, sparse.coo_array)
  np.testing.assert_allclose(result.toarray(), np.broadcast_to(np.sin(source.toarray()), (2, *pattern.shape)))


@pytest.mark.parametrize("shape,index", [((3,), 1), ((), ())])
def test_scalar_mapped_views_keep_the_selected_value(shape, index) -> None:
  stage = sc.function(sc.arg("x", ()), name="scalar_view_stage")(lambda x: x * x)
  mapped = sc.vmap(stage, 3)
  wrapper = sc.function(sc.arg("z", shape), name="scalar_view_outer")(lambda z: mapped(sc.broadcast(z[index])))
  source = np.arange(3.0) + 2.0 if shape else np.array(2.0)
  np.testing.assert_array_equal(wrapper(source), np.repeat(source[index] ** 2, 3))


def test_view_windows_enumerate_the_selected_region(monkeypatch: pytest.MonkeyPatch) -> None:
  base = sc.sym("base", 10_000_000_000)
  selected = base[100_000_000:100_000_004].reshape((2, 2))
  arange = np.arange

  def bounded_arange(stop, *args, **kwargs):
    assert stop <= 4
    return arange(stop, *args, **kwargs)

  monkeypatch.setattr(np, "arange", bounded_arange)
  outer, start, stride = _view_window(selected, 2, 2)
  assert outer is base and start == 100_000_000 and stride == 2


@pytest.mark.parametrize("layout", ["csr", COORDINATE_LAYOUT])
def test_sparse_seeded_functions_and_factory_preserve_source_layouts(layout) -> None:
  type_ = sc.TensorType(PATTERN.shape, pattern=PATTERN)
  inp, out = sc.arg("x", type_, layout=layout), sc.arg("y", type_, layout=layout)
  stage = sc.function(inp, outputs=out, name="sparse_seeded_stage")(lambda x: x.sin())
  assert stage.inputs is not None
  forward: Any = sc.forward(stage)
  adjoint: Any = sc.adjoint(stage)
  assert forward.inputs.layouts == (*stage.inputs.layouts, inp.layouts[0])
  assert adjoint.inputs.layouts == (*stage.inputs.layouts, out.layouts[0])
  source = sparse.coo_array(([0.2, -0.5, 1.2, 0.7], (PATTERN.rows, PATTERN.cols)), shape=PATTERN.shape)
  direction = sparse.coo_array(([1.0, -2.0, 0.0, 3.0], (PATTERN.rows, PATTERN.cols)), shape=PATTERN.shape)
  declared = inp.layouts[0]
  assert isinstance(declared, SparseLayout)
  x = declared.numerical_values(source, PATTERN.shape, np.dtype(np.float64))
  seed = declared.numerical_values(direction, PATTERN.shape, np.dtype(np.float64))
  expected = np.cos(source.toarray()) * direction.toarray()
  for derivative in (forward, adjoint):
    result = derivative(x, seed)
    assert isinstance(result, sparse.csc_array | sparse.csr_array | sparse.coo_array)
    np.testing.assert_allclose(result.toarray(), expected)
  factory = stage.factory("sparse_seeded_factory", ["x", "fwd:x", "lam:y"], ["y", sc.factory.Fwd("y", "x"), sc.factory.Adj("y", "x")])
  assert factory.inputs is not None and factory.outputs is not None
  assert factory.inputs.layouts == (declared,) * 3
  assert factory.outputs.layouts[0] is out.layouts[0]
  original, tangent, pulled = factory(x, seed, seed)
  np.testing.assert_allclose(original.toarray(), np.sin(source.toarray()))
  np.testing.assert_allclose(tangent.toarray(), expected)
  np.testing.assert_allclose(pulled.toarray(), expected)


@pytest.mark.parametrize("layout", ["csr", COORDINATE_LAYOUT])
def test_sparse_factory_with_shape_holes_preserves_fixed_leaf_layouts(layout) -> None:
  type_ = sc.TensorType(PATTERN.shape, pattern=PATTERN)
  inp, out = sc.arg("x", type_, layout=layout), sc.arg("y", type_, layout=layout)
  source = sc.function(inp, sc.arg("scale"), outputs=out, name="sparse_factory_template")(lambda x, scale: x * scale.sum())
  factory = source.factory("sparse_factory_bound", ["x", "scale", "fwd:x"], ["y", sc.factory.Fwd("y", "x")])
  assert factory.inputs is not None
  assert factory.inputs.layouts == (inp.layouts[0], None, inp.layouts[0])
  declared = inp.layouts[0]
  assert isinstance(declared, SparseLayout)
  canonical = np.array([0.2, -0.5, 1.2, 0.7])
  seed = np.array([1.0, -2.0, 0.0, 3.0])
  original, tangent = factory(canonical[declared.permutation], np.array([2.0, 3.0]), seed[declared.permutation])
  np.testing.assert_allclose(original.toarray()[PATTERN.rows, PATTERN.cols], canonical * 5)
  np.testing.assert_allclose(tangent.toarray()[PATTERN.rows, PATTERN.cols], seed * 5)


@pytest.mark.parametrize("layout", ["csr", COORDINATE_LAYOUT])
def test_sparse_weighted_hessians_preserve_dual_layouts(layout) -> None:
  type_ = sc.TensorType(PATTERN.shape, pattern=PATTERN)
  inp, out = sc.arg("x", type_, layout=layout), sc.arg("y", type_, layout=layout)
  stage = sc.function(inp, outputs=out, name="sparse_weighted_stage")(lambda x: x * x)
  dense: Any = sc.lagrangian_hessian(stage)
  compact: Any = sc.sparse_lagrangian_hessian(stage)
  assert dense.inputs.layouts[-1] is out.layouts[0]
  assert compact.inputs.layouts[-1] is out.layouts[0]
  declared = inp.layouts[0]
  assert isinstance(declared, SparseLayout)
  primal = np.array([0.2, -0.5, 1.2, 0.7])[declared.permutation]
  multipliers = np.array([1.0, -2.0, 0.0, 3.0])
  weights = multipliers[declared.permutation]
  logical = np.ravel_multi_index((PATTERN.rows, PATTERN.cols), PATTERN.shape)
  expected = np.zeros((12, 12))
  expected[logical, logical] = 2 * multipliers
  np.testing.assert_allclose(dense(primal, weights), expected)
  np.testing.assert_allclose(compact(primal, weights).toarray(), expected)


@pytest.mark.parametrize("layout", ["csr", COORDINATE_LAYOUT])
def test_sparse_custom_rules_preserve_boundaries_and_reject_different_patterns(layout) -> None:
  type_ = sc.TensorType(PATTERN.shape, pattern=PATTERN)
  x, dx = sc.arg("x", type_, layout=layout), sc.arg("dx", type_, layout=layout)
  out = sc.arg("y", type_, layout=layout)
  source = sc.function(x, outputs=out, name="sparse_custom_source")(lambda x: x * x)
  jvp = sc.function(x, dx, outputs=out, name="sparse_custom_jvp")(lambda x, dx: 3 * x * dx)
  fwd = sc.function(x, outputs=sc.group(out, x.relabel("res_")), name="sparse_custom_fwd")(lambda x: (x * x, x))
  bwd = sc.function(x.relabel("res_"), out.relabel("lam_"), outputs=x, name="sparse_custom_bwd")(lambda res, cot: 3 * res * cot)
  ruled = sc.custom_derivative(source, jvp=jvp, fwd=fwd, bwd=bwd)
  values = np.array([0.2, -0.5, 1.2, 0.7])
  seed_values = np.array([1.0, -2.0, 0.0, 3.0])
  primal = sparse.coo_array((values, (PATTERN.rows, PATTERN.cols)), shape=PATTERN.shape)
  seed = sparse.coo_array((seed_values, (PATTERN.rows, PATTERN.cols)), shape=PATTERN.shape)
  declared = x.layouts[0]
  assert isinstance(declared, SparseLayout)
  for derivative in (sc.forward(ruled), sc.adjoint(ruled)):
    call: Any = derivative
    result = call(
      declared.numerical_values(primal, PATTERN.shape, np.dtype(np.float64)), declared.numerical_values(seed, PATTERN.shape, np.dtype(np.float64))
    )
    assert isinstance(result, sparse.csc_array | sparse.csr_array | sparse.coo_array)
    assert result.format == declared.format
    np.testing.assert_allclose(result.toarray(), 3 * primal.toarray() * seed.toarray())
  other = CanonicalPattern(PATTERN.shape, (0, 1, 2, 2), (0, 1, 0, 3))
  wrong_type = replace(type_, pattern=other)
  wrong = sc.function(x, dx, outputs=sc.arg("bad", wrong_type), name="sparse_custom_wrong")(lambda x, dx: sc.pack(sc.values(dx), other))
  with pytest.raises(TypeError, match="rule returns types"):
    sc.custom_derivative(source, jvp=wrong)


@pytest.mark.parametrize("input_order,output_order", [("csr", "csc"), (COORDINATE_LAYOUT, "csr"), ("csc", COORDINATE_LAYOUT)])
def test_sparse_derived_results_inherit_the_primal_leaf_layout(input_order, output_order):
  type_ = sc.TensorType(PATTERN.shape, pattern=PATTERN)
  inp = sc.arg("x", type_, layout=input_order)
  out = sc.arg("y", type_, layout=output_order)
  stage = sc.function(inp, outputs=out, name="derived_sparse_layout")(lambda x: x.sin())
  objective = sc.function(inp, name="derived_sparse_objective")(lambda x: (x * x).sum())
  source = sparse.coo_array(([0.2, -0.5, 1.2, 0.7], (PATTERN.rows, PATTERN.cols)), shape=PATTERN.shape)
  seed = sparse.coo_array(([1.0, -2.0, 0.0, 3.0], (PATTERN.rows, PATTERN.cols)), shape=PATTERN.shape)
  expected = np.cos(source.toarray()) * seed.toarray()
  cases = (
    (sc.forward(stage), out.layouts[0], (source, seed), expected),
    (sc.adjoint(stage), inp.layouts[0], (source, seed), expected),
    (sc.gradient(objective), inp.layouts[0], (source,), 2 * source.toarray()),
  )
  for derivative, layout, args, reference in cases:
    assert isinstance(layout, SparseLayout)
    assert derivative.instantiate().output_tree.layouts[0] is layout
    result = derivative(*args)
    assert isinstance(result, sparse.csc_array | sparse.csr_array | sparse.coo_array)
    np.testing.assert_allclose(result.toarray(), reference)
    np.testing.assert_allclose(result.data, reference[layout.rows, layout.cols])
    concrete = derivative.instantiate()
    concrete_result = concrete.numerical_call(*args)
    assert isinstance(concrete_result, sparse.csc_array | sparse.csr_array | sparse.coo_array)
    np.testing.assert_allclose(concrete_result.toarray(), reference)
    mapped = sc.vmap(concrete, 2)(*(sc.broadcast(value) for value in args))
    assert isinstance(mapped, sparse.coo_array)
    np.testing.assert_allclose(mapped.toarray(), np.stack([reference, reference]))


def test_dense_input_with_packed_output_keeps_sparse_forward_and_dense_adjoint():
  @sc.function(sc.arg("x", PATTERN.nnz), outputs=sc.arg("out"))
  def stage(x: sc.Expr) -> sc.Expr:
    return sc.pack(x.sin(), PATTERN)

  values = np.array([0.2, -0.5, 1.2, 0.7])
  direction = np.array([1.0, -2.0, 0.0, 3.0])
  weights = sparse.csc_array((direction, (PATTERN.rows, PATTERN.cols)), shape=PATTERN.shape)
  expected_values = np.cos(values) * direction
  expected = sparse.csc_array((expected_values, (PATTERN.rows, PATTERN.cols)), shape=PATTERN.shape).toarray()
  fwd, adj = sc.forward(stage.instantiate()), sc.adjoint(stage.instantiate())
  assert fwd.instantiate().output_tree.layouts[0] is stage.instantiate().output_tree.layouts[0]
  assert adj.instantiate().output_tree.layouts == (None,)
  result = fwd(values, direction)
  assert isinstance(result, sparse.csc_array)
  np.testing.assert_allclose(result.toarray(), expected)
  np.testing.assert_allclose(adj(values, weights), expected_values)
  mapped = sc.vmap(fwd, 2)(sc.broadcast(values), sc.broadcast(direction))
  assert isinstance(mapped, sparse.coo_array)
  np.testing.assert_allclose(mapped.toarray(), np.stack([expected, expected]))


def test_sparse_ad_helper_boundaries_use_canonical_order():
  from scaly.ir.expr import topo

  type_ = sc.TensorType(PATTERN.shape, pattern=PATTERN)
  stage = sc.function(sc.arg("x", type_, layout="csr"), name="canonical_helper_stage")(lambda x: x.sin())
  mapped = sc.vmap(stage, 3)
  objective = sc.function(sc.arg("x", sc.TensorType((3, *PATTERN.shape), pattern=PATTERN), layout="csr"), name="canonical_helper_loss")(
    lambda x: mapped(x).sum()
  )
  derivative = sc.gradient(objective).instantiate()
  helpers = []
  pending = [derivative]
  visited = set()
  while pending:
    function = pending.pop()
    if function in visited:
      continue
    visited.add(function)
    if function.role is not None:
      helpers.append(function)
      assert all(
        layout is None or isinstance(layout, SparseLayout) and layout.format == "csr"
        for layout in (*function.input_tree.layouts, *function.output_tree.layouts)
      )
    pending.extend(node.attrs["callee"] for node in topo(function.outputs) if node.op in {sc.ExprOp.CALL, sc.ExprOp.LOOP})
  assert helpers
