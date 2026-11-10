"""Pattern bindings share graph instances across symbolic and numerical calls."""

from typing import Any

import numpy as np
import pytest
from scipy import sparse

import scaly as sc
from scaly.codegen import render_c_source
from scaly.ir.layout import SparseLayout
from scaly.ir.types import CanonicalPattern


SHAPE = (3, 4)
PATTERNS = (
  CanonicalPattern(SHAPE, (0, 0, 1, 2, 2), (0, 3, 1, 0, 2)),
  CanonicalPattern(SHAPE, (0, 1, 1, 2, 2), (1, 0, 2, 1, 3)),
  CanonicalPattern(SHAPE, (), ()),
)


@pytest.mark.parametrize("layout", ["csc", "csr"])
def test_pattern_instances_reuse_symbolic_numerical_and_explicit_bindings(layout: str) -> None:
  traces = []
  leaf = sc.arg("x", SHAPE, pattern=..., layout=layout)

  @sc.function(leaf, outputs=leaf.relabel("out_"))
  def negate(x: sc.Expr) -> sc.Expr:
    traces.append(x.type.pattern)
    return -x

  assert leaf.has_holes and leaf.shapes == (SHAPE,) and not negate.instances
  instances = []
  for pattern in PATTERNS:
    source = sparse.coo_array((np.arange(pattern.nnz, dtype=float), (pattern.rows, pattern.cols)), shape=SHAPE)
    source.sum_duplicates()
    result: Any = negate(source)
    np.testing.assert_array_equal(result.toarray(), -source.toarray())
    symbol = sc.arg("actual", sc.TensorType(SHAPE, pattern=pattern)).symbols()
    instance = negate.symbolic_call(symbol).attrs["callee"]
    assert negate.instantiate((symbol.type,)) is instance
    assert instance.input_tree.types == (symbol.type,)
    assert instance.input_tree.layouts[0].format == layout
    assert f"p{pattern.digest[:8]}" in instance.name and "__" not in instance.name
    again: Any = negate(source * 2)
    np.testing.assert_array_equal(again.toarray(), -source.toarray() * 2)
    stored = instance.input_tree.layouts[0].numerical_values(source, SHAPE)
    direct: Any = instance(stored)
    np.testing.assert_array_equal(direct.toarray(), -source.toarray())
    instances.append(instance)
  assert len(negate.instances) == 3 and len({instance.name for instance in instances}) == 3
  assert traces == list(PATTERNS)


@pytest.mark.parametrize("format_", ["coo", "csr", "csc"])
def test_scipy_binding_canonicalizes_a_copy_and_recomputes_mutable_patterns(format_: str) -> None:
  identity: Any = sc.function(sc.arg("x", SHAPE, pattern=...))(lambda x: x)
  source = sparse.coo_array(([2.0, 3.0, -2.0, 0.0], ([2, 0, 2, 1], [0, 3, 0, 1])), shape=SHAPE).asformat(format_)
  before = source.copy()
  result = identity(source)
  np.testing.assert_array_equal(result.toarray(), source.toarray())
  assert result.nnz == 3
  np.testing.assert_array_equal(source.data, before.data)
  if format_ == "coo":
    np.testing.assert_array_equal(source.row, before.row)
    np.testing.assert_array_equal(source.col, before.col)
  else:
    np.testing.assert_array_equal(source.indices, before.indices)
    np.testing.assert_array_equal(source.indptr, before.indptr)
  source = sparse.coo_array(([4.0], ([0], [0])), shape=SHAPE)
  identity(source)
  source.col[0] = 1
  changed = identity(source)
  np.testing.assert_array_equal(changed.toarray(), source.toarray())
  assert len(identity.instances) == 3


def test_unbound_pattern_errors_and_dense_binding() -> None:
  leaf = sc.arg("x", SHAPE, pattern=...)
  identity: Any = sc.function(leaf)(lambda x: x)
  for operation in (identity.instantiate, identity.compile, leaf.symbols):
    with pytest.raises(TypeError, match="pattern|holes"):
      operation()
  with pytest.raises(TypeError, match="pattern"):
    identity.instantiate((SHAPE,))
  with pytest.raises(ValueError, match="shape"):
    identity(sparse.coo_array((4, 3)))
  with pytest.raises(ValueError, match="shape"):
    identity.symbolic_call(sc.sym("wrong", (4, 3)))
  with pytest.raises(ValueError, match="shape"):
    identity(np.ones(5))
  dense = np.arange(12.0).reshape(SHAPE)
  np.testing.assert_array_equal(identity(dense), dense)
  assert next(iter(identity.instances.values())).inputs[0].type.pattern is None
  assert leaf.has_holes


def test_two_bound_patterns_generate_distinct_procedures_and_match_numpy() -> None:
  negate = sc.function(sc.arg("x", SHAPE, pattern=...), name="pattern_negate")(lambda x: -x)
  left, right = (sc.arg(name, sc.TensorType(SHAPE, pattern=pattern)) for name, pattern in zip(("a", "b"), PATTERNS[:2], strict=True))

  @sc.function(left, right)
  def host(a: sc.Expr, b: sc.Expr) -> sc.Expr:
    return negate(a).to_dense() + negate(b).to_dense()

  a, b = (sparse.coo_array((np.arange(pattern.nnz, dtype=float), (pattern.rows, pattern.cols)), shape=SHAPE) for pattern in PATTERNS[:2])
  evaluate: Any = host
  np.testing.assert_array_equal(evaluate(a, b), -a.toarray() - b.toarray())
  source = render_c_source(host, lanes=1)
  assert len(negate.instances) == 2
  assert all(instance.name in source for instance in negate.instances.values())


def test_concrete_values_path_does_not_canonicalize_or_copy(monkeypatch: pytest.MonkeyPatch) -> None:
  leaf = sc.arg("x", sc.TensorType(SHAPE, pattern=PATTERNS[0]))
  identity: Any = sc.function(leaf)(lambda x: x)
  instance = identity.instantiate()
  values = np.arange(PATTERNS[0].nnz, dtype=float)
  assert leaf.flatten_numerical(values, "values")[0] is values

  def forbidden(*args, **kwargs):
    raise AssertionError("the values path canonicalized a sparse argument")

  monkeypatch.setattr(sparse, "coo_array", forbidden)
  for function in (identity, instance):
    np.testing.assert_array_equal(function(values).data, values)


def test_hole_binding_canonicalizes_only_once(monkeypatch: pytest.MonkeyPatch) -> None:
  identity: Any = sc.function(sc.arg("x", SHAPE, pattern=...))(lambda x: x)
  source = sparse.coo_array(([1.0], ([0], [1])), shape=SHAPE)
  original = SparseLayout.numerical_values

  def values_only(self, value, shape):
    assert isinstance(value, np.ndarray)
    return original(self, value, shape)

  monkeypatch.setattr(SparseLayout, "numerical_values", values_only)
  np.testing.assert_array_equal(identity(source).toarray(), source.toarray())


@pytest.mark.parametrize("layout", ["csc", "csr"])
def test_lifted_holes_keep_seed_factory_and_mapped_layouts(layout: str) -> None:
  leaf = sc.arg("x", SHAPE, pattern=..., layout=layout)
  source: Any = sc.function(leaf, outputs=leaf.relabel("out_"))(lambda x: x.sin())
  forward: Any = sc.forward(source, name="pattern_fwd")
  adjoint: Any = sc.adjoint(source, name="pattern_adj")
  factory: Any = source.factory("pattern_factory", ["x"], ["out_x"])
  mapped: Any = sc.vmap(source, 2)
  with pytest.raises(ValueError, match="bind open source inputs"):
    source.factory("missing_pattern", [], ["out_x"])
  for pattern in PATTERNS[:2]:
    x = sparse.coo_array((np.arange(pattern.nnz, dtype=float) / 10, (pattern.rows, pattern.cols)), shape=SHAPE)
    seed = sparse.coo_array((np.ones(pattern.nnz), (pattern.rows, pattern.cols)), shape=SHAPE)
    expected = np.cos(x.toarray()) * seed.toarray()
    for derivative in (forward, adjoint):
      result = derivative(x, seed)
      np.testing.assert_allclose(result.toarray(), expected)
      instance = derivative.instantiate((sc.TensorType(SHAPE, pattern=pattern),) * 2)
      assert all(item is not None and item.format == layout for item in instance.input_tree.layouts)
      assert f"p{pattern.digest[:8]}" in instance.name
    np.testing.assert_allclose(factory(x).toarray(), np.sin(x.toarray()))
    np.testing.assert_allclose(mapped(sc.broadcast(x)).toarray(), np.broadcast_to(np.sin(x.toarray()), (2, *SHAPE)))
  for function in (source, forward, adjoint, factory, mapped):
    assert len(function.instances) == 2
  with pytest.raises(ValueError, match="seed or argument types"):
    forward(x, sparse.coo_array(SHAPE))


def test_nested_shape_and_pattern_holes_and_declared_metadata() -> None:
  leaf = sc.arg("x", pattern=..., layout="csr")
  grouped = sc.group(leaf, sc.group(sc.arg("scale", ())))
  function: Any = sc.function(grouped)(lambda args: args[0] * args[1][0])
  source = sparse.eye_array(3, format="csr")
  np.testing.assert_array_equal(function((source, (np.array(2.0),))).toarray(), source.toarray() * 2)
  instance = next(iter(function.instances.values()))
  assert instance.inputs[0].shape == (3, 3) and instance.input_tree.layouts[0].format == "csr"
  assert "3x3_p" in instance.name
  with pytest.raises(TypeError, match="inferred shapes"):
    _ = leaf.shapes
  with pytest.raises(ValueError, match="structure"):
    function((source, np.array(2.0)))
  typed = sc.arg("x", sc.TensorType(SHAPE, diff=False), pattern=...)
  identity: Any = sc.function(typed)(lambda x: x)
  concrete = identity.instantiate((sc.TensorType(SHAPE, diff=False, pattern=PATTERNS[0]),))
  assert not concrete.inputs[0].type.diff
  for wrong in (sc.TensorType(SHAPE, pattern=PATTERNS[0]), sc.TensorType(SHAPE, sc.dtypes.float32, False, PATTERNS[0])):
    with pytest.raises(TypeError, match="contradicts"):
      identity.instantiate((wrong,))


def test_batched_hole_requires_the_same_local_pattern() -> None:
  identity: Any = sc.function(sc.arg("x", (2, *SHAPE), pattern=...))(lambda x: x)
  pattern = PATTERNS[0]
  source = sparse.coo_array(
    (np.arange(2 * pattern.nnz, dtype=float), (np.repeat([0, 1], pattern.nnz), np.tile(pattern.rows, 2), np.tile(pattern.cols, 2))),
    shape=(2, *SHAPE),
  )
  np.testing.assert_array_equal(identity(source).toarray(), source.toarray())
  source.coords[2][-1] = 3
  with pytest.raises(ValueError, match="share one stored pattern"):
    identity(source)


def test_pattern_name_collision_raises_instead_of_aliasing() -> None:
  first, other = PATTERNS[:2]
  second = CanonicalPattern(SHAPE, other.rows, other.cols)
  original = second.digest
  identity: Any = sc.function(sc.arg("x", SHAPE, pattern=...))(lambda x: x)
  try:
    object.__setattr__(second, "digest", first.digest)
    identity.instantiate((sc.TensorType(SHAPE, pattern=first),))
    with pytest.raises(ValueError, match="instance name collision"):
      identity.instantiate((sc.TensorType(SHAPE, pattern=second),))
  finally:
    object.__setattr__(second, "digest", original)


@pytest.mark.parametrize("outputs,grouped", [(None, False), (sc.arg("y"), False), (sc.group(sc.arg("y")), True)])
def test_inferred_sparse_output_multipliers_bind_the_whole_type(outputs, grouped: bool) -> None:
  options = {} if outputs is None else {"outputs": outputs}
  source: Any = sc.function(sc.arg("x", SHAPE, pattern=..., layout="csr"), name="inferred_sparse", **options)(
    lambda x: (x.sin(),) if grouped else x.sin()
  )
  output = "inferred_sparse" if outputs is None else "y"
  adjoint: Any = sc.adjoint(source)
  hess: Any = sc.lagrangian_hessian(source)
  factory: Any = source.factory("inferred_sparse_factory", ["x", f"lam:{output}"], [sc.factory.Adj(output, "x")])
  for pattern in PATTERNS[:2]:
    x = sparse.coo_array((np.arange(pattern.nnz, dtype=float) / 10 + 0.2, (pattern.rows, pattern.cols)), shape=SHAPE)
    seed = sparse.coo_array((np.ones(pattern.nnz), (pattern.rows, pattern.cols)), shape=SHAPE)
    expected = np.cos(x.toarray()) * seed.toarray()
    np.testing.assert_allclose(adjoint(x, seed).toarray(), expected)
    np.testing.assert_allclose(factory(x, seed).toarray(), expected)
    multiplier = (seed,) if grouped else seed
    np.testing.assert_allclose(hess(x, multiplier), np.diag((-np.sin(x.toarray()) * seed.toarray()).reshape(-1)))
    symbol = sc.arg("actual", sc.TensorType(SHAPE, pattern=pattern)).symbols()
    assert adjoint.symbolic_call(symbol, symbol).attrs["callee"] is next(reversed(adjoint.instances.values()))
  assert len(source.instances) == len(adjoint.instances) == len(factory.instances) == len(hess.instances) == 2


@pytest.mark.parametrize("shape", [(2,), (2, 5), (2, 3, 4)])
def test_dense_shape_and_pattern_holes_map_like_shape_holes(shape) -> None:
  source: Any = sc.function(sc.arg("x", pattern=...))(lambda x: x * 2)
  mapped: Any = sc.vmap(source, 2)
  values = np.arange(np.prod(shape), dtype=float).reshape(shape)
  np.testing.assert_array_equal(mapped(values), values * 2)
  symbol = sc.sym("actual", shape)
  assert mapped.symbolic_call(symbol).shape == shape
  assert len(source.instances) == len(mapped.instances) == 1


@pytest.mark.parametrize("counts", [(1, 3), (1, 2)])
def test_unequal_batch_pattern_sizes_report_the_shared_pattern_error(counts) -> None:
  identity: Any = sc.function(sc.arg("x", (2, *SHAPE), pattern=...))(lambda x: x)
  nnz = sum(counts)
  source = sparse.coo_array((np.ones(nnz), (np.repeat([0, 1], counts), np.zeros(nnz, dtype=int), np.r_[0, np.arange(counts[1])])), shape=(2, *SHAPE))
  with pytest.raises(ValueError, match="share one stored pattern"):
    identity(source)


@pytest.mark.parametrize("derived", [False, True])
def test_bare_and_inferred_output_derivatives_map_sparse_arguments(derived: bool) -> None:
  source: Any = sc.adjoint(sc.function(sc.arg("x", SHAPE, pattern=..., layout="csr"))(lambda x: x.sin())) if derived else sc.function()(lambda x: -x)
  mapped: Any = sc.vmap(source, 2)
  for pattern in PATTERNS[:2]:
    x = sparse.coo_array((np.arange(pattern.nnz, dtype=float) / 10, (pattern.rows, pattern.cols)), shape=SHAPE)
    actuals = (x, x) if derived else (x,)
    expected = np.cos(x.toarray()) * x.toarray() if derived else -x.toarray()
    np.testing.assert_allclose(mapped(*(sc.broadcast(value) for value in actuals)).toarray(), np.broadcast_to(expected, (2, *SHAPE)))
    symbol = sc.arg("actual", sc.TensorType(SHAPE, pattern=pattern)).symbols()
    symbols = (symbol, symbol) if derived else (symbol,)
    result = mapped.symbolic_call(*(sc.broadcast(value) for value in symbols))
    assert result.shape == (2, *SHAPE) and result.type.pattern is pattern
  assert len(source.instances) == len(mapped.instances) == 2


@pytest.mark.parametrize("shape", [(3,), (3, 4)])
def test_dense_unbound_pattern_maps_require_a_leading_axis_or_broadcast(shape) -> None:
  source: Any = sc.function(sc.arg("x", pattern=...))(lambda x: x * 2)
  mapped: Any = sc.vmap(source, 2)
  values = np.arange(np.prod(shape), dtype=float).reshape(shape)
  with pytest.raises(ValueError, match="cannot map shape"):
    mapped(values)
  with pytest.raises(ValueError, match="cannot map shape"):
    mapped.symbolic_call(sc.sym("actual", shape))
  np.testing.assert_array_equal(mapped(sc.broadcast(values)), np.broadcast_to(values * 2, (2, *shape)))


def test_inferred_derivatives_keep_declared_differentiability() -> None:
  source: Any = sc.function(sc.arg("x", sc.TensorType(SHAPE, diff=False), pattern=...))(lambda x: x.sin())
  adjoint: Any = sc.adjoint(source)
  mapped: Any = sc.vmap(adjoint, 2)
  pattern = PATTERNS[0]
  value = sparse.coo_array((np.ones(pattern.nnz), (pattern.rows, pattern.cols)), shape=SHAPE)
  expected = np.cos(value.toarray()) * value.toarray()
  np.testing.assert_allclose(adjoint(value, value).toarray(), expected)
  np.testing.assert_allclose(mapped(sc.broadcast(value), sc.broadcast(value)).toarray(), np.broadcast_to(expected, (2, *SHAPE)))
  assert not next(iter(source.instances.values())).inputs[0].type.diff
  assert not next(iter(adjoint.instances.values())).inputs[0].type.diff
