"""Dense boundary orders preserve logical values through JIT, calls, maps and derivatives."""

import numpy as np
import pytest
from scipy import sparse

import scaly as sc
from scaly.codegen.jit import get_compiled
from scaly.ir.layout import DenseLayout


@pytest.mark.parametrize("shape", [(4,), (2, 3, 4)])
def test_dense_layout_numerical_calls_and_shape_binding(shape):
  @sc.function(sc.arg("x", layout="column-major"), outputs=sc.arg("y", layout="column-major"), name="dense_order")
  def fun(x):
    return x.sin()

  x = np.arange(np.prod(shape, dtype=int), dtype=float).reshape(shape) / 10
  for value in (x, np.asfortranarray(x)):
    np.testing.assert_allclose(fun(value), np.sin(x))
  instance = fun.instantiate((shape,))
  if len(shape) >= 2:
    assert isinstance(instance.input_tree.layouts[0], DenseLayout)
    assert isinstance(instance.output_tree.layouts[0], DenseLayout)
    assert instance.input_tree.layouts[0].shape == shape
  else:
    assert instance.input_tree.layouts == instance.output_tree.layouts == (None,)
  assert fun.inputs is not None
  assert fun.inputs.relabel("prefix_").layouts == fun.inputs.layouts


def test_dense_layout_reshape_uses_logical_order():
  @sc.function(sc.arg("x", (2, 3), layout="column-major"), name="dense_reshape")
  def fun(x):
    return x.reshape((6,))

  x = np.arange(6, dtype=float).reshape(2, 3)
  np.testing.assert_array_equal(fun(x), x.reshape(6))


def test_dense_layout_calls_maps_and_derivatives():
  @sc.function(sc.arg("x", (2, 3), layout="column-major"), outputs=sc.arg("y", layout="column-major"), name="dense_stage")
  def stage(x):
    return (x.sin() * np.array([[0, 1, 0], [2, 0, 3]], dtype=float)).block()

  @sc.function(sc.arg("x", (2, 3)), name="dense_caller")
  def caller(x):
    return stage(x) + x

  @sc.function(sc.arg("x", (2, 2, 3)), name="dense_map_caller")
  def mapped_caller(x):
    return sc.vmap(stage, 2)(x)

  x = np.arange(6, dtype=float).reshape((2, 3)) / 10
  weights = np.array([[0, 1, 0], [2, 0, 3]])
  np.testing.assert_allclose(caller(x), np.sin(x) * weights + x)
  batch = np.stack([x, x + 1])
  np.testing.assert_allclose(sc.vmap(stage, 2)(batch), np.sin(batch) * weights)
  np.testing.assert_allclose(mapped_caller(batch), np.sin(batch) * weights)
  seed = x + 2
  np.testing.assert_allclose(sc.forward(stage)(x, seed), np.cos(x) * seed * weights)
  np.testing.assert_allclose(sc.adjoint(stage)(x, seed), np.cos(x) * seed * weights)


def test_dense_layout_constant_output_support():
  @sc.function(outputs=sc.arg("y", layout="column-major"), name="dense_constants")
  def fun():
    return sc.const([[0.0, 2, 0], [3, 0, 4]])

  np.testing.assert_array_equal(fun(), [[0, 2, 0], [3, 0, 4]])


def test_dense_column_major_maps_do_not_reorder_scalar_or_vector_batches():
  @sc.function(sc.arg("x", layout="column-major"), outputs=sc.arg("y", layout="column-major"), name="dense_vector_map")
  def fun(x):
    return x * 2.0

  x = np.arange(6, dtype=float).reshape((2, 3))
  np.testing.assert_array_equal(sc.vmap(fun, 2)(x), x * 2)


def test_dense_layout_row_major_default_and_invalid_orders():
  assert sc.arg("x", (2, 3)).layouts == sc.arg("x", (2, 3), layout="row-major").layouts == (None,)
  with pytest.raises(ValueError, match="dense layout"):
    sc.arg("x", (2, 3), layout="csc")
  pattern, _ = sc.SparsityPattern((2, 3), (0,), (1,)).canonicalize()
  with pytest.raises(ValueError, match="sparse layout"):
    sc.arg("x", sc.TensorType((2, 3), pattern=pattern), layout="column-major")

  fun = sc.function(sc.arg("x", (2, 3), layout="column-major"), name="dense_bad_shape")(lambda x: x)
  with pytest.raises(ValueError, match="shape"):
    get_compiled(fun).run([np.arange(6, dtype=float)])


def test_dense_layout_with_pattern_holes_in_factories_maps_and_adjoints():
  @sc.function(
    sc.arg("x", layout="column-major"),
    sc.arg("s", (2, 3), pattern=..., layout="csc"),
    outputs=sc.arg("y", layout="column-major"),
    name="mixed_layout_holes",
  )
  def fun(x, s):
    return (x + s.to_dense()).sin()

  x = np.arange(6, dtype=float).reshape(2, 3) / 10
  s = sparse.coo_array(([1.0, 2.0], ([0, 1], [2, 0])), shape=(2, 3))
  expected = np.sin(x + s.toarray())
  factory = fun.factory("mixed_layout_factory", ["x", "s"], ["y"])
  for evaluate in (fun, factory):
    np.testing.assert_allclose(evaluate(x, s), expected)
  seed = x + 2
  result = sc.adjoint(fun, of="y", wrt="x")(x, s, seed)
  assert isinstance(result, np.ndarray)
  np.testing.assert_allclose(result, np.cos(x + s.toarray()) * seed)
  adjoint = fun.factory("mixed_layout_adj_factory", ["x", "s", "lam:y"], [sc.factory.Adj("y", "x")])
  np.testing.assert_allclose(adjoint(x, s, seed), np.cos(x + s.toarray()) * seed)
  batch = np.stack([x, x + 1])
  np.testing.assert_allclose(sc.vmap(fun, 2)(batch, sc.broadcast(s)), np.sin(batch + s.toarray()))
