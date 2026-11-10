"""Dense boundary orders preserve logical values through JIT, calls, maps and derivatives."""

import numpy as np
import pytest

import scaly as sc
from scaly.codegen.jit import get_compiled
from scaly.ir.layout import DenseLayout


@pytest.mark.parametrize("shape", [(), (4,), (2, 3), (2, 3, 4)])
def test_dense_layout_numerical_calls_and_shape_binding(shape):
  @sc.function(sc.arg("x", layout="column-major"), outputs=sc.arg("y", layout="column-major"), name="dense_order")
  def fun(x):
    return x.sin()

  x = np.arange(np.prod(shape, dtype=int), dtype=float).reshape(shape) / 10
  for value in (x, np.asfortranarray(x) if shape else x):
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


@pytest.mark.parametrize("shape", [(), (3,)])
def test_dense_column_major_maps_do_not_reorder_scalar_or_vector_batches(shape):
  @sc.function(sc.arg("x", layout="column-major"), outputs=sc.arg("y", layout="column-major"), name="dense_vector_map")
  def fun(x):
    return x * 2.0

  x = np.arange(2 * np.prod(shape, dtype=int), dtype=float).reshape((2, *shape))
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
