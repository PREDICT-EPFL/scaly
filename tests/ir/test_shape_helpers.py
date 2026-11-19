import numpy as np
import pytest

import scaly as sc
from scaly.codegen import render_c_module
from scaly.function.concrete import ConcreteFunction
from scaly.passes.expr import simplify_cse_fixpoint


CASES = [
  ("squeeze", (1, 2, 1), lambda x: x.squeeze(axis=-1), lambda x: np.squeeze(x, axis=-1), lambda x: sc.squeeze(x), np.squeeze),
  (
    "expand_dims",
    (2, 3),
    lambda x: sc.expand_dims(x, (0, -1)),
    lambda x: np.expand_dims(x, (0, -1)),
    lambda x: sc.expand_dims(x, -1),
    lambda x: np.expand_dims(x, -1),
  ),
  (
    "broadcast_to",
    (2, 1),
    lambda x: sc.broadcast_to(x, (3, 2, 4)),
    lambda x: np.broadcast_to(x, (3, 2, 4)),
    lambda x: sc.broadcast_to(x, (2, 3)),
    lambda x: np.broadcast_to(x, (2, 3)),
  ),
  ("flatten", (2, 3), lambda x: x.T.flatten(), lambda x: x.T.flatten(), lambda x: x.flatten(order="C"), lambda x: x.flatten(order="C")),
  ("flip", (2, 3), lambda x: sc.flip(x, (-1, 0)), lambda x: np.flip(x, (-1, 0)), lambda x: sc.flip(x), np.flip),
  (
    "pad",
    (2, 3),
    lambda x: sc.pad(x, ((1, 0), (0, 2)), constant_values=-2),
    lambda x: np.pad(x, ((1, 0), (0, 2)), constant_values=-2),
    lambda x: sc.pad(x, 1),
    lambda x: np.pad(x, 1),
  ),
  (
    "swapaxes",
    (2, 1, 3),
    lambda x: x.swapaxes(0, -1),
    lambda x: x.swapaxes(0, -1),
    lambda x: sc.swapaxes(x, -2, -1),
    lambda x: np.swapaxes(x, -2, -1),
  ),
  (
    "moveaxis",
    (2, 1, 3),
    lambda x: sc.moveaxis(x, (0, -1), (-1, 0)),
    lambda x: np.moveaxis(x, (0, -1), (-1, 0)),
    lambda x: sc.moveaxis(x, -1, 0),
    lambda x: np.moveaxis(x, -1, 0),
  ),
  (
    "newaxis",
    (2, 3),
    lambda x: x[None, ..., -1, None],
    lambda x: x[None, ..., -1, None],
    lambda x: x[::-1, None, ..., None],
    lambda x: x[::-1, None, ..., None],
  ),
]


@pytest.mark.parametrize(("name", "shape", "helper", "reference", "other", "other_reference"), CASES, ids=[case[0] for case in CASES])
def test_shape_helpers_numpy_and_gradient(name, shape, helper, reference, other, other_reference):
  x = sc.sym("x", shape)
  y = helper(x)
  weights = np.arange(y.size, dtype=float).reshape(y.shape) / 7 + 1
  cost = (y * sc.const(weights)).sumsqr()
  grad = sc.vjp((cost,), (x,), (sc.const(1.0),))[0]
  fn = ConcreteFunction._from_exprs(f"shape_{name}", [x], [y, grad], ["x"], ["y", "grad"])
  data = np.linspace(-0.7, 1.1, x.size).reshape(shape)
  actual, actual_grad = fn(data)
  expected = reference(data)
  assert np.asarray(actual).shape == expected.shape
  assert np.asarray(actual).dtype == expected.dtype
  np.testing.assert_array_equal(actual, expected)
  finite = np.empty(shape)
  for i in range(x.size):
    step = np.zeros(shape)
    step.flat[i] = 1e-5
    finite.flat[i] = (np.square(reference(data + step) * weights).sum() - np.square(reference(data - step) * weights).sum()) / 2e-5
  np.testing.assert_allclose(actual_grad, finite, rtol=1e-7, atol=1e-8)

  dtype = {"broadcast_to": np.bool_, "flip": np.float32, "pad": np.float32}.get(name, np.int32)
  typed = np.arange(x.size).reshape(shape).astype(dtype)
  if name == "flip":
    typed.flat[0] = -0.0
  result = simplify_cse_fixpoint(other(sc.const(typed)))
  expected = other_reference(typed)
  assert result.value is not None
  assert result.type.dtype.numpy() == expected.dtype
  assert result.shape == expected.shape
  np.testing.assert_array_equal(result.value, expected)
  if name == "flip":
    np.testing.assert_array_equal(np.signbit(result.value), np.signbit(expected))


def test_shape_helpers_invalid_inputs():
  x = sc.sym("x", (2, 3))
  cases = [
    ("squeeze", lambda: x.squeeze(-1)),
    ("squeeze", lambda: sc.squeeze(x, 2)),
    ("expand_dims", lambda: sc.expand_dims(x, 4)),
    ("expand_dims", lambda: sc.expand_dims(x, (0, -4))),
    ("broadcast_to", lambda: sc.broadcast_to(x, (2, 4))),
    ("broadcast_to", lambda: sc.broadcast_to(x, (-1, 3))),
    ("flatten", lambda: x.flatten("F")),
    ("flip", lambda: sc.flip(x, -3)),
    ("flip", lambda: sc.flip(x, (0, -2))),
    ("pad", lambda: sc.pad(x, -1)),
    ("pad", lambda: sc.pad(x, ((1, 2, 3),))),
    ("pad", lambda: sc.pad(x, 1, mode="reflect")),
    ("pad", lambda: sc.pad(x, 1, constant_values=(1, 2))),
    ("pad.*numerical scalar", lambda: sc.pad(x, 1, constant_values=sc.sym("padding"))),
    ("swapaxes", lambda: x.swapaxes(0, 2)),
    ("moveaxis", lambda: sc.moveaxis(x, 0, -3)),
    ("moveaxis", lambda: sc.moveaxis(x, (0, 1), (1,))),
    ("moveaxis", lambda: sc.moveaxis(x, (0, -2), (0, 1))),
    ("indices", lambda: x[None, 0, 0, 0]),
    ("index", lambda: x[None, ..., ...]),
  ]
  for name, helper in cases:
    with pytest.raises((ValueError, IndexError), match=name):
      helper()


def test_reshape_helpers_lower_without_workspace():
  @sc.function(sc.arg("x", (17, 19)), outputs=sc.arg("y"))
  def shape_views(x):
    return sc.expand_dims(x, -1).squeeze(-1)[None, ...].flatten().block()

  module = render_c_module(shape_views, frame_budget=0)
  assert module.workspace_size == 0
  assert "#define shape_views_WORKSPACE_BYTES 0" in module.header
