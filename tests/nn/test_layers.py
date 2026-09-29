"""Multilayer perceptrons: every activation and bias-free layers against NumPy, weights as constants and
as slices of one vector, the packed layout and its round trip, the gradient through the layers, and
a state dict's layers."""

from __future__ import annotations

import numpy as np
import pytest

import scaly as sc
from scaly import nn

RNG = np.random.default_rng(3)
WIDTHS = (3, 5, 4, 2)
LAYERS = [(RNG.standard_normal((o, i)), RNG.standard_normal(o)) for i, o in zip(WIDTHS[:-1], WIDTHS[1:], strict=True)]
X = RNG.standard_normal(3)

NUMPY = {
  "tanh": np.tanh,
  "sigmoid": lambda z: 1.0 / (1.0 + np.exp(-z)),
  "silu": lambda z: z / (1.0 + np.exp(-z)),
  "relu": lambda z: np.maximum(z, 0.0),
}


def _numpy_mlp(x, layers, act, output=None):
  h = x
  for W, b in layers[:-1]:
    h = act(W @ h + (0.0 if b is None else b))
  W, b = layers[-1]
  out = W @ h + (0.0 if b is None else b)
  return out if output is None else output(out)


@pytest.mark.parametrize("name", list(NUMPY))
def test_each_activation_against_numpy(name: str) -> None:
  act = getattr(nn, name)

  @sc.function(3, name=f"nn_mlp_{name}")
  def net(x):
    return nn.mlp(x, LAYERS, act)

  np.testing.assert_allclose(net(X), _numpy_mlp(X, LAYERS, NUMPY[name]), rtol=1e-13, atol=1e-14)


def test_the_smooth_relu_bias_free_layers_and_an_output_activation() -> None:
  eps = 0.3
  layers = [(W, None) for W, _ in LAYERS[:-1]] + [LAYERS[-1]]

  @sc.function(3, name="nn_mlp_smooth")
  def net(x):
    return nn.mlp(x, layers, lambda z: nn.smooth_relu(z, eps), output=nn.sigmoid)

  smooth = lambda z: 0.5 * (z + np.sqrt(z * z + eps**2))  # noqa: E731
  np.testing.assert_allclose(net(X), _numpy_mlp(X, layers, smooth, NUMPY["sigmoid"]), rtol=1e-13)


def test_weights_from_one_vector_and_the_packed_layout() -> None:
  bias = [True, False, True]
  layers = [(W, b if keep else None) for (W, b), keep in zip(LAYERS, bias, strict=True)]
  flat = nn.pack(layers)
  assert flat.size == nn.size(WIDTHS, bias=bias) == 3 * 5 + 5 + 5 * 4 + 4 * 2 + 2

  @sc.function(sc.L("w", flat.size + 7), 3, name="nn_mlp_packed")
  def net(w, x):
    return nn.mlp(x, nn.unpack(w, WIDTHS, offset=7, bias=bias))

  np.testing.assert_allclose(net(np.r_[np.zeros(7), flat], X), _numpy_mlp(X, layers, np.tanh), rtol=1e-13)
  # The gradient in the weights is one vector, as a training loop wants it.
  grad = sc.gradient(
    sc.function(sc.L("w", flat.size), 3, output="y", name="nn_mlp_scalar")(lambda w, x: nn.mlp(x, nn.unpack(w, WIDTHS, bias=bias)).sum()), wrt="w"
  )
  h = 1e-6
  e = np.zeros(flat.size)
  e[11] = h
  numeric = (
    _numpy_mlp(X, [(W, b) for W, b in _unpacked(flat + e, bias)], np.tanh).sum()
    - _numpy_mlp(X, [(W, b) for W, b in _unpacked(flat - e, bias)], np.tanh).sum()
  ) / (2 * h)
  np.testing.assert_allclose(grad(flat, X)[11], numeric, rtol=1e-6)


def _unpacked(flat: np.ndarray, bias: list[bool]) -> list[tuple[np.ndarray, np.ndarray | None]]:
  out, at = [], 0
  for (i, o), keep in zip(zip(WIDTHS[:-1], WIDTHS[1:], strict=True), bias, strict=True):
    W = flat[at : at + i * o].reshape(o, i)
    at += i * o
    b = None
    if keep:
      b, at = flat[at : at + o], at + o
    out.append((W, b))
  return out


def test_a_state_dicts_layers() -> None:
  state = {"fc.0.weight": np.ones((2, 3), np.float32), "fc.0.bias": np.zeros(2, np.float32), "fc.1.weight": np.eye(2)}
  (w0, b0), (w1, b1) = nn.layers_from_state_dict(state, ["fc.0", "fc.1"])
  assert w0.dtype == np.float64 and b0 is not None and b0.dtype == np.float64 and b1 is None
  np.testing.assert_array_equal(w1, np.eye(2))


def test_what_the_builders_refuse() -> None:
  with pytest.raises(ValueError, match="at least one layer"):
    nn.mlp(sc.sym("x", 3), [])
  with pytest.raises(ValueError, match="bias names 2 layers, widths 3"):
    nn.unpack(sc.sym("w", 100), WIDTHS, bias=[True, False])
