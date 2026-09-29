"""Multilayer perceptrons as expressions: dense layers, the activations workloads use, and weights unpacked from one flat vector."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from ..ir.expr import Expr, as_expr, maximum

type Layer = tuple[Any, Any]
"""A linear layer ``(W, b)``: ``W`` of shape ``(out, in)``, PyTorch's order, and ``b`` of ``out`` or
``None`` for none; each an array (a constant of the generated code) or an ``Expr``."""


def tanh(x: Expr) -> Expr:
  """``tanh(x)``."""
  return x.tanh()


def sigmoid(x: Expr) -> Expr:
  """The logistic function ``1 / (1 + exp(-x))``."""
  return 1.0 / (1.0 + (-x).exp())


def silu(x: Expr) -> Expr:
  """SiLU (swish), ``x sigmoid(x)``, spelled ``x / (1 + exp(-x))``."""
  return x / (1.0 + (-x).exp())


def relu(x: Expr) -> Expr:
  """``max(x, 0)``; its derivative at 0 is 0."""
  return maximum(x, 0.0)


def smooth_relu(x: Expr, eps: float) -> Expr:
  """``(x + sqrt(x^2 + eps^2)) / 2``, a ReLU smoothed over a width ``eps`` around 0."""
  return 0.5 * (x + (x * x + eps**2).sqrt())


def dense(x: Expr, W: Any, b: Any = None) -> Expr:
  """``W @ x + b``, without the bias when ``b`` is ``None``."""
  z = as_expr(W) @ x
  return z if b is None else z + as_expr(b)


def mlp(x: Expr, layers: Sequence[Layer], activation: Callable[[Expr], Expr] = tanh, *, output: Callable[[Expr], Expr] | None = None) -> Expr:
  """The multilayer perceptron of ``layers`` at ``x``: ``activation`` after every layer but the last,
  and ``output`` (none by default) after the last. Arrays become constants of the generated code,
  so the layers are dense matrix-vector loops there and their derivatives go through them; weights
  passed as ``Expr`` (a parameter, slices of one flat vector) make the network an input."""
  if not layers:
    raise ValueError("an mlp needs at least one layer")
  h = x
  for W, b in layers[:-1]:
    h = activation(dense(h, W, b))
  W, b = layers[-1]
  out = dense(h, W, b)
  return out if output is None else output(out)


def unpack(flat: Expr, widths: Sequence[int], *, offset: int = 0, bias: bool | Sequence[bool] = True) -> list[Layer]:
  """The layers of widths ``widths`` (input first, as ``(in, h1, ..., out)``) read from ``flat``
  starting at ``offset``: per layer ``W`` row-major, then ``b`` unless ``bias`` says the layer has
  none (one flag, or one per layer)."""
  count = len(widths) - 1
  if count < 1:
    raise ValueError("widths needs the input and at least one layer's width")
  flags = [bias] * count if isinstance(bias, bool) else list(bias)
  if len(flags) != count:
    raise ValueError(f"bias names {len(flags)} layers, widths {count}")
  out: list[Layer] = []
  at = offset
  for (n_in, n_out), has_bias in zip(zip(widths[:-1], widths[1:], strict=True), flags, strict=True):
    W = flat[at : at + n_in * n_out].reshape((n_out, n_in))
    at += n_in * n_out
    b = None
    if has_bias:
      b = flat[at : at + n_out]
      at += n_out
    out.append((W, b))
  return out


def size(widths: Sequence[int], *, bias: bool | Sequence[bool] = True) -> int:
  """The length of the flat vector ``unpack`` reads for ``widths``."""
  count = len(widths) - 1
  flags = [bias] * count if isinstance(bias, bool) else list(bias)
  return int(sum(a * b + (b if f else 0) for a, b, f in zip(widths[:-1], widths[1:], flags, strict=True)))


def pack(layers: Sequence[tuple[np.ndarray, np.ndarray | None]]) -> np.ndarray:
  """``unpack``'s layout of NumPy layers: per layer ``W`` row-major, then ``b`` if it has one."""
  parts = []
  for W, b in layers:
    parts.append(np.asarray(W, np.float64).reshape(-1))
    if b is not None:
      parts.append(np.asarray(b, np.float64).reshape(-1))
  return np.concatenate(parts)


__all__ = ["Layer", "dense", "mlp", "pack", "relu", "sigmoid", "silu", "size", "smooth_relu", "tanh", "unpack"]
