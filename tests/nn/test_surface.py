"""The public surface of ``scaly.nn``: its names, and that ``sc.nn`` is the package, loaded on first use."""

from __future__ import annotations

import importlib

import scaly as sc


def test_the_nn_surface() -> None:
  nn = importlib.import_module("scaly.nn")
  assert sc.nn is nn
  assert nn.__all__ == [
    "Layer",
    "dense",
    "layers_from_state_dict",
    "load_torch_state_dict",
    "mlp",
    "pack",
    "relu",
    "sigmoid",
    "silu",
    "size",
    "smooth_relu",
    "tanh",
    "unpack",
  ]
  assert "nn" not in sc.__all__ and not hasattr(importlib.import_module("scaly.utils"), "load_torch_state_dict")
  for name in nn.__all__:
    assert getattr(nn, name).__doc__, name
