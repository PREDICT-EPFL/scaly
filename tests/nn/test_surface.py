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


def test_the_experimental_modules_warn_on_import() -> None:
  """scaly-experimental makes no stability promise, and says so once per module, on import."""
  import subprocess
  import sys

  strict = "import warnings, scaly; warnings.simplefilter('error', scaly.ExperimentalWarning); "
  for module in ("scaly.nn", "scaly.geometry", "scaly.ocp.altro", "scaly.ocp.scvx"):
    proc = subprocess.run([sys.executable, "-c", strict + f"import {module}"], capture_output=True, text=True, check=False)
    assert proc.returncode != 0 and f"{module} is experimental (scaly-experimental)" in proc.stderr, module
  quiet = subprocess.run(
    [sys.executable, "-c", strict + "import scaly.ocp, scaly.linalg; scaly.ocp.Direct"], capture_output=True, text=True, check=False
  )
  assert quiet.returncode == 0, quiet.stderr
