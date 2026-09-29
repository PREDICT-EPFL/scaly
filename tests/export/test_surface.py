"""The public surface of ``scaly.export``: its names, that ``sc.export`` is the package, loaded on first
use, and that the output adapters load from it by name."""

from __future__ import annotations

import importlib
import importlib.util

import scaly as sc
from scaly.codegen.adapter import get_adapter


def test_the_export_surface() -> None:
  export = importlib.import_module("scaly.export")
  assert sc.export is export
  assert export.__all__ == [
    "CASADI_QUERIES",
    "acados_functions",
    "casadi_scratch",
    "casadi_sparsity",
    "check_casadi_layout",
    "install_dropin",
    "render_cpp_header",
  ]
  assert "export" not in sc.__all__
  assert get_adapter("cpp") is not None and get_adapter("casadi") is not None
  assert importlib.util.find_spec("scaly.codegen.cpp") is None and importlib.util.find_spec("scaly.codegen.casadi") is None
