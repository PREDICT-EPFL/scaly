"""The installed output adapters: ``cpp`` and ``casadi`` resolve through their entry points, a
second header adapter is refused beside ``cpp``, and ``cpp`` leaves the kernel alone."""

from __future__ import annotations

import numpy as np
import pytest

import scaly as sc
from scaly.codegen import render_c_module
from scaly.codegen import adapter as adapters_module
from scaly.codegen.adapter import available_adapters, get_adapter, register_adapter


def _fun() -> sc.ConcreteFunction:
  x = sc.sym("x", 3)
  return sc.Function.from_exprs("export_adapter_probe", [x], [x * 2.0], ["x"], ["y"])


def test_the_installed_adapters_resolve_by_name() -> None:
  assert {"cpp", "casadi"} <= set(available_adapters())
  assert get_adapter("cpp").header_suffix == "hpp"
  with pytest.raises(ValueError, match="already registered"):
    register_adapter("cpp")


def test_a_second_header_adapter_is_refused_beside_cpp() -> None:
  get_adapter("cpp")
  register_adapter("export_probe_header", header=lambda spec: "// a header\n", header_suffix="hh")
  try:
    with pytest.raises(ValueError, match="export_probe_header, cpp each set the header"):
      render_c_module(_fun(), adapters=("export_probe_header", "cpp"))
  finally:
    del adapters_module._ADAPTERS["export_probe_header"]


def test_the_cpp_adapter_leaves_the_kernel_alone() -> None:
  fun = _fun()
  np.testing.assert_allclose(fun(np.array([1.0, 2.0, 3.0])), [2.0, 4.0, 6.0])
  assert render_c_module(fun, adapters=("cpp",)).body == render_c_module(fun).body
