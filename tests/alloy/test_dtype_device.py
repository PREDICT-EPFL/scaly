"""Phase 1: dtype model and device placement policy.

Covers:
- ``DType`` registry and back-compat string equality (so legacy ``"float64"`` code paths keep working).
- ``DeviceSpec.parse`` / ``Function.with_device`` placement policy and its
  diagnostics (loud failure on unsupported placement).
- dtype propagation through expression construction and the rejection of
  mixed-dtype binary ops without an explicit cast.
"""

from __future__ import annotations

import numpy as np
import pytest

import alloy as al
from alloy.types import dtypes


def test_dtype_registry_and_string_compat() -> None:
  d = dtypes.float64
  assert isinstance(d, al.DType)
  assert d == "float64"  # back-compat string equality
  assert d.bits == 64
  assert d.c_type == "double"
  assert d.itemsize == 8
  assert d.is_floating
  assert dtypes.float32.c_type == "float"
  assert dtypes.int32.c_type == "int32_t"
  assert dtypes.bool_.is_bool
  with pytest.raises(ValueError):
    al.as_dtype("complex64")


def test_dtype_propagation_through_unary_and_binary() -> None:
  x = al.sym("x", 3, dtype=dtypes.float32)
  y = x.sin() + x
  assert y.type.dtype == dtypes.float32
  z = al.sym("z", 3)  # defaults to float64
  with pytest.raises(TypeError):
    _ = x + z  # mixed dtype refused without an explicit cast


def test_const_dtype_round_trip() -> None:
  c = al.const(np.array([1.0, 2.0], dtype=np.float32), dtype=dtypes.float32)
  assert c.type.dtype == dtypes.float32
  assert c.value is not None and c.value.dtype == np.float32


def test_devicespec_parse_and_str() -> None:
  assert al.DeviceSpec.parse(None) == al.DeviceSpec("host", 0)
  assert al.DeviceSpec.parse("host") == al.DeviceSpec("host", 0)
  assert al.DeviceSpec.parse("cuda:1") == al.DeviceSpec("cuda", 1)
  assert str(al.DeviceSpec("cuda", 0)) == "cuda:0"
  with pytest.raises(ValueError):
    al.DeviceSpec("tpu", 0)


def test_function_with_device_repr_and_lower_diagnostic() -> None:
  x = al.sym("x", 3)
  y = x.sum()
  fn = al.Function("f", [x], [y], ["x"], ["y"])
  assert fn.device.kind == "host"
  assert "device=" not in repr(fn)
  gpu = fn.with_device("cuda:0")
  assert gpu.device == al.DeviceSpec("cuda", 0)
  assert "device=cuda:0" in repr(gpu)
  # only host lowers today; cuda placement should fail loudly via the JIT path
  from alloy.jit import JitError

  with pytest.raises(JitError, match="only host lowering"):
    gpu(np.array([1.0, 2.0, 3.0]))


def test_backend_capability_table_rejects_unsupported_dtype() -> None:
  x = al.sym("x", 3, dtype=dtypes.float64)
  y = x.sum()
  with pytest.raises(ValueError, match="cannot lower dtype float64"):
    al.Function("f", [x], [y], ["x"], ["y"], device="metal:0")  # Metal has no float64
  # but float32 on metal is fine
  x32 = al.sym("x", 3, dtype=dtypes.float32)
  y32 = x32.sum()
  al.Function("f32", [x32], [y32], ["x"], ["y"], device="metal:0")


def test_float32_construction_keeps_dtype_metadata() -> None:
  x = al.sym("x", 3, dtype=dtypes.float32)
  y = (x * x).sum()
  fn = al.Function("f32", [x], [y], ["x"], ["y"])
  assert fn.inputs[0].type.dtype == dtypes.float32
  assert fn.outputs[0].type.dtype == dtypes.float32
