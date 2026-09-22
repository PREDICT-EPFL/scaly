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

import scaly as sc
from scaly.ir.types import dtypes


def test_dtype_registry_and_string_compat() -> None:
  d = dtypes.float64
  assert isinstance(d, sc.DType)
  assert d == "float64"  # back-compat string equality
  assert d.bits == 64
  assert d.c_type == "double"
  assert d.itemsize == 8
  assert d.is_floating
  assert dtypes.float32.c_type == "float"
  assert dtypes.int32.c_type == "int32_t"
  assert dtypes.bool_.is_bool
  with pytest.raises(ValueError):
    sc.as_dtype("complex64")


def test_dtype_propagation_through_unary_and_binary() -> None:
  x = sc.sym("x", 3, dtype=dtypes.float32)
  y = x.sin() + x
  assert y.type.dtype == dtypes.float32
  z = sc.sym("z", 3)  # defaults to float64
  with pytest.raises(TypeError):
    _ = x + z  # mixed dtype refused without an explicit cast


def test_const_dtype_round_trip() -> None:
  c = sc.const(np.array([1.0, 2.0], dtype=np.float32), dtype=dtypes.float32)
  assert c.type.dtype == dtypes.float32
  assert c.value is not None and c.value.dtype == np.float32


def test_devicespec_parse_and_str() -> None:
  assert sc.DeviceSpec.parse(None) == sc.DeviceSpec("host", 0)
  assert sc.DeviceSpec.parse("host") == sc.DeviceSpec("host", 0)
  assert sc.DeviceSpec.parse("cuda:1") == sc.DeviceSpec("cuda", 1)
  assert str(sc.DeviceSpec("cuda", 0)) == "cuda:0"
  with pytest.raises(ValueError):
    sc.DeviceSpec("tpu", 0)


def test_function_with_device_repr_and_lower_diagnostic() -> None:
  @sc.function(sc.L("x", 3), sc.L("y", ...), name="f")
  def fn(x: sc.Expr) -> sc.Expr:
    return x.sum()

  assert fn.device.kind == "host"
  assert "device=" not in repr(fn)
  gpu = fn.with_device("cuda:0")
  assert gpu.device == sc.DeviceSpec("cuda", 0)
  assert "device=cuda:0" in repr(gpu)
  # only host lowers today; cuda placement should fail loudly via the JIT path
  from scaly.codegen.jit import JitError

  with pytest.raises(JitError, match="only host lowering"):
    gpu(np.array([1.0, 2.0, 3.0]))


def test_backend_capability_table_rejects_unsupported_dtype() -> None:
  with pytest.raises(ValueError, match="cannot lower dtype float64"):
    sc.Function("f", lambda x: x.sum(), sc.L("x", sc.TensorType((3,), dtype=dtypes.float64)), sc.L("y", ...), device="metal:0")
  # but float32 on metal is fine
  sc.Function("f32", lambda x: x.sum(), sc.L("x", sc.TensorType((3,), dtype=dtypes.float32)), sc.L("y", ...), device="metal:0")


def test_float32_construction_keeps_dtype_metadata() -> None:
  @sc.function(sc.L("x", sc.TensorType((3,), dtype=dtypes.float32)), sc.L("y", ...), name="f32")
  def fn(x: sc.Expr) -> sc.Expr:
    return (x * x).sum()

  assert fn.inputs[0].type.dtype == dtypes.float32
  assert fn.outputs[0].type.dtype == dtypes.float32


def test_type_shapes_reject_negative_dimensions_and_mismatched_sparsity() -> None:
  for make in [lambda: sc.sym("bad", -1), lambda: sc.sym("bad", (2, -1)), lambda: sc.TensorType((-1,))]:
    try:
      _ = make()
    except ValueError as e:
      assert "negative dimensions" in str(e)
    else:  # pragma: no cover
      raise AssertionError("negative shape should fail")

  try:
    _ = sc.SparsityType((-1, 2), (), ())
  except ValueError as e:
    assert "negative dimensions" in str(e)
  else:  # pragma: no cover
    raise AssertionError("negative sparsity shape should fail")

  try:
    _ = sc.TensorType((2, 2), sparsity=sc.SparsityType.dense((2, 3)))
  except ValueError as e:
    assert "does not match sparsity shape" in str(e)
  else:  # pragma: no cover
    raise AssertionError("mismatched tensor sparsity shape should fail")
