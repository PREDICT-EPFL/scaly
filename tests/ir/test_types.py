"""Phase 1: dtype model and device placement policy.

Covers:
- ``DType`` registry and back-compat string equality (so legacy ``"float64"`` code paths keep working).
- ``DeviceSpec.parse``, which accepts only the host.
- dtype propagation through expression construction and the rejection of
  mixed-dtype binary ops.
- the scalar coercion rule: weak Python scalars, strong NumPy values, ``sc.const`` keeping a
  NumPy dtype, no truth value, NumPy integer indices.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import scaly as sc
from scaly.ir.expr import ExprOp, topo
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
  assert dtypes.uint8.c_type == "uint8_t" and dtypes.uint8.is_integer and dtypes.uint8.is_unsigned
  assert not dtypes.int32.is_unsigned
  assert sc.as_dtype("uint8") is dtypes.uint8 and dtypes.uint8 in dtypes.all()
  assert dtypes.uint8.numpy() == np.uint8
  with pytest.raises(ValueError):
    sc.as_dtype("complex64")


@pytest.mark.parametrize(
  "field,value",
  [
    ("name", "float_alias"),
    ("bits", 32),
    ("c_type", "float"),
    ("is_floating", False),
    ("is_integer", True),
    ("is_bool", True),
    ("is_unsigned", True),
  ],
)
def test_interning_distinguishes_every_dtype_field(field: str, value: object) -> None:
  original = dtypes.float64
  distinct = replace(original, **{field: value})
  first_dtype, second_dtype = original, distinct
  name = f"dtype_intern_{field}"
  first = sc.sym(name, 3, dtype=first_dtype)
  second = sc.sym(name, 3, dtype=second_dtype)
  assert first is not second
  assert first.type.dtype is first_dtype
  assert second.type.dtype is second_dtype
  assert first.type.dtype.c_type == first_dtype.c_type
  assert second.type.dtype.c_type == second_dtype.c_type
  assert first_dtype != second_dtype
  assert len({first_dtype, second_dtype}) == 2
  assert replace(first_dtype) == first_dtype
  assert hash(replace(first_dtype)) == hash(first_dtype) == hash(first_dtype.name)
  assert first_dtype == first_dtype.name and first_dtype.name == first_dtype
  assert sc.sym(name, 3, dtype=replace(first_dtype)) is first
  assert sc.sym(name, 3, dtype=replace(second_dtype)) is second


def test_unsigned_constants_fold_within_their_own_range() -> None:
  from scaly.ir.program import ProgramOp
  from scaly.passes.arith import evaluate

  assert evaluate(ProgramOp.ADD, [200, 55], dtypes.uint8) == 255
  assert evaluate(ProgramOp.ADD, [200, 56], dtypes.uint8) is None
  assert evaluate(ProgramOp.SUB, [1, 2], dtypes.uint8) is None
  assert evaluate(ProgramOp.ADD, [100, 27], dtypes.int64) == 127


@pytest.mark.parametrize("dtype", ["uint8", replace(dtypes.uint8, name="u1")], ids=["name", "alias"])
def test_uint8_is_not_an_expression_dtype(dtype: sc.DType | str) -> None:
  for make in (
    lambda: sc.TensorType((2,), dtype=sc.as_dtype(dtype)),
    lambda: sc.sym("bytes", 2, dtype=dtype, diff=False),
    lambda: sc.const([0, 255], dtype=dtype),
  ):
    with pytest.raises(ValueError, match="unsupported expression dtype uint8"):
      make()


@pytest.mark.parametrize("value", [0, 255])
def test_numpy_uint8_constants_are_rejected(value: int) -> None:
  for data in (np.uint8(value), np.array([value], dtype=np.uint8)):
    with pytest.raises(ValueError, match="unsupported expression dtype uint8"):
      sc.const(data)
    converted = sc.const(data, dtype="float64")
    assert converted.type.dtype == dtypes.float64
    np.testing.assert_array_equal(converted.value, np.asarray(data, dtype=np.float64))


def test_dtype_propagation_through_unary_and_binary() -> None:
  x = sc.sym("x", 3, dtype=dtypes.float32)
  y = x.sin() + x
  assert y.type.dtype == dtypes.float32
  z = sc.sym("z", 3)  # defaults to float64
  with pytest.raises(TypeError):
    _ = x + z  # mixed dtype refused


def test_devicespec_parse_and_str() -> None:
  assert sc.DeviceSpec.parse(None) == sc.DeviceSpec("host")
  assert sc.DeviceSpec.parse("host") == sc.DeviceSpec("host")
  assert str(sc.DeviceSpec()) == "host"
  for kind in ("cuda", "opencl", "metal", "cuda:0"):
    with pytest.raises(ValueError, match="expected host"):
      sc.DeviceSpec.parse(kind)
  for name in ("BACKEND_SUPPORT", "BackendSupport", "backend_supports"):
    assert not hasattr(sc, name)


def test_function_rejects_a_dtype_the_host_cannot_lower() -> None:
  half = sc.DType("float16", 16, "float16_t", is_floating=True)
  with pytest.raises(ValueError, match="cannot lower dtype float16"):
    sc.function(sc.arg("x", sc.TensorType((3,), dtype=half)), outputs=sc.arg("y"), name="f16")(lambda x: x.sum()).instantiate()


def test_type_shapes_reject_negative_dimensions_and_mismatched_sparsity() -> None:
  for make in [lambda: sc.sym("bad", -1), lambda: sc.sym("bad", (2, -1)), lambda: sc.TensorType((-1,))]:
    try:
      _ = make()
    except ValueError as e:
      assert "negative dimensions" in str(e)
    else:  # pragma: no cover
      raise AssertionError("negative shape should fail")

  try:
    _ = sc.SparsityPattern((-1, 2), (), ())
  except ValueError as e:
    assert "negative dimensions" in str(e)
  else:  # pragma: no cover
    raise AssertionError("negative sparsity shape should fail")


def test_python_scalars_take_the_dtype_of_the_expression_beside_them() -> None:
  x32 = sc.sym("x32", 3, dtype=dtypes.float32)
  i64 = sc.sym("i64", 3, dtype=dtypes.int64, diff=False)
  for y in [x32 * 2.0, 2 + x32, x32 - True, x32 + [1.0, 2.0, 3.0], sc.minimum(x32, 0.0), sc.atan2(1.0, x32), x32**2, sc.dot(x32, [1, 2, 3])]:
    assert y.type.dtype == dtypes.float32
    assert all(np.asarray(leaf.value).dtype == np.float32 for leaf in topo([y]) if leaf.op == ExprOp.CONST)
  assert sc.stack([x32[0], 0.0]).type.dtype == dtypes.float32 and sc.concat([[0.0], x32]).type.dtype == dtypes.float32
  assert (x32.reshape((3, 1)) @ [[1.0, 2.0]]).type.dtype == dtypes.float32
  assert (i64 + 3).type.dtype == dtypes.int64
  assert sc.concat([[], i64]).type.dtype == dtypes.int64 and (x32 + 2**100).type.dtype == dtypes.float32
  x64 = sc.sym("x64", 3)
  assert np.asarray((x64 * 2).args[1].value).dtype == np.float64
  with pytest.raises(TypeError, match="float"):
    _ = i64 * 1.5
  with pytest.raises(OverflowError):
    _ = sc.sym("i32", dtype=dtypes.int32, diff=False) + 2**40
  with pytest.raises(OverflowError):
    _ = sc.sym("i32", dtype=dtypes.int32, diff=False) + (-(2**31) - 1)


def test_numpy_values_are_strong_and_must_match() -> None:
  x32 = sc.sym("x32", 3, dtype=dtypes.float32)
  x64 = sc.sym("x64", 3)
  assert (x32 * np.float32(2)).type.dtype == dtypes.float32
  assert (x64 + np.arange(3.0)).type.dtype == dtypes.float64
  assert sc.stack([np.float32(1), 2.0]).type.dtype == dtypes.float32
  for bad in [lambda: x32 * np.float64(2), lambda: np.float32(2) * x64, lambda: x64 + np.arange(3), lambda: x32 + x64, lambda: sc.stack([x32, x64])]:
    with pytest.raises(TypeError, match="mixed dtypes"):
      bad()


def test_const_keeps_numpy_dtypes_and_defaults_to_float64() -> None:
  assert sc.const(np.arange(3)).type.dtype == dtypes.int64
  assert sc.const(np.ones(2, dtype=np.float32)).type.dtype == dtypes.float32
  assert sc.const(np.float32(1.0)).type.dtype == dtypes.float32
  assert sc.const(np.array([True])).type.dtype == dtypes.bool_
  assert np.asarray(sc.const(np.arange(3), dtype="float64").value).dtype == np.float64
  assert sc.const([1, 2]).type.dtype == dtypes.float64 and sc.const(3).type.dtype == dtypes.float64


def test_an_expression_has_no_truth_value() -> None:
  x = sc.sym("x")
  with pytest.raises(TypeError, match="truth value"):
    bool(x)
  with pytest.raises(TypeError, match="truth value"):
    _ = x and 1.0


def test_indexing_accepts_numpy_integers() -> None:
  x = sc.sym("x", (4, 3))
  row, block = x[np.int64(2)], x[np.int32(1) : np.int64(3), np.uint8(0)]
  assert row is x[2] and block is x[1:3, 0]
  assert [type(item) for item in row.attrs["index"][:1] + (block.attrs["index"][0].stop, block.attrs["index"][1])] == [int, int, int]
  for bad in [True, np.True_, 1.0]:
    with pytest.raises(TypeError, match="index"):
      _ = x[bad]
