"""Dtype holes bind each leaf's dtype from the call, alongside shape and pattern holes."""

from typing import Any

import numpy as np
import pytest
from scipy import sparse

import scaly as sc
from scaly.codegen import render_c_source

f32, f64, i64 = sc.dtypes.float32, sc.dtypes.float64, sc.dtypes.int64


def test_a_dtype_hole_is_declared_with_an_ellipsis_only() -> None:
  leaf = sc.arg("x", dtype=...)
  assert leaf.has_holes
  with pytest.raises(TypeError, match="has shape, dtype or pattern holes"):
    leaf.types
  with pytest.raises(TypeError, match=r"dtype=\.\.\."):
    sc.arg("x", dtype=f32)  # ty: ignore[invalid-argument-type]
  assert sc.arg("x", 3, dtype=...).has_holes and sc.arg("x", 3, dtype=...).shapes == ((3,),)


def test_each_dtype_traces_once_and_shares_symbolic_and_numerical_instances() -> None:
  traces = []

  @sc.function(sc.arg("x", dtype=...))
  def double(x: sc.Expr) -> sc.Expr:
    traces.append(x.type.dtype)
    return x * 2 + 1

  for dtype in (np.float32, np.float64, np.int64, np.float32):
    value = np.arange(3, dtype=dtype)
    result: Any = double(value)
    assert result.dtype == dtype
    np.testing.assert_array_equal(result, value * dtype(2) + dtype(1))
  assert traces == [f32, f64, i64]
  names = sorted(instance.name for instance in double.instances.values())
  assert names == sorted(["double_3_f32_t772c30", "double_3_t772c30", "double_3_i64_t772c30"])
  symbol = sc.sym("y", 3, dtype=f32)
  instance = double.symbolic_call(symbol).attrs["callee"]
  assert instance.name == "double_3_f32_t772c30" and len(double.instances) == 3


def test_float64_instances_keep_the_names_of_a_plain_shape_hole() -> None:
  holed: Any = sc.function(sc.arg("x", dtype=...), name="same")(lambda x: x * 2)
  plain: Any = sc.function(sc.arg("x"), name="same")(lambda x: x * 2)
  holed(np.ones((2, 3)))
  plain(np.ones((2, 3)))
  assert [i.name for i in holed.instances.values()] == [i.name for i in plain.instances.values()] == ["same_2x3_t772c30"]
  fixed: Any = sc.function(sc.arg("x", (2, 3), dtype=...))(lambda x: x * 2)
  fixed(np.ones((2, 3), np.float32))
  fixed(np.ones((2, 3)))
  assert sorted(i.name for i in fixed.instances.values()) == ["<lambda>_f32_t772c30", "<lambda>_t772c30"]


def test_a_plain_arg_still_converts_to_float64() -> None:
  plain: Any = sc.function(sc.arg("x"))(lambda x: x * 2)
  assert plain(np.ones(2, np.float32)).dtype == np.float64
  assert plain(np.arange(2)).dtype == np.float64
  assert len(plain.instances) == 1


@pytest.mark.parametrize(("value", "dtype"), [(2.5, np.float64), (3, np.int64), (True, np.bool_), ([1.0, 2.0], np.float64)])
def test_python_values_bind_like_np_asarray(value: Any, dtype: type) -> None:
  identity: Any = sc.function(sc.arg("x", dtype=...))(lambda x: x)
  result = identity(value)
  assert result.dtype == np.asarray(value).dtype == dtype
  np.testing.assert_array_equal(result, np.asarray(value))


def test_mixed_float_leaves_raise_unless_the_body_casts() -> None:
  matrix, vector = np.eye(2, dtype=np.float32) * 3, np.array([1.0, 2.0])
  mixed: Any = sc.function(sc.arg("A", dtype=...), sc.arg("b", dtype=...))(lambda A, b: A @ b)
  with pytest.raises(TypeError, match="mixed dtypes float32 and float64; convert one with sc.cast"):
    mixed(matrix, vector)
  assert not mixed.instances
  cast: Any = sc.function(sc.arg("A", dtype=...), sc.arg("b", dtype=...))(lambda A, b: A @ sc.cast(b, A.type.dtype))
  result = cast(matrix, vector)
  assert result.dtype == np.float32
  np.testing.assert_array_equal(result, matrix @ vector.astype(np.float32))
  instance = next(iter(cast.instances.values()))
  assert instance.input_tree.types == (sc.TensorType((2, 2), f32), sc.TensorType((2,), f64))


def test_an_integer_binding_matches_a_concrete_integer_leaf() -> None:
  def body(n: sc.Expr, x: sc.Expr) -> sc.Expr:
    return (sc.cast(n, f64) * x).sum()

  holed: Any = sc.function(sc.arg("n", dtype=...), sc.arg("x"), outputs=sc.arg("y", ()))(body)
  fixed: Any = sc.function(sc.arg("n", sc.TensorType((3,), i64)), sc.arg("x", 3), outputs=sc.arg("y", ()))(body)
  n, x = np.array([1, 2, 3]), np.array([0.5, 1.0, 2.0])
  assert holed(n, x) == fixed(n, x) == 8.5
  assert next(iter(holed.instances.values())).input_tree.types == fixed.instantiate().input_tree.types
  for wrt, expected in (("x", n.astype(float)), ("n", np.zeros(3, np.int64))):
    holed_grad, fixed_grad = sc.gradient(holed, wrt=wrt)(n, x), sc.gradient(fixed, wrt=wrt)(n, x)
    assert holed_grad.dtype == fixed_grad.dtype == expected.dtype
    np.testing.assert_array_equal(holed_grad, expected)
    np.testing.assert_array_equal(fixed_grad, expected)
  mixing: Any = sc.function(sc.arg("n", dtype=...), sc.arg("x"))(lambda n, x: n * x)
  with pytest.raises(TypeError, match="mixed dtypes int64 and float64"):
    mixing(n, x)


def test_float32_values_and_derivatives_follow_numpy() -> None:
  @sc.function(sc.arg("x", dtype=...), outputs=sc.arg("y", (), dtype=...))
  def energy(x: sc.Expr) -> sc.Expr:
    return (x.sin() * x).sum()

  x = np.linspace(-1, 1, 5, dtype=np.float32)
  value: Any = energy(x)
  grad: Any = sc.gradient(energy)(x)
  assert value.dtype == grad.dtype == np.float32
  np.testing.assert_allclose(value, np.sum(np.sin(x) * x), rtol=1e-6)
  np.testing.assert_allclose(grad, np.cos(x) * x + np.sin(x), rtol=1e-6)
  assert len(energy.instances) == 1


def test_the_generated_entry_takes_the_bound_c_type() -> None:
  scale: Any = sc.function(sc.arg("x", dtype=...))(lambda x: x * 2)
  instance = scale.instantiate((sc.TensorType((4,), f32),))
  source = render_c_source(instance, lanes=1)
  assert "(const float* x, float* " in source and "const double*" not in source
  with pytest.raises(TypeError, match="a dtype hole needs a TensorType binding"):
    scale.instantiate(((4,),))
  assert scale(np.ones(4, np.float32)).dtype == np.float32 and len(scale.instances) == 1


def test_dtype_and_pattern_holes_bind_together() -> None:
  identity: Any = sc.function(sc.arg("x", (2, 3), pattern=..., dtype=...))(lambda x: x * 2)
  source = sparse.coo_array((np.array([1.0, 2.0], np.float32), ([0, 1], [2, 0])), shape=(2, 3))
  result: Any = identity(source)
  assert result.dtype == np.float32
  np.testing.assert_array_equal(result.toarray(), source.toarray() * 2)
  (instance,) = identity.instances.values()
  assert instance.input_tree.types[0].dtype == f32 and "_f32_" in instance.name and "_p" in instance.name


def test_dtype_holes_bind_through_vmap() -> None:
  scale: Any = sc.function(sc.arg("x", dtype=...))(lambda x: x * 3)
  mapped: Any = sc.vmap(scale, 4)
  values = np.arange(8, dtype=np.float32).reshape(4, 2)
  result = mapped(values)
  assert result.dtype == np.float32
  np.testing.assert_array_equal(result, values * 3)
  symbol = sc.sym("v", (4, 2), dtype=f32)
  assert mapped.symbolic_call(symbol).type.dtype == f32
  shared = mapped(sc.broadcast(np.arange(2, dtype=np.int64)))
  assert shared.dtype == np.int64
  np.testing.assert_array_equal(shared, np.broadcast_to(np.arange(2) * 3, (4, 2)))
  assert sorted(instance.name.split("_")[2] for instance in scale.instances.values()) == ["f32", "i64"]


def test_dtype_holes_bind_through_scan() -> None:
  step: Any = sc.function(sc.arg("c", dtype=...), sc.arg("x", dtype=...))(lambda c, x: (c + x, c * x))

  @sc.function(sc.arg("c0", sc.TensorType((2,), f32)), sc.arg("xs", sc.TensorType((3, 2), f32)))
  def run(c0: sc.Expr, xs: sc.Expr) -> Any:
    return sc.scan(step, c0, (xs,), length=3)

  c0, xs = np.array([1.0, -1.0], np.float32), np.arange(6, dtype=np.float32).reshape(3, 2)
  carry, (ys,) = run(c0, xs)
  expected_c, expected_y = c0.copy(), []
  for x in xs:
    expected_y.append(expected_c * x)
    expected_c = expected_c + x
  assert carry.dtype == ys.dtype == np.float32
  np.testing.assert_array_equal(carry, expected_c)
  np.testing.assert_array_equal(ys, np.stack(expected_y))
  (instance,) = step.instances.values()
  assert instance.input_tree.types == (sc.TensorType((2,), f32), sc.TensorType((2,), f32))


def test_outputs_may_leave_their_dtype_to_the_trace() -> None:
  half: Any = sc.function(sc.arg("x", dtype=...), outputs=sc.arg("y", 2, dtype=...))(lambda x: x / 2)
  assert half(np.ones(2, np.float32)).dtype == np.float32
  fixed: Any = sc.function(sc.arg("x", dtype=...), outputs=sc.arg("y", 2))(lambda x: x / 2)
  assert fixed(np.ones(2)).dtype == np.float64
  with pytest.raises(TypeError, match="expected dtype float64 for 'y', got float32"):
    fixed(np.ones(2, np.float32))
