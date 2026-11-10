from __future__ import annotations

import numpy as np
import pytest

from scaly.function.concrete import ConcreteFunction
from scaly.function.model import as_concrete
from scaly.function.sugar import _mapped_call
import scaly as sc
from scaly.ir.expr import Expr, ExprOp, substitute
from scaly.ir.types import TensorType, frozen
from scaly.passes.expr import cse_many


@pytest.mark.parametrize(
  ("first", "second"),
  [
    ([13.0, 17.0], [19.0, 23.0]),
    ([0.0, 13.0], [-0.0, 13.0]),
    (
      np.array([0x7FF8000000000001, 0x3FF0000000000000], dtype=np.uint64).view(np.float64),
      np.array([0x7FF8000000000002, 0x3FF0000000000000], dtype=np.uint64).view(np.float64),
    ),
  ],
)
def test_constant_keys_compare_bytes_when_digests_collide(monkeypatch, first, second) -> None:
  from scaly.ir.expr import hashlib

  blake2b = hashlib.blake2b
  monkeypatch.setattr(hashlib, "blake2b", lambda data, **kwargs: blake2b(b"", **kwargs))
  a, b = sc.const(first), sc.const(second)
  assert a is not b
  assert a.structural_key() != b.structural_key()
  assert cse_many([a, b]) == (a, b)
  assert sc.const(np.array(first)) is a
  assert sc.const(np.array(second)) is b
  for node, expected in ((a, first), (b, second)):
    assert node.value is not None
    np.testing.assert_array_equal(node.value.view(np.uint64), np.asarray(expected).view(np.uint64))


def test_colliding_array_attributes_and_constants_evaluate_separately(monkeypatch) -> None:
  from scaly.ir.expr import hashlib

  blake2b = hashlib.blake2b
  monkeypatch.setattr(hashlib, "blake2b", lambda data, **kwargs: blake2b(b"", **kwargs))
  x = sc.sym("collision_x", 3)
  forward, backward = x.gather([0, 2]), x.gather([2, 0])
  assert forward is not backward
  assert forward.structural_key() != backward.structural_key()
  assert x.gather([0, 2]) is forward
  a, b = sc.const([13.0, 17.0]), sc.const([19.0, 23.0])
  outputs = cse_many([a, b, forward, backward])
  assert outputs == (a, b, forward, backward)
  fn = ConcreteFunction._from_exprs("constant_digest_collision", [x], outputs, ["x"], ["a", "b", "forward", "backward"])
  values = np.array([2.0, 3.0, 5.0])
  for actual, expected in zip(fn(values), ([13.0, 17.0], [19.0, 23.0], values[[0, 2]], values[[2, 0]]), strict=True):
    np.testing.assert_array_equal(actual, expected)


def test_array_keys_serialize_only_compacted_bytes(monkeypatch) -> None:
  from scaly.ir.expr import _array_key, _attrs_key, hashlib

  blake2b = hashlib.blake2b
  serialized = []

  def digest(data, **kwargs):
    assert len(data) == 16
    serialized.append(data)
    return blake2b(data, **kwargs)

  monkeypatch.setattr(hashlib, "blake2b", digest)
  tile = np.array([13.0, 17.0])
  large = np.broadcast_to(tile, (100_000, 100_000, 2))
  constant = sc.const(large)
  constant.structural_key()
  assert cse_many([constant]) == (constant,)
  assert len(serialized) == 1
  attributed = Expr(ExprOp.NEG, (sc.sym("compact_attrs", 2),), TensorType((2,)), attrs={"tile": large})
  attributed.structural_key()
  assert cse_many([attributed]) == (attributed,)
  assert len(serialized) == 2
  assert _attrs_key({"tile": large}) == (("tile", ("ndarray", _array_key(large))),)
  repeated = np.broadcast_to(tile, (4, 3, 2))
  assert _array_key(repeated) == _array_key(repeated.copy())
  assert serialized and all(data == tile.tobytes() for data in serialized)


def test_uniform_constants_keep_compact_immutable_storage_and_bitwise_keys() -> None:
  from scaly.ir.expr import _uniform, zeros_like

  shape = (100_000, 100_000)
  zero = zeros_like(sc.sym("uniform_large", shape, dtype=sc.dtypes.float32))
  assert zero.type.dtype == sc.dtypes.float64
  assert zero.value is not None and zero.value.strides == (0, 0)
  assert not zero.value.flags.writeable and zero.support == frozenset()
  assert zero.structural_key() is zero.structural_key()
  assert sc.cse(zero) is zero
  assert _uniform(-0.0, (3,)) is sc.const(np.full(3, -0.0))
  assert _uniform(0.0, (3,)) is not _uniform(-0.0, (3,))
  tiled = np.broadcast_to([1.0, -2.0], (4, 3, 2))
  assert sc.const(tiled) is sc.const(tiled.copy())
  source = np.array([7.0])
  constant = sc.const(np.broadcast_to(source, shape))
  source[0] = 9
  assert constant.value is not None and constant.value[99_999, 99_999] == 7


@pytest.mark.parametrize("dense_first", [False, True])
def test_uniform_constant_storage_does_not_depend_on_creation_order(dense_first) -> None:
  values = np.full(2048 + dense_first, 0.1)
  view = np.broadcast_to(values[:1], values.shape)
  first, second = (values, view) if dense_first else (view, values)
  constant = sc.const(first)
  assert constant is sc.const(second)
  assert constant.value is not None and constant.value.strides == (0,)
  assert constant.support is None
  assert sc.simplify(constant.sum()).value == 0.1 * values.size


@pytest.mark.parametrize("index", [(3, slice(None)), (-4, slice(None)), (slice(None), 0), (slice(None, None, 0), slice(None))])
def test_slice_rejects_invalid_indices_without_a_shape_allocation(index) -> None:
  shape = (3, 0)
  with pytest.raises((IndexError, ValueError)):
    sc.sym("invalid_slice", shape)[index]


def test_binary_nonlinear_method_helpers_eval() -> None:
  @sc.function(sc.group(sc.arg("x", 3), sc.arg("y", 3)), outputs=sc.group(sc.arg("atan"), sc.arg("min"), sc.arg("max")), name="binary_helpers")
  def f(xy: tuple[sc.Expr, sc.Expr]) -> tuple[sc.Expr, sc.Expr, sc.Expr]:
    x, y = xy
    return x.atan2(y), x.minimum(y), x.maximum(y)

  atan, mn, mx = as_concrete(f).outputs
  xv = np.array([0.5, -1.0, 2.0])
  yv = np.array([1.5, 2.0, -0.25])

  atan_v, mn_v, mx_v = f((xv, yv))
  np.testing.assert_allclose(atan_v, np.arctan2(xv, yv))
  np.testing.assert_allclose(mn_v, np.minimum(xv, yv))
  np.testing.assert_allclose(mx_v, np.maximum(xv, yv))
  assert atan.type.diff
  assert not mn.type.diff
  assert not mx.type.diff


def test_dot_sumsqr_and_norm_2() -> None:
  @sc.function(sc.arg("x", (2, 2)), outputs=sc.group(sc.arg("dot"), sc.arg("sumsqr"), sc.arg("norm")))
  def f(x: sc.Expr) -> tuple[sc.Expr, sc.Expr, sc.Expr]:
    return sc.dot(x, x.T), x.sumsqr(), sc.norm_2(x)

  xv = np.array([[1.0, 2.0], [3.0, 4.0]])

  dot_val, sumsqr_val, norm_val = f(xv)
  np.testing.assert_allclose(dot_val, np.dot(xv.reshape(-1), xv.T.reshape(-1)))
  np.testing.assert_allclose(sumsqr_val, np.sum(xv * xv))
  np.testing.assert_allclose(norm_val, np.linalg.norm(xv.reshape(-1)))

  try:
    _ = sc.dot(sc.sym("a", 2), sc.sym("b", 3))
  except ValueError as e:
    assert "dot size mismatch" in str(e)
  else:  # pragma: no cover
    raise AssertionError("dot size mismatch should fail")


def test_shape_checks_for_structural_ops() -> None:
  x = sc.sym("x", (2, 3))
  y = sc.sym("y", (4, 2))

  try:
    _ = x @ y
  except ValueError as e:
    assert "cannot matmul shapes (2, 3) and (4, 2)" in str(e)
  else:  # pragma: no cover
    raise AssertionError("invalid matmul should fail")

  try:
    _ = x.transpose((0, 0))
  except ValueError as e:
    assert "not a permutation" in str(e)
  else:  # pragma: no cover
    raise AssertionError("invalid transpose should fail")

  try:
    _ = sc.concat([x, sc.sym("z", (2, 4))], axis=0)
  except ValueError as e:
    assert "cannot concat shapes" in str(e)
  else:  # pragma: no cover
    raise AssertionError("invalid concat should fail")


def test_debug_printing_uses_stable_topological_names() -> None:
  x = sc.sym("x", 2)
  text = ((x + 1.0) * x).debug()

  assert "%0 = input x : float64(2,)" in text
  assert "add(%0, %1)" in text
  assert text.endswith("outputs %3")


def test_structural_equality_collapses_to_identity() -> None:
  # Construction-time interning: two ``Expr``s with the same structural key are the same
  # Python object, so structural equality is the same as ``is`` equality.
  x0 = sc.sym("x", 2)
  x1 = sc.sym("x", 2)
  y = sc.sym("y", 2)

  assert x0 is x1
  assert x0.id == x1.id
  assert x0.structurally_equal(x1)
  assert x0.structural_hash() == x1.structural_hash()
  assert x0 is not y
  assert not x0.structurally_equal(y)


def test_interning_hit_returns_the_node_as_it_was_built() -> None:
  """The key matches values that are equal and not identical, so a hit must not assign them:
  Functions already hold the node."""
  x = sc.sym("intern_x", 2)
  node = Expr(ExprOp.NEG, (x,), TensorType((2,)), attrs={"axis": 0, "scale": np.float64(1.5)})
  attrs, key = node.attrs, node.structural_key()

  again = Expr(ExprOp.NEG, (x,), TensorType((2,)), attrs={"axis": np.int64(0), "scale": 1.5})

  assert again is node and node.attrs is attrs and node.structural_key() is key
  assert type(attrs["axis"]) is int and type(attrs["scale"]) is np.float64


def test_constants_and_attributes_are_frozen_copies() -> None:
  given = np.array([1.0, 2.0])
  c = sc.const(given)
  given[0] = 5.0
  assert c.value is not None and c.value.tolist() == [1.0, 2.0] and not c.value.flags.writeable
  assert sc.const(np.array([1.0, 2.0])) is c and sc.const(given) is not c

  indices = np.array([1, 0])
  g = sc.sym("intern_x", 2).gather(indices)
  indices[0] = 0
  assert g.attrs["indices"].tolist() == [1, 0] and not g.attrs["indices"].flags.writeable
  with pytest.raises(TypeError):
    g.attrs["indices"] = indices  # ty: ignore[invalid-assignment]


@pytest.mark.parametrize(
  ("first", "second"),
  [(0.0, -0.0), (1, True), (1, 1.0), (1.0, True), ((0.0, 1), (-0.0, 1)), ({"a": 1}, {"a": True})],
)
def test_interning_tells_apart_attributes_that_differ_in_bits_or_kind(first, second) -> None:
  """``==`` holds between each pair, and generated code can tell them apart."""
  assert first == second
  x = sc.sym("intern_x", 2)
  a = Expr(ExprOp.NEG, (x,), TensorType((2,)), attrs={"v": first})
  b = Expr(ExprOp.NEG, (x,), TensorType((2,)), attrs={"v": second})

  assert a is not b and not a.structurally_equal(b)
  assert repr(a.attrs["v"]) == repr(frozen(first)) and repr(b.attrs["v"]) == repr(frozen(second))
  assert len({n.id for n in cse_many([a, b])}) == 2
  assert Expr(ExprOp.NEG, (x,), TensorType((2,)), attrs={"v": first}) is a


def test_interning_shares_attributes_with_the_same_bits() -> None:
  x = sc.sym("intern_x", 2)

  def node(v):
    return Expr(ExprOp.NEG, (x,), TensorType((2,)), attrs={"v": v})

  nan = node(float("nan"))
  assert node(float("nan")) is nan
  assert node(-float("nan")) is not nan
  assert node(np.float64(1.5)) is node(1.5) and node(np.float32(1.5)) is node(1.5)
  assert node(np.int64(3)) is node(3) and node(np.True_) is node(True)


def test_differentiability_metadata_propagates_through_exprs() -> None:
  x = sc.sym("x", 3)
  p = sc.sym("p", 3, diff=False)
  c = sc.const([1.0, 2.0, 3.0])

  assert x.type.diff
  assert not p.type.diff
  assert not c.type.diff
  assert (x + c).type.diff
  assert not (p + c).type.diff
  assert x.reshape((3, 1)).T.type.diff
  assert x.gather([2, 0]).type.diff
  assert sc.scatter(p.gather([1, 2]), [0, 2], 3).type.diff is False
  assert sc.stack([p, c]).type.diff is False
  assert sc.concat([x[:1], p[:1]]).type.diff
  assert not x.floor().type.diff
  assert not sc.minimum(x, p).type.diff

  @sc.function(sc.arg("u", 3), outputs=sc.arg("y"))
  def inner(u: sc.Expr) -> sc.Expr:
    return u * u

  diff_call = inner(x)
  const_call = inner(c)
  assert diff_call.type.diff
  assert not const_call.type.diff


def test_substitute_rebuilds_changed_ancestors_and_preserves_sharing() -> None:
  x = sc.sym("sub_x", 3)
  z = sc.sym("sub_z", 3)
  expr = (x * x + x).scalar()

  actual = substitute(expr, {x: z})
  expected = (z * z + z).scalar()

  assert actual is expected
  assert actual.lowering == expr.lowering
  assert substitute(expr, {}) is expr


def test_substitute_rejects_incompatible_shape_or_dtype() -> None:
  x = sc.sym("sub_bad_x", 2)
  with pytest.raises(ValueError, match="cannot substitute"):
    substitute(x + 1.0, {x: sc.sym("sub_bad_z", 3)})


def test_substitute_rebuilds_call_and_vmap_actuals_without_entering_callees() -> None:
  @sc.function(sc.arg("u", 2), outputs=sc.arg("y"), name="sub_callee")
  def callee(formal: sc.Expr) -> sc.Expr:
    return formal * formal

  x = sc.sym("sub_actual_x", 2)
  z = sc.sym("sub_actual_z", 2)
  called = callee(x)
  rewritten_call = substitute(called, {x: z})
  assert rewritten_call is callee(z)
  assert rewritten_call.attrs["callee"] is callee.instantiate()

  xs = sc.sym("sub_vmap_x", 4)
  zs = sc.sym("sub_vmap_z", 4)
  mapped = _mapped_call(callee, 2, [(xs, 0, 2)])
  rewritten_vmap = substitute(mapped, {xs: zs})
  assert rewritten_vmap is _mapped_call(callee, 2, [(zs, 0, 2)])
  assert rewritten_vmap.attrs["callee"] is callee.instantiate()


@pytest.mark.parametrize(
  ("fmt", "values", "error", "message"),
  [
    ("x", (), ValueError, "at least one value"),
    ("x={} y={}", (sc.sym("x"),), ValueError, "2 placeholders for 1 value"),
    ("x={0}", (sc.sym("x"),), ValueError, "only empty '{}' placeholders"),
    ("x={:.3f}", (sc.sym("x"),), ValueError, "only empty '{}' placeholders"),
    ("x={}", (sc.sym("i", dtype="int64"),), TypeError, "float64"),
    ("x=\0{}", (sc.sym("x"),), ValueError, "NUL"),
  ],
)
def test_print_rejects_a_format_it_cannot_render(fmt, values, error, message) -> None:
  with pytest.raises(error, match=message):
    sc.print(fmt, *values)
