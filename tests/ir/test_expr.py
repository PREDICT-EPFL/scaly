from __future__ import annotations

import numpy as np
import pytest

from scaly.function.model import as_concrete
from scaly.function.sugar import _mapped_call
import scaly as sc
from scaly.ir.expr import Expr, ExprOp, substitute, topo
from scaly.ir.types import TensorType, frozen
from scaly.passes.expr import cse_many


def test_elementwise_eval_and_topological_order() -> None:
  @sc.function(sc.arg("x", 3), outputs=sc.arg("y"))
  def f(x: sc.Expr) -> sc.Expr:
    return (x.sin() + x * x).sum()

  np.testing.assert_allclose(f(np.array([1.0, 2.0, 3.0])), np.sin([1.0, 2.0, 3.0]).sum() + 14.0)

  nodes = topo(as_concrete(f).outputs)
  loc = {e.id: i for i, e in enumerate(nodes)}
  assert nodes[-1].op == sc.ExprOp.SUM
  assert [e.op for e in nodes].count(sc.ExprOp.INPUT) == 1
  assert all(loc[arg.id] < loc[e.id] for e in nodes for arg in e.args)


def test_common_ops_contains_modeling_basics() -> None:
  for op in [
    sc.ExprOp.SIN,
    sc.ExprOp.COS,
    sc.ExprOp.TAN,
    sc.ExprOp.ATAN2,
    sc.ExprOp.SINH,
    sc.ExprOp.COSH,
    sc.ExprOp.TANH,
    sc.ExprOp.ERF,
    sc.ExprOp.EXP,
    sc.ExprOp.LOG,
    sc.ExprOp.SQRT,
    sc.ExprOp.TRANSPOSE,
    sc.ExprOp.SLICE,
    sc.ExprOp.GATHER,
    sc.ExprOp.SCATTER,
    sc.ExprOp.CONCAT,
    sc.ExprOp.MATMUL,
    sc.ExprOp.CALL,
  ]:
    assert op in sc.COMMON_OPS
  assert sc.ExprOp.SIN.value == "sin"
  assert sc.ExprOp.ERF.value == "erf"


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
  [(0.0, -0.0), (-0.0, 0.0), (1, True), (True, 1), (1, 1.0), (1.0, True), (0, False), ((0.0, 1), (-0.0, 1)), ({"a": 1}, {"a": True})],
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


def test_mixed_lowering_hints_survive_expr_graph() -> None:
  @sc.function(sc.arg("x", 3), outputs=sc.arg("y"), name="mixed")
  def f(x: sc.Expr) -> sc.Expr:
    scalar_region = (x.sin() + x * x).scalar()
    block_region = (sc.const(np.eye(3)) @ x).block()
    opaque_region = (x + 1.0).opaque()
    return scalar_region + block_region + opaque_region

  lowerings = [e.lowering for e in topo(as_concrete(f).outputs) if e.lowering != "auto"]
  assert "scalar" in lowerings
  assert "block" in lowerings
  assert "opaque" in lowerings


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


def test_print_is_an_identity_node_carrying_its_format() -> None:
  x, k = sc.sym("x", 2), sc.sym("k")
  y = sc.print("x={} k={}", x, k)
  assert y.op == ExprOp.PRINT and y.args == (x, k) and y.attrs["format"] == "x={} k={}"
  assert y.type == x.type
  assert sc.print("x={} k={}", x, k) is y
  assert sc.print("{{x}}={}", x).attrs["format"] == "{{x}}={}"
  sc.verify_expr(y)


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
