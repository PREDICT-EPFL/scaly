from __future__ import annotations

import numpy as np
import pytest

import scaly as sc
from scaly.ir.expr import Expr, ExprOp, substitute, topo
from scaly.ir.types import TensorType
from scaly.passes.expr import cse_many


def test_elementwise_eval_and_topological_order() -> None:
  x = sc.sym("x", 3)
  y = (x.sin() + x * x).sum()
  f = sc.Function.from_exprs("f", [x], [y], ["x"], ["y"])

  np.testing.assert_allclose(f(np.array([1.0, 2.0, 3.0])), np.sin([1.0, 2.0, 3.0]).sum() + 14.0)

  nodes = topo(f.outputs)
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
    sc.ExprOp.SEGMENT_REDUCE,
    sc.ExprOp.CONCAT,
    sc.ExprOp.MATMUL,
    sc.ExprOp.CALL,
  ]:
    assert op in sc.COMMON_OPS
  assert sc.ExprOp.SIN.value == "sin"
  assert sc.ExprOp.ERF.value == "erf"


def test_binary_nonlinear_method_helpers_eval() -> None:
  x = sc.sym("x", 3)
  y = sc.sym("y", 3)
  atan = x.atan2(y)
  mn = x.minimum(y)
  mx = x.maximum(y)
  f = sc.Function.from_exprs("binary_helpers", [x, y], [atan, mn, mx], ["x", "y"], ["atan", "min", "max"])
  xv = np.array([0.5, -1.0, 2.0])
  yv = np.array([1.5, 2.0, -0.25])

  atan_v, mn_v, mx_v = f((xv, yv))
  np.testing.assert_allclose(atan_v, np.arctan2(xv, yv))
  np.testing.assert_allclose(mn_v, np.minimum(xv, yv))
  np.testing.assert_allclose(mx_v, np.maximum(xv, yv))
  assert atan.type.diff
  assert mn.type.diff
  assert mx.type.diff


def test_dot_sumsqr_and_norm_2() -> None:
  x = sc.sym("x", (2, 2))
  f = sc.Function.from_exprs("f", [x], [sc.dot(x, x.T), x.sumsqr(), sc.norm_2(x)], ["x"], ["dot", "sumsqr", "norm"])
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
  """The interning key matches values that are equal and not identical, so a hit must not assign
  them: Functions already hold the node."""
  x = sc.sym("intern_x", 2)
  args, type_, attrs = (x,), TensorType((2,)), {"axis": 0, "table": [1, 2], "scale": np.float64(1.5)}
  node = Expr(ExprOp.NEG, args, type_, attrs=attrs)
  value = np.array([1.0, -0.0])
  const = Expr(ExprOp.CONST, type=TensorType((2,), diff=False), value=value)
  key = node.structural_key()

  again = Expr(ExprOp.NEG, (x,), TensorType((2,)), attrs={"axis": np.int64(0), "table": (1, 2), "scale": 1.5})
  const_again = Expr(ExprOp.CONST, type=TensorType((2,), diff=False), value=np.array([1.0, -0.0]))

  assert again is node and const_again is const
  assert node.args is args and node.type is type_ and node.attrs is attrs
  assert type(attrs["table"]) is list and type(attrs["axis"]) is int and type(attrs["scale"]) is np.float64
  assert const.value is value and const.type.diff is False
  assert node.structural_key() is key


@pytest.mark.parametrize(
  ("first", "second"),
  [(0.0, -0.0), (-0.0, 0.0), (1, True), (True, 1), (1, 1.0), (1.0, True), (0, False), (0.0, False), ((0.0, 1), (-0.0, 1)), ({"a": 1}, {"a": True})],
)
def test_interning_tells_apart_attributes_that_differ_in_bits_or_kind(first, second) -> None:
  """``==`` holds between each pair, and generated code can tell them apart: a sign bit, or an
  integer where there was a flag or a real."""
  assert first == second
  x = sc.sym("intern_x", 2)
  a = Expr(ExprOp.NEG, (x,), TensorType((2,)), attrs={"v": first})
  b = Expr(ExprOp.NEG, (x,), TensorType((2,)), attrs={"v": second})

  assert a is not b and not a.structurally_equal(b)
  assert a.attrs["v"] is first and b.attrs["v"] is second
  assert len({n.id for n in cse_many([a, b])}) == 2  # the pass's own key keeps them apart too
  assert Expr(ExprOp.NEG, (x,), TensorType((2,)), attrs={"v": first}) is a


def test_interning_shares_attributes_with_the_same_bits() -> None:
  x = sc.sym("intern_x", 2)

  def node(v):
    return Expr(ExprOp.NEG, (x,), TensorType((2,)), attrs={"v": v})

  nan = node(float("nan"))
  assert node(float("nan")) is nan  # by bits, where ``nan == nan`` is false
  assert node(-float("nan")) is not nan
  assert node(np.float64(1.5)) is node(1.5) and node(np.float32(1.5)) is node(1.5)
  assert node(np.int64(3)) is node(3) and node(np.True_) is node(True)


def test_function_keeps_its_fill_after_one_with_the_other_zero_is_built() -> None:
  """A ``take`` filling with ``0.0`` and one filling with ``-0.0`` are two nodes. They were one,
  rewritten in place by the second construction, so the Function built first changed its sign."""
  x, idx = sc.sym("intern_x", 3), sc.sym("intern_idx", 2, dtype="int64")
  xv, iv = np.array([1.0, 2.0, 3.0]), np.array([0.0, 7.0])  # the second index is out of range

  def fn(name, *outputs):
    return sc.Function.from_exprs(name, [x, idx], list(outputs), ["x", "idx"], [f"o{k}" for k in range(len(outputs))])

  plus = sc.take(x, idx, fill=0.0)
  f_plus = fn("intern_fill_plus", plus)  # built, and not compiled before the other node exists
  minus = sc.take(x, idx, fill=-0.0)
  f_minus = fn("intern_fill_minus", minus)
  f_both = fn("intern_fill_both", plus, minus)

  def signs(f):
    out = f._flat_numerical_call(xv, iv)
    np.testing.assert_array_equal([o[0] for o in out], 1.0)
    return [bool(np.signbit(o[1])) for o in out]

  assert signs(f_plus) == [False]
  assert signs(f_minus) == [True]
  assert signs(f_both) == [False, True]
  assert plus is not minus and plus.attrs["fill"] == 0.0 and not np.signbit(plus.attrs["fill"])


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
  assert x.floor().type.diff  # a zero derivative, following the nonsmooth option like minimum's
  assert sc.minimum(x, p).type.diff  # differentiable since the nonsmooth option (tie convention) exists
  assert not sc.minimum(p, c).type.diff

  u = sc.sym("u", 3)
  inner = sc.Function.from_exprs("inner", [u], [u * u], ["u"], ["y"])
  diff_call = inner(x)
  const_call = inner(c)
  assert diff_call.type.diff
  assert not const_call.type.diff


def test_mixed_lowering_hints_survive_expr_graph() -> None:
  x = sc.sym("x", 3)
  scalar_region = (x.sin() + x * x).scalar()
  block_region = (sc.const(np.eye(3)) @ x).block()
  opaque_region = (x + 1.0).opaque()
  f = sc.Function.from_exprs("mixed", [x], [scalar_region + block_region + opaque_region], ["x"], ["y"])

  lowerings = [e.lowering for e in topo(f.outputs) if e.lowering != "auto"]
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
  formal = sc.sym("sub_formal", 2)
  callee = sc.Function.from_exprs("sub_callee", [formal], [formal * formal], ["u"], ["y"])
  x = sc.sym("sub_actual_x", 2)
  z = sc.sym("sub_actual_z", 2)
  called = callee(x)
  rewritten_call = substitute(called, {x: z})
  assert rewritten_call is callee(z)
  assert rewritten_call.attrs["callee"] is callee

  xs = sc.sym("sub_vmap_x", 4)
  zs = sc.sym("sub_vmap_z", 4)
  mapped = sc.vmap(callee, 2, [(xs, 0, 2)])
  rewritten_vmap = substitute(mapped, {xs: zs})
  assert rewritten_vmap is sc.vmap(callee, 2, [(zs, 0, 2)])
  assert rewritten_vmap.attrs["callee"] is callee
