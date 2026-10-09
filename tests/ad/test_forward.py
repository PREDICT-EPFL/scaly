"""The forward traversal shares seeds, preserves typed zeros, and stays iterative."""

from __future__ import annotations

import numpy as np
import pytest

import scaly as sc
from scaly.function.model import as_concrete
from scaly.ir.expr import ExprOp, topo


def _wrap_linear(inner, name):
  @sc.function(sc.arg("linear_u", 3), outputs=sc.arg("y"), name=name)
  def wrapper(u):
    return inner(u)

  return wrapper.symbolic_call


@pytest.mark.parametrize("nseed", [0, 1, 3])
def test_joint_inputs_and_outputs_against_numpy(nseed):
  from scaly.ad.forward import _pushforward

  @sc.function(
    sc.group(sc.arg("joint_s", ()), sc.arg("joint_m", (2, 3)), sc.arg("joint_v", 4)),
    outputs=sc.group(sc.arg("a"), sc.arg("b"), sc.arg("c")),
  )
  def products(inputs):
    s, m, v = inputs
    outputs = ((m * s).sin() @ v[:3], sc.stack([s * v.sum(), (m.sum() * v[3]).exp()]), (s * s).reshape((1,)))
    seeds = {x: sc.const(np.random.default_rng(i).normal(size=(nseed, *x.shape))) for i, x in enumerate(inputs)}
    return _pushforward(outputs, seeds, nseed)

  s, m, v = np.array(0.3), np.arange(6.0).reshape(2, 3) / 5, np.linspace(-0.2, 0.6, 4)
  ds, dm, dv = [np.random.default_rng(i).normal(size=(nseed, *x.shape)) for i, x in enumerate((s, m, v))]
  expected = (
    np.stack([((np.cos(m * s) * (m * ds[i] + s * dm[i])) @ v[:3] + np.sin(m * s) @ dv[i, :3]) for i in range(nseed)]) if nseed else np.empty((0, 2)),
    np.stack([ds * v.sum() + s * dv.sum(axis=1), np.exp(m.sum() * v[3]) * (dm.sum(axis=(1, 2)) * v[3] + m.sum() * dv[:, 3])], axis=1),
    (2 * s * ds).reshape(nseed, 1),
  )
  for actual, reference in zip(products((s, m, v)), expected, strict=True):
    np.testing.assert_allclose(actual, reference, atol=1e-12, rtol=1e-12)


def test_forward_handles_deep_expression_graph():
  x = sc.sym("deep_forward_x", ())
  value = x
  for _ in range(2000):
    value = value + x
  tangent = sc.jvp_many(value, x, sc.const([1.0, -0.5]))

  @sc.function(sc.arg("deep_forward_x", ()), outputs=sc.arg("dy"))
  def fn(x):
    return tangent

  np.testing.assert_array_equal(fn(np.array(0.2)), [2001.0, -1000.5])


@pytest.mark.parametrize("case", ["scalar_times", "times_const", "over_const", "matmul"])
@pytest.mark.parametrize("depth", [0, 1, 3])
def test_linear_tangent_never_depends_on_primal(case, depth):
  constants = np.array([2.0, -3.0, 0.5])
  matrix = np.array([[1.0, 2.0, 0.0], [0.0, -1.0, 4.0], [3.0, 0.0, 1.0]])
  build = {
    "scalar_times": lambda u: 2.0 * u,
    "times_const": lambda u: u * sc.const(constants),
    "over_const": lambda u: u / sc.const(constants),
    "matmul": lambda u: sc.const(matrix) @ u,
  }[case]
  reference = {"scalar_times": 2 * np.eye(3), "times_const": np.diag(constants), "over_const": np.diag(1 / constants), "matmul": matrix}[case]
  for level in range(depth):
    build = _wrap_linear(build, f"linear_{case}_{level}")

  @sc.function(sc.group(sc.arg("linear_u", 3), sc.arg("linear_seed", (2, 3))), outputs=sc.group(sc.arg("one"), sc.arg("many")))
  def products(inputs):
    u, seed = inputs
    y = build(u)
    return sc.jvp(y, u, seed[0]), sc.jvp_many(y, u, seed)

  u = as_concrete(products).inputs[0]
  for tangent in as_concrete(products).outputs:
    assert sc.jacobian_sparsity(tangent, u).nnz == 0
  seeds = np.arange(6.0).reshape(2, 3) / 5
  one, many = products((np.array([0.3, -1.2, 2.0]), seeds))
  np.testing.assert_allclose(one, seeds[0] @ reference.T)
  np.testing.assert_allclose(many, seeds @ reference.T)


@pytest.mark.parametrize("dtype", [sc.dtypes.float32, sc.dtypes.float64])
def test_zero_tangents_and_empty_seed_axis_keep_dtype(dtype):
  x = sc.sym("typed_forward", (2, 3), dtype=dtype)
  for nseed in (0, 2):
    seeds = sc.const(np.zeros((nseed, *x.shape), dtype=dtype.numpy()), dtype=dtype)
    tangent = sc.jvp_many(x.log(), x, seeds)
    assert tangent.type.dtype == dtype
    assert tangent.shape == (nseed, *x.shape)
    assert tangent.op == ExprOp.CONST
    np.testing.assert_array_equal(tangent.value, np.zeros(tangent.shape))
  seed = sc.sym("typed_seed", x.shape, dtype=dtype)
  tangent = sc.jvp(x * sc.const(2, dtype=dtype), x, seed)
  assert all(node.type.dtype == dtype for node in topo((tangent,)))


@pytest.mark.parametrize("dtype", [sc.dtypes.int32, sc.dtypes.int64, sc.dtypes.bool_])
def test_nonfloating_values_have_no_tangent(dtype):
  from scaly.ad.forward import _pushforward

  x = sc.sym("no_tangent", 2, dtype=dtype)
  assert _pushforward((x, x.reshape((1, 2))), {}, 3) == (None, None)
  with pytest.raises(TypeError, match="no tangent"):
    sc.jvp_many(x, x, sc.const(np.ones((3, 2)), dtype=dtype))


@pytest.mark.parametrize("side", ["left", "right", "both"])
def test_matmul_graph_does_not_grow_with_seed_count(side):
  from scaly.ad.forward import _pushforward

  x, y = sc.sym("folded_x", (2, 3)), sc.sym("folded_y", (3, 4))
  counts = []
  for nseed in (1, 7):
    seeds = {}
    if side != "right":
      seeds[x] = sc.sym("folded_dx", (nseed, *x.shape))
    if side != "left":
      seeds[y] = sc.sym("folded_dy", (nseed, *y.shape))
    (tangent,) = _pushforward((x @ y,), seeds, nseed)
    assert tangent is not None
    nodes = topo((tangent,))
    assert not any(node.op in {ExprOp.SLICE, ExprOp.STACK} for node in nodes)
    counts.append(len(nodes))
  assert counts[0] == counts[1]


@pytest.mark.parametrize("mapped", [False, True])
def test_missing_rule_names_the_original_callee(mapped):
  @sc.function(sc.arg("refused_body_x", 2), outputs=sc.arg("y"), name="refused_leaf")
  def leaf(x):
    return x.floor()

  x = sc.sym("refused_outer_x", 4 if mapped else 2)
  y = sc.vmap(leaf, 2)(x).vec() if mapped else leaf(x)
  with pytest.raises(NotImplementedError, match=r"FLOOR.*refused_leaf"):
    sc.jvp_many(y, x, sc.sym("refused_outer_seed", (3, *x.shape)))


@pytest.mark.parametrize("nseed", [1, 7])
def test_primal_alignment_shares_one_cosine_without_seed_copies(nseed):
  x = sc.sym("aligned_forward_x", (2, 3))
  tangent = sc.jvp_many(x.sin(), x, sc.sym("aligned_forward_seed", (nseed, 2, 3)))
  nodes = topo((tangent,))
  assert sum(node.op == ExprOp.COS for node in nodes) == 1
  assert not any(node.op == ExprOp.STACK for node in nodes)
  (aligned,) = [node for node in nodes if node.op == ExprOp.RESHAPE]
  assert aligned.shape == (1, 2, 3)


@pytest.mark.parametrize("shapes", [((3,), (3,)), ((2, 3), (3,)), ((3,), (3, 4)), ((2, 3), (3, 4))])
def test_joint_matmul_both_operands_against_numpy(shapes):
  from scaly.ad.forward import _pushforward

  sx, sy = shapes
  rng = np.random.default_rng(19)
  xv, yv = rng.normal(size=sx), rng.normal(size=sy)
  dx, dy = rng.normal(size=(3, *sx)), rng.normal(size=(3, *sy))

  @sc.function(sc.group(sc.arg("mm_x", sx), sc.arg("mm_y", sy)), outputs=sc.arg("dy"))
  def products(inputs):
    x, y = inputs
    (tangent,) = _pushforward((x @ y,), {x: sc.const(dx), y: sc.const(dy)}, 3)
    assert tangent is not None
    return tangent

  expected = np.stack([dx[i] @ yv + xv @ dy[i] for i in range(3)])
  np.testing.assert_allclose(products((xv, yv)), expected, atol=1e-12, rtol=1e-12)


def test_runtime_callee_seeds_use_one_body_traversal():
  from scaly.ad.calls import _call_jvp_many_function
  from scaly.ad.forward import _pushforward

  @sc.function(sc.group(sc.arg("body_x", (2, 3)), sc.arg("body_y", (3, 4))), outputs=sc.arg("y"), name="body_shares_seeds")
  def stage(inputs):
    x, y = inputs
    return (x @ y).sin()

  counts = []
  for nseed in (1, 7):
    helper, _, _, _ = _call_jvp_many_function(stage.instantiate(), 0, (0, 1), nseed, (None, None), pushforward=_pushforward)
    nodes = topo(helper.outputs)
    assert not any(node.op in {ExprOp.SLICE, ExprOp.STACK} for node in nodes)
    assert sum(node.op == ExprOp.COS for node in nodes) == 1
    counts.append(len(nodes))
  assert counts[0] == counts[1]


def test_print_differentiates_only_its_returned_value():
  x = sc.sym("print_tangent_x", 2)
  y = sc.print("returned {} diagnostic {}", x.sin(), x.floor())
  tangent = sc.jvp_many(y, x, sc.const(np.eye(2)))

  @sc.function(sc.arg("print_tangent_x", 2), outputs=sc.arg("dy"))
  def products(x):
    return tangent

  point = np.array([0.2, -0.7])
  np.testing.assert_allclose(products(point), np.diag(np.cos(point)))
