"""Forward loop derivatives agree with unrolled graphs and finite differences."""

import numpy as np
import pytest

import scaly as sc
from scaly.function.concrete import ConcreteFunction


def evaluate(inputs, outputs, point):
  fn = ConcreteFunction._from_exprs("loop_forward", inputs, outputs, [x.name for x in inputs], [f"out{i}" for i in range(len(outputs))])
  return fn(*point)


@pytest.mark.parametrize("length", [0, 1, 4])
@pytest.mark.parametrize("reverse", [False, True])
def test_scan_forward_matches_unrolled(length, reverse):
  @sc.function(sc.arg("c", 2), sc.arg("x", ()), sc.arg("p", 2))
  def step(c, x, p):
    return (c * p + x).sin(), c * x

  c, xs, p = sc.sym("c", 2), sc.sym("xs", length), sc.sym("p", 2)
  final, (ys,) = sc.scan(step, c, (xs, sc.broadcast(p)), length=length, reverse=reverse)
  ref, rows = c, [None] * length
  for k in reversed(range(length)) if reverse else range(length):
    ref, rows[k] = step(ref, xs[k], p)
  stacked = sc.stack(rows) if length else sc.const(np.zeros((0, 2)))
  point = [np.array([0.2, -0.4]), np.linspace(0.1, 0.3, length), np.array([0.7, 0.9])]
  for wrt in (c, xs, p):
    for actual, expected in ((final, ref), (ys, stacked)):
      got, want = evaluate([c, xs, p], [sc.jacobian(actual, wrt), sc.jacobian(expected, wrt)], point)
      np.testing.assert_allclose(got, want, atol=1e-12)


def check_derivatives(inputs, outputs, point):
  from scaly.ad.derivatives import finite_difference

  for k, wrt in enumerate(inputs):
    if not wrt.type.dtype.is_floating:
      continue
    for out in outputs:
      seed = sc.const(np.ones(wrt.shape), dtype=wrt.type.dtype)
      jac, single = evaluate(inputs, [sc.jacobian(out, wrt), sc.jvp(out, wrt, seed)], point)

      def at(value):
        return evaluate(inputs, [out], [*point[:k], value, *point[k + 1 :]])

      fd = finite_difference(at, point[k]).reshape(out.size, wrt.size)
      np.testing.assert_allclose(jac, fd, rtol=2e-5, atol=2e-7)
      np.testing.assert_allclose(single.reshape(-1), jac @ np.ones(wrt.size), atol=1e-11)


@pytest.mark.parametrize("bound,limit", [(0, 5.0), (1, 5.0), (6, -1.0), (6, 2.5), (6, 100.0)])
def test_while_primal_steps(bound, limit):
  @sc.function(sc.arg("c", 2), sc.arg("p", 2), sc.arg("limit", ()), outputs=sc.arg("go", sc.TensorType((), sc.dtypes.bool_)))
  def cond(c, p, limit):
    return c[1] < limit

  @sc.function(sc.arg("c", 2), sc.arg("p", 2), sc.arg("limit", ()))
  def step(c, p, limit):
    return sc.stack([c[0].sin() * p[0] + p[1], c[1] + 1.0])

  c, p, lim = sc.sym("c", 2), sc.sym("p", 2), sc.sym("limit", ())
  out, count = sc.while_loop(cond, step, c, max_iter=bound, params=(p, lim))
  steps = min(bound, max(0, int(np.ceil(limit))))
  ref = c
  for _ in range(steps):
    ref = step(ref, p, lim)
  point = [np.array([0.2, 0.0]), np.array([0.7, 0.1]), np.array(limit)]
  jac, expected = evaluate([c, p, lim], [sc.jacobian(out, c), sc.jacobian(ref, c)], point)
  np.testing.assert_allclose(jac, expected, atol=1e-12)
  check_derivatives([c, p, lim], [out], point)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("trip", [0, 1, 3, 9])
def test_trip_index_windows_mixed_carries_and_trajectories(reverse, trip):
  from scaly.function.model import as_concrete
  from scaly.function.sugar import _loop_node

  i64 = sc.TensorType((), sc.dtypes.int64)
  boolean = sc.TensorType((), sc.dtypes.bool_)

  @sc.function(sc.arg("c", 2), sc.arg("n", i64), sc.arg("b", boolean), sc.arg("k", i64), sc.arg("x", ()), sc.arg("p", 2))
  def step(c, n, b, k, x, p):
    cn = sc.where(b, c * p + x, c + p * x) + 0.01 * k.cast(sc.dtypes.float64)
    return cn, n + 1, ~b, c * x

  c, x, p, t = sc.sym("c", 2), sc.sym("xs", 10), sc.sym("p", 2), sc.sym("trip", ())
  body = as_concrete(step)
  attrs = dict(n_carry=3, length=4, starts=(1, 0), strides=(2, 0), index=True, reverse=reverse, in_place=())
  args = (c, sc.const(np.int64(0)), sc.const(np.bool_(True)), x, p, t.cast(sc.dtypes.int64))
  out = [_loop_node(body, None, args, attrs, i) for i in (0, 3, 5)]
  check_derivatives([c, x, p, t], out, [np.array([0.2, -0.3]), np.linspace(0.1, 0.5, 10), np.array([0.7, 0.8]), np.array(float(trip) + 0.25)])
  tangent = sc.jvp(out[0], c, sc.const(np.ones(2)))
  from scaly.ir.expr import topo, ExprOp

  loop = next(node for node in topo((tangent,)) if node.op == ExprOp.LOOP)
  assert loop.attrs["n_carry"] == 4


@pytest.mark.parametrize("kind", ["scan", "while"])
def test_nested_loop_and_loop_inside_map(kind):
  boolean = sc.TensorType((), sc.dtypes.bool_)

  @sc.function(sc.arg("c", 2), sc.arg("p", 2))
  def inner(c, p):
    return (c * p).sin() + 0.1

  @sc.function(sc.arg("c", 2), sc.arg("p", 2), outputs=sc.arg("go", boolean))
  def cond(c, p):
    return c[0] < 0.9

  @sc.function(sc.arg("c", 2), sc.arg("p", 2))
  def outer(c, p):
    if kind == "scan":
      return sc.scan(inner, c, (sc.broadcast(p),), length=3)[0]
    return sc.while_loop(cond, inner, c, max_iter=3, params=(p,))[0]

  @sc.function(sc.arg("x", 2), sc.arg("p", 2))
  def nested(x, p):
    return sc.scan(outer, x, (sc.broadcast(p),), length=2)[0]

  xs, p = sc.sym("xs", (3, 2)), sc.sym("p", 2)
  y = sc.vmap(nested, 3)(xs, sc.broadcast(p))
  check_derivatives([xs, p], [y], [np.array([[0.2, 0.3], [0.4, -0.1], [1.0, 0.5]]), np.array([0.7, 0.8])])


@pytest.mark.parametrize("dtype", [sc.dtypes.int64, sc.dtypes.bool_])
def test_nonfloating_carry_with_floating_stacked_output(dtype):
  """Port of devrush test_loop_edge_cases.py's non-float scan carry test."""

  @sc.function(sc.arg("c", sc.TensorType((), dtype)), sc.arg("x", ()))
  def step(c, x):
    if dtype == sc.dtypes.int64:
      return c + 1, x * c.cast(sc.dtypes.float64)
    return ~c, sc.where(c, x, -x)

  c, x = sc.sym("c", ()), sc.sym("xs", 4)
  _, (ys,) = sc.scan(step, c.cast(dtype), (x,), length=4)
  init = np.array(2.0) if dtype == sc.dtypes.int64 else np.array(1.0)
  jac = evaluate([c, x], [sc.jacobian(ys, x)], [init, np.arange(4.0)])
  np.testing.assert_array_equal(jac, np.diag([2.0, 3.0, 4.0, 5.0] if dtype == sc.dtypes.int64 else [1.0, -1.0, 1.0, -1.0]))


def test_symbolic_seeds_and_forward_over_forward():
  """Port of devrush test_multiseed_loops.py's arbitrary seeds and nested forward tests."""
  from scaly.ad.forward import jvp_many

  @sc.function(sc.arg("c", 2), sc.arg("x", ()))
  def step(c, x):
    return (c + x).sin()

  c, x, s = sc.sym("c", 2), sc.sym("xs", 3), sc.sym("seeds", (3, 2))
  out, _ = sc.scan(step, c, (x,), length=3)
  deriv = jvp_many(out, c, s)
  jac = sc.jacobian(out, c)
  got, want = evaluate(
    [c, x, s], [deriv, s @ jac.T], [np.array([0.2, 0.4]), np.array([0.1, 0.3, 0.2]), np.array([[1.0, 2.0], [0.0, 0.0], [-0.5, 0.7]])]
  )
  np.testing.assert_allclose(got, want, atol=1e-12)
  check_derivatives([c, x], [jac], [np.array([0.2, 0.4]), np.array([0.1, 0.3, 0.2])])


def test_tangent_support_and_sparsity_follow_loop_fixed_point():
  @sc.function(sc.arg("c", 3), sc.arg("x", ()))
  def shift(c, x):
    return sc.stack([c[1], c[0] * x, c[2]])

  c, x = sc.sym("c", 3), sc.sym("xs", 4)
  final, _ = sc.scan(shift, c, (x,), length=4)
  tangent = sc.jvp(final, c, sc.const(np.array([1.0, 0.0, 0.0])))
  assert tangent.support == frozenset({0, 1})
  pattern = sc.jacobian_sparsity(tangent, x)
  assert set(pattern.rows) == {0, 1}
  check_derivatives([c, x], [final], [np.array([0.2, 0.3, 0.0]), np.array([0.4, 0.5, 0.6, 0.7])])
