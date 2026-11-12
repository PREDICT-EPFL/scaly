"""Reverse loop derivatives against unrolled graphs, finite differences and forward mode."""

import numpy as np
import pytest

import scaly as sc
from scaly.function.concrete import ConcreteFunction


def evaluate(inputs, outputs, point):
  fn = ConcreteFunction._from_exprs("loop_reverse", inputs, outputs, [x.name for x in inputs], [f"out{i}" for i in range(len(outputs))])
  return fn(*point)


@pytest.mark.parametrize("length", [0, 1, 4])
@pytest.mark.parametrize("reverse", [False, True])
def test_scan_reverse_matches_unrolled(length, reverse):
  @sc.function(sc.arg("c", 2), sc.arg("x", ()), sc.arg("p", 2))
  def step(c, x, p):
    return (c * p + x).sin(), c * x

  c, xs, p = sc.sym("c", 2), sc.sym("xs", length), sc.sym("p", 2)
  final, (ys,) = sc.scan(step, c, (xs, sc.broadcast(p)), length=length, reverse=reverse)
  ref, rows = c, [None] * length
  for k in reversed(range(length)) if reverse else range(length):
    ref, rows[k] = step(ref, xs[k], p)
  cost = sc.sumsqr(final) + ys.sum()
  reference = sc.sumsqr(ref) + sum(row.sum() for row in rows)
  point = [np.array([0.2, -0.4]), np.linspace(0.1, 0.3, length), np.array([0.7, 0.9])]
  for wrt in (c, xs, p):
    grad = sc.gradient(cost, wrt)
    got, want, tangent = evaluate([c, xs, p], [grad, sc.gradient(reference, wrt), sc.jvp(cost, wrt, sc.const(np.ones(wrt.shape)))], point)
    np.testing.assert_allclose(got, want, atol=1e-12)
    np.testing.assert_allclose(got.sum(), tangent, atol=1e-12)


@pytest.mark.parametrize("limit", [-1.0, 2.5, 100.0])
def test_while_reverse_and_second_derivative(limit):
  @sc.function(sc.arg("c", 2), sc.arg("p", ()), sc.arg("limit", ()), outputs=sc.arg("go", sc.TensorType((), sc.dtypes.bool_)))
  def cond(c, p, limit):
    return c[1] < limit

  @sc.function(sc.arg("c", 2), sc.arg("p", ()), sc.arg("limit", ()))
  def step(c, p, limit):
    return sc.stack([(c[0] * p).sin() + 0.1, c[1] + 1.0])

  c, p, lim = sc.sym("c", 2), sc.sym("p", ()), sc.sym("limit", ())
  final, _ = sc.while_loop(cond, step, c, max_iter=6, params=(p, lim))
  ref = c
  for _ in range(min(6, max(0, int(np.ceil(limit))))):
    ref = step(ref, p, lim)
  point = [np.array([0.2, 0.0]), np.array(0.8), np.array(limit)]
  for wrt in (c, p):
    actual, expected = sc.sumsqr(final), sc.sumsqr(ref)
    got, want, hess, href = evaluate(
      [c, p, lim], [sc.gradient(actual, wrt), sc.gradient(expected, wrt), sc.hessian(actual, wrt), sc.hessian(expected, wrt)], point
    )
    np.testing.assert_allclose(got, want, atol=1e-12)
    np.testing.assert_allclose(hess, href, atol=1e-12)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("trip", [0, 1, 3, 9])
def test_trajectory_windows_index_and_mixed_carries(reverse, trip):
  """Port of devrush's loop edge cases: windows, step numbers and mixed carries."""
  from scaly.function.model import as_concrete
  from scaly.function.sugar import _loop_node

  i64, boolean = sc.TensorType((), sc.dtypes.int64), sc.TensorType((), sc.dtypes.bool_)

  @sc.function(sc.arg("c", 2), sc.arg("n", i64), sc.arg("b", boolean), sc.arg("k", i64), sc.arg("x", 2), sc.arg("p", 2))
  def step(c, n, b, k, x, p):
    return sc.where(b, (c * p + x).sin(), c + x * p) + 0.01 * k.cast(sc.dtypes.float64), n + 1, ~b, c * x

  c, x, p, t = sc.sym("c", 2), sc.sym("xs", 7), sc.sym("p", 2), sc.sym("trip", ())
  body = as_concrete(step)
  attrs = dict(n_carry=3, length=4, starts=(1, 0), strides=(1, 0), index=True, reverse=reverse, in_place=())
  args = (c, sc.const(np.int64(0)), sc.const(np.bool_(True)), x, p, t.cast(sc.dtypes.int64))
  final, ys, trajectory = [_loop_node(body, None, args, attrs, i) for i in (0, 3, 5)]
  actual = sc.sumsqr(final) + ys.sum() + sc.sumsqr(trajectory)
  ref, n, b, rows = c, sc.const(np.int64(0)), sc.const(np.bool_(True)), []
  for k in reversed(range(min(trip, 4))) if reverse else range(min(trip, 4)):
    rows.append(sc.sumsqr(ref))
    ref, n, b, y = step(ref, n, b, sc.const(np.int64(k)), x[k + 1 : k + 3], p)
    rows.append(y.sum())
  expected = sc.sumsqr(ref) + sum(rows)
  point = [np.array([0.2, -0.3]), np.linspace(0.1, 0.5, 7), np.array([0.7, 0.8]), np.array(float(trip))]
  from scaly.ad.derivatives import finite_difference

  for j, wrt in enumerate((c, x, p)):
    got, want = evaluate([c, x, p, t], [sc.gradient(actual, wrt), sc.gradient(expected, wrt)], point)
    fd = finite_difference(lambda value: evaluate([c, x, p, t], [actual], [*point[:j], value, *point[j + 1 :]]), point[j])
    np.testing.assert_allclose(got, want, atol=1e-12)
    np.testing.assert_allclose(got.reshape(-1), fd.reshape(-1), atol=2e-7, rtol=2e-5)


def test_one_backward_invocation_and_primal_trip():
  from scaly.ir.expr import ExprOp, invocation_key, topo

  @sc.function(sc.arg("c", ()), sc.arg("p", ()))
  def step(c, p):
    return c * p, c.sin()

  c, p = sc.sym("c", ()), sc.sym("p", ())
  final, (ys,) = sc.scan(step, c, (sc.broadcast(p),), length=5)
  grads = sc.vjp((final, ys), (c, p), (sc.const(1.0), sc.const(np.ones(5))))
  loops = [e for e in topo(grads) if e.op == ExprOp.LOOP]
  backward = {invocation_key(e) for e in loops if e.attrs["reverse"]}
  primal = {invocation_key(e) for e in loops if not e.attrs["reverse"]}
  assert len(backward) == len(primal) == 1
  assert next(e for e in loops if e.attrs["reverse"]).attrs["n_carry"] == 2


@pytest.mark.parametrize("bound", [0, 4])
def test_max_trajectory_counts_carries_per_loop(bound):
  @sc.function(sc.arg("c", 2))
  def step(c):
    return c.sin(), sc.concat([c] * 20)

  c = sc.sym("c", 2)
  final, (ys,) = sc.scan(step, c, length=bound, max_trajectory=2 * bound)
  grad = sc.gradient(final.sum() + ys.sum(), c)
  evaluate([c], [grad], [np.array([0.2, 0.4])])
  if bound:
    final, _ = sc.scan(step, c, length=bound, max_trajectory=2 * bound - 1)
    with pytest.raises(ValueError, match="max_trajectory=.*sc.custom_derivative"):
      sc.gradient(final.sum(), c)
  with pytest.raises(ValueError, match="non-negative"):
    sc.scan(step, c, length=bound, max_trajectory=-1)


@pytest.mark.parametrize("limit,steps", [(-1.0, 0), (2.5, 3), (100.0, 6)])
def test_backward_uses_primal_trip_count(limit, steps):
  from scaly.ir.expr import ExprOp, topo

  @sc.function(sc.arg("c", 2), sc.arg("limit", ()), outputs=sc.arg("go", sc.TensorType((), sc.dtypes.bool_)))
  def cond(c, limit):
    return c[1] < limit

  @sc.function(sc.arg("c", 2), sc.arg("limit", ()))
  def step(c, limit):
    return sc.stack([c[0] * c[0] + 0.1, c[1] + 1.0])

  c, lim = sc.sym("c", 2), sc.sym("limit", ())
  final, _ = sc.while_loop(cond, step, c, max_iter=6, params=(lim,))
  grad = sc.gradient(final.sum(), c)
  backward = next(e for e in topo((grad,)) if e.op == ExprOp.LOOP and e.attrs["reverse"])
  trip = backward.args[-1]
  assert backward.attrs["cond"] is None
  assert trip.op == ExprOp.LOOP and trip.attrs["output"] == len(trip.attrs["callee"].results)
  actual = evaluate([c, lim], [trip.cast(sc.dtypes.float64)], [np.array([0.2, 0.0]), np.array(limit)])
  assert actual == steps


def test_newton_early_exit_zero_trip_and_second_derivative():
  """Milestone 3 Newton iteration, differentiated through the steps actually taken."""

  @sc.function(sc.arg("x", ()), sc.arg("target", ()), outputs=sc.arg("go", sc.TensorType((), sc.dtypes.bool_)))
  def unsolved(x, target):
    return (x * x - target).abs() > 1e-12

  @sc.function(sc.arg("x", ()), sc.arg("target", ()))
  def newton(x, target):
    return 0.5 * (x + target / x)

  x, target = sc.sym("x", ()), sc.sym("target", ())
  final, count = sc.while_loop(unsolved, newton, x, max_iter=30, params=(target,))
  gradient, hessian = sc.gradient(final, target), sc.hessian(final, target)
  for initial, value in [(1.0, 2.0), (2.0, 4.0)]:
    got, n, grad, hess = evaluate([x, target], [final, count.cast(sc.dtypes.float64), gradient, hessian], [np.array(initial), np.array(value)])
    np.testing.assert_allclose(got, np.sqrt(value), atol=1e-12)
    if n == 0:
      assert grad == 0.0 and hess == 0.0
    else:
      assert n < 30
      np.testing.assert_allclose(grad, 0.5 / np.sqrt(value), atol=1e-11)
      np.testing.assert_allclose(hess, -0.25 / value**1.5, atol=1e-10)


def test_nested_loop_limits_and_reverse_over_reverse():
  """Port of devrush's per-loop trajectory limit and nested derivative problems."""

  @sc.function(sc.arg("c", 3), sc.arg("u", ()))
  def inner(c, u):
    return (c * u).sin() + 0.1

  @sc.function(sc.arg("c", 2), sc.arg("u", ()))
  def outer(c, u):
    state = sc.concat([c, u.reshape((1,))])
    return sc.scan(inner, state, (sc.broadcast(u),), length=5, max_trajectory=15)[0][:2]

  c, us = sc.sym("c", 2), sc.sym("us", 6)
  final = sc.scan(outer, c, (us,), length=6, max_trajectory=12)[0]
  cost = sc.sumsqr(final)
  grad = sc.gradient(cost, c)
  second_reverse = sc.gradient(grad.sum(), c)
  forward_reverse = sc.jvp(grad, c, sc.const(np.ones(2)))
  point = [np.array([0.3, -0.2]), np.linspace(-0.6, 0.6, 6)]
  got, want = evaluate([c, us], [second_reverse, forward_reverse], point)
  np.testing.assert_allclose(got, want, atol=1e-12)


def test_rk4_value_and_reverse_gradients_have_constant_program_size():
  from tests.function.test_loop import Z0, rk4, rk4_numpy
  from scaly.ad.derivatives import finite_difference
  from scaly.codegen import render_c_module
  from scaly.ir.program import walk_program

  sizes, counts = [], []
  for length in (20, 200, 2000):
    z, us = sc.sym("z", 4), sc.sym("us", length)
    final, (stages,) = sc.scan(rk4, z, (us,), length=length)
    cost = sc.sumsqr(final) + stages.sum()
    dz, du = sc.vjp((cost,), (z, us), (sc.const(1.0),))
    fn = ConcreteFunction._from_exprs("rk4_reverse", [z, us], [cost, dz, du], ["z", "us"], ["cost", "dz", "du"])
    module = render_c_module(fn)
    sizes.append(len(module.source.encode()))
    counts.append(len(list(walk_program(module.program))))
    controls = 0.05 * np.sin(np.arange(length) * 0.01)
    value, grad_z, grad_u = fn(Z0, controls)
    ref_final, ref_stages = rk4_numpy(Z0, controls)
    np.testing.assert_allclose(value, ref_final @ ref_final + ref_stages.sum(), rtol=1e-12)
    if length == 20:

      def reference(initial, controls):
        end, rows = rk4_numpy(initial, controls)
        return end @ end + rows.sum()

      np.testing.assert_allclose(grad_z, finite_difference(lambda initial: reference(initial, controls), Z0).reshape(-1), rtol=2e-5, atol=1e-6)
      np.testing.assert_allclose(grad_u, finite_difference(lambda controls: reference(Z0, controls), controls).reshape(-1), rtol=2e-5, atol=1e-6)
    direction = sc.jvp(cost, us, sc.const(np.ones(length)))
    tangent = evaluate([z, us], [direction], [Z0, controls])
    np.testing.assert_allclose(grad_u.sum(), tangent, rtol=1e-10, atol=1e-8)
  assert counts[0] == counts[1] == counts[2]
  assert max(sizes) - min(sizes) < 300


def test_sparse_carries_and_parameters_match_unrolled():
  from scaly.ir.types import CanonicalPattern

  pattern = CanonicalPattern((2, 2), (0, 1), (0, 1))
  type_ = sc.TensorType((2, 2), pattern=pattern)

  @sc.function(sc.arg("c", type_), sc.arg("p", type_))
  def step(c, p):
    return c * p, c * c

  c, p = sc.sym("c", 2), sc.sym("p", 2)
  initial, parameter = sc.pack(c, pattern), sc.pack(p, pattern)
  final, (ys,) = sc.scan(step, initial, (sc.broadcast(parameter),), length=3)
  ref, rows = initial, []
  for _ in range(3):
    ref, y = step(ref, parameter)
    rows.append(sc.values(y).sum())
  actual, expected = sc.values(final).sum() + sc.values(ys).sum(), sc.values(ref).sum() + sum(rows)
  point = [np.array([0.2, 0.4]), np.array([0.7, 0.9])]
  for wrt in (c, p):
    got, want = evaluate([c, p], [sc.gradient(actual, wrt), sc.gradient(expected, wrt)], point)
    np.testing.assert_allclose(got, want, atol=1e-12)


def test_loop_respects_custom_reverse_rule():
  @sc.function(sc.arg("c", ()), sc.arg("x", ()), outputs=sc.arg("cn", ()))
  def raw(c, x):
    return c * x

  @sc.function(sc.arg("c", ()), sc.arg("x", ()), outputs=sc.group(sc.arg("cn", ()), sc.group(sc.arg("res_c", ()), sc.arg("res_x", ()))))
  def forward(c, x):
    return c * x, (c, x)

  @sc.function(sc.group(sc.arg("res_c", ()), sc.arg("res_x", ())), sc.arg("lam", ()), outputs=sc.group(sc.arg("dc", ()), sc.arg("dx", ())))
  def backward(residuals, lam):
    c, x = residuals
    return 3 * x * lam, 3 * c * lam

  step = sc.custom_derivative(raw, fwd=forward, bwd=backward)
  c, xs = sc.sym("c", ()), sc.sym("xs", 3)
  final = sc.scan(step, c, (xs,), length=3)[0]
  ref = c
  for k in range(3):
    ref = step(ref, xs[k])
  for wrt in (c, xs):
    got, want = evaluate([c, xs], [sc.gradient(final, wrt), sc.gradient(ref, wrt)], [np.array(0.3), np.array([0.5, 0.7, -0.4])])
    np.testing.assert_allclose(got, want, atol=1e-12)


def test_adjoint_cache_distinguishes_broadcast_partitions():
  @sc.function(sc.arg("c", ()), sc.arg("a", ()), sc.arg("b", ()))
  def step(c, a, b):
    return c * a + b

  c, a, bs = sc.sym("c", ()), sc.sym("a", ()), sc.sym("bs", 3)
  first = sc.scan(step, c, (sc.broadcast(a), bs), length=3)[0]
  sc.vjp((first,), (a, bs), (sc.const(1.0),))
  aa, b = sc.sym("aa", 3), sc.sym("b", ())
  final = sc.scan(step, c, (aa, sc.broadcast(b)), length=3)[0]
  ga, gb = sc.vjp((final,), (aa, b), (sc.const(1.0),))
  got_a, got_b = evaluate([c, aa, b], [ga, gb], [np.array(2.0), np.array([3.0, 4.0, 5.0]), np.array(7.0)])
  np.testing.assert_array_equal(got_a, [40.0, 65.0, 59.0])
  assert got_b == 26.0
  both = sc.scan(step, c, (sc.broadcast(a), sc.broadcast(b)), length=3)[0]
  ga, gb = sc.vjp((both,), (a, b), (sc.const(1.0),))
  got_a, got_b = evaluate([c, a, b], [ga, gb], [np.array(2.0), np.array(3.0), np.array(7.0)])
  assert got_a == 103.0 and got_b == 13.0


def test_scalar_parameter_reverse_over_reverse():
  @sc.function(sc.arg("c", ()), sc.arg("p", ()))
  def step(c, p):
    return c * p

  c, p = sc.sym("c", ()), sc.sym("p", ())
  final = sc.scan(step, c, (sc.broadcast(p),), length=3)[0]
  grad = sc.gradient(final, p)
  reverse, forward = evaluate([c, p], [sc.gradient(grad, p), sc.jvp(grad, p, sc.const(1.0))], [np.array(2.0), np.array(3.0)])
  assert reverse == forward == 36.0


def test_unused_singular_carry_does_not_contaminate_gradients():
  @sc.function(sc.arg("c", ()), sc.arg("d", ()), sc.arg("p", ()))
  def step(c, d, p):
    return c * p, d.sqrt() * p

  c, d, p = sc.sym("c", ()), sc.sym("d", ()), sc.sym("p", ())
  (final, _), _ = sc.scan(step, (c, d), (sc.broadcast(p),), length=2)
  grads = sc.vjp((final,), (c, d, p), (sc.const(1.0),))
  actual = evaluate([c, d, p], grads, [np.array(2.0), np.array(0.0), np.array(0.5)])
  np.testing.assert_array_equal(actual, [0.25, 0.0, 2.0])


def test_loop_result_cotangents_accumulate_across_lowering_hints():
  @sc.function(sc.arg("c", ()))
  def step(c):
    return 2.0 * c

  c = sc.sym("c", ())
  final = sc.scan(step, c, length=3)[0]
  grad = sc.gradient(final + final.scalar(), c)
  actual = evaluate([c], [grad], [np.array(2.0)])
  assert actual == 16.0


def test_carry_activity_follows_coupling_across_steps():
  @sc.function(sc.arg("c", ()), sc.arg("d", ()), sc.arg("p", ()))
  def step(c, d, p):
    return c + d, d * p

  c, d, p = sc.sym("c", ()), sc.sym("d", ()), sc.sym("p", ())
  (final, _), _ = sc.scan(step, (c, d), (sc.broadcast(p),), length=3)
  ref_c, ref_d = c, d
  for _ in range(3):
    ref_c, ref_d = step(ref_c, ref_d, p)
  grads = sc.vjp((final,), (c, d, p), (sc.const(1.0),))
  reference = sc.vjp((ref_c,), (c, d, p), (sc.const(1.0),))
  result = evaluate([c, d, p], [*grads, *reference], [np.array(2.0), np.array(0.4), np.array(0.5)])
  np.testing.assert_allclose(result[:3], result[3:], atol=1e-12)


@pytest.mark.parametrize("mode", ["forward_reverse", "reverse_reverse"])
def test_second_derivative_at_primal_trajectory_limit(mode):
  @sc.function(sc.arg("c", 2), sc.arg("x", ()), sc.arg("p", ()))
  def step(c, x, p):
    return (c * p + x).sin()

  c, xs, p = sc.sym("c", 2), sc.sym("xs", 4), sc.sym("p", ())
  final = sc.scan(step, c, (xs, sc.broadcast(p)), length=4, max_trajectory=8)[0]
  ref = c
  for k in range(4):
    ref = step(ref, xs[k], p)
  point = [np.array([0.2, -0.3]), np.linspace(0.1, 0.3, 4), np.array(0.8)]
  for wrt in (c, xs, p):
    grad, gref = sc.gradient(final.sum(), wrt), sc.gradient(ref.sum(), wrt)
    if mode == "forward_reverse":
      actual, expected = sc.jacobian(grad, wrt), sc.jacobian(gref, wrt)
    else:
      actual, expected = sc.gradient(grad.sum(), wrt), sc.gradient(gref.sum(), wrt)
    got, want = evaluate([c, xs, p], [actual, expected], point)
    np.testing.assert_allclose(got, want, atol=1e-12)


def test_multiple_map_outputs_keep_mapped_adjoints():
  from scaly.ir.expr import ExprOp, is_map, topo

  @sc.function(sc.arg("x", ()), sc.arg("p", ()))
  def stage(x, p):
    return x * p, x.sin() + p * p

  xs, p = sc.sym("xs", 4), sc.sym("p", ())
  a, b = sc.vmap(stage, 4)(xs, sc.broadcast(p))
  grad = sc.vjp((a, b), (xs, p), (sc.const(np.ones(4)), sc.const(np.ones(4))))
  loops = [node for node in topo(grad) if node.op == ExprOp.LOOP]
  assert loops and all(is_map(node) for node in loops)
  dx, dp = evaluate([xs, p], grad, [np.arange(4.0) / 10, np.array(0.7)])
  np.testing.assert_allclose(dx, 0.7 + np.cos(np.arange(4.0) / 10))
  np.testing.assert_allclose(dp, np.arange(4.0).sum() / 10 + 8 * 0.7)


def test_while_inside_map_reverse_matches_finite_differences():
  from scaly.ad.derivatives import finite_difference

  @sc.function(sc.arg("c", ()), sc.arg("p", ()), outputs=sc.arg("go", sc.TensorType((), sc.dtypes.bool_)))
  def cond(c, p):
    return c < p

  @sc.function(sc.arg("c", ()), sc.arg("p", ()))
  def step(c, p):
    return c * 1.2 + 0.1 * p

  @sc.function(sc.arg("c", ()), sc.arg("p", ()))
  def run(c, p):
    return sc.while_loop(cond, step, c, max_iter=4, params=(p,))[0]

  xs, p = sc.sym("xs", 3), sc.sym("p", ())
  cost = sc.sumsqr(sc.vmap(run, 3)(xs, sc.broadcast(p)))
  point = [np.array([0.1, 0.8, 1.3]), np.array(1.0)]
  for j, wrt in enumerate((xs, p)):
    got = evaluate([xs, p], [sc.gradient(cost, wrt)], point)
    want = finite_difference(lambda value: evaluate([xs, p], [cost], [*point[:j], value, *point[j + 1 :]]), point[j])
    np.testing.assert_allclose(got.reshape(-1), want.reshape(-1), rtol=1e-6)


@pytest.mark.parametrize("indexed_first", [False, True])
def test_adjoint_cache_distinguishes_step_number_formal(indexed_first):
  @sc.function(sc.arg("c", ()), sc.arg("k", sc.TensorType((), sc.dtypes.int64)), sc.arg("x", ()))
  def step(c, k, x):
    return c * x + 0.01 * k.cast(sc.dtypes.float64)

  c, xs = sc.sym("c", ()), sc.sym("xs", 3)
  for indexed in (indexed_first, not indexed_first):
    final = sc.scan(step, c, (xs,) if indexed else (sc.broadcast(sc.const(np.int64(2))), xs), length=3, index=indexed)[0]
    ref = c
    for k in range(3):
      ref = step(ref, sc.const(np.int64(k if indexed else 2)), xs[k])
    actual = sc.vjp((final,), (c, xs), (sc.const(1.0),))
    reference = sc.vjp((ref,), (c, xs), (sc.const(1.0),))
    result = evaluate([c, xs], [*actual, *reference], [np.array(0.3), np.array([0.5, 0.7, 0.9])])
    for got, want in zip(result[:2], result[2:], strict=True):
      np.testing.assert_allclose(got, want, atol=1e-12)


def test_runtime_index_reverse_preserves_primal_trajectory():
  from scaly.codegen import render_c_module
  from scaly.ir.program import ProgramOp, walk_program

  @sc.function(sc.arg("carry", 4), sc.arg("pivot", sc.TensorType((), sc.dtypes.int64)), sc.arg("scale", ()))
  def step(carry, pivot, scale):
    return sc.scatter(carry, pivot, sc.gather(carry, pivot) * scale, combine="set")

  initial, pivot, scale = sc.sym("initial", 4), sc.sym("pivot", ()), sc.sym("scale", ())
  index = pivot.cast(sc.dtypes.int64)
  final, _ = sc.scan(step, initial, (sc.broadcast(index), sc.broadcast(scale)), length=3)
  reference = initial
  for _ in range(3):
    reference = step(reference, index, scale)
  actual = sc.vjp((sc.sumsqr(final),), (initial, scale), (sc.const(1.0),))
  expected = sc.vjp((sc.sumsqr(reference),), (initial, scale), (sc.const(1.0),))
  point = [np.array([0.2, 0.4, 0.6, 0.8]), np.array(1.25), np.array(0.7)]
  got = evaluate([initial, pivot, scale], [final, *actual, *expected], point)
  for a, b in zip(got[1:3], got[3:], strict=True):
    np.testing.assert_allclose(a, b, atol=1e-12)
  fn = ConcreteFunction._from_exprs("index_reverse", [initial, pivot, scale], [final, *actual], ["initial", "pivot", "scale"], ["final", "di", "ds"])
  module = render_c_module(fn)
  assert not any(node.op == ProgramOp.PROC and node.attrs["name"].startswith("step_inplace") for node in walk_program(module.program))


def test_carried_reverse_drops_select_activity():
  from scaly.ad.rules import _mask_of
  from scaly.ir.expr import ExprOp, topo

  @sc.function(sc.arg("carry", 2))
  def step(carry):
    return carry.sin()

  x = sc.sym("x", 2)
  final, _ = sc.scan(step, x, length=2)
  seed = sc.where(x > 0.0, sc.const(np.ones(2)), 0.0)
  (grad,) = sc.vjp((final,), (x,), (seed,))
  assert _mask_of(grad) is None
  backward = next(node for node in topo((grad,)) if node.op == ExprOp.LOOP and node.attrs["reverse"])
  assert all(not name.startswith("mask:") for name in backward.attrs["callee"].input_names)
  point = np.array([-0.2, 0.4])
  expected = np.where(point > 0.0, np.cos(np.sin(point)) * np.cos(point), 0.0)
  np.testing.assert_allclose(evaluate([x], [grad], [point]), expected, atol=1e-12)
