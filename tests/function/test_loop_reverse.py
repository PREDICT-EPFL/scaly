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
