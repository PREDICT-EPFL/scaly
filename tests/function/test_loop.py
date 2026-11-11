"""``scan`` and ``while_loop`` build one ``LOOP`` node that computes what a NumPy loop and the unrolled chain of calls compute."""

from __future__ import annotations

import numpy as np
import pytest

import scaly as sc
from scaly.function.concrete import ConcreteFunction
from scaly.function.model import as_concrete
from scaly.function.sugar import _loop_node
from scaly.ir.expr import callees_of
from scaly.ir.spec import VerifyError

DT = 0.1
from_exprs = ConcreteFunction._from_exprs
I64 = sc.TensorType((), sc.dtypes.int64)
BOOL = sc.TensorType((), sc.dtypes.bool_)


@sc.function(sc.arg("z", 4), sc.arg("u", ()), outputs=sc.group(sc.arg("znext", 4), sc.arg("cost", ())))
def rk4(z, u):
  def f(s):
    return sc.stack([s[2], s[3], u - 0.1 * s[2], -9.81 * s[1].sin() - u * s[1].cos()])

  k1 = f(z)
  k2 = f(z + 0.5 * DT * k1)
  k3 = f(z + 0.5 * DT * k2)
  k4 = f(z + DT * k3)
  return z + (DT / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), sc.sumsqr(z) + 0.01 * u * u


def rk4_numpy(z: np.ndarray, us: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  def f(s, u):
    return np.array([s[2], s[3], u - 0.1 * s[2], -9.81 * np.sin(s[1]) - u * np.cos(s[1])])

  costs = []
  for u in us:
    costs.append(z @ z + 0.01 * u * u)
    k1 = f(z, u)
    k2 = f(z + 0.5 * DT * k1, u)
    k3 = f(z + 0.5 * DT * k2, u)
    k4 = f(z + DT * k3, u)
    z = z + (DT / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
  return z, np.array(costs)


def rollout(length: int, *, unrolled: bool = False) -> sc.Function:
  @sc.function(sc.arg("z0", 4), sc.arg("us", length), outputs=sc.group(sc.arg("zN", 4), sc.arg("costs", length)), name=f"roll_{unrolled:d}_{length}")
  def roll(z0, us):
    if not unrolled:
      zn, (costs,) = sc.scan(rk4, z0, (us,), length=length)
      return zn, costs
    z, costs = z0, []
    for k in range(length):
      z, c = rk4(z, us[k])
      costs.append(c)
    return z, sc.stack(costs) if costs else sc.const(np.zeros(0))

  return roll


Z0 = np.array([0.1, 0.4, -0.2, 0.05])


@pytest.mark.parametrize("length", [0, 1, 2, 7])
def test_scan_matches_unrolled_calls_and_numpy(length: int) -> None:
  us = np.sin(np.arange(length) * 0.3)
  zn, costs = rk4_numpy(Z0, us)
  for unrolled in (False, True):
    got = rollout(length, unrolled=unrolled)(Z0, us)
    np.testing.assert_allclose(got[0], zn, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(got[1], costs, rtol=1e-12, atol=1e-12)


def test_long_scan_matches_numpy() -> None:
  us = np.sin(np.arange(1000) * 0.01) * 0.2
  zn, costs = rk4_numpy(Z0, us)
  got = rollout(1000)(Z0, us)
  np.testing.assert_allclose(got[0], zn, rtol=1e-10, atol=1e-10)
  np.testing.assert_allclose(got[1], costs, rtol=1e-10, atol=1e-10)


def test_scan_of_length_zero_returns_the_init_bit_for_bit() -> None:
  z0 = np.array([0.1, -0.0, np.inf, np.nan])
  got = rollout(0)(z0, np.zeros(0))
  assert got[0].tobytes() == z0.tobytes() and got[1].shape == (0,)


def test_matrix_carry_kalman_filter_matches_numpy() -> None:
  """A matrix carry beside a vector carry: the covariance recursion of a Kalman filter."""
  n = 3
  a = np.array([[1.0, DT, 0.0], [0.0, 1.0, DT], [0.0, 0.0, 0.98]])
  c = np.array([1.0, 0.0, 0.0])
  q, r = 0.01 * np.eye(n), 0.1

  @sc.function(sc.arg("x", n), sc.arg("P", (n, n)), sc.arg("y", ()), outputs=sc.group(sc.arg("xn", n), sc.arg("Pn", (n, n)), sc.arg("xhat", ())))
  def kf_step(x, p, y):
    xp, pp = sc.const(a) @ x, sc.const(a) @ p @ sc.const(a.T) + sc.const(q)
    gain = (pp @ sc.const(c)) / (sc.dot(sc.const(c), pp @ sc.const(c)) + r)
    xn = xp + gain * (y - sc.dot(sc.const(c), xp))
    return xn, pp - gain.reshape((n, 1)) @ (sc.const(c).reshape((1, n)) @ pp), xn[0]

  @sc.function(sc.arg("x0", n), sc.arg("P0", (n, n)), sc.arg("ys", 40))
  def kf(x0, p0, ys):
    (x, p), (xhat,) = sc.scan(kf_step, (x0, p0), (ys,), length=40)
    return x, p, xhat

  meas = np.sin(np.arange(40) * 0.2) + 0.05 * np.random.default_rng(0).normal(size=40)
  xv, pv, ref = np.zeros(n), np.eye(n), []
  for yk in meas:
    xv, pv = a @ xv, a @ pv @ a.T + q
    k = pv @ c / (c @ pv @ c + r)
    xv = xv + k * (yk - c @ xv)
    pv = pv - np.outer(k, c @ pv)
    ref.append(xv[0])
  got = kf(np.zeros(n), np.eye(n), meas)
  np.testing.assert_allclose(got[0], xv, rtol=1e-12, atol=1e-12)
  np.testing.assert_allclose(got[1], pv, rtol=1e-12, atol=1e-12)
  np.testing.assert_allclose(got[2], ref, rtol=1e-12, atol=1e-12)


@sc.function(sc.arg("c", 2), sc.arg("w", 2), sc.arg("v", ()), outputs=sc.group(sc.arg("cn", 2), sc.arg("y", ())))
def mix(c, w, v):
  return c * w + v, c[0] - v


def mix_numpy(c: np.ndarray, w: np.ndarray, vs: list[float]) -> tuple[np.ndarray, list[float]]:
  ys = []
  for v in vs:
    ys.append(c[0] - v)
    c = c * w + v
  return c, ys


@pytest.mark.parametrize("reverse", [False, True])
def test_parameters_windows_and_reverse(reverse: bool) -> None:
  """``w`` is a parameter read whole by every step; ``v`` is every third entry from 2, as a window."""

  @sc.function(sc.arg("c0", 2), sc.arg("w", 2), sc.arg("v", 9))
  def f(c0, w, v):
    c, (ys,) = sc.scan(mix, c0, (sc.broadcast(w), sc.window(v, 2, 3)), length=3, reverse=reverse)
    return c, ys

  c0, w, v = np.array([1.0, -1.0]), np.array([0.5, 2.0]), np.arange(9.0)
  order = [2, 1, 0] if reverse else [0, 1, 2]
  c, ys = mix_numpy(c0, w, [v[2 + 3 * k] for k in order])
  got = f(c0, w, v)
  np.testing.assert_allclose(got[0], c, rtol=1e-15)
  np.testing.assert_allclose(got[1], [ys[order.index(k)] for k in range(3)], rtol=1e-15)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("trip", [-2, 0, 1, 3, 5, 9])
def test_run_time_trip_count(trip: int, reverse: bool) -> None:
  """``trip`` runs the first ``min(trip, length)`` steps; the stacked outputs past them are zero."""

  @sc.function(sc.arg("c0", 2), sc.arg("w", 2), sc.arg("v", 5), sc.arg("n", ()), outputs=sc.group(sc.arg("c", 2), sc.arg("ys", 5)))
  def f(c0, w, v, n):
    c, (ys,) = sc.scan(mix, c0, (sc.broadcast(w), v), length=5, trip=n.cast(sc.dtypes.int64), reverse=reverse)
    return c, ys

  c0, w, v = np.array([1.0, -1.0]), np.array([0.5, 2.0]), np.arange(5.0) + 1.0
  steps = max(0, min(trip, 5))
  order = list(range(steps))[::-1] if reverse else list(range(steps))
  c, ys = mix_numpy(c0, w, [v[k] for k in order])
  expected = np.zeros(5)
  expected[order] = ys
  got = f(c0, w, v, np.array(float(trip)))
  np.testing.assert_allclose(got[0], c, rtol=1e-15)
  np.testing.assert_array_equal(got[1], expected)


@sc.function(sc.arg("c", 2), sc.arg("k", I64), sc.arg("x", ()), outputs=sc.group(sc.arg("cn", 2), sc.arg("y", ())))
def counted(c, k, x):
  kf = k.cast(sc.dtypes.float64)
  return c + kf * x, kf


@pytest.mark.parametrize("reverse", [False, True])
def test_step_number(reverse: bool) -> None:
  @sc.function(sc.arg("c0", 2), sc.arg("xs", 4))
  def f(c0, xs):
    c, (ks,) = sc.scan(counted, c0, (xs,), length=4, index=True, reverse=reverse)
    return c, ks

  xs = np.array([1.0, 10.0, 100.0, 1000.0])
  got = f(np.zeros(2), xs)
  np.testing.assert_array_equal(got[1], [0.0, 1.0, 2.0, 3.0])
  np.testing.assert_array_equal(got[0], [3210.0, 3210.0])


@sc.function(sc.arg("c", 1), outputs=sc.arg("cn", 1))
def inc(c):
  return c + 1.0


@sc.function(sc.arg("c", 1), outputs=sc.arg("cn", 1))
def square(c):
  return c * c


def go(test, name: str) -> sc.Function:
  return sc.function(sc.arg("c", 1), outputs=sc.arg("go", BOOL), name=name)(lambda c: test(c[0]))


def test_while_loop_edge_counts() -> None:
  """Exit at step 0, reach ``max_iter``, stop on a condition that reads a non-finite value."""
  never, always, finite = go(lambda c: c > 1e9, "never"), go(lambda c: c > -1e9, "always"), go(sc.isfinite, "finite")

  @sc.function(sc.arg("c0", 1))
  def f(c0):
    outs = []
    for cond, body, max_iter in ((never, inc, 5), (always, inc, 5), (finite, square, 40), (always, inc, 0), (always, inc, 1)):
      c, count = sc.while_loop(cond, body, c0, max_iter=max_iter)
      outs += [c, count.cast(sc.dtypes.float64)]
    return tuple(outs)

  a, na, b, nb, d, nd, z, nz, o, no = f(np.array([3.0]))
  assert (a[0], na, b[0], nb) == (3.0, 0.0, 8.0, 5.0)
  assert d[0] == np.inf and nd == 10
  assert (z[0], nz, o[0], no) == (3.0, 0.0, 4.0, 1.0)


def test_while_loop_newton_matches_a_python_loop() -> None:
  """Newton's method for ``x**3 = target`` elementwise, with the target a parameter."""

  @sc.function(sc.arg("x", 5), sc.arg("target", 5), outputs=sc.arg("xn", 5))
  def newton(x, target):
    return x - (x**3 - target) / (3.0 * x * x)

  @sc.function(sc.arg("x", 5), sc.arg("target", 5), outputs=sc.arg("go", BOOL))
  def unsolved(x, target):
    return sc.sumsqr(x**3 - target) > 1e-24

  @sc.function(sc.arg("x0", 5), sc.arg("target", 5))
  def solve(x0, target):
    x, count = sc.while_loop(unsolved, newton, x0, max_iter=60, params=(target,))
    return x, count.cast(sc.dtypes.float64)

  targets = np.array([8.0, 27.0, 0.5, 2.0, 100.0])
  x, steps = np.full(5, 1.5), 0
  while steps < 60 and np.sum((x**3 - targets) ** 2) > 1e-24:
    x = x - (x**3 - targets) / (3 * x * x)
    steps += 1
  got, n = solve(np.full(5, 1.5), targets)
  np.testing.assert_allclose(got, x, rtol=1e-15)
  assert n == steps and 3 < steps < 60


def test_mixed_dtype_carries() -> None:
  """A float64, an int64 and a bool carry, with the step number, counted and flipped every step."""

  @sc.function(
    sc.group(sc.arg("x", 2), sc.arg("n", I64), sc.arg("b", BOOL)),
    sc.arg("k", I64),
    outputs=sc.group(sc.group(sc.arg("xn", 2), sc.arg("nn", I64), sc.arg("bn", BOOL))),
  )
  def step(c, k):
    x, n, b = c
    return ((sc.where(b, x * 2.0, x + 1.0), n + k, ~b),)

  @sc.function(sc.group(sc.arg("x", 2), sc.arg("n", I64), sc.arg("b", BOOL)), sc.arg("k", I64), outputs=sc.arg("go", BOOL))
  def small(c, k):
    return c[1] < 6

  @sc.function(sc.arg("x0", 2))
  def f(x0):
    init = ((x0, sc.const(np.int64(0)), sc.const(np.bool_(True))),)
    ((x, n, b),), count = sc.while_loop(small, step, init, max_iter=20, index=True)
    ((xs, ns, bs),), _ = sc.scan(step, init, length=3, index=True)
    return (
      x,
      n.cast(sc.dtypes.float64),
      b.cast(sc.dtypes.float64),
      count.cast(sc.dtypes.float64),
      xs,
      ns.cast(sc.dtypes.float64),
      bs.cast(sc.dtypes.float64),
    )

  def reference(cond, steps):
    x, n, b, k = np.array([1.0, -1.0]), 0, True, 0
    while k < steps and cond(n):
      x, n, b, k = (x * 2.0 if b else x + 1.0), n + k, not b, k + 1
    return x, n, b, k

  x, n, b, k = reference(lambda n: n < 6, 20)
  got = f(np.array([1.0, -1.0]))
  np.testing.assert_array_equal(got[0], x)
  assert (got[1], got[2], got[3]) == (n, float(b), k)
  x, n, b, _ = reference(lambda n: True, 3)
  np.testing.assert_array_equal(got[4], x)
  assert (got[5], got[6]) == (n, float(b))


def test_nested_loops_and_two_while_loops_in_one_function() -> None:
  """A while loop in a scan body takes a different count each step; two while loops sit side by side."""

  @sc.function(sc.arg("c", 1), sc.arg("limit", ()), outputs=sc.arg("go", BOOL))
  def below(c, limit):
    return c[0] < limit

  @sc.function(sc.arg("c", 1), sc.arg("limit", ()), outputs=sc.arg("cn", 1))
  def double(c, limit):
    return c * 2.0

  @sc.function(sc.arg("total", ()), sc.arg("limit", ()), outputs=sc.group(sc.arg("total2", ()), sc.arg("count", ())))
  def outer_step(total, limit):
    c, count = sc.while_loop(below, double, sc.const(np.ones(1)), max_iter=30, params=(limit,))
    return total + c[0], count.cast(sc.dtypes.float64)

  @sc.function(sc.arg("c", 1), sc.arg("limit", ()), outputs=sc.arg("cn", 1))
  def inc_with_limit(c, limit):
    return c + 1.0

  @sc.function(sc.arg("limits", 4), sc.arg("p", ()))
  def f(limits, p):
    total, (counts,) = sc.scan(outer_step, sc.const(0.0), (limits,), length=4)
    a, na = sc.while_loop(below, double, sc.const(np.ones(1)), max_iter=30, params=(p,))
    b, nb = sc.while_loop(below, inc_with_limit, sc.const(np.ones(1)), max_iter=30, params=(p,))
    return total, counts, a, na.cast(sc.dtypes.float64), b, nb.cast(sc.dtypes.float64)

  limits = np.array([1.0, 5.0, 100.0, 0.5])
  counts = [int(np.ceil(np.log2(lim))) if lim > 1 else 0 for lim in limits]
  total, got_counts, a, na, b, nb = f(limits, np.array(10.0))
  assert total == sum(2.0**c for c in counts)
  np.testing.assert_array_equal(got_counts, counts)
  assert (a[0], na, b[0], nb) == (16.0, 4.0, 10.0, 9.0)


@pytest.mark.parametrize("loop", ["scan", "while"])
def test_loop_inside_a_map_body(loop: str) -> None:
  """Each trip of a map runs its own loop: scans of 0 to 3 steps, and while loops that stop at different counts."""

  @sc.function(sc.arg("x", ()), sc.arg("n", I64), outputs=sc.group(sc.arg("y", ()), sc.arg("count", ())))
  def body(x, n):
    if loop == "scan":
      c, _ = sc.scan(inc, x.reshape((1,)), length=3, trip=n)
      return c[0], n.cast(sc.dtypes.float64)
    c, count = sc.while_loop(go(lambda c: c < 10.0, "below_ten"), square, x.reshape((1,)), max_iter=8)
    return c[0], count.cast(sc.dtypes.float64)

  @sc.function(sc.arg("xs", 4), sc.arg("ns", 4))
  def mapped(xs, ns):
    return sc.vmap(body, 4)(xs, ns.cast(sc.dtypes.int64))

  xs, ns = np.array([1.5, 2.0, 3.5, 20.0]), np.array([0.0, 1.0, 3.0, 2.0])
  ys, counts = mapped(xs, ns)
  if loop == "scan":
    np.testing.assert_array_equal(ys, xs + ns)
  else:
    expected = []
    for x in xs:
      k = 0
      while k < 8 and x < 10.0:
        x, k = x * x, k + 1
      expected.append((x, k))
    np.testing.assert_array_equal(ys, [e[0] for e in expected])
    np.testing.assert_array_equal(counts, [e[1] for e in expected])


def raw_loop(body, cond, args, *, n_carry, length, starts=(), strides=(), index=False, reverse=False, in_place=()):
  attrs = {"n_carry": n_carry, "length": length, "starts": starts, "strides": strides, "index": index, "reverse": reverse, "in_place": in_place}
  body, cond = as_concrete(body), None if cond is None else as_concrete(cond)
  return [_loop_node(body, cond, tuple(args), attrs, i) for i in range(2 * n_carry + len(body.outputs) - n_carry + 1)]


@sc.function(sc.arg("c", 2), sc.arg("x", ()), outputs=sc.arg("go", BOOL))
def mix_go(c, x):
  return c[0] < 6.0


@pytest.mark.parametrize(
  ("cond", "trip", "reverse"),
  [(False, None, False), (False, None, True), (True, None, False), (False, 2, False), (False, 2, True), (False, -2, True)],
)
def test_trajectories_and_zero_fill(cond: bool, trip: int | None, reverse: bool) -> None:
  """``trajectory[k]`` is the carry entering step ``k``; with a condition or a trip operand, slots past the trip count are zero."""
  c0, xs = sc.sym("c0", 2), sc.sym("xs", 4)
  args = [c0, xs] + ([] if trip is None else [sc.const(np.int64(trip))])
  final, ys, count, traj = raw_loop(
    sc.function(sc.arg("c", 2), sc.arg("x", ()), outputs=sc.group(sc.arg("cn", 2), sc.arg("y", ())), name="mix_one")(
      lambda c, x: (c * 1.5 + x, c[0] * x)
    ),
    mix_go if cond else None,
    args,
    n_carry=1,
    length=4,
    starts=(0,),
    strides=(1,),
    reverse=reverse,
  )
  f = from_exprs("traj", [c0, xs], [final, ys, count.cast(sc.dtypes.float64), traj.reshape((4, 2))], ["c0", "xs"], ["c", "ys", "n", "traj"])
  c, x = np.array([1.0, -1.0]), np.array([0.5, 1.0, 2.0, -1.0])
  bound = 4 if trip is None else max(trip, 0)
  want_traj, want_ys, n = np.zeros((4, 2)), np.zeros(4), 0
  for k in range(bound)[::-1] if reverse else range(bound):
    if cond and not c[0] < 6.0:
      break
    want_traj[k], want_ys[k] = c, c[0] * x[k]
    c, n = c * 1.5 + x[k], n + 1
  got = f(np.array([1.0, -1.0]), x)
  np.testing.assert_array_equal(got[0], c)
  np.testing.assert_array_equal(got[1], want_ys)
  assert got[2] == n
  np.testing.assert_array_equal(got[3], want_traj)
  assert (n < 4) == (cond or trip is not None)


def test_callees_and_the_verifier() -> None:
  c0, xs = sc.sym("c0", 2), sc.sym("xs", 4)
  final = raw_loop(mix_one := as_concrete(rk4), None, [sc.sym("z", 4), xs], n_carry=1, length=4, starts=(0,), strides=(1,))[0]
  assert callees_of(final) == (mix_one,)
  loop = raw_loop(
    mix,
    mix_go3 := as_concrete(
      sc.function(sc.arg("c", 2), sc.arg("w", 2), sc.arg("v", ()), outputs=sc.arg("go", BOOL), name="mix_go3")(lambda c, w, v: c[0] < v)
    ),
    [c0, c0, xs],
    n_carry=1,
    length=4,
    starts=(0, 0),
    strides=(0, 1),
  )
  assert callees_of(loop[0]) == (as_concrete(mix), mix_go3)
  assert [node.shape for node in loop] == [(2,), (4,), (), (8,)] and loop[2].type.dtype == sc.dtypes.int64

  def refused(match: str, **kwargs) -> None:
    defaults = {"body": mix, "cond": None, "args": [c0, c0, xs], "n_carry": 1, "length": 4, "starts": (0, 0), "strides": (0, 1)}
    nodes = raw_loop(**{**defaults, **kwargs})
    with pytest.raises(VerifyError, match=match):
      from_exprs("bad", [c0, xs], [nodes[0]], ["c0", "xs"], ["c"])

  refused("must keep its type", args=[sc.sym("c3", 3), c0, xs])
  refused("reads past", length=5)
  refused("reads past", starts=(1, 0))
  refused("cannot run in reverse", cond=mix_go3, reverse=True)
  refused("trip operand must be an int64 scalar", args=[c0, c0, xs, sc.const(2.0)])
  refused("must name distinct carries", in_place=(1,))
  refused("step number", index=True, args=[c0, xs], starts=(0,), strides=(1,))
  with pytest.raises(NotImplementedError, match="in_place"):
    sc.scan(mix, c0, (sc.broadcast(c0), xs), length=4, in_place=(0,))
  with pytest.raises(ValueError, match=r"expected slice shape \(2,\), got \(3,\)"):
    sc.function(sc.arg("c0", 3), sc.arg("xs", 4))(lambda c, xs: sc.scan(mix, c, (sc.broadcast(sc.const(np.ones(2))), xs), length=4)[0])


def test_derivatives_raise_and_an_inactive_loop_is_a_constant() -> None:
  p, x = sc.sym("p", 2), sc.sym("x", 2)
  (c, _) = sc.scan(mix, p, (sc.broadcast(sc.const(np.array([0.5, 2.0]))), sc.const(np.arange(3.0))), length=3)
  for derivative in (lambda: sc.jvp(c, p, sc.const(np.ones(2))), lambda: sc.vjp((c,), (p,), (sc.const(np.ones(2)),))):
    with pytest.raises(NotImplementedError, match="scan and while_loop"):
      derivative()
  y = x * c
  f = from_exprs("inactive", [x, p], [sc.jvp(y, x, sc.const(np.ones(2))), sc.vjp((y,), (x,), (sc.const(np.ones(2)),))[0]], ["x", "p"], ["fwd", "rev"])
  expected, _ = mix_numpy(np.array([1.0, 3.0]), np.array([0.5, 2.0]), [0.0, 1.0, 2.0])
  for got in f(np.zeros(2), np.array([1.0, 3.0])):
    np.testing.assert_allclose(got, expected, rtol=1e-15)


def dense(pattern: sc.SparsityPattern) -> list[list[int]]:
  out = np.zeros(pattern.shape, dtype=int)
  out[list(pattern.rows), list(pattern.cols)] = 1
  return out.tolist()


def test_sparsity_and_support_fixed_points() -> None:
  """A carry entry the body never touches keeps its own dependency and zero support; stacked outputs see their own slice."""

  @sc.function(sc.arg("c", 3), sc.arg("x", ()), outputs=sc.group(sc.arg("cn", 3), sc.arg("y", ())))
  def shift(c, x):
    return sc.stack([c[1], c[0] * x, c[2]]), c[1] + x

  c0, xs = sc.sym("c0", 3), sc.sym("xs", 4)
  final, (ys,) = sc.scan(shift, c0, (xs,), length=4)
  assert dense(sc.jacobian_sparsity(final, c0)) == [[1, 1, 0], [1, 1, 0], [0, 0, 1]]
  assert dense(sc.jacobian_sparsity(final, xs)) == [[1] * 4, [1] * 4, [0] * 4]
  assert dense(sc.jacobian_sparsity(ys, xs)) == [[1, 1, 1, 1]] * 4
  assert [row[2] for row in dense(sc.jacobian_sparsity(ys, c0))] == [0] * 4
  init = sc.stack([c0[0], sc.const(0.0), sc.const(0.0)])
  final, (ys,) = sc.scan(shift, init, (xs,), length=4)
  assert final.support == frozenset({0, 1}) and ys.support is None
  (once,), _ = sc.scan(shift, (init,), (xs[:1],), length=1)
  assert once.support == frozenset({0, 1})
  (zero,), _ = sc.scan(shift, (init,), (sc.sym("none", 0),), length=0)
  assert zero.support == frozenset({0})


def test_generated_c_is_constant_in_the_number_of_steps() -> None:
  from scaly.codegen import render_c_source

  def source(length: int) -> str:
    return render_c_source(rollout(length)).replace(f"_{length}", "_N").replace(f" < {length};", " < N;").replace(f"(res[1] + {length - 1}", "")

  assert source(20) == source(200) == source(2000)
  assert source(200).count("for (long long i_loop") == 1


def test_a_scan_in_a_scalar_procedure_stays_a_loop() -> None:
  """``scalarize.work`` refuses a carried range, so a procedure under the scalar hint does not unroll its scan."""
  from scaly.codegen import render_c_source

  @sc.function(sc.arg("z0", 4), sc.arg("us", 5))
  def scalar_roll(z0, us):
    zn, (costs,) = sc.scan(rk4, z0, (us,), length=5)
    return zn.with_lowering("scalar"), costs

  us = np.sin(np.arange(5) * 0.7)
  assert render_c_source(scalar_roll).count("rk4_raw(") == 2
  for got, want in zip(scalar_roll(Z0, us), rk4_numpy(Z0, us), strict=True):
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)


def test_a_mapped_invariant_while_loop_keeps_its_exit_variable_with_its_reader() -> None:
  """``hoist_invariant`` does not hoist a carried loop away from a store of its trip count that stays in the mapped body."""

  @sc.function(sc.arg("x", ()), sc.arg("w", 3), outputs=sc.group(sc.arg("y", ()), sc.arg("count", I64)))
  def invariant_while(x, w):
    c, count = sc.while_loop(go3, scale3, w, max_iter=6, params=(w,))
    return x + c.sum(), count

  @sc.function(sc.arg("xs", 5), sc.arg("w", 3))
  def mapped(xs, w):
    y, count = sc.vmap(invariant_while, 5)(xs, sc.broadcast(w))
    return y, count.cast(sc.dtypes.float64)

  w, c, k = np.array([0.7, -1.3, 0.2]), np.array([0.7, -1.3, 0.2]), 0
  while k < 6 and c @ c < 40.0:
    c, k = c * 1.5 + w, k + 1
  y, count = mapped(np.linspace(-1, 1, 5), w)
  np.testing.assert_allclose(y, np.linspace(-1, 1, 5) + c.sum(), rtol=1e-15)
  np.testing.assert_array_equal(count, np.full(5, k))


@sc.function(sc.arg("c", 3), sc.arg("w", 3), outputs=sc.arg("go", BOOL))
def go3(c, w):
  return sc.sumsqr(c) < 40.0


@sc.function(sc.arg("c", 3), sc.arg("w", 3), outputs=sc.arg("cn", 3))
def scale3(c, w):
  return c * 1.5 + w


# Ported from devrush's tests/core/integration/test_loop_edge_cases.py, primal parts only.

MAX_ITERS = (0, 1, 3, 4, 5, 6, 7)


def test_while_step_counts_at_every_boundary() -> None:
  """Each bound against a condition that holds for 0 to 7 steps, forever, or reads a NaN limit, on both sides of the slot parity."""

  @sc.function(sc.arg("c", 3), sc.arg("k", I64), sc.arg("lim", ()), outputs=sc.arg("cn", 3))
  def body(c, k, lim):
    return sc.stack([c[0] * 0.5 + c[2], c[1] + 1.0, c[2] + k.cast(sc.dtypes.float64)])

  @sc.function(sc.arg("c", 3), sc.arg("k", I64), sc.arg("lim", ()), outputs=sc.arg("go", BOOL))
  def cond(c, k, lim):
    return c[1] < lim

  @sc.function(sc.arg("c0", 3), sc.arg("lim", ()))
  def f(c0, lim):
    loops = [sc.while_loop(cond, body, c0, max_iter=m, index=True, params=(lim,)) for m in MAX_ITERS]
    return tuple(e for c, n in loops for e in (c, n.cast(sc.dtypes.float64)))

  c0, seen = np.array([1.0, 0.0, 0.25]), set()
  for lim in (np.nan, -1.0, 0.0, 1.0, 3.0, 4.0, 5.0, 6.0, 7.0, 100.0):
    got = f(c0, np.array(lim))
    for m, fin, n in zip(MAX_ITERS, got[::2], got[1::2], strict=True):
      c, steps = c0.copy(), 0
      while steps < m and c[1] < lim:
        c, steps = np.array([c[0] * 0.5 + c[2], c[1] + 1.0, c[2] + steps]), steps + 1
      np.testing.assert_array_equal(fin, c, err_msg=f"lim={lim} max_iter={m}")
      assert n == steps
      seen.add((steps == 0, steps == m, steps < m))
  assert seen == {(True, True, False), (True, False, True), (False, True, False), (False, False, True)}


def test_a_condition_that_reads_nan() -> None:
  """A comparison with NaN is false: ``y >= 0`` stops where ``y`` turns NaN, ``not y < 0`` runs to ``max_iter``, ``isfinite`` stops at once."""

  @sc.function(sc.arg("c", 2), outputs=sc.arg("cn", 2))
  def body(c):
    return sc.stack([c[0] - 1.0, (c[0] - 1.0).sqrt()])

  conds = {
    "ge": (lambda c: c[1] >= 0.0, lambda x, y: y >= 0.0),
    "not_lt": (lambda c: ~(c[1] < 0.0), lambda x, y: not y < 0.0),
    "eq_self": (lambda c: sc.equal(c[1], c[1]), lambda x, y: y == y),
    "finite": (lambda c: sc.isfinite(c[0]), lambda x, y: bool(np.isfinite(x))),
  }
  tests = [sc.function(sc.arg("c", 2), outputs=sc.arg("go", BOOL), name=f"nan_{name}")(go) for name, (go, _) in conds.items()]

  @sc.function(sc.arg("c0", 2))
  def f(c0):
    return tuple(e for test in tests for c, n in [sc.while_loop(test, body, c0, max_iter=8)] for e in (c, n.cast(sc.dtypes.float64)))

  counts = {}
  for start in (np.array([3.0, 0.0]), np.array([np.nan, 0.0]), np.array([np.inf, 1.0]), np.array([0.5, np.nan])):
    got = f(start)
    for k, (name, (_, go)) in enumerate(conds.items()):
      x, y, n = start[0], start[1], 0
      with np.errstate(invalid="ignore"):
        while n < 8 and go(x, y):
          x, y, n = x - 1.0, np.sqrt(x - 1.0), n + 1
      np.testing.assert_array_equal(got[2 * k], [x, y], err_msg=f"{name} from {start}")
      assert got[2 * k + 1] == n
      counts[name, start[0]] = n
  assert counts["ge", 3.0] == 4 and counts["not_lt", 3.0] == 8 and counts["finite", np.inf] == 0


def test_a_body_that_ignores_or_keeps_its_carry() -> None:
  """A kept carry stays bit for bit over odd and even step counts, an ignored one leaves only the last slice, and an empty carry still steps."""

  @sc.function(sc.arg("c", 3), sc.arg("u", ()), outputs=sc.group(sc.arg("cn", 3), sc.arg("y", 3)))
  def keep(c, u):
    return c, c * u

  @sc.function(sc.arg("c", 3), sc.arg("u", 3), outputs=sc.arg("cn", 3))
  def ignore(c, u):
    return u.sin() * 2.0

  @sc.function(sc.arg("c", 0), sc.arg("u", ()), outputs=sc.group(sc.arg("cn", 0), sc.arg("y", ())))
  def empty(c, u):
    return c, (u * 2.0).sin()

  @sc.function(sc.arg("c", 0), outputs=sc.arg("go", BOOL))
  def empty_go(c):
    return c.sum() < 1.0

  @sc.function(sc.arg("c", 0), outputs=sc.arg("cn", 0))
  def empty_body(c):
    return c * 2.0

  @sc.function(sc.arg("c0", 3), sc.arg("us", 3), sc.arg("ws", 12))
  def f(c0, us, ws):
    kept = [sc.scan(keep, c0, (us[:n],), length=n) for n in (1, 2, 3)]
    ignored, _ = sc.scan(ignore, c0, (ws.reshape((4, 3)),), length=4, reverse=True)
    nothing = sc.const(np.zeros(0))
    e_fin, (e_ys,) = sc.scan(empty, nothing, (us,), length=3)
    w_fin, w_n = sc.while_loop(empty_go, empty_body, nothing, max_iter=5)
    return (*(e for c, (ys,) in kept for e in (c, ys)), ignored, e_fin, e_ys, w_fin, w_n.cast(sc.dtypes.float64))

  odd, uv, wv = np.array([np.nan, -0.0, np.inf]), np.array([2.0, -1.0, 0.5]), np.linspace(-1.0, 1.0, 12)
  got = f(odd, uv, wv)
  for n in range(3):
    assert got[2 * n].tobytes() == odd.tobytes()
    np.testing.assert_array_equal(got[2 * n + 1], np.stack([odd * u for u in uv[: n + 1]]))
  np.testing.assert_array_equal(got[6], np.sin(wv[:3]) * 2.0)
  assert got[7].shape == got[9].shape == (0,) and got[10] == 5.0
  np.testing.assert_array_equal(got[8], np.sin(uv * 2.0))


@pytest.mark.parametrize("layout", ["csr", "csc"])
def test_sparse_carries_and_outputs(layout: str) -> None:
  """A carry with a stored pattern keeps it through every step; a sparse per-step output is stacked in its declared layout."""
  from scipy import sparse

  from scaly.ir.types import CanonicalPattern
  from scaly.codegen.jit import JitUnavailable

  pattern = CanonicalPattern((3, 3), (0, 1, 2, 2), (0, 1, 0, 2))
  t = sc.TensorType((3, 3), pattern=pattern)

  @sc.function(
    sc.arg("A", t, layout="csr"),
    sc.arg("x", ()),
    outputs=sc.group(sc.arg("An", t, layout="csr"), sc.arg("y", t, layout=layout)),
    name=f"sparse_step_{layout}",
  )
  def step(a, x):
    return a * 0.5 + a * x, a * x

  @sc.function(sc.arg("A0", t, layout="csr"), sc.arg("xs", 3), name=f"sparse_roll_{layout}")
  def roll(a0, xs):
    a, (ys,) = sc.scan(step, a0, (xs,), length=3)
    return sc.to_dense(a), sc.to_dense(ys)

  a0 = np.zeros((3, 3))
  a0[pattern.rows, pattern.cols] = [1.0, 2.0, 3.0, 4.0]
  xs, ref, ys = np.array([0.5, -1.0, 2.0]), a0.copy(), []
  for x in xs:
    ys.append(ref * x)
    ref = ref * 0.5 + ref * x
  got = roll(sparse.csr_array(a0), xs)
  np.testing.assert_allclose(got[0], ref, rtol=1e-15)
  np.testing.assert_allclose(got[1], np.stack(ys), rtol=1e-15)

  @sc.function(sc.arg("A", t, layout="csc"), outputs=sc.arg("An", t, layout="csc"), name=f"csc_step_{layout}")
  def csc_step(a):
    return a * 0.5

  @sc.function(sc.arg("A0", t, layout="csr"), name=f"csc_roll_{layout}")
  def csc_roll(a0):
    return sc.to_dense(sc.scan(csc_step, a0, length=2)[0])

  with pytest.raises(JitUnavailable, match="canonical sparse layout"):
    csc_roll(sparse.csr_array(a0))


def test_sparse_layouts_at_every_boundary() -> None:
  """A final carry into a CSC output, a condition whose parameter layout differs from the body's, and a SciPy slice in a declared layout."""
  from scipy import sparse

  from scaly.ir.types import CanonicalPattern

  t = sc.TensorType((2, 2), pattern=CanonicalPattern((2, 2), (0, 0, 1), (0, 1, 0)))
  a = sparse.csr_array([[1.0, 2.0], [5.0, 0.0]])

  @sc.function(sc.arg("a", t, layout="csr"), outputs=sc.arg("an", t, layout="csr"))
  def double(a):
    return a * 2.0

  @sc.function(sc.arg("a", t, layout="csr"), outputs=sc.arg("out", t, layout="csc"))
  def into_csc(a):
    return sc.scan(double, a, length=2)[0]

  np.testing.assert_array_equal(sparse.csr_array(into_csc(a)).toarray(), [[4.0, 8.0], [20.0, 0.0]])

  @sc.function(sc.arg("c", ()), sc.arg("a", t, layout="csr"))
  def advance(c, a):
    return c + 1.0

  @sc.function(sc.arg("c", ()), sc.arg("a", t, layout="csc"), outputs=sc.arg("go", BOOL))
  def below(c, a):
    return c < sc.to_dense(a)[0, 1]

  @sc.function(sc.arg("a", t, layout="csr"))
  def mixed(a):
    c, n = sc.while_loop(below, advance, sc.const(0.0), max_iter=20, params=(a,))
    return c, n.cast(sc.dtypes.float64)

  assert mixed(a) == (2.0, 2.0)

  @sc.function(sc.arg("c", ()), sc.arg("a", t, layout="csc"))
  def add_entry(c, a):
    return c + sc.to_dense(a)[0, 1]

  @sc.function(sc.arg("x", ()))
  def scipy_slice(x):
    return sc.scan(add_entry, x, (sc.broadcast(a),), length=2)[0]

  assert scipy_slice(np.array(0.0)) == 4.0


def test_inactive_slices_and_grouped_params() -> None:
  """An inactive slice before an active one keeps the active one's dependencies; a grouped parameter is read whole."""

  @sc.function(sc.arg("c", ()), sc.arg("unused", ()), sc.arg("b", ()))
  def add_last(c, unused, b):
    return c + b, b

  x = sc.sym("x", 3)
  c, (ys,) = sc.scan(add_last, sc.const(0.0), (sc.const(np.zeros(3)), x), length=3)
  assert sc.jacobian_sparsity(c, x).nnz == 3 and sc.jacobian_sparsity(ys, x).nnz == 3

  @sc.function(sc.arg("c", ()), sc.group(sc.arg("a", ()), sc.arg("b", ())))
  def grouped_step(c, pair):
    return c + pair[0]

  @sc.function(sc.arg("c", ()), sc.group(sc.arg("a", ()), sc.arg("b", ())), outputs=sc.arg("go", BOOL))
  def grouped_cond(c, pair):
    return c < pair[1]

  @sc.function(sc.arg("lim", ()))
  def grouped(lim):
    c, n = sc.while_loop(grouped_cond, grouped_step, sc.const(0.0), max_iter=10, params=((sc.const(1.0), lim),))
    return c, n.cast(sc.dtypes.float64)

  assert grouped(np.array(3.0)) == (3.0, 3.0)


def test_a_loop_on_the_variables_is_a_decision_for_the_quadratic_proof() -> None:
  """A while loop whose condition alone reads ``x`` makes its result piecewise in ``x``, which derivatives miss."""
  from scaly.solvers.qp import _decides_on

  @sc.function(sc.arg("c", ()), sc.arg("x", ()), outputs=sc.arg("go", BOOL))
  def under(c, x):
    return c < x

  @sc.function(sc.arg("c", ()), sc.arg("x", ()))
  def step(c, x):
    return c + 1.0

  x = sc.sym("x", ())
  c, _ = sc.while_loop(under, step, sc.const(0.0), max_iter=5, params=(x,))
  assert _decides_on((c,), (x,))
  assert not _decides_on((c,), (sc.sym("y", ()),))
