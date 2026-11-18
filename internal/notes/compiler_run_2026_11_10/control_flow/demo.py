"""Section 9 demo: the three Milestone 3 rows the section owns, a conditional call, and a clipped rollout.

Run from a checkout with ``uv run internal/notes/compiler_run_2026_11_10/control_flow/demo.py``. It takes source sizes
and generation times, and no run times.
"""

import re
import time
from unittest.mock import patch

import numpy as np

import scaly as sc
from scaly.ad import finite_difference
from scaly.codegen import render_c_module

DT = 0.05


def dynamics(z, u, sin, cos, stack):
  return stack([z[2], z[3], u - 0.1 * z[2], -9.81 * sin(z[1]) - u * cos(z[1])])


def rk4_step(z, u, sin, cos, stack):
  k1 = dynamics(z, u, sin, cos, stack)
  k2 = dynamics(z + 0.5 * DT * k1, u, sin, cos, stack)
  k3 = dynamics(z + 0.5 * DT * k2, u, sin, cos, stack)
  k4 = dynamics(z + DT * k3, u, sin, cos, stack)
  return z + (DT / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


@sc.function(sc.arg("z", 4), sc.arg("u", ()), outputs=sc.group(sc.arg("znext", 4), sc.arg("cost", ())))
def rk4(z, u):
  return rk4_step(z, u, lambda a: a.sin(), lambda a: a.cos(), sc.stack), sc.sumsqr(z) + 0.01 * u * u


def scanned(n):
  @sc.function(sc.arg("z0", 4), sc.arg("us", n), outputs=sc.arg("cost", ()), name=f"rollout_{n}")
  def rollout(z0, us):
    final, (costs,) = sc.scan(rk4, z0, (us,), length=n)
    return sc.sumsqr(final) + costs.sum()

  return rollout


def unrolled(n):
  @sc.function(sc.arg("z0", 4), sc.arg("us", n), outputs=sc.arg("cost", ()), name=f"unrolled_{n}")
  def rollout(z0, us):
    z, total = z0, 0.0
    for k in range(n):
      znext, cost = rk4(z, us[k])
      z, total = znext, total + cost
    return sc.sumsqr(z) + total

  return rollout


def rollout_np(z0, us):
  z, total = z0, 0.0
  for u in us:
    total += z @ z + 0.01 * u * u
    z = rk4_step(z, u, np.sin, np.cos, np.array)
  return z @ z + total


print("== 1. an RK4 rollout with its gradient, written with scan ==")
z0 = np.array([0.1, 0.4, -0.2, 0.05])
print("    N   value against NumPy   C bytes of value + both gradients   generation")
for n in (20, 200, 2000):
  us = 0.3 * np.sin(0.7 * np.arange(n))
  fun = scanned(n)
  both = fun.factory(f"value_and_gradients_{n}", ["z0", "us"], ["cost", sc.factory.Grad("cost", "z0"), sc.factory.Grad("cost", "us")])
  started = time.perf_counter()
  body = render_c_module(both).body
  elapsed = time.perf_counter() - started
  loops = len(re.findall(r"\bfor \(", body))
  print(
    f"{n:5d}   {abs(fun(z0, us) - rollout_np(z0, us)):.1e}               {len(body.encode()):6d} bytes, {loops} loops              {elapsed:.2f} s"
  )
n = 20
us = 0.3 * np.sin(0.7 * np.arange(n))
grad_z0, grad_us = sc.gradient(scanned(n), "cost", "z0"), sc.gradient(scanned(n), "cost", "us")
ref_z0, ref_us = sc.gradient(unrolled(n), "cost", "z0"), sc.gradient(unrolled(n), "cost", "us")
fd_z0 = finite_difference(lambda z: scanned(n)(z, us), z0).reshape(-1)
fd_us = finite_difference(lambda u: scanned(n)(z0, u), us).reshape(-1)
print(
  f"N = 20, d cost / d z0: max |scan - unrolled| {np.abs(grad_z0(z0, us) - ref_z0(z0, us)).max():.1e}, max |scan - finite differences| {np.abs(grad_z0(z0, us) - fd_z0).max():.1e}"
)
print(
  f"N = 20, d cost / d us: max |scan - unrolled| {np.abs(grad_us(z0, us) - ref_us(z0, us)).max():.1e}, max |scan - finite differences| {np.abs(grad_us(z0, us) - fd_us).max():.1e}"
)
forward = sc.jacobian(scanned(n), "cost", "z0")
print(f"N = 20, forward mode against reverse mode: {np.abs(np.asarray(forward(z0, us)).reshape(-1) - grad_z0(z0, us)).max():.1e}")

print("\n== 2. a Newton iteration inside while_loop ==")
flag = sc.TensorType((), sc.dtypes.bool_)
count_type = sc.TensorType((), sc.dtypes.int64)


@sc.function(sc.arg("x", ()), sc.arg("a", ()), outputs=sc.arg("go", flag))
def unsolved(x, a):
  return (x * x * x - a).abs() > 1e-13


@sc.function(sc.arg("x", ()), sc.arg("a", ()))
def newton(x, a):
  return x - (x * x * x - a) / (3.0 * x * x)


def cube_root(max_iter):
  @sc.function(sc.arg("a", ()), outputs=sc.group(sc.arg("root", ()), sc.arg("count", count_type)), name=f"cube_root_{max_iter}")
  def f(a):
    return sc.while_loop(unsolved, newton, sc.const(1.0), max_iter=max_iter, params=(a,))

  return f


solve = cube_root(50)
for a, label in ((1.0, "condition false at entry"), (5.0, "early exit"), (1e9, "early exit, far start")):
  root, count = solve(np.array(a))
  print(f"a = {a:g}: root {root:.12f} after {count} steps ({label}); error {abs(root - a ** (1 / 3)):.1e}")
root, count = cube_root(3)(np.array(5.0))
print(f"a = 5 with max_iter = 3: {root:.6f} after {count} steps (the bound is reached and nothing is raised)")
a = 5.0
first, second = sc.gradient(solve, "root", "a"), sc.hessian(solve, "root", "a")
fd_second = (first(np.array(a + 1e-5)) - first(np.array(a - 1e-5))) / 2e-5
print(f"d root / d a: {first(np.array(a)):.10f} (exact {a ** (-2 / 3) / 3:.10f})")
print(
  f"second derivative, forward over reverse: {np.asarray(second(np.array(a))).item():.10f} (exact {-2 / 9 * a ** (-5 / 3):.10f}, finite difference of the gradient {fd_second:.10f})"
)

print("\n== 3. run-time indices, and a carry updated in place ==")
i64 = sc.TensorType((3,), sc.dtypes.int64)


@sc.function(sc.arg("x", 6), sc.arg("idx", i64), sc.arg("start", count_type), outputs=sc.group(sc.arg("taken"), sc.arg("piece"), sc.arg("sums")))
def indexed(x, idx, start):
  return sc.take(x, idx, fill_value=-1.0), sc.dynamic_slice(x, start, 4), sc.segment_sum(x, sc.const(np.array([0, 2, 0, 2, 1, 2])), 4)


x = np.arange(1.0, 7.0)
taken, piece, sums = indexed(x, np.array([5, 9, 0]), np.int64(4))
ids = np.array([0, 2, 0, 2, 1, 2])
print(f"take(x, [5, 9, 0], fill -1) = {taken}   dynamic_slice(x, 4, 4) = {piece}")
print(f"segment_sum = {sums}, np.add.at gives {np.bincount(ids, weights=x, minlength=4)}")
SIZE = 100_000


@sc.function(sc.arg("carry", SIZE), sc.arg("k", count_type), name="bump")
def bump(carry, k):
  at = sc.stack([k, k + 7]) % SIZE
  return sc.scatter(carry, at, sc.gather(carry, at) + 1.0, combine="set")


@sc.function(sc.arg("carry", SIZE), sc.arg("k", count_type), name="bump_fill")
def bump_fill(carry, k):
  return sc.index_add(carry, sc.stack([k, k + 7]) % SIZE, sc.const(np.ones(2)))


def bumped(step, in_place):
  @sc.function(sc.arg("initial", SIZE), sc.arg("ks", sc.TensorType((50,), sc.dtypes.int64)), name=f"{step.name}_loop_{len(in_place)}")
  def f(initial, ks):
    return sc.scan(step, initial, (ks,), length=50, in_place=in_place)[0]

  return f


def whole_carry_loops(fun):
  module = render_c_module(fun)
  return f"workspace {module.workspace_size} bytes, {len(re.findall(rf'< {SIZE}\b', module.body))} loops over all {SIZE} entries in the C"


ks = (np.arange(50) * 37) % SIZE
expected = np.zeros(SIZE)
for k in ks:
  expected[[k, (k + 7) % SIZE]] += 1.0
strict = bumped(bump, (0,))
print(
  f"50 steps, each adding one to two entries, chosen at run time, of a carry of {SIZE} entries: max error {np.abs(strict(np.zeros(SIZE), ks) - expected).max():.1e}"
)
print("reused carry (in_place=(0,), also the default):", whole_carry_loops(strict))
with patch("scaly.passes.lowering.calls.carry_reuse", return_value={}):
  print("reuse switched off for comparison:             ", whole_carry_loops(bumped(bump, ())))
try:
  render_c_module(bumped(bump_fill, (0,)))
  print("the same step written with index_add (fill mode): proved")
except Exception as e:  # noqa: BLE001
  print(f"the same step written with index_add (fill mode), in_place=(0,): {type(e).__name__}: {e}")
print("without in_place=, index_add:                  ", whole_carry_loops(bumped(bump_fill, ())))

print("\n== 4. a conditional call runs one branch ==")


@sc.function(sc.arg("x", ()), outputs=sc.arg("root", ()))
def negative_root(x):
  return sc.print("negative branch, x={}", x * -1.0).sqrt()


@sc.function(sc.arg("x", ()), outputs=sc.arg("root", ()))
def positive_root(x):
  return sc.print("positive branch, x={}", x).sqrt()


@sc.function(sc.arg("x", ()), outputs=sc.arg("root", ()))
def absolute_root(x):
  return sc.cond(x > 0.0, (negative_root, positive_root), x)


print("value at -4:", absolute_root(np.array(-4.0)))
print("gradient at -4:", sc.gradient(absolute_root, "root", "x")(np.array(-4.0)))

print("\n== 5. a clipped rollout: select masks across iterations ==")


@sc.function(sc.arg("c", 2), sc.arg("u", ()), name="clipped_step")
def clipped_step(c, u):
  return sc.where(c > 0.0, c, 0.0).sqrt() + sc.where(u > 0.0, u, 0.0)


def clipped(n, loop):
  @sc.function(sc.arg("x", 2), sc.arg("us", n), outputs=sc.arg("y", ()), name=f"clipped_{n}_{loop:d}")
  def f(x, us):
    if loop:
      c = sc.scan(clipped_step, x, (us,), length=n)[0]
    else:
      c = x
      for k in range(n):
        c = clipped_step(c, us[k])
    return c.sum()

  return f


x, us = np.array([-1.0, 16.0]), np.array([0.0, -0.5, 0.25, 0.0, 1.0, -2.0, 0.0, 0.5])
for wrt, point in (("x", x), ("us", us)):
  rev_loop = sc.gradient(clipped(8, True), "y", wrt)(x, us)
  rev_ref = sc.gradient(clipped(8, False), "y", wrt)(x, us)
  fwd_loop = np.asarray(sc.jacobian(clipped(8, True), "y", wrt)(x, us)).reshape(-1)
  print(f"d y / d {wrt}: reverse scan {np.array2string(rev_loop, precision=5)}")
  print(
    f"           finite: {bool(np.isfinite(rev_loop).all())}; max |reverse scan - unrolled| {np.abs(rev_loop - rev_ref).max():.1e}; max |forward scan - unrolled| {np.abs(fwd_loop - rev_ref).max():.1e}"
  )
