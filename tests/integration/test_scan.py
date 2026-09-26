"""``scan`` is a loop in the generated C that computes what the unrolled chain of calls computes.

Each workload is checked three ways: against the same body called ``length`` times in a chain of
``CALL`` nodes, against a NumPy loop, and in its derivatives against finite differences and the
unrolled graph. The generated code must not grow with the number of steps.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

import scaly as sc
from scaly.ad import finite_difference
from scaly.codegen import render_c_source
from scaly.ir.program import ProgramNode, ProgramOp
from scaly.passes.lowering import lower_function
from scaly.passes.program._common import _walk

DT = 0.1


def _rk4_body(name: str = "rk4_step") -> sc.Function:
  """One RK4 step of a pendulum on a cart, with the stage cost as a second output."""
  z, u = sc.sym("z", 4), sc.sym("u", 1)

  def f(s: sc.Expr) -> sc.Expr:
    return sc.stack([s[2], s[3], u[0] - 0.1 * s[2], -9.81 * s[1].sin() - u[0] * s[1].cos()])

  k1 = f(z)
  k2 = f(z + 0.5 * DT * k1)
  k3 = f(z + 0.5 * DT * k2)
  k4 = f(z + DT * k3)
  nxt = z + (DT / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
  return sc.Function._from_exprs(name, [z, u], [nxt, sc.stack([sc.sumsqr(z) + 0.01 * u[0] * u[0]])], ["z", "u"], ["znext", "cost"])


def _rk4_numpy(z: np.ndarray, us: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  def f(s: np.ndarray, u: float) -> np.ndarray:
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


def _rollout(length: int, *, unrolled: bool = False) -> sc.Function:
  body = _rk4_body()
  z0, us = sc.sym("z0", 4), sc.sym("us", length)
  if unrolled:
    z, costs = z0, []
    for k in range(length):
      z, c = body._flat_symbolic_call([z, us[k : k + 1]])
      costs.append(c)
    outputs = [z, sc.concat(costs) if costs else sc.const(np.zeros(0))]
  else:
    outputs = list(sc.scan(body, z0, [(us, 0, 1)], length=length))
  return sc.Function._from_exprs(f"roll_{'u' if unrolled else 's'}{length}", [z0, us], outputs, ["z0", "us"], ["zN", "costs"])


Z0 = np.array([0.1, 0.4, -0.2, 0.05])


@pytest.mark.parametrize("length", [0, 1, 2, 3, 50])
def test_scan_matches_unrolled_calls_and_numpy(length: int) -> None:
  us = np.sin(np.arange(length) * 0.3)
  scanned, unrolled = _rollout(length)((Z0, us)), _rollout(length, unrolled=True)((Z0, us))
  zn, costs = _rk4_numpy(Z0, us)
  for got in (scanned, unrolled):
    np.testing.assert_allclose(got[0], zn, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(got[1], costs, rtol=1e-12, atol=1e-12)


def test_long_scan_matches_numpy() -> None:
  us = np.sin(np.arange(1000) * 0.01) * 0.2
  zn, costs = _rk4_numpy(Z0, us)
  got = _rollout(1000)((Z0, us))
  np.testing.assert_allclose(got[0], zn, rtol=1e-10, atol=1e-10)
  np.testing.assert_allclose(got[1], costs, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("length", [1, 2, 7])
def test_scan_derivatives_match_the_unrolled_graph_and_finite_differences(length: int) -> None:
  us = np.cos(np.arange(length) * 0.7) * 0.5
  scanned, unrolled = _rollout(length), _rollout(length, unrolled=True)
  for out in ("zN", "costs"):
    for wrt in ("z0", "us"):
      js, ju = sc.jacobian(scanned, out, wrt)((Z0, us)), sc.jacobian(unrolled, out, wrt)((Z0, us))
      np.testing.assert_allclose(js, ju, rtol=1e-11, atol=1e-12)
      fd = finite_difference(lambda v: scanned((v, us) if wrt == "z0" else (Z0, v))[0 if out == "zN" else 1], Z0 if wrt == "z0" else us)
      np.testing.assert_allclose(js, fd.reshape(js.shape), rtol=1e-6, atol=1e-7)
      np.testing.assert_allclose(
        sc.sparse_jacobian(scanned, out, wrt)((Z0, us)),
        _compact(js, sc.jacobian_sparsity(scanned.outputs[0 if out == "zN" else 1], scanned.inputs[0 if wrt == "z0" else 1])),
      )
  z0, u = scanned.inputs
  lam = np.linspace(-1.0, 1.0, 4)
  mu = np.linspace(0.5, 1.5, length)
  grads = sc.vjp(tuple(scanned.outputs), (z0, u), (sc.const(lam), sc.const(mu)))
  g = sc.Function._from_exprs(f"roll_vjp{length}", [z0, u], list(grads), ["z0", "us"], ["gz", "gu"])((Z0, us))
  jz, ju_ = (sc.jacobian(scanned, "zN", w)((Z0, us)) for w in ("z0", "us"))
  cz, cu = (sc.jacobian(scanned, "costs", w)((Z0, us)) for w in ("z0", "us"))
  np.testing.assert_allclose(g[0], lam @ jz + mu @ cz, rtol=1e-11, atol=1e-12)
  np.testing.assert_allclose(g[1], lam @ ju_ + mu @ cu, rtol=1e-11, atol=1e-12)
  cost = sc.Function._from_exprs(f"roll_cost{length}", [z0, u], [sc.sumsqr(scanned.outputs[0]) + scanned.outputs[1].sum()], ["z0", "us"], ["c"])
  hess = sc.hessian(cost, "c", "us")((Z0, us))
  np.testing.assert_allclose(hess, hess.T, atol=1e-12)
  np.testing.assert_allclose(
    hess,
    sc.hessian(
      sc.Function._from_exprs(
        f"roll_ucost{length}", list(unrolled.inputs), [sc.sumsqr(unrolled.outputs[0]) + unrolled.outputs[1].sum()], ["z0", "us"], ["c"]
      ),
      "c",
      "us",
    )((Z0, us)),
    rtol=1e-10,
    atol=1e-12,
  )


def _compact(dense: np.ndarray, pattern: sc.SparsityType) -> np.ndarray:
  return dense.reshape(pattern.shape)[list(pattern.rows), list(pattern.cols)]


def test_matrix_carry_kalman_filter_matches_numpy() -> None:
  """A carry that is a matrix and a vector packed flat: the covariance recursion of a Kalman filter."""
  n = 3
  a = np.array([[1.0, DT, 0.0], [0.0, 1.0, DT], [0.0, 0.0, 0.98]])
  c = np.array([1.0, 0.0, 0.0])
  q, r = 0.01 * np.eye(n), 0.1
  state = sc.sym("state", n + n * n)
  y = sc.sym("y", 1)
  x, p = state[:n], state[n:].reshape((n, n))
  xp, pp = sc.const(a) @ x, sc.const(a) @ p @ sc.const(a.T) + sc.const(q)
  s = sc.dot(sc.const(c), pp @ sc.const(c)) + r
  gain = (pp @ sc.const(c)) / s
  xn = xp + gain * (y[0] - sc.dot(sc.const(c), xp))
  pn = pp - gain.reshape((n, 1)) @ (sc.const(c).reshape((1, n)) @ pp)
  body = sc.Function._from_exprs("kf_step", [state, y], [sc.concat([xn, pn.reshape((n * n,))]), xn[:1]], ["state", "y"], ["next", "xhat"])
  init, ys = sc.sym("init", n + n * n), sc.sym("ys", 40)
  final, xhat = sc.scan(body, init, [(ys, 0, 1)], length=40)
  kf = sc.Function._from_exprs("kf", [init, ys], [final, xhat], ["init", "ys"], ["final", "xhat"])
  meas = np.sin(np.arange(40) * 0.2) + 0.05 * np.random.default_rng(0).normal(size=40)
  xv, pv = np.zeros(n), np.eye(n)
  ref = []
  for yk in meas:
    xv, pv = a @ xv, a @ pv @ a.T + q
    k = pv @ c / (c @ pv @ c + r)
    xv = xv + k * (yk - c @ xv)
    pv = pv - np.outer(k, c @ pv)
    ref.append(xv[0])
  got_final, got_xhat = kf((np.concatenate([np.zeros(n), np.eye(n).reshape(-1)]), meas))
  np.testing.assert_allclose(got_xhat, ref, rtol=1e-12, atol=1e-12)
  np.testing.assert_allclose(got_final, np.concatenate([xv, pv.reshape(-1)]), rtol=1e-12, atol=1e-12)


def test_slicing_broadcast_strided_and_backwards() -> None:
  c, w, v = sc.sym("c", 2), sc.sym("w", 2), sc.sym("v", 1)
  body = sc.Function._from_exprs("mix_step", [c, w, v], [c * w + v[0], sc.stack([c[0] - v[0]])], ["c", "w", "v"], ["n", "y"])
  c0, w_all, v_all = sc.sym("c0", 2), sc.sym("w_all", 2), sc.sym("v_all", 9)
  # w is the same at every step (stride 0); v is read backwards every third entry, from the end.
  final, ys = sc.scan(body, c0, [(w_all, 0, 0), (v_all, 8, -3)], length=3)
  f = sc.Function._from_exprs("mix", [c0, w_all, v_all], [final, ys], ["c0", "w", "v"], ["final", "ys"])
  c0v, wv, vv = np.array([1.0, -1.0]), np.array([0.5, 2.0]), np.arange(9.0)
  cv, yv = c0v.copy(), []
  for k in range(3):
    vk = vv[8 - 3 * k]
    yv.append(cv[0] - vk)
    cv = cv * wv + vk
  got = f((c0v, wv, vv))
  np.testing.assert_allclose(got[0], cv)
  np.testing.assert_allclose(got[1], yv)
  gv = sc.gradient(sc.Function._from_exprs("mix_c", [c0, w_all, v_all], [sc.sumsqr(final) + ys.sum()], ["c0", "w", "v"], ["c"]), "c", "v")(
    (c0v, wv, vv)
  )
  fd = finite_difference(lambda x: float(np.sum(f((c0v, wv, x))[0] ** 2) + np.sum(f((c0v, wv, x))[1])), vv)
  np.testing.assert_allclose(gv, fd.reshape(-1), rtol=1e-7, atol=1e-8)
  assert gv[[0, 1, 3, 4, 6, 7]].tolist() == [0.0] * 6


def test_generated_code_is_constant_in_the_number_of_steps() -> None:
  def shape(length: int) -> tuple[int, int, str]:
    fun = _rollout(length)
    prog = lower_function(fun)
    nodes = sum(1 for _ in _walk(prog))
    src = render_c_source(fun).replace(f" < {length};", " < N;").replace(f"roll_s{length}", "roll")
    return nodes, len(src), src

  n10, n1k, n100k = shape(10), shape(1000), shape(100_000)
  assert n10[0] == n1k[0] == n100k[0]
  assert n1k[2] == n100k[2]  # identical source apart from the trip count


def test_reverse_mode_stores_the_carry_at_every_step_and_forward_keeps_two() -> None:
  def carry_slots(fun: sc.Function) -> set[int]:
    return {int(np.prod(n.attrs["shape"])) for n in _walk(lower_function(fun)) if n.op == ProgramOp.BUFFER and n.attrs["address_space"] == "private"}

  assert 2 * 4 in carry_slots(_rollout(30))
  scanned = _rollout(30)
  z0, us = scanned.inputs
  (g,) = sc.vjp((scanned.outputs[0],), (us,), (sc.const(np.ones(4)),))
  assert 31 * 4 in carry_slots(sc.Function._from_exprs("roll_traj", [z0, us], [g], ["z0", "us"], ["g"]))


def test_scan_contract_errors() -> None:
  body = _rk4_body("rk4_err")
  with pytest.raises(ValueError, match="init"):
    sc.scan(body, sc.sym("z", 3), [(sc.sym("us", 4), 0, 1)], length=4)
  with pytest.raises(ValueError, match="reads outside"):
    sc.scan(body, sc.sym("z", 4), [(sc.sym("us", 4), 0, 1)], length=5)
  with pytest.raises(ValueError, match="sliced inputs"):
    sc.scan(body, sc.sym("z", 4), [], length=1)
  c = sc.sym("c", 2)
  bad = sc.Function._from_exprs("grow", [c], [sc.concat([c, c])], ["c"], ["n"])
  with pytest.raises(ValueError, match="carry like its input"):
    sc.scan(bad, c, [], length=2)


def _newton(n: int, name: str) -> tuple[sc.Function, sc.Function]:
  """Newton's method for ``x**3 = target`` elementwise; the carry is ``[x, target]``."""
  c = sc.sym("c", 2 * n)
  x, target = c[:n], c[n:]
  body = sc.Function._from_exprs(f"{name}_step", [c], [sc.concat([x - (x**3 - target) / (3.0 * x * x), target])], ["c"], ["cn"])
  cond = sc.Function._from_exprs(f"{name}_go", [c], [sc.norm_inf(x**3 - target) > 1e-12], ["c"], ["go"])
  return cond, body


def test_while_loop_newton_matches_a_python_loop() -> None:
  cond, body = _newton(5, "wn")
  c0 = sc.sym("c0", 10)
  final, count = sc.while_loop(cond, body, c0, max_iter=60)
  f = sc.Function._from_exprs("wn_solve", [c0], [final, count], ["c0"], ["c", "n"])
  targets = np.array([8.0, 27.0, 0.5, 2.0, 100.0])
  start = np.concatenate([np.full(5, 1.5), targets])
  got, n = f(start)
  x, steps = start[:5].copy(), 0
  while steps < 60 and np.max(np.abs(x**3 - targets)) > 1e-12:
    x = x - (x**3 - targets) / (3 * x * x)
    steps += 1
  np.testing.assert_array_equal(got[:5], x)
  assert n == steps and 3 < steps < 60
  np.testing.assert_allclose(got[:5], np.cbrt(targets), rtol=1e-12)


def test_while_loop_edge_counts() -> None:
  c = sc.sym("c", 1)
  inc = sc.Function._from_exprs("we_inc", [c], [c + 1.0], ["c"], ["cn"])
  never = sc.Function._from_exprs("we_never", [c], [c[0] > 1e9], ["c"], ["go"])
  always = sc.Function._from_exprs("we_always", [c], [c[0] > -1e9], ["c"], ["go"])
  finite = sc.Function._from_exprs("we_finite", [c], [sc.isfinite(c[0])], ["c"], ["go"])
  double = sc.Function._from_exprs("we_double", [c], [c * c], ["c"], ["cn"])
  c0 = sc.sym("c0", 1)
  outs = [*sc.while_loop(never, inc, c0, max_iter=5), *sc.while_loop(always, inc, c0, max_iter=5), *sc.while_loop(finite, double, c0, max_iter=40)]
  outs += [*sc.while_loop(always, inc, c0, max_iter=0), *sc.while_loop(always, inc, c0, max_iter=1)]
  f = sc.Function._from_exprs("we", [c0], outs, ["c0"], ["a", "na", "b", "nb", "d", "nd", "z", "nz", "o", "no"])
  a, na, b, nb, d, nd, z, nz, o, no = f(np.array([3.0]))
  assert (a[0], na, b[0], nb) == (3.0, 0.0, 8.0, 5.0)  # zero steps return init; max_iter steps equal a scan
  assert d[0] == np.inf and nd == 10  # 3**(2**k) overflows at the 10th squaring and the guard stops it
  assert (z[0], nz, o[0], no) == (3.0, 0.0, 4.0, 1.0)
  (scanned,) = sc.scan(inc, c0, [], length=5)
  np.testing.assert_array_equal(sc.Function._from_exprs("we_scan", [c0], [scanned], ["c0"], ["s"])(np.array([3.0])), b)


def test_while_loop_derivatives_through_the_steps() -> None:
  """A contractive fixed-point iteration ``x = 0.5 cos x + p``: the derivative through the steps taken
  matches finite differences, and it approaches the implicit derivative as the tolerance shrinks."""
  c = sc.sym("c", 2)
  x, p = c[0], c[1]
  body = sc.Function._from_exprs("wd_step", [c], [sc.stack([0.5 * x.cos() + p, p])], ["c"], ["cn"])

  def solve(tol: float) -> sc.Function:
    cond = sc.Function._from_exprs(f"wd_go_{tol:g}", [c], [(0.5 * x.cos() + p - x).abs() > tol], ["c"], ["go"])
    c0 = sc.sym("c0", 2)
    final, _ = sc.while_loop(cond, body, c0, max_iter=200)
    return sc.Function._from_exprs(f"wd_{tol:g}", [c0], [final[:1] * 2.0], ["c0"], ["x"])

  start = np.array([0.0, 0.3])
  loose, tight = solve(1e-3), solve(1e-13)
  x_star = tight(start)[0] / 2.0
  implicit = 2.0 / (1.0 + 0.5 * np.sin(x_star))
  for fun in (loose, tight):
    jac = sc.jacobian(fun, "x", "c0")(start)
    np.testing.assert_allclose(jac, finite_difference(lambda v: fun(v), start).reshape(jac.shape), rtol=1e-6, atol=1e-8)
    g = sc.gradient(sc.Function._from_exprs(f"{fun.name}_s", list(fun.inputs), [fun.outputs[0].sum()], ["c0"], ["s"]), "s", "c0")(start)
    np.testing.assert_allclose(g, jac.reshape(-1), rtol=1e-12, atol=1e-14)
  assert abs(sc.jacobian(loose, "x", "c0")(start)[0, 1] - implicit) > 1e-5
  assert abs(sc.jacobian(tight, "x", "c0")(start)[0, 1] - implicit) < 1e-10


def test_while_loop_code_is_constant_in_max_iter_and_copies_once() -> None:
  cond, body = _newton(3, "wc")

  def render(max_iter: int) -> str:
    c0 = sc.sym("c0", 6)
    final, count = sc.while_loop(cond, body, c0, max_iter=max_iter)
    return render_c_source(sc.Function._from_exprs("wc", [c0], [final, count], ["c0"], ["c", "n"])).replace(f" < {max_iter};", " < K;")

  assert render(10) == render(100_000)
  src = render(10)
  assert src.count("break;") == 1 and src.count("long long k_") == 1


def _accumulator(size: int, name: str) -> sc.Function:
  """A carry of ``size`` entries of which each step changes four: two sums, a running product and
  a copy of the freshly updated first sum, the last read after the write in the same step."""
  c, x = sc.sym("c", size), sc.sym("x", 2)
  u1 = sc.index_add(c, [0, 1], x * c[2:4])
  u2 = sc.index_set(u1, [size - 1, size - 2], sc.stack([u1[0] * 0.5, u1[1] + u1[2]]))
  return sc.Function._from_exprs(name, [c, x], [u2], ["c", "x"], ["cn"])


def _procs(fun: sc.Function) -> dict[str, ProgramNode]:
  return {pr.attrs["name"]: pr for pr in lower_function(fun).args}


@pytest.mark.parametrize("size", [6, 400])
def test_in_place_carry_matches_the_two_slot_carry(size: int, monkeypatch: pytest.MonkeyPatch) -> None:
  import scaly.passes.lowering as lowering

  body = _accumulator(size, f"acc{size}")
  c0, xs = sc.sym("c0", size), sc.sym("xs", 40)
  (scanned,) = sc.scan(body, c0, [(xs, 0, 2)], length=20)
  start = np.linspace(0.5, 1.5, size)
  data = np.sin(np.arange(40.0))
  donated = sc.Function._from_exprs(f"acc_ip{size}", [c0, xs], [scanned], ["c0", "xs"], ["c"])
  assert f"acc{size}_inplace" in _procs(donated)
  monkeypatch.setattr(lowering, "DONATE_CARRIES", False)
  two_slot = sc.Function._from_exprs(f"acc_2s{size}", [c0, xs], [scanned], ["c0", "xs"], ["c"])
  assert f"acc{size}_inplace" not in _procs(two_slot)
  np.testing.assert_array_equal(donated((start, data)), two_slot((start, data)))
  c, ref = start.copy(), None
  for k in range(20):
    c[0:2] += data[2 * k : 2 * k + 2] * c[2:4]
    c[size - 1], c[size - 2] = c[0] * 0.5, c[1] + c[2]
  ref = c
  np.testing.assert_allclose(donated((start, data)), ref, rtol=1e-13)


def test_in_place_body_touches_only_its_indices_and_holds_one_carry() -> None:
  size = 400
  body = _accumulator(size, "acc_struct")
  c0, xs = sc.sym("c0", size), sc.sym("xs", 40)
  (scanned,) = sc.scan(body, c0, [(xs, 0, 2)], length=20)
  procs = _procs(sc.Function._from_exprs("acc_struct_host", [c0, xs], [scanned], ["c0", "xs"], ["c"]))
  inplace = procs["acc_struct_inplace"]
  assert inplace.attrs.get("in_place") and inplace.attrs["scalarize_mode"] == "disabled"
  writes = 0
  for node in _walk(inplace):
    if node.op == ProgramOp.FOR:
      start, stop, _ = (a.attrs["value"] for a in node.args[0].args)
      writes += (stop - start) * sum(1 for s in node.args[1:] if s.op == ProgramOp.STORE and s.args[0].attrs["buffer"] == "cn")
    elif (
      node.op == ProgramOp.STORE
      and node.args[0].attrs["buffer"] == "cn"
      and not any(node in set(_walk(f)) for f in _walk(inplace) if f.op == ProgramOp.FOR)
    ):
      writes += 1
  assert writes == 4  # per step, whatever the carry size
  host = procs["acc_struct_host"]
  sizes = [int(np.prod(n.attrs["shape"])) for n in _walk(host) if n.op == ProgramOp.BUFFER and n.attrs["address_space"] == "private"]
  assert size in sizes and 2 * size not in sizes


def test_bodies_that_read_what_they_overwrite_keep_two_slots() -> None:
  c, x = sc.sym("c", 4), sc.sym("x", 1)
  cases = {
    # another output reads the carry after it may have been overwritten
    "ip_neg_y": ([sc.index_add(c, [0], x), c[:1]], ["cn", "y"]),
    # the second update reads the carry before the first update rather than after it
    "ip_neg_stale": ([sc.index_add(sc.index_add(c, [0], x), [1], c[:1]), None], ["cn"]),
    # the values read an entry the same update writes
    "ip_neg_overlap": ([sc.index_add(c, [0, 1], c[1:3] * x[0]), None], ["cn"]),
    # a read through a comparison cannot be traced entry by entry
    "ip_neg_select": ([sc.index_set(c, [3], sc.where(c[0] > 0.0, x, -x)), None], ["cn"]),
    # the carry is recomputed, not updated
    "ip_neg_dense": ([c * 2.0, None], ["cn"]),
    # a later update reads entries it writes (a swap): the read set is taken against the link itself
    "ip_neg_swap": ([sc.index_set(sc.index_add(c, [0], x), [2, 1], sc.index_add(c, [0], x)[1:3]), None], ["cn"]),
    # copysign's sign operand is read although its derivative pattern omits it
    "ip_neg_copysign": ([sc.index_set(c, [0, 1], sc.copysign(sc.concat([x, x]), c[1::-1])), None], ["cn"]),
  }
  import scaly.passes.lowering as lowering

  for name, (outs, names) in cases.items():
    fun = sc.Function._from_exprs(name, [c, x], [o for o in outs if o is not None], ["c", "x"], names)
    assert lowering.in_place_chain(lowering._normalize_function(fun)) is None, name
    c0, xs = sc.sym("c0", 4), sc.sym("xs", 3)
    results = sc.scan(fun, c0, [(xs, 0, 1)], length=3)
    host = sc.Function._from_exprs(f"{name}_host", [c0, xs], list(results), ["c0", "xs"], [f"o{i}" for i in range(len(results))])
    assert f"{name}_inplace" not in _procs(host)
    start, data = np.array([1.0, -2.0, 3.0, 0.5]), np.array([0.3, -0.7, 1.1])
    cv = start.copy()
    for k in range(3):
      step = fun._flat_numerical_call(cv, data[k : k + 1])
      cv = step[0]
    np.testing.assert_allclose(host._flat_numerical_call(start, data)[0], cv, rtol=1e-13)
  ok = sc.Function._from_exprs("ip_pos", [c, x], [sc.index_add(sc.index_add(c, [0], x), [1], sc.index_add(c, [0], x)[:1])], ["c", "x"], ["cn"])
  assert lowering.in_place_chain(lowering._normalize_function(ok)) is not None


def test_in_place_while_loop_and_reverse_mode_fall_back() -> None:
  size = 50
  c = sc.sym("c", size)
  body = sc.Function._from_exprs("ipw_step", [c], [sc.index_add(c, [0, 1], sc.stack([1.0 + 0.0 * c[2], c[0] * 0.0 + 2.0]))], ["c"], ["cn"])
  cond = sc.Function._from_exprs("ipw_go", [c], [c[0] < 7.5], ["c"], ["go"])
  c0 = sc.sym("c0", size)
  final, count = sc.while_loop(cond, body, c0, max_iter=100)
  fun = sc.Function._from_exprs("ipw", [c0], [final, count], ["c0"], ["c", "n"])
  assert "ipw_step_inplace" in _procs(fun)
  got, n = fun(np.zeros(size))
  assert n == 8 and got[0] == 8.0 and got[1] == 16.0 and not got[2:].any()
  (grad,) = sc.vjp((final.sum(),), (c0,), (sc.const(1.0),))
  rev = sc.Function._from_exprs("ipw_grad", [c0], [grad], ["c0"], ["g"])
  assert "ipw_step_inplace" not in _procs(rev)  # reverse mode needs every carry, so two slots are not enough either
  np.testing.assert_array_equal(rev(np.zeros(size)), np.ones(size))


def test_index_updates_differentiate() -> None:
  c, v = sc.sym("c", 5), sc.sym("v", 3)
  y = sc.index_set(sc.index_add(c * c, [1, 1, 4], v.sin()), [0, 2], v[:2] * c[3])
  f = sc.Function._from_exprs("idx_d", [c, v], [y], ["c", "v"], ["y"])
  cv, vv = np.array([0.3, -1.0, 2.0, 0.7, 1.5]), np.array([0.2, -0.4, 0.9])
  expected = cv * cv
  np.add.at(expected, [1, 1, 4], np.sin(vv))
  expected[[0, 2]] = vv[:2] * cv[3]
  np.testing.assert_allclose(f((cv, vv)), expected)
  for wrt, point in (("c", cv), ("v", vv)):
    jac = sc.jacobian(f, "y", wrt)((cv, vv))
    fd = finite_difference(lambda p: f((p, vv) if wrt == "c" else (cv, p)), point)
    np.testing.assert_allclose(jac, fd.reshape(jac.shape), rtol=1e-7, atol=1e-9)
    pattern = sc.jacobian_sparsity(y, f.inputs[0 if wrt == "c" else 1])
    assert set(zip(pattern.rows, pattern.cols)) == {tuple(ix) for ix in np.argwhere(np.abs(jac) > 0)}
  lam = np.arange(1.0, 6.0)
  gc, gv = sc.vjp((y,), tuple(f.inputs), (sc.const(lam),))
  g = sc.Function._from_exprs("idx_g", list(f.inputs), [gc, gv], ["c", "v"], ["gc", "gv"])((cv, vv))
  np.testing.assert_allclose(g[0], lam @ sc.jacobian(f, "y", "c")((cv, vv)), rtol=1e-12, atol=1e-14)
  np.testing.assert_allclose(g[1], lam @ sc.jacobian(f, "y", "v")((cv, vv)), rtol=1e-12, atol=1e-14)
  with pytest.raises(ValueError, match="distinct"):
    sc.index_set(c, [1, 1], v[:2])


def _cubic_solver(n: int, name: str) -> sc.Function:
  """``x**3 + x = p`` elementwise by Newton inside a ``while_loop``; ``dx/dp = 1 / (3x^2 + 1)``."""
  c = sc.sym("c", 2 * n)
  x, p = c[:n], c[n:]
  body = sc.Function._from_exprs(f"{name}_step", [c], [sc.concat([x - (x**3 + x - p) / (3 * x * x + 1), p])], ["c"], ["cn"])
  cond = sc.Function._from_exprs(f"{name}_go", [c], [sc.norm_inf(x**3 + x - p) > 1e-14], ["c"], ["go"])
  pp = sc.sym("p", n)
  final, _ = sc.while_loop(cond, body, sc.concat([sc.const(np.zeros(n)), pp]), max_iter=60)
  return sc.Function._from_exprs(name, [pp], [final[:n]], ["p"], ["x"])


def _implicit_rules(n: int, name: str, scale: float = 1.0) -> tuple[sc.Function, sc.Function]:
  p, x, bar, dp = sc.sym("p", n), sc.sym("x", n), sc.sym("xbar", n), sc.sym("dp", n)
  slope = scale / (3 * x * x + 1)
  # The forward rule gets only the inputs, so it recomputes the solution through the solver itself.
  solver = _cubic_solver(n, f"{name}_inner")
  xs = solver(p)
  jvp = sc.Function._from_exprs(f"{name}_jvp", [p, dp], [dp * scale / (3 * xs * xs + 1)], ["p", "dp"], ["dx"])
  vjp = sc.Function._from_exprs(f"{name}_vjp", [p, x, bar], [bar * slope], ["p", "x", "xbar"], ["pbar"])
  return jvp, vjp


P = np.array([0.5, 2.0, 10.0])


def _cubic_root(p: np.ndarray) -> np.ndarray:
  return np.array([np.real([r for r in np.roots([1.0, 0.0, 1.0, -v]) if abs(r.imag) < 1e-12][0]) for v in p])


def test_custom_derivatives_replace_differentiating_the_solver_steps() -> None:
  solver = _cubic_solver(3, "cd_solve")
  jvp, vjp = _implicit_rules(3, "cd_rules")
  custom = sc.custom_derivative(solver, jvp=jvp, vjp=vjp)
  q = sc.sym("q", 3)
  host = sc.Function._from_exprs("cd_host", [q], [custom(q)], ["q"], ["x"])
  xs = _cubic_root(P)
  exact = np.diag(1.0 / (3 * xs * xs + 1))
  np.testing.assert_allclose(host(P), xs, rtol=1e-13)
  np.testing.assert_allclose(sc.jacobian(host, "x", "q")(P), exact, rtol=1e-12, atol=1e-15)
  cost = sc.Function._from_exprs("cd_cost", [q], [(custom(q) ** 2).sum()], ["q"], ["c"])
  grad = sc.gradient(cost, "c", "q")
  np.testing.assert_allclose(grad(P), 2 * xs * np.diag(exact), rtol=1e-12)
  np.testing.assert_allclose(sc.sparse_jacobian(host, "x", "q")(P), np.diag(exact), rtol=1e-12)
  # The gradient reads the solution and never runs a backward pass over the solver's steps.
  names = {pr.attrs["name"] for pr in lower_function(grad).args}
  assert not any("whileadj" in name for name in names)


def test_custom_rules_are_the_ones_used_in_both_modes_and_under_vmap() -> None:
  solver = _cubic_solver(1, "cd_wrong_solve")
  jvp, vjp = _implicit_rules(1, "cd_wrong", scale=2.0)  # deliberately twice the true derivative
  custom = sc.custom_derivative(solver, jvp=jvp, vjp=vjp)
  q = sc.sym("q", 3)
  mapped = sc.vmap(custom, 3, [(q, 0, 1)])
  host = sc.Function._from_exprs("cd_wrong_host", [q], [mapped], ["q"], ["x"])
  xs = _cubic_root(P)
  doubled = np.diag(2.0 / (3 * xs * xs + 1))
  np.testing.assert_allclose(sc.jacobian(host, "x", "q")(P), doubled, rtol=1e-12, atol=1e-15)
  (g,) = sc.vjp((mapped,), (q,), (sc.const(np.ones(3)),))
  np.testing.assert_allclose(sc.Function._from_exprs("cd_wrong_g", [q], [g], ["q"], ["g"])(P), np.diag(doubled), rtol=1e-12)


def test_a_missing_direction_differentiates_the_body() -> None:
  solver = _cubic_solver(3, "cd_half_solve")
  _, vjp = _implicit_rules(3, "cd_half")
  custom = sc.custom_derivative(solver, vjp=vjp)
  q = sc.sym("q", 3)
  host = sc.Function._from_exprs("cd_half_host", [q], [custom(q)], ["q"], ["x"])
  xs = _cubic_root(P)
  # Forward mode goes through the Newton steps; at this tolerance that matches the implicit value.
  np.testing.assert_allclose(sc.jacobian(host, "x", "q")(P), np.diag(1.0 / (3 * xs * xs + 1)), rtol=1e-9, atol=1e-12)
  with pytest.raises(ValueError, match="must map shapes"):
    sc.custom_derivative(solver, vjp=sc.Function._from_exprs("cd_bad", [sc.sym("a", 3)], [sc.sym("a", 3)], ["a"], ["b"]))
  with pytest.raises(TypeError, match="scaly Function"):
    sc.custom_derivative(solver, jvp=lambda p: p)


def test_custom_derivative_copies_keep_their_own_derivatives() -> None:
  """A Function and its custom-derivative copy in one graph each keep their own derivative, in a
  call, under ``vmap`` in either order, and as a ``scan`` body."""
  x, t, bar = sc.sym("x", 1), sc.sym("t", 1), sc.sym("bar", 1)
  plain = sc.Function._from_exprs("cdn_sq", [x], [x * x], ["x"], ["y"])
  jvp = sc.Function._from_exprs("cdn_jvp", [x, t], [20.0 * x * t], ["x", "t"], ["dy"])
  vjp = sc.Function._from_exprs("cdn_vjp", [x, sc.sym("y", 1), bar], [20.0 * x * bar], ["x", "y", "bar"], ["xbar"])
  custom = sc.custom_derivative(plain, jvp=jvp, vjp=vjp)
  assert custom.name != plain.name
  q = sc.sym("q", 1)
  for first, second in ((plain, custom), (custom, plain)):
    host = sc.Function._from_exprs(f"cdn_host_{first.name}", [q], [first(q) + second(q)], ["q"], ["y"])
    assert sc.jacobian(host, "y", "q")(np.array([3.0]))[0, 0] == 66.0  # 2x + 20x at x = 3
    assert (
      sc.gradient(sc.Function._from_exprs(f"cdn_s_{first.name}", [q], [(first(q) + second(q)).sum()], ["q"], ["s"]), "s", "q")(np.array([3.0]))[0]
      == 66.0
    )
  qs = sc.sym("qs", 2)
  mapped = sc.vmap(plain, 2, [(qs, 0, 1)]) + sc.vmap(custom, 2, [(qs, 0, 1)])
  host = sc.Function._from_exprs("cdn_vmap", [qs], [mapped], ["qs"], ["y"])
  np.testing.assert_array_equal(np.diag(sc.jacobian(host, "y", "qs")(np.array([3.0, 1.0]))), [66.0, 22.0])
  # As a scan body, the rule of the Function being scanned is the one differentiated.
  c, u = sc.sym("c", 1), sc.sym("u", 1)
  step = sc.Function._from_exprs("cdn_step", [c, u], [c * c + u], ["c", "u"], ["cn"])
  rule = sc.Function._from_exprs(
    "cdn_step_jvp", [c, u, sc.sym("dc", 1), sc.sym("du", 1)], [5.0 * sc.sym("dc", 1) + sc.sym("du", 1)], ["c", "u", "dc", "du"], ["dcn"]
  )
  back = sc.Function._from_exprs("cdn_step_vjp", [c, u, sc.sym("cn", 1), bar], [5.0 * bar, bar], ["c", "u", "cn", "bar"], ["cbar", "ubar"])
  stepped = sc.custom_derivative(step, jvp=rule, vjp=back)
  c0, us = sc.sym("c0", 1), sc.sym("us", 3)
  (final,) = sc.scan(stepped, c0, [(us, 0, 1)], length=3)
  loop = sc.Function._from_exprs("cdn_loop", [c0, us], [final], ["c0", "us"], ["c"])
  assert sc.jacobian(loop, "c", "c0")((np.array([0.5]), np.zeros(3)))[0, 0] == 125.0
  (g,) = sc.vjp((final,), (c0,), (sc.const(np.ones(1)),))
  assert sc.Function._from_exprs("cdn_loop_g", [c0, us], [g], ["c0", "us"], ["g"])((np.array([0.5]), np.zeros(3)))[0] == 125.0


# --- Multi-seed forward mode through loops (C-93) -----------------------------------------------


def _loop_nodes(exprs: list[sc.Expr]) -> list[sc.Expr]:
  """Every ``scan`` and ``while_loop`` node in the graph of ``exprs``, callees included."""
  from scaly.ir.expr import CALLEE_OPS, ExprOp, callees_of, topo

  found, seen, todo = [], set(), list(exprs)
  while todo:
    for node in topo(todo):
      if node.id in seen:
        continue
      seen.add(node.id)
      if node.op in (ExprOp.SCAN, ExprOp.WHILE):
        found.append(node)
      if node.op in CALLEE_OPS:
        todo.extend(out for fn in callees_of(node) for out in fn.outputs)
    todo = [e for e in todo if e.id not in seen]
  return found


def test_jacobian_of_a_scan_is_one_scan_carrying_every_seed(monkeypatch: pytest.MonkeyPatch) -> None:
  """``jacobian`` pushes all its seeds through one tangent scan instead of one scan per seed; strict
  mode proves no per-seed fallback happens."""
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  length = 6
  scanned = _rollout(length)
  us = np.cos(np.arange(length) * 0.7) * 0.5
  for i, out in enumerate(("zN", "costs")):
    for j, wrt in enumerate(("z0", "us")):
      js = sc.jacobian(scanned, out, wrt)((Z0, us))
      fd = finite_difference(lambda v: scanned((v, us) if wrt == "z0" else (Z0, v))[i], Z0 if wrt == "z0" else us)
      np.testing.assert_allclose(js, fd.reshape(js.shape), rtol=1e-6, atol=1e-7)
      nseed = scanned.inputs[j].size
      loops = _loop_nodes([sc.jacobian(scanned.outputs[i], scanned.inputs[j])])
      assert {n.attrs["callee"].name for n in loops} == {f"rk4_step_scanfwd{nseed}_{'c' if wrt == 'z0' else '0'}"}


def test_multi_seed_tangents_of_broadcast_strided_and_backward_slices(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  c, w, v = sc.sym("c", 2), sc.sym("w", 2), sc.sym("v", 1)
  body = sc.Function._from_exprs("mixm_step", [c, w, v], [c * w + v[0] * c[::-1], sc.stack([c[0] * v[0]])], ["c", "w", "v"], ["n", "y"])
  c0, w_all, v_all = sc.sym("c0", 2), sc.sym("w_all", 2), sc.sym("v_all", 9)
  final, ys = sc.scan(body, c0, [(w_all, 0, 0), (v_all, 8, -3)], length=3)
  f = sc.Function._from_exprs("mixm", [c0, w_all, v_all], [final, ys], ["c0", "w", "v"], ["final", "ys"])
  point = (np.array([1.0, -1.0]), np.array([0.5, 2.0]), np.linspace(-1.0, 1.0, 9))
  for k, wrt in enumerate(("c0", "w", "v")):
    for i, out in enumerate(("final", "ys")):
      jac = sc.jacobian(f, out, wrt)(point)

      def at(x: np.ndarray, k: int = k, i: int = i) -> np.ndarray:
        return f(tuple(x if m == k else p for m, p in enumerate(point)))[i]

      np.testing.assert_allclose(jac, finite_difference(at, point[k]).reshape(jac.shape), rtol=1e-7, atol=1e-9)


def _single_shooting_cost(n: int) -> sc.Function:
  """The double-integrator MPC cost of the user example: states simulated by a scan inside a
  Function that the cost calls, with every state and input penalised."""
  a = np.array([[1.0, DT], [0.0, 1.0]])
  b = np.array([[0.5 * DT**2], [DT]])
  z, u = sc.sym("z", 2), sc.sym("u", 1)
  zn = a @ z + b @ u
  step = sc.Function._from_exprs(f"ss{n}_step", [z, u], [zn, zn, sc.stack([sc.sumsqr(z) + 0.1 * sc.sumsqr(u)])], ["z", "u"], ["zn", "zo", "c"])
  x0, big_u = sc.sym("x0", 2), sc.sym("U", n)
  _, xs, costs = sc.scan(step, x0, [(big_u, 0, 1)], length=n)
  shoot = sc.Function._from_exprs(f"ss{n}_shoot", [x0, big_u], [xs, costs], ["x0", "U"], ["X", "costs"])
  xs_c, costs_c = shoot._flat_symbolic_call([x0, big_u])
  return sc.Function._from_exprs(f"ss{n}_cost", [x0, big_u], [costs_c.sum() + 10.0 * sc.sumsqr(xs_c)], ["x0", "U"], ["f"])


def _condensed_hessian(n: int) -> np.ndarray:
  a = np.array([[1.0, DT], [0.0, 1.0]])
  b = np.array([[0.5 * DT**2], [DT]])
  phi = [np.linalg.matrix_power(a, k) for k in range(n + 1)]
  gam = [np.hstack([phi[k - 1 - j] @ b if j < k else np.zeros((2, 1)) for j in range(n)]) for k in range(n + 1)]
  return sum(2 * g.T @ g for g in gam[:n]) + sum(20 * g.T @ g for g in gam[1:]) + 0.2 * np.eye(n)


def test_hessian_through_a_scan_is_a_fixed_number_of_loops(monkeypatch: pytest.MonkeyPatch) -> None:
  """Forward over reverse through single shooting: one tangent scan forwards and the adjoint scans
  backwards, each carrying every seed, however long the horizon. The adjoint scans read the stored
  tangent carries in place, and the call around the scan gets no separate tangent helper."""
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  from scaly.ir.expr import ExprOp

  shapes = []
  for n in (5, 11):
    cost = _single_shooting_cost(n)
    hess = sc.hessian(cost, "f", "U")
    point = (np.array([1.0, -0.5]), np.sin(np.arange(n)))
    np.testing.assert_allclose(hess(point), _condensed_hessian(n), rtol=1e-12, atol=1e-12)
    loops = _loop_nodes(list(hess.outputs))
    assert len({n.attrs["callee"].name for n in loops}) == 4  # primal, tangent, and one adjoint per scan output
    for node in loops:
      for outer in node.args[1:]:
        copied = False
        while outer.op in (ExprOp.RESHAPE, ExprOp.GATHER):
          copied |= outer.op == ExprOp.GATHER
          outer = outer.args[0]
        assert not (copied and outer.op == ExprOp.SCAN), "a tangent trajectory is copied before a scan reads it"
    src = render_c_source(hess)
    assert f"ss{n}_shoot_fwd" not in src
    shapes.append((src.count("for ("), sorted(p.replace(str(n), "N") for p in re.findall(r"void (\w+)_raw\(", src))))
  assert shapes[0] == shapes[1]


def test_jacobian_of_a_while_loop_is_one_loop_carrying_every_seed(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  cond, body = _newton(3, "wm")
  c0 = sc.sym("c0", 6)
  final, _ = sc.while_loop(cond, body, c0, max_iter=60)
  f = sc.Function._from_exprs("wm_solve", [c0], [final[:3]], ["c0"], ["x"])
  start = np.array([1.5, 1.5, 1.5, 8.0, 27.0, 2.0])
  jac = sc.jacobian(f, "x", "c0")(start)
  x = f(start)
  np.testing.assert_allclose(jac[:, 3:], np.diag(1.0 / (3.0 * x * x)), rtol=1e-9, atol=1e-12)  # the implicit derivative at convergence
  np.testing.assert_allclose(jac, finite_difference(f, start).reshape(jac.shape), rtol=1e-5, atol=1e-6)
  loops = _loop_nodes([sc.jacobian(f.outputs[0], c0)])
  assert {n.attrs["callee"].name for n in loops} == {"wm_step_whilefwd6"}
