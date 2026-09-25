"""``scan`` is a loop in the generated C that computes what the unrolled chain of calls computes.

Each workload is checked three ways: against the same body called ``length`` times in a chain of
``CALL`` nodes, against a NumPy loop, and in its derivatives against finite differences and the
unrolled graph. The generated code must not grow with the number of steps.
"""

from __future__ import annotations

import numpy as np
import pytest

import scaly as sc
from scaly.ad import finite_difference
from scaly.codegen import render_c_source
from scaly.ir.program import ProgramOp
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
