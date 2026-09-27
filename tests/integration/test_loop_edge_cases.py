"""Edge cases of ``scan``, ``while_loop`` and ``vmap`` over Functions that hold them.

The step count at its boundaries (zero steps, one step, stopped by ``max_iter``, stopped by the
condition exactly at the bound), slices that end exactly at the edge of their outer tensor, NaN
reaching a condition, integer and boolean carries, ``max_trajectory`` at its exact value, and loops
nested in loops and maps. Values are compared with plain Python loops; derivatives with finite
differences and with the same body called step by step.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

import scaly as sc
from scaly.ad import finite_difference
from scaly.ad.derivatives import gradient, hessian, jacobian
from scaly.codegen import render_c_source


def _fn(name, inputs, outputs):
  return sc.Function._from_exprs(name, list(inputs), list(outputs), [str(x.name) for x in inputs], [f"o{k}" for k in range(len(outputs))])


def _step(name: str, *, index: bool) -> sc.Function:
  """A carry of two and a sliced input of two; with ``index`` the step number enters the next carry
  and the stacked output, without it a zero stands in."""
  c, k, u = sc.sym("c", 2), sc.sym("k", (), dtype="int64"), sc.sym("u", 2)
  t = k.cast("float64") if index else sc.const(0.0)
  nxt = sc.stack([c[0] * u[1] + 0.5 * t, (c[1] * u[0]).sin() + c[0]])
  return _fn(name, [c, k, u] if index else [c, u], [nxt, sc.stack([c[0] - u[0], c[1] * c[0], t])])


def _np_scan(c: np.ndarray, us: np.ndarray, start: int, stride: int, length: int, *, index: bool) -> tuple[np.ndarray, np.ndarray]:
  ys = []
  for k in range(length):
    u, t = us[start + k * stride : start + k * stride + 2], float(k) if index else 0.0
    ys.append([c[0] - u[0], c[1] * c[0], t])
    c = np.array([c[0] * u[1] + 0.5 * t, np.sin(c[1] * u[0]) + c[0]])
  return c, np.array(ys).reshape(-1)


def _loop_counters(src: str) -> list[str]:
  """The scan loops of a rendered source: lowering names a scan's counter ``k_`` after its carry."""
  return re.findall(r"for \(long long k_\w+", src)


# --- scan: the number of steps ---------------------------------------------------------------------


def test_scan_of_length_zero_returns_the_init_bit_for_bit() -> None:
  """Zero steps: the final carry is the init itself (NaN, infinities and a negative zero included), the
  stacked outputs are empty, and each derivative is the identity's or zero, the index or not."""
  c0, us = sc.sym("c0", 2), sc.sym("us", 5)
  outs = []
  for index in (False, True):
    fin, ys = sc.scan(_step(f"le_z{int(index)}_step", index=index), c0, [(us, 3, 1)], length=0, index=index)
    cost = sc.sumsqr(fin) + ys.sum()
    outs += [fin, ys, jacobian(fin, c0), jacobian(fin, us), jacobian(ys, c0), gradient(cost, c0), gradient(cost, us), hessian(cost, c0)]
  f = _fn("le_zero", [c0, us], outs)
  for odd in (np.array([np.nan, -0.0]), np.array([np.inf, -np.inf])):
    got = f._flat_numerical_call(odd, np.arange(5.0))
    assert got[0].tobytes() == got[8].tobytes() == odd.tobytes()
  point = np.array([0.3, -1.2])
  got = f._flat_numerical_call(point, np.arange(5.0))
  for base in (0, 8):
    fin, ys, j_c, j_u, j_y, g_c, g_u, h_c = got[base : base + 8]
    assert ys.shape == (0,) and j_y.shape == (0, 2)
    np.testing.assert_array_equal(fin, point)
    np.testing.assert_array_equal(j_c, np.eye(2))
    np.testing.assert_array_equal(j_u, np.zeros((2, 5)))
    np.testing.assert_array_equal(g_c, 2.0 * point)
    np.testing.assert_array_equal(g_u, np.zeros(5))
    np.testing.assert_array_equal(h_c, 2.0 * np.eye(2))


@pytest.mark.parametrize("index", [False, True])
def test_scan_of_length_one_is_one_call_of_the_body(index: bool) -> None:
  """One step is the body called once on the first slice, in value and every derivative. The loop is
  gone from the generated C; a step read at the wrong counter value would read ``us[1:3]``."""
  body = _step(f"le_one{int(index)}_step", index=index)
  c0, us = sc.sym("c0", 2), sc.sym("us", 5)
  fin, ys = sc.scan(body, c0, [(us, 3, -2)], length=1, index=index)
  k0 = [sc.const(np.array(0), dtype="int64")] if index else []
  ref_fin, ref_ys = body._flat_symbolic_call([c0, *k0, us[3:5]])
  outs = []
  for y_fin, y_ys in ((fin, ys), (ref_fin, ref_ys)):
    cost = sc.sumsqr(y_fin) + (y_ys * y_ys).sum()
    outs += [y_fin, y_ys, jacobian(y_fin, us), jacobian(y_ys, c0), gradient(cost, c0), gradient(cost, us), hessian(cost, us)]
  f = _fn(f"le_one{int(index)}", [c0, us], outs)
  src = render_c_source(f)
  assert not _loop_counters(src), _loop_counters(src)
  two = sc.scan(body, c0, [(us, 3, -2)], length=2, index=index)
  assert _loop_counters(render_c_source(_fn(f"le_two{int(index)}", [c0, us], list(two))))
  point = (np.array([0.3, -0.7]), np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
  got = f._flat_numerical_call(*point)
  for a, b in zip(got[:7], got[7:], strict=True):
    np.testing.assert_allclose(a, b, rtol=1e-14, atol=1e-15)
  want_fin, want_ys = _np_scan(point[0], point[1], 3, -2, 1, index=index)
  np.testing.assert_allclose(got[0], want_fin, rtol=1e-14)
  np.testing.assert_allclose(got[1], want_ys, rtol=1e-14)
  fd = finite_difference(lambda v: _np_scan(point[0], v, 3, -2, 1, index=index)[0], point[1])
  np.testing.assert_allclose(got[2], fd, rtol=1e-7, atol=1e-9)
  assert not got[2][:, :3].any()


def test_scan_step_number_at_the_first_and_last_step() -> None:
  """The step number runs ``0 .. length - 1`` whichever way and however far the slices walk: seven
  steps backwards and forwards through the whole array, two steps reading only its first and last
  windows, and one step."""
  body = _step("le_sn_step", index=True)
  c0, us = sc.sym("c0", 2), sc.sym("us", 14)
  walks = [(12, -2, 7), (0, 2, 7), (12, -12, 2), (12, 5, 1)]
  outs, cost = [], sc.const(0.0)
  for start, stride, length in walks:
    fin, ys = sc.scan(body, c0, [(us, start, stride)], length=length, index=True)
    outs += [fin, ys]
    cost = cost + sc.sumsqr(fin) + ys.sum()
  f = _fn("le_sn", [c0, us], [*outs, gradient(cost, us), gradient(cost, c0)])
  point = (np.array([0.4, -0.3]), 0.8 * np.sin(np.arange(14.0)))
  got = f._flat_numerical_call(*point)

  def total(c: np.ndarray, u: np.ndarray) -> np.ndarray:
    parts = [_np_scan(c, u, s, st, n, index=True) for s, st, n in walks]
    return np.array([sum(fin @ fin + ys.sum() for fin, ys in parts)])

  for k, (start, stride, length) in enumerate(walks):
    fin, ys = got[2 * k], got[2 * k + 1]
    np.testing.assert_array_equal(ys[2::3], np.arange(length))
    want_fin, want_ys = _np_scan(point[0], point[1], start, stride, length, index=True)
    np.testing.assert_allclose(fin, want_fin, rtol=1e-14)
    np.testing.assert_allclose(ys, want_ys, rtol=1e-14, atol=1e-15)
  np.testing.assert_allclose(got[-2], finite_difference(lambda v: total(point[0], v), point[1])[0], rtol=1e-6, atol=1e-8)
  np.testing.assert_allclose(got[-1], finite_difference(lambda v: total(v, point[1]), point[0])[0], rtol=1e-6, atol=1e-8)


# --- scan: slices at the edges of their outer tensors ---------------------------------------------

# Outer size, start and stride of each sliced input: windows of two with gaps ending exactly at the
# end, overlapping windows of three walking backwards to exactly index 0, and the last window of
# two broadcast to every step.
EDGES = {"a": (12, 1, 3), "b": (6, 3, -1), "w": (5, 3, 0)}
EDGE_STEPS = 4


def _edge_step() -> sc.Function:
  c, a, b, w = sc.sym("c", 2), sc.sym("a", 2), sc.sym("b", 3), sc.sym("w", 2)
  nxt = sc.stack([c[0] * a[0] + b[2] * w[1], (c[1] * b[0]).sin() + a[1] * w[0] - b[1]])
  return _fn("le_edge_step", [c, a, b, w], [nxt, sc.stack([c[0] * b[1] + a[0], c[1] * w[1]])])


def _np_edges(c: np.ndarray, a: np.ndarray, b: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  ys = []
  (_, sa, da), (_, sb, db), (_, sw, dw) = EDGES.values()
  for k in range(EDGE_STEPS):
    ak, bk, wk = a[sa + k * da : sa + k * da + 2], b[sb + k * db : sb + k * db + 3], w[sw + k * dw : sw + k * dw + 2]
    ys.append([c[0] * bk[1] + ak[0], c[1] * wk[1]])
    c = np.array([c[0] * ak[0] + bk[2] * wk[1], np.sin(c[1] * bk[0]) + ak[1] * wk[0] - bk[1]])
  return c, np.array(ys).reshape(-1)


def test_scan_slices_that_end_exactly_at_the_array_edges(monkeypatch: pytest.MonkeyPatch) -> None:
  """Every slice reaches the first or last entry of its outer tensor: the values, the Jacobians in
  one multi-seed pass, and the gradients, whose cotangents are scattered back to those entries."""
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  c0 = sc.sym("c0", 2)
  outer = {name: sc.sym(f"{name}_all", size) for name, (size, _, _) in EDGES.items()}
  fin, ys = sc.scan(_edge_step(), c0, [(outer[n], start, stride) for n, (_, start, stride) in EDGES.items()], length=EDGE_STEPS)
  cost = sc.sumsqr(fin) + (ys * ys).sum()
  inputs = [c0, *outer.values()]
  exprs = [fin, ys, *(gradient(cost, x) for x in inputs), *(e for x in inputs[1:] for e in (jacobian(fin, x), jacobian(ys, x)))]
  rng = np.random.default_rng(7)
  point = [np.array([0.5, -0.4]), *(0.3 + 0.5 * rng.random(size) for size, _, _ in EDGES.values())]
  got = _fn("le_edges", inputs, exprs)._flat_numerical_call(*point)
  want_fin, want_ys = _np_edges(*point)
  np.testing.assert_allclose(got[0], want_fin, rtol=1e-14)
  np.testing.assert_allclose(got[1], want_ys, rtol=1e-14)

  def at(k: int, v: np.ndarray) -> list[np.ndarray]:
    return [v if m == k else p for m, p in enumerate(point)]

  for k in range(4):
    fd_cost = finite_difference(lambda v, k=k: np.array([sum(np.sum(y * y) for y in _np_edges(*at(k, v)))]), point[k])
    np.testing.assert_allclose(got[2 + k], fd_cost[0], rtol=1e-6, atol=1e-8)
  for k in range(1, 4):
    for i in range(2):
      fd = finite_difference(lambda v, k=k, i=i: _np_edges(*at(k, v))[i], point[k])
      np.testing.assert_allclose(got[6 + 2 * (k - 1) + i], fd, rtol=1e-6, atol=1e-8)
  # The first entry of ``a`` and the entries of ``w`` before its window are never read.
  assert not got[3][0] and not got[5][:3].any()


@pytest.mark.parametrize(
  ("which", "start"), [("a", 2), ("b", 4), ("b", 2), ("w", 4)], ids=["a_past_the_end", "b_past_the_end", "b_before_zero", "w_past_the_end"]
)
def test_scan_slices_one_entry_past_the_edge_are_refused(which: str, start: int) -> None:
  specs = {name: (sc.sym(f"{name}_all", size), s, stride) for name, (size, s, stride) in EDGES.items()}
  outer, _, stride = specs[which]
  specs[which] = (outer, start, stride)
  position = list(EDGES).index(which) + 1
  with pytest.raises(ValueError, match=rf"scan input {position} reads outside its outer tensor of size {outer.size}: start={start}"):
    sc.scan(_edge_step(), sc.sym("c0", 2), list(specs.values()), length=EDGE_STEPS)


# --- scan: carries of other dtypes and bodies that ignore or keep their carry ---------------------


def test_integer_and_boolean_carries_and_slices() -> None:
  """A step counter in an ``int64`` carry, a flag flipped in a ``bool`` carry, and ``int64`` and
  ``bool`` slices of run-time inputs walking backwards and forwards; the gradients of the float
  outputs in reverse mode; a while loop over an ``int64`` carry."""
  ci, x = sc.sym("ci", (), dtype="int64"), sc.sym("x", 1)
  count = _fn("le_int_step", [ci, x], [ci + 1, x * ci.cast("float64")])
  cb = sc.sym("cb", (), dtype="bool")
  flip = _fn("le_bool_step", [cb, x], [sc.logical_not(cb), sc.where(cb, x, -x)])
  cf, i, b = sc.sym("cf", ()), sc.sym("i", 1, dtype="int64"), sc.sym("b", 1, dtype="bool")
  mixed = _fn("le_mixed_step", [cf, i, b, x], [cf * 0.5 + i[0].cast("float64") * x[0], sc.where(b[0], x, -x)])
  jump = _fn("le_int_jump", [ci], [ci + 3])
  below = _fn("le_int_below", [ci], [sc.less(ci, 10)])
  ci0, cb0, cf0 = sc.sym("ci0", (), dtype="int64"), sc.sym("cb0", (), dtype="bool"), sc.sym("cf0", ())
  ii, bb, xs = sc.sym("ii", 7, dtype="int64"), sc.sym("bb", 3, dtype="bool"), sc.sym("xs", 4)
  n_fin, n_ys = sc.scan(count, ci0, [(xs, 0, 1)], length=4)
  b_fin, b_ys = sc.scan(flip, cb0, [(xs, 0, 1)], length=3)
  m_fin, m_ys = sc.scan(mixed, cf0, [(ii, 6, -3), (bb, 0, 1), (xs, 1, 1)], length=3)
  w_fin, w_n = sc.while_loop(below, jump, ci0, max_iter=50)
  grads = [gradient(n_ys.sum(), xs), gradient(b_ys.sum(), xs), gradient(m_fin + m_ys.sum(), xs), gradient(m_fin.reshape((1,))[0], cf0)]
  f = _fn("le_dtypes", [ci0, cb0, cf0, ii, bb, xs], [n_fin, n_ys, b_fin, b_ys, m_fin, m_ys, w_fin, w_n, *grads])
  xv = np.array([1.0, 2.0, 3.0, 4.0])
  got = f._flat_numerical_call(np.array(2), np.array(True), np.array(1.0), np.arange(7), np.array([True, False, True]), xv)
  n_fin, n_ys, b_fin, b_ys, m_fin, m_ys, w_fin, w_n, g_n, g_b, g_m, g_c = got
  assert n_fin == 6 and b_fin == 0.0  # integers and bools cross the ABI as doubles
  np.testing.assert_array_equal(n_ys, xv * [2, 3, 4, 5])
  np.testing.assert_array_equal(b_ys, [1.0, -2.0, 3.0])
  c, ref_ys = 1.0, []
  for k, (iv, bv) in enumerate(zip([6, 3, 0], [True, False, True], strict=True)):
    c = c * 0.5 + iv * xv[1 + k]
    ref_ys.append(xv[1 + k] if bv else -xv[1 + k])
  assert m_fin == c
  np.testing.assert_array_equal(m_ys, ref_ys)
  assert (w_fin, w_n) == (11.0, 3.0)
  np.testing.assert_array_equal(g_n, [2.0, 3.0, 4.0, 5.0])
  np.testing.assert_array_equal(g_b, [1.0, -1.0, 1.0, 0.0])
  np.testing.assert_array_equal(g_m, [0.0, 6 * 0.25 + 1.0, 3 * 0.5 - 1.0, 1.0])
  assert g_c == 0.125


@pytest.mark.parametrize("dtype", ["int64", "bool"])
def test_forward_mode_through_a_scan_with_a_non_float_carry(dtype: str) -> None:
  """The float stacked output is differentiable in its float slices (reverse mode agrees above), so
  its Jacobian must build; the carry itself has no tangent."""
  c, x = sc.sym("c", (), dtype=dtype), sc.sym("x", 1)
  nxt, y = (c + 1, x * c.cast("float64")) if dtype == "int64" else (sc.logical_not(c), sc.where(c, x, -x))
  body = _fn(f"le_fm_{dtype}_step", [c, x], [nxt, y])
  c0, xs = sc.sym("c0", (), dtype=dtype), sc.sym("xs", 4)
  _, ys = sc.scan(body, c0, [(xs, 0, 1)], length=4)
  init, slope = (np.array(2), [2.0, 3.0, 4.0, 5.0]) if dtype == "int64" else (np.array(True), [1.0, -1.0, 1.0, -1.0])
  (jac,) = _fn(f"le_fm_{dtype}", [c0, xs], [jacobian(ys, xs)])._flat_numerical_call(init, np.arange(4.0))
  np.testing.assert_array_equal(jac, np.diag(slope))


@pytest.mark.parametrize("dtype", ["int64", "bool"])
def test_forward_mode_through_a_while_loop_with_a_non_float_carry(dtype: str) -> None:
  """A while loop over an integer or bool carry has no tangent at all: every output is the carry or
  its step count. Forward mode gives zeros where it used to raise, as reverse mode does."""
  c, p = sc.sym("c", (), dtype=dtype), sc.sym("p", 1)
  nxt = c + 1 if dtype == "int64" else sc.logical_not(c)
  body = _fn(f"le_wnf_{dtype}_body", [c, p], [nxt])
  go = _fn(f"le_wnf_{dtype}_go", [c, p], [sc.less(p[0], 10.0)])
  c0, ps = sc.sym("c0", (), dtype=dtype), sc.sym("ps", 1)
  final, count = sc.while_loop(go, body, c0, max_iter=3, params=[ps])
  value = sc.cast(final, "float64") * ps[0]
  outs = [value, sc.jvp(value, ps, sc.const(np.ones(1))), jacobian(value.reshape((1,)), ps), gradient(value, ps), count]
  init = np.array(2) if dtype == "int64" else np.array(True)
  got = _fn(f"le_wnf_{dtype}", [c0, ps], outs)._flat_numerical_call(init, np.array([1.5]))
  final_v = 5.0 if dtype == "int64" else 0.0
  np.testing.assert_array_equal([np.ravel(g)[0] for g in got], [final_v * 1.5, final_v, final_v, final_v, 3.0])


def _loops_over(dtype: str) -> tuple[sc.Expr, sc.Expr, sc.Expr]:
  """A scan over float64 slices then a while loop over params, with carries of ``dtype``."""
  c, u, p = sc.sym("c", 2, dtype=dtype), sc.sym("u", 1), sc.sym("p", 1, dtype=dtype)
  step = _fn(f"le_{dtype}_step", [c, u], [c * sc.cast(u[0], dtype) + c[::-1] * 0.5, sc.cast(c.sum(), "float64").reshape((1,)) * u])
  body = _fn(f"le_{dtype}_body", [c, p], [c * p[0] + c[::-1] * 0.5])
  go = _fn(f"le_{dtype}_go", [c, p], [sc.less(sc.cast(c.sum(), "float64"), 50.0)])
  c0, us = sc.sym("c0", 2, dtype=dtype), sc.sym("us", 3)
  _, ys = sc.scan(step, c0, [(us, 0, 1)], length=3)
  final, _ = sc.while_loop(go, body, sc.cast(ys[:2], dtype), max_iter=4, params=[sc.cast(us[:1], dtype)])
  return c0, us, sc.cast(final, "float64").sum()


def test_float32_carries_differentiate_in_forward_and_reverse_mode() -> None:
  """A float32 carry and its tangent share the tangent loop's carry, and its cotangent is float32 too:
  single-seed forward and reverse mode through a scan over float64 slices and a while loop over
  float32 params agree with the same loops over float64 carries, to float32 rounding."""
  c0v, usv = np.array([1.0, 0.5]), np.array([1.25, 0.75, 1.5])
  got = {}
  for dtype in ("float32", "float64"):
    c0, us, cost = _loops_over(dtype)
    f = _fn(f"le_{dtype}_loops", [c0, us], [cost, sc.jvp(cost, us, sc.const(np.ones(3))), gradient(cost, us)])
    got[dtype] = f._flat_numerical_call(c0v, usv)
  for g32, g64 in zip(got["float32"], got["float64"], strict=True):
    np.testing.assert_allclose(g32, g64, rtol=1e-5)
  np.testing.assert_allclose(got["float64"][1], got["float64"][2].sum(), rtol=1e-14)


@pytest.mark.xfail(strict=True, raises=TypeError, reason="the while adjoint packs a float32 carry's cotangent with a float64 param's in one vector")
def test_reverse_mode_through_a_float32_carry_beside_float64_params() -> None:
  c, p = sc.sym("c", 2, dtype="float32"), sc.sym("p", 1)
  body = _fn("le_f32mix_body", [c, p], [c * sc.cast(p[0], "float32")])
  go = _fn("le_f32mix_go", [c, p], [sc.less(p[0], 10.0)])
  ps = sc.sym("ps", 1)
  final, _ = sc.while_loop(go, body, sc.sym("c0", 2, dtype="float32"), max_iter=3, params=[ps])
  gradient(sc.cast(final, "float64").sum(), ps)


@pytest.mark.xfail(
  strict=True, raises=TypeError, reason="multi-seed forward mode joins the tangents of a float32 carry and float64 slices in one float64 vector"
)
def test_multi_seed_forward_mode_through_a_float32_carry() -> None:
  c, u = sc.sym("c", (), dtype="float32"), sc.sym("u", 1)
  step = _fn("le_f32m_step", [c, u], [c * sc.cast(u[0], "float32"), u * sc.cast(c, "float64")])
  c0, us = sc.sym("c0", (), dtype="float32"), sc.sym("us", 3)
  _, ys = sc.scan(step, c0, [(us, 0, 1)], length=3)
  jacobian(ys, us)


def test_a_body_that_ignores_or_keeps_its_carry() -> None:
  """A body that returns its carry unchanged keeps the init bit for bit over odd and even step counts
  (the two carry slots alternate) and differentiates to the identity; one that ignores its carry
  depends only on the last slice; NaN and infinity in one entry of an elementwise carry stay there. A
  carry of no entries still steps: a scan stacks its outputs and a while loop runs to its bound."""
  c3, u1, u3 = sc.sym("c", 3), sc.sym("u", 1), sc.sym("u", 3)
  keep = _fn("le_keep_step", [c3, u1], [c3, c3 * u1[0]])
  halve = _fn("le_halve_step", [c3, u3], [c3 * 0.5 + u3])
  ignore = _fn("le_ignore_step", [c3, u3], [u3.sin() * 2.0])
  c0, us, ws = sc.sym("c0", 3), sc.sym("us", 3), sc.sym("ws", 12)
  kept = [sc.scan(keep, c0, [(us, 0, 1)], length=n) for n in (1, 2, 3)]
  (halved,) = sc.scan(halve, c0, [(ws, 0, 3)], length=4)
  (ignored,) = sc.scan(ignore, c0, [(ws, 9, -3)], length=4)
  derivs = [jacobian(kept[2][0], c0), gradient(kept[1][0].sum(), c0), jacobian(ignored, c0), gradient(sc.sumsqr(ignored), c0), jacobian(ignored, ws)]
  c_none, nothing = sc.sym("c", 0), sc.const(np.zeros(0))
  e_fin, e_ys = sc.scan(_fn("le_none_step", [c_none, u1], [c_none, (u1 * 2.0).sin()]), nothing, [(us, 0, 1)], length=3)
  w_fin, w_n = sc.while_loop(
    _fn("le_none_go", [c_none], [sc.less(c_none.sum(), 1.0)]), _fn("le_none_body", [c_none], [c_none * 2.0]), nothing, max_iter=5
  )
  empty = [e_fin, e_ys, gradient(e_ys.sum(), us), w_fin, w_n]
  f = _fn("le_keep", [c0, us, ws], [*(e for pair in kept for e in pair), halved, ignored, *derivs, *empty])
  odd = np.array([np.nan, -0.0, np.inf])
  uv, wv = np.array([2.0, -1.0, 0.5]), np.linspace(-1.0, 1.0, 12)
  got = f._flat_numerical_call(odd, uv, wv)
  for n in range(3):
    fin, ys = got[2 * n], got[2 * n + 1]
    assert fin.tobytes() == odd.tobytes()
    np.testing.assert_array_equal(ys, np.concatenate([odd * u for u in uv[: n + 1]]))
  ref = odd.copy()
  for k in range(4):
    ref = ref * 0.5 + wv[3 * k : 3 * k + 3]
  np.testing.assert_array_equal(got[6], ref)
  np.testing.assert_array_equal(got[7], np.sin(wv[:3]) * 2.0)
  j_keep, g_keep, j_ign, g_ign, j_ign_w = got[8:13]
  np.testing.assert_array_equal(j_keep, np.eye(3))
  np.testing.assert_array_equal(g_keep, np.ones(3))
  assert not j_ign.any() and not g_ign.any()
  np.testing.assert_allclose(j_ign_w[:, :3], np.diag(2.0 * np.cos(wv[:3])), rtol=1e-15)
  assert not j_ign_w[:, 3:].any()
  assert sc.jacobian_sparsity(ignored, c0).nnz == 0
  e_fin, e_ys, e_grad, w_fin, w_n = got[13:]
  assert e_fin.shape == w_fin.shape == (0,) and w_n == 5.0
  np.testing.assert_array_equal(e_ys, np.sin(uv * 2.0))
  np.testing.assert_allclose(e_grad, 2.0 * np.cos(uv * 2.0), rtol=1e-15)


# --- while_loop: the number of steps -------------------------------------------------------------

MAX_ITERS = (0, 1, 3, 4, 5, 6, 7)


def _count_function(in_place: bool) -> sc.Function:
  """One loop per bound in ``MAX_ITERS`` over the carry ``[x | m | s]``: the condition ``m < lim`` on a
  param, ``m`` counting the steps and ``s`` summing the step numbers. The in-place body updates two
  entries through ``index_add`` and the loop reuses one carry slot; the other rewrites every entry."""
  tag = "ip" if in_place else "ts"
  c, k, lim = sc.sym("c", 3), sc.sym("k", (), dtype="int64"), sc.sym("lim", ())
  t = k.cast("float64")
  nxt = sc.index_add(c, [1, 2], sc.stack([sc.const(1.0), t])) if in_place else sc.stack([c[0] * 0.5 + c[2], c[1] + 1.0, c[2] + t])
  body = _fn(f"le_cnt_body_{tag}", [c, k, lim], [nxt])
  cond = _fn(f"le_cnt_cond_{tag}", [c, lim], [sc.less(c[1], lim)])
  c0, lv = sc.sym("c0", 3), sc.sym("lv", ())
  outs = [e for m in MAX_ITERS for e in sc.while_loop(cond, body, c0, max_iter=m, index=True, params=(lv,))]
  return _fn(f"le_cnt_{tag}", [c0, lv], outs)


def _count_reference(c: np.ndarray, lim: float, max_iter: int, in_place: bool) -> tuple[np.ndarray, int]:
  c, n = c.copy(), 0
  while n < max_iter and c[1] < lim:
    c = np.array([c[0], c[1] + 1.0, c[2] + n]) if in_place else np.array([c[0] * 0.5 + c[2], c[1] + 1.0, c[2] + n])
    n += 1
  return c, n


@pytest.mark.parametrize("in_place", [False, True], ids=["two_slots", "in_place"])
def test_while_step_counts_at_every_boundary(in_place: bool) -> None:
  """Each bound against a condition that holds for 0 .. 7 steps or forever, or reads a NaN limit:
  zero steps return the init, the condition turning false exactly at ``max_iter`` counts
  ``max_iter``, and the bound cuts a longer run off. The count, the final carry and the step
  numbers the body saw (their sum) match a Python loop exactly, on either side of the slot parity."""
  f = _count_function(in_place)
  assert ("_inplace" in render_c_source(f)) == in_place
  c0 = np.array([1.0, 0.0, 0.25])
  seen = set()
  for lim in (np.nan, -1.0, 0.0, 1.0, 3.0, 4.0, 5.0, 6.0, 7.0, 100.0):
    got = f._flat_numerical_call(c0, np.array(lim))
    for m, fin, n in zip(MAX_ITERS, got[::2], got[1::2], strict=True):
      want, steps = _count_reference(c0, lim, m, in_place)
      np.testing.assert_array_equal(fin, want, err_msg=f"lim={lim} max_iter={m}")
      assert n == steps, (lim, m, n, steps)
      seen.add((steps == 0, steps == m, steps < m))
  assert seen == {(True, True, False), (True, False, True), (False, True, False), (False, False, True)}


def test_a_condition_that_reads_nan() -> None:
  """A comparison with NaN is false, so ``y >= 0`` stops at the step that makes ``y`` NaN while
  ``not (y < 0)`` runs on to ``max_iter``; ``y == y`` is a NaN test, and ``isfinite`` stops at once
  on a NaN or infinite start."""
  c = sc.sym("c", 2)
  body = _fn("le_nan_body", [c], [sc.stack([c[0] - 1.0, (c[0] - 1.0).sqrt()])])
  conds = {
    "ge": (sc.greater_equal(c[1], 0.0), lambda x, y: y >= 0.0),
    "not_lt": (sc.logical_not(sc.less(c[1], 0.0)), lambda x, y: not y < 0.0),
    "eq_self": (sc.equal(c[1], c[1]), lambda x, y: y == y),
    "finite": (sc.isfinite(c[0]), lambda x, y: bool(np.isfinite(x))),
  }
  c0 = sc.sym("c0", 2)
  outs = [e for name, (go, _) in conds.items() for e in sc.while_loop(_fn(f"le_nan_{name}", [c], [go]), body, c0, max_iter=8)]
  f = _fn("le_nan", [c0], outs)
  counts = {}
  for start in (np.array([3.0, 0.0]), np.array([np.nan, 0.0]), np.array([np.inf, 1.0]), np.array([0.5, np.nan])):
    got = f._flat_numerical_call(start)
    for k, (name, (_, go)) in enumerate(conds.items()):
      x, y, n = start[0], start[1], 0
      with np.errstate(invalid="ignore"):
        while n < 8 and go(x, y):
          x, y, n = x - 1.0, np.sqrt(x - 1.0), n + 1
      np.testing.assert_array_equal(got[2 * k], [x, y], err_msg=f"{name} from {start}")
      assert got[2 * k + 1] == n, (name, start, got[2 * k + 1], n)
      counts[name, str(start)] = n
  assert counts["ge", str(np.array([3.0, 0.0]))] == 4 and counts["not_lt", str(np.array([3.0, 0.0]))] == 8
  assert counts["finite", str(np.array([np.nan, 0.0]))] == 0 and counts["eq_self", str(np.array([0.5, np.nan]))] == 0


def test_params_and_the_same_values_in_the_carry_agree() -> None:
  """The same loop with its invariants passed as params and carried unchanged in the carry: equal
  values and counts at zero steps, part way and at ``max_iter``, and equal derivatives in the
  invariants (a param's cotangent summed over the steps against a carry entry's)."""
  c, k, w, lim = sc.sym("c", 2), sc.sym("k", (), dtype="int64"), sc.sym("w", 2), sc.sym("lim", ())
  t = k.cast("float64")
  body_p = _fn("le_pc_body_p", [c, k, w, lim], [(c * w + 0.1 * t).tanh() + 0.5 * c])
  cond_p = _fn("le_pc_cond_p", [c, w, lim], [sc.less(c[0] + c[1], lim)])
  s = sc.sym("s", 5)
  body_c = _fn("le_pc_body_c", [s, k], [sc.concat([(s[:2] * s[2:4] + 0.1 * t).tanh() + 0.5 * s[:2], s[2:]])])
  cond_c = _fn("le_pc_cond_c", [s], [sc.less(s[0] + s[1], s[4])])
  x0, wv, lv = sc.sym("x0", 2), sc.sym("wv", 2), sc.sym("lv", ())
  a, na = sc.while_loop(cond_p, body_p, x0, max_iter=6, index=True, params=(wv, lv))
  b, nb = sc.while_loop(cond_c, body_c, sc.concat([x0, wv, lv.reshape((1,))]), max_iter=6, index=True)
  b = b[:2]
  outs = [a, na, b, nb]
  for y in (a, b):
    cost = sc.sumsqr(y) + y[0] * wv[1]
    outs += [jacobian(y, wv), jacobian(y, x0), gradient(cost, wv), gradient(cost, x0), hessian(cost, wv)]
  f = _fn("le_pc", [x0, wv, lv], outs)
  xv, w_v = np.array([0.2, 0.1]), np.array([0.9, 1.3])
  for lim, steps in ((0.1, 0), (0.5, 2), (2.0, 4), (10.0, 6)):
    got = f._flat_numerical_call(xv, w_v, np.array(lim))
    assert got[1] == got[3] == steps
    np.testing.assert_array_equal(got[0], got[2])
    for p, q in zip(got[4:9], got[9:], strict=True):
      np.testing.assert_allclose(p, q, rtol=1e-13, atol=1e-15)
    if steps == 4:

      def run(v: np.ndarray) -> np.ndarray:
        z = xv.copy()
        for n in range(4):
          z = np.tanh(z * v + 0.1 * n) + 0.5 * z
        return z

      np.testing.assert_allclose(got[4], finite_difference(run, w_v), rtol=1e-6, atol=1e-8)


# (max_iter, lim, steps): a condition false at the start, one step, a stop part way, a stop by the
# bound, and a bound of zero.
BOUNDARIES = [(4, -1.0, 0), (1, 2.5, 1), (4, 2.5, 3), (4, 10.0, 4), (0, 2.5, 0)]


def _smooth_step(c: sc.Expr, t: sc.Expr | float, p: sc.Expr) -> sc.Expr:
  return sc.stack([c[0] * p[0] + c[1].sin() * 0.3 + 0.1 * t, c[1] * p[1] + 0.2 * c[0] * c[0], c[2] + 1.0])


def _np_smooth(x: np.ndarray, p: np.ndarray, steps: int) -> np.ndarray:
  c = x.copy()
  for k in range(steps):
    c = np.array([c[0] * p[0] + np.sin(c[1]) * 0.3 + 0.1 * k, c[1] * p[1] + 0.2 * c[0] * c[0]])
  return c


@pytest.mark.parametrize(("max_iter", "lim", "steps"), BOUNDARIES, ids=["false_at_start", "one", "part_way", "cut_off", "bound_zero"])
def test_while_derivatives_at_the_boundaries(max_iter: int, lim: float, steps: int, monkeypatch: pytest.MonkeyPatch) -> None:
  """Forward (every seed in one loop), reverse and forward-over-reverse derivatives in the init and in
  the params equal those of the body called ``steps`` times, and finite differences. The step count
  has none: adding it to the cost changes no gradient."""
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  c, k, p, lm = sc.sym("c", 3), sc.sym("k", (), dtype="int64"), sc.sym("p", 2), sc.sym("lm", ())
  tag = f"{max_iter}_{steps}"
  body = _fn(f"le_bd_body_{tag}", [c, k, p, lm], [_smooth_step(c, k.cast("float64"), p)])
  cond = _fn(f"le_bd_cond_{tag}", [c, p, lm], [sc.less(c[2], lm)])
  x, pv = sc.sym("x", 2), sc.sym("pv", 2)
  init = sc.concat([x, sc.const(np.zeros(1))])
  fin, n = sc.while_loop(cond, body, init, max_iter=max_iter, index=True, params=(pv, sc.const(lim)))
  ref = init
  for step in range(steps):
    ref = _smooth_step(ref, float(step), pv)
  outs = [fin, n, gradient(sc.sumsqr(fin[:2]) + fin[0] * pv[1] + 3.0 * n, pv), jacobian(n.reshape((1,)), x)]
  for y in (fin[:2], ref[:2]):
    cost = sc.sumsqr(y) + y[0] * pv[1]
    outs += [jacobian(y, x), jacobian(y, pv), gradient(cost, x), gradient(cost, pv), hessian(cost, pv), hessian(cost, x)]
  f = _fn(f"le_bd_{tag}", [x, pv], outs)
  point = (np.array([0.4, -0.3]), np.array([0.9, 1.1]))
  got = f._flat_numerical_call(*point)
  assert got[1] == steps
  np.testing.assert_allclose(got[0][:2], _np_smooth(point[0], point[1], steps), rtol=1e-14)
  np.testing.assert_array_equal(got[2], got[7])
  assert not got[3].any()
  for a, b in zip(got[4:10], got[10:], strict=True):
    np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-13)
  j_x, j_p, _, g_p, h_p, _ = got[4:10]
  np.testing.assert_allclose(j_x, finite_difference(lambda v: _np_smooth(v, point[1], steps), point[0]), rtol=1e-6, atol=1e-8)
  np.testing.assert_allclose(j_p, finite_difference(lambda v: _np_smooth(point[0], v, steps), point[1]), rtol=1e-6, atol=1e-8)
  if steps == 0:
    np.testing.assert_array_equal(j_x, np.eye(2))
    np.testing.assert_array_equal(g_p, [0.0, point[0][0]])  # only the cost's own ``y[0] * p[1]``
    assert not j_p.any() and not h_p.any()
  np.testing.assert_allclose(h_p, finite_difference(lambda v: f._flat_numerical_call(point[0], v)[7], point[1]), rtol=1e-5, atol=1e-7)


# --- max_trajectory ------------------------------------------------------------------------------


def _limit_is(limit: int, build, loop: str):
  """``build`` is refused one value under ``limit``, naming ``loop``, and accepted at exactly ``limit``."""
  with sc.options(max_trajectory=limit - 1), pytest.raises(ValueError, match=rf"'{loop}' would store .*max_trajectory={limit - 1}"):
    build()
  with sc.options(max_trajectory=limit):
    return build()


def _nested(tag: str, outer_length: int) -> tuple[sc.Expr, sc.Expr, sc.Expr, str, str]:
  """A scan of ``outer_length`` steps over a carry of two whose body runs a scan of five steps over a
  carry of three: the inner loop stores 15 values, the outer ``2 * outer_length``."""
  ic, iu = sc.sym("ic", 3), sc.sym("iu", 1)
  inner = _fn(f"le_{tag}_inner", [ic, iu], [ic * iu[0] + ic[::-1].sin()])
  c, u = sc.sym("c", 2), sc.sym("u", 1)
  (ifin,) = sc.scan(inner, sc.concat([c, u]), [(u, 0, 0)], length=5)
  outer = _fn(f"le_{tag}_outer", [c, u], [ifin[:2] * 0.5 + c])
  c0, us = sc.sym("c0", 2), sc.sym("us", outer_length)
  (fin,) = sc.scan(outer, c0, [(us, 0, 1)], length=outer_length)
  return fin, c0, us, inner.name, outer.name


def _np_nested(c: np.ndarray, us: np.ndarray) -> np.ndarray:
  for u in us:
    ic = np.array([c[0], c[1], u])
    for _ in range(5):
      ic = ic * u + np.sin(ic[::-1])
    c = ic[:2] * 0.5 + c
  return c


def test_max_trajectory_is_per_loop_and_counts_the_bound(monkeypatch: pytest.MonkeyPatch) -> None:
  """The limit is the steps a loop may take times its carry size, for each loop on its own: a while
  loop is charged its bound however soon it stops, stacked outputs are not charged, loops that take
  no step store nothing, forward mode stores nothing, and nested loops are each checked against the
  limit rather than their sum. A gradient accepted at exactly the limit is right."""
  c = sc.sym("c", 4)
  x = sc.sym("x", 4)
  walked, _ = sc.while_loop(_fn("le_mt_above", [c], [c[0] > 1.0]), _fn("le_mt_half", [c], [c * 0.5]), x, max_iter=100)
  _limit_is(400, lambda: gradient(walked.sum(), x), "le_mt_half")
  cc, u = sc.sym("cc", 2), sc.sym("u", 1)
  wide = _fn("le_mt_wide", [cc, u], [cc * u[0], sc.concat([(cc * u[0]).sin()] * 25)])
  c2, us = sc.sym("c2", 2), sc.sym("us", 10)
  w_fin, w_ys = sc.scan(wide, c2, [(us, 0, 1)], length=10)
  _limit_is(20, lambda: gradient(w_fin.sum() + w_ys.sum(), us), "le_mt_wide")
  with sc.options(max_trajectory=0):
    none, _ = sc.while_loop(_fn("le_mt_go", [c], [c[0] > -1e300]), _fn("le_mt_none", [c], [c.sin()]), x, max_iter=0)
    (empty,) = sc.scan(_fn("le_mt_empty", [c], [c.cos()]), x, length=0)
    gradient((none + empty).sum(), x)
  fin, c0, us, inner, outer = _nested("mt", 8)
  with sc.options(max_trajectory=14), pytest.raises(ValueError, match=rf"'{inner}' would store 5 carries of 3 values"):
    gradient(fin.sum(), us)
  grad = _limit_is(16, lambda: gradient(fin.sum(), us), outer)
  _limit_is(16, lambda: hessian(fin.sum(), us), outer)
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  with sc.options(max_trajectory=0):
    jac = jacobian(fin, us)
  point = (np.array([0.3, -0.2]), np.linspace(-0.6, 0.6, 8))
  got_grad, got_jac = _fn("le_mt_nested", [c0, us], [grad, jac])._flat_numerical_call(*point)
  fd = finite_difference(lambda v: _np_nested(point[0], v), point[1])
  np.testing.assert_allclose(got_jac, fd, rtol=1e-6, atol=1e-8)
  np.testing.assert_allclose(got_grad, fd.sum(axis=0), rtol=1e-6, atol=1e-8)


def test_max_trajectory_is_checked_again_under_a_stricter_limit() -> None:
  """The inner loop stores 15 values and the outer 12. A gradient built under a limit of 15, then the
  same gradient under 14, must be refused the second time as it is when built first under 14."""
  fin, _, us, inner, _ = _nested("mtc", 6)
  with sc.options(max_trajectory=15):
    gradient(fin.sum(), us)
  with sc.options(max_trajectory=14), pytest.raises(ValueError, match=rf"'{inner}' would store"):
    gradient(fin.sum(), us)


# --- nesting --------------------------------------------------------------------------------------


def _np_inner_while(z: float, u: float) -> tuple[float, int]:
  m = 0
  while m < 4 and m < u:
    z, m = 0.5 * z + 0.1 * u + 0.3 * np.sin(z), m + 1
  return z, m


def _np_while_in_scan(c: np.ndarray, us: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  ys = []
  for u in us:
    z, m = _np_inner_while(c[0], u)
    ys += [m, z]
    c = np.array([z + 0.2 * c[1], 0.9 * c[1] + z * u])
  return c, np.array(ys, dtype=float)


def test_a_while_in_a_scan_body_takes_a_different_count_each_step(monkeypatch: pytest.MonkeyPatch) -> None:
  """Each step's slice is the inner loop's param and bounds its count: no step, one, two, three, and
  the inner ``max_iter``. Values, counts and derivatives in both slices and init match per step."""
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  w, u = sc.sym("w", 2), sc.sym("u", 1)
  inner_body = _fn("le_ws_inner", [w, u], [sc.stack([0.5 * w[0] + 0.1 * u[0] + 0.3 * w[0].sin(), w[1] + 1.0])])
  inner_cond = _fn("le_ws_go", [w, u], [sc.less(w[1], u[0])])
  c = sc.sym("c", 2)
  wfin, wn = sc.while_loop(inner_cond, inner_body, sc.stack([c[0], sc.const(0.0)]), max_iter=4, params=(u,))
  body = _fn("le_ws_step", [c, u], [sc.stack([wfin[0] + 0.2 * c[1], 0.9 * c[1] + wfin[0] * u[0]]), sc.stack([wn, wfin[0]])])
  c0, us = sc.sym("c0", 2), sc.sym("us", 5)
  fin, ys = sc.scan(body, c0, [(us, 0, 1)], length=5)
  cost = sc.sumsqr(fin) + (ys * ys).sum()
  f = _fn("le_ws", [c0, us], [fin, ys, jacobian(fin, us), jacobian(ys, c0), gradient(cost, us), gradient(cost, c0)])
  point = (np.array([0.4, -0.3]), np.array([-0.5, 1.5, 2.5, 10.0, 0.7]))
  got = f._flat_numerical_call(*point)
  want_fin, want_ys = _np_while_in_scan(*point)
  np.testing.assert_array_equal(got[1][::2], [0, 2, 3, 4, 1])
  np.testing.assert_allclose(got[0], want_fin, rtol=1e-14)
  np.testing.assert_allclose(got[1], want_ys, rtol=1e-14)
  np.testing.assert_allclose(got[2], finite_difference(lambda v: _np_while_in_scan(point[0], v)[0], point[1]), rtol=1e-6, atol=1e-8)
  np.testing.assert_allclose(got[3], finite_difference(lambda v: _np_while_in_scan(v, point[1])[1], point[0]), rtol=1e-6, atol=1e-8)
  assert not got[3][::2].any()  # the counts carry no derivative

  def total(c: np.ndarray, v: np.ndarray) -> np.ndarray:
    fin, ys = _np_while_in_scan(c, v)
    return np.array([fin @ fin + ys @ ys])

  np.testing.assert_allclose(got[4], finite_difference(lambda v: total(point[0], v), point[1])[0], rtol=1e-6, atol=1e-8)
  np.testing.assert_allclose(got[5], finite_difference(lambda v: total(v, point[1]), point[0])[0], rtol=1e-6, atol=1e-8)


def _np_lane(x: float, p: np.ndarray) -> tuple[float, int]:
  r = np.tanh(0.3 * p) + np.array([0.3, 0.1])
  n = 0
  while n < 6 and x > 1.0:
    x, n = x * r[0] + r[1], n + 1
  return x, n


def test_vmap_of_a_while_whose_lanes_stop_at_different_counts(monkeypatch: pytest.MonkeyPatch) -> None:
  """Lanes that take no step, two, four and the bound of six, with the loop's param computed from a
  broadcast input in a prologue hoisted out of the map. Each lane matches its own Python loop in
  value, count and derivative."""
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  c, r = sc.sym("c", 1), sc.sym("r", 2)
  body = _fn("le_vw_body", [c, r], [c * r[0] + r[1]])
  cond = _fn("le_vw_cond", [c, r], [c[0] > 1.0])
  x, p = sc.sym("x", 1), sc.sym("p", 2)
  fin, n = sc.while_loop(cond, body, x, max_iter=6, params=((p * 0.3).tanh() + sc.const(np.array([0.3, 0.1])),))
  lane = _fn("le_vw_lane", [x, p], [fin, n.reshape((1,))])
  xs, pb = sc.sym("xs", 4), sc.sym("pb", 2)
  values = sc.vmap(lane, 4, [(xs, 0, 1), (pb, 0, 0)], output=0)
  counts = sc.vmap(lane, 4, [(xs, 0, 1), (pb, 0, 0)], output=1)
  cost = sc.sumsqr(values)
  f = _fn("le_vw", [xs, pb], [values, counts, jacobian(values, xs), jacobian(values, pb), gradient(cost, xs), gradient(cost, pb), hessian(cost, pb)])
  assert "le_vw_lane_hoist" in render_c_source(f)
  point = (np.array([0.5, 1.5, 3.0, 1e3]), np.array([1.0, 0.5]))
  got = f._flat_numerical_call(*point)

  def lanes(xv: np.ndarray, pv: np.ndarray) -> np.ndarray:
    return np.array([_np_lane(v, pv)[0] for v in xv])

  np.testing.assert_array_equal(got[1], [0, 2, 4, 6])
  np.testing.assert_array_equal(got[1], [_np_lane(v, point[1])[1] for v in point[0]])
  np.testing.assert_allclose(got[0], lanes(*point), rtol=1e-14)
  np.testing.assert_allclose(got[2], finite_difference(lambda v: lanes(v, point[1]), point[0]), rtol=1e-6, atol=1e-8)
  np.testing.assert_allclose(got[3], finite_difference(lambda v: lanes(point[0], v), point[1]), rtol=1e-6, atol=1e-6)
  np.testing.assert_allclose(got[4], 2.0 * got[0] @ got[2], rtol=1e-12)
  np.testing.assert_allclose(got[5], 2.0 * got[0] @ got[3], rtol=1e-12)
  np.testing.assert_allclose(got[6], finite_difference(lambda v: f._flat_numerical_call(point[0], v)[5], point[1]), rtol=1e-5, atol=1e-4)


def _np_inner_scan(ic: np.ndarray, iu: float, length: int) -> np.ndarray:
  for _ in range(length):
    ic = np.array([ic[0] + 0.3 * ic[1] * iu, np.sin(ic[1]) + iu])
  return ic


def test_scans_of_zero_one_and_three_steps_inside_a_map(monkeypatch: pytest.MonkeyPatch) -> None:
  """A mapped callee holding a scan of 0, 1 or 3 steps, and a map of no iterations over one: values
  and Jacobians per lane, and empty results of the right shape."""
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  ic, iu = sc.sym("ic", 2), sc.sym("iu", 1)
  inner = _fn("le_sv_inner", [ic, iu], [sc.stack([ic[0] + 0.3 * ic[1] * iu[0], ic[1].sin() + iu[0]])])
  x, p = sc.sym("x", 2), sc.sym("p", 1)
  xs, ps = sc.sym("xs", 6), sc.sym("ps", 3)
  outs = []
  for length in (0, 1, 3):
    (fin,) = sc.scan(inner, x, [(p, 0, 0)], length=length)
    mapped = sc.vmap(_fn(f"le_sv_callee{length}", [x, p], [fin * p[0]]), 3, [(xs, 0, 2), (ps, 0, 1)])
    outs += [mapped, jacobian(mapped, xs), jacobian(mapped, ps)]
  none = sc.vmap(_fn("le_sv_callee_none", [x, p], [sc.scan(inner, x, [(p, 0, 0)], length=3)[0]]), 0, [(xs, 0, 2), (ps, 0, 1)])
  outs += [none, jacobian(none, xs), gradient(none.sum(), ps)]
  point = (np.array([0.3, -0.2, 0.5, 0.1, -0.4, 0.8]), np.array([0.9, 1.1, 0.7]))
  got = _fn("le_sv", [xs, ps], outs)._flat_numerical_call(*point)

  def mapped_np(xv: np.ndarray, pv: np.ndarray, length: int) -> np.ndarray:
    return np.concatenate([_np_inner_scan(xv[2 * i : 2 * i + 2], pv[i], length) * pv[i] for i in range(3)])

  for k, length in enumerate((0, 1, 3)):
    value, j_x, j_p = got[3 * k : 3 * k + 3]
    np.testing.assert_allclose(value, mapped_np(*point, length), rtol=1e-14)
    np.testing.assert_allclose(j_x, finite_difference(lambda v, n=length: mapped_np(v, point[1], n), point[0]), rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(j_p, finite_difference(lambda v, n=length: mapped_np(point[0], v, n), point[1]), rtol=1e-6, atol=1e-8)
  np.testing.assert_array_equal(got[1], np.diag(np.repeat(point[1], 2)))
  assert got[9].shape == (0,) and got[10].shape == (0, 6)
  np.testing.assert_array_equal(got[11], np.zeros(3))


def test_loops_of_no_steps_nested_in_a_loop_body_change_nothing() -> None:
  """A while body that runs a scan of no steps and a while of bound zero on its carry computes what
  the body without them computes, bit for bit, and differentiates the same way."""
  c = sc.sym("c", 2)

  def grow(v: sc.Expr) -> sc.Expr:
    return 1.3 * v + 0.2 * v[::-1].sin()

  plain = _fn("le_zi_plain", [c], [grow(c)])
  (skipped,) = sc.scan(_fn("le_zi_scan", [c], [c * 2.0]), c, length=0)
  held, _ = sc.while_loop(_fn("le_zi_always", [c], [c[0] > -1e300]), _fn("le_zi_triple", [c], [c * 3.0]), skipped, max_iter=0)
  nested = _fn("le_zi_nested", [c], [grow(held)])
  go = _fn("le_zi_go", [c], [sc.less(sc.sumsqr(c), 25.0)])
  c0 = sc.sym("c0", 2)
  outs = []
  for body in (plain, nested):
    fin, n = sc.while_loop(go, body, c0, max_iter=12)
    outs += [fin, n, gradient(sc.sumsqr(fin), c0), jacobian(fin, c0)]
  f = _fn("le_zi", [c0], outs)
  for start, steps in ((np.array([1.0, 0.5]), 5), (np.array([6.0, 0.0]), 0), (np.array([0.01, 0.0]), 12)):
    got = f._flat_numerical_call(start)
    np.testing.assert_array_equal(got[0], got[4])
    assert got[1] == got[5] == steps
    np.testing.assert_allclose(got[2], got[6], rtol=1e-14)
    np.testing.assert_allclose(got[3], got[7], rtol=1e-14)


# --- sparsity ---------------------------------------------------------------------------------------


def test_sparse_jacobian_of_loops_whose_pattern_holds_every_step_count() -> None:
  """A while loop's pattern is the union over every count up to its bound; its sparse Jacobian must
  still carry the exact values when the loop takes no step and when it takes all of them. A scan of
  one step has exactly the step's pattern."""
  c, lm = sc.sym("c", 4), sc.sym("lm", ())
  shift = sc.stack([c[0], c[0] + c[1], c[1] * c[2], c[3]])
  body = _fn("le_sp_body", [c, lm], [shift])
  cond = _fn("le_sp_cond", [c, lm], [sc.less(c[3], lm)])
  c0, lv = sc.sym("c0", 4), sc.sym("lv", ())
  fin, n = sc.while_loop(cond, body, c0, max_iter=2, params=(lv,))
  (one,) = sc.scan(_fn("le_sp_step", [c], [shift]), c0, length=1)
  sparse = {"while": sc.sparse_jacobian(fin, c0), "scan": sc.sparse_jacobian(one, c0)}
  union = np.array([[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [0, 0, 0, 1]], dtype=bool)
  np.testing.assert_array_equal(sparse["while"].sparsity.to_mask(), union)
  np.testing.assert_array_equal(sparse["scan"].sparsity.to_mask(), [[1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 0, 1]])
  f = _fn("le_sp", [c0, lv], [n, sparse["while"].values, jacobian(fin, c0), sparse["scan"].values, jacobian(one, c0)])
  start = np.array([0.5, -1.5, 2.0, 1.0])
  for lim, steps in ((0.0, 0), (5.0, 2)):
    got = f._flat_numerical_call(start, np.array(lim))
    assert got[0] == steps
    for values, dense, pattern in ((got[1], got[2], sparse["while"].sparsity), (got[3], got[4], sparse["scan"].sparsity)):
      scattered = np.zeros((4, 4))
      scattered[list(pattern.rows), list(pattern.cols)] = values
      np.testing.assert_allclose(scattered, dense, rtol=1e-15, atol=0.0)
    if steps == 0:
      np.testing.assert_array_equal(got[2], np.eye(4))


# --- validation -----------------------------------------------------------------------------------


def test_scan_refuses_what_it_cannot_loop() -> None:
  c, u = sc.sym("c", ()), sc.sym("u", 1)
  body = _fn("le_bad_scan", [c, u], [c + u[0]])
  c0, us = sc.sym("c0", ()), sc.sym("us", 3)
  with pytest.raises(TypeError, match="scan body must be an scaly Function, got function"):
    sc.scan(lambda c: c, c0, length=2)
  with pytest.raises(ValueError, match="scan length must be non-negative, got -1"):
    sc.scan(body, c0, [(us, 0, 1)], length=-1)
  with pytest.raises(ValueError, match="needs the carry as its first input"):
    sc.scan(_fn("le_bad_no_out", [c], []), c0, length=2)
  with pytest.raises(ValueError, match="needs the carry as its first input"):
    sc.scan(_fn("le_bad_no_in", [], [sc.const(1.0)]), c0, length=2)
  with pytest.raises(ValueError, match=r"must return a carry like its input: float64\(\) -> int64\(\)"):
    sc.scan(_fn("le_bad_cast", [c, u], [(c + u[0]).cast("int64")]), c0, [(us, 0, 1)], length=2)
  with pytest.raises(ValueError, match=r"scan init int64\(\) does not match the carry float64\(\)"):
    sc.scan(body, sc.sym("ci", (), dtype="int64"), [(us, 0, 1)], length=2)
  with pytest.raises(NotImplementedError, match=r"rank-1 outer tensors, got \(3, 1\)"):
    sc.scan(body, c0, [(sc.sym("m", (3, 1)), 0, 1)], length=2)
  with pytest.raises(ValueError, match="takes 1 sliced inputs after the carry, got 2"):
    sc.scan(body, c0, [(us, 0, 1), (us, 0, 1)], length=2)
  # An exact fit on either side is accepted.
  sc.scan(body, c0, [(us, 0, 1)], length=3)
  sc.scan(body, c0, [(us, 2, -1)], length=3)


def test_while_loop_refuses_what_it_cannot_loop() -> None:
  c, p = sc.sym("c", 2), sc.sym("p", 2)
  body, cond = _fn("le_bad_body", [c], [c * 0.5]), _fn("le_bad_cond", [c], [c[0] > 1.0])
  x = sc.sym("x", 2)
  with pytest.raises(TypeError, match="cond and body must be scaly Functions"):
    sc.while_loop(cond, lambda c: c, x, max_iter=2)
  with pytest.raises(ValueError, match="max_iter must be non-negative, got -1"):
    sc.while_loop(cond, body, x, max_iter=-1)
  with pytest.raises(ValueError, match=r"while_loop body output float64\(3,\) does not match the carry float64\(2,\)"):
    sc.while_loop(cond, _fn("le_bad_grow", [c], [sc.concat([c, c[:1]])]), x, max_iter=2)
  c3 = sc.sym("c3", 3)
  with pytest.raises(ValueError, match=r"while_loop cond input float64\(3,\) does not match"):
    sc.while_loop(_fn("le_bad_cond3", [c3], [c3[0] > 1.0]), body, x, max_iter=2)
  with pytest.raises(ValueError, match=r"while_loop init int64\(2,\) does not match"):
    sc.while_loop(cond, body, sc.sym("xi", 2, dtype="int64"), max_iter=2)
  with pytest.raises(ValueError, match=r"cond must return one bool, got bool\(2,\)"):
    sc.while_loop(_fn("le_bad_two", [c], [c > 1.0]), body, x, max_iter=2)
  with pytest.raises(ValueError, match=r"cond must return one bool, got float64\(\)"):
    sc.while_loop(_fn("le_bad_float", [c], [c[0] * 1.0]), body, x, max_iter=2)
  body_p, cond_p = _fn("le_bad_body_p", [c, p], [c * p]), _fn("le_bad_cond_p", [c, p], [c[0] > p[0]])
  with pytest.raises(ValueError, match=r"param 0 is int64\(2,\), but body takes float64\(2,\)"):
    sc.while_loop(cond_p, body_p, x, max_iter=2, params=(sc.sym("q", 2, dtype="int64"),))
  with pytest.raises(ValueError, match=r"param 0 is float64\(2,\), but cond takes float64\(3,\)"):
    sc.while_loop(_fn("le_bad_cond_r", [c, c3], [c[0] > 1.0]), body_p, x, max_iter=2, params=(p,))
  fin, n = sc.while_loop(cond, body, x, max_iter=0)
  assert fin.shape == (2,) and n.shape == () and n.type.dtype.name == "float64" and not n.type.diff


def test_vmap_refuses_what_it_cannot_map() -> None:
  c, p = sc.sym("c", 2), sc.sym("p", 2)
  f = _fn("le_bad_map", [c, p], [c * p, c[:1]])
  z = sc.sym("z", 6)
  with pytest.raises(TypeError, match="vmap callee must be an scaly Function"):
    sc.vmap(lambda a: a, 2, [(z, 0, 2), (z, 0, 2)])
  for output in (-1, 2):
    with pytest.raises(ValueError, match=f"output index {output} out of range for callee with 2 outputs"):
      sc.vmap(f, 2, [(z, 0, 2), (z, 0, 2)], output=output)
  with pytest.raises(ValueError, match="input 0 start must be non-negative, got -1"):
    sc.vmap(f, 0, [(z, -1, 2), (z, 0, 2)])
  with pytest.raises(NotImplementedError, match=r"rank-1 outer tensors, got \(3, 2\)"):
    sc.vmap(f, 2, [(sc.sym("m", (3, 2)), 0, 2), (z, 0, 2)])
  with pytest.raises(ValueError, match="input 1 reads past outer tensor of size 6: start=3"):
    sc.vmap(f, 3, [(z, 0, 2), (z, 3, 1)])
  with pytest.raises(ValueError, match="input 0 reads past outer tensor of size 6: start=5, stride=0"):
    sc.vmap(f, 3, [(z, 5, 0), (z, 0, 2)])
  # Exact fits are accepted, and a map of no iterations reads nothing.
  assert sc.vmap(f, 3, [(z, 0, 2), (z, 2, 1)]).shape == (6,)
  assert sc.vmap(f, 3, [(z, 4, 0), (z, 0, 2)], output=1).shape == (3,)
  assert sc.vmap(f, 0, [(z, 100, 7), (z, 0, 2)]).shape == (0,)
