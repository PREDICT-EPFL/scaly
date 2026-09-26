"""``while_loop(..., params=...)``: tensors every step reads unchanged, passed to the body and the
condition where they are instead of travelling in the carry."""

from __future__ import annotations

import re

import numpy as np
import pytest

import scaly as sc
from scaly.codegen import render_c_module
from scaly.ir.expr import CALLEE_OPS, Expr, ExprOp, callees_of, format_expr, topo
from scaly.ir.expr_spec import verify_expr
from scaly.linalg import SparseLDL, SparseMatrix

STEPS = 5


def _counted_loop(tag: str, *, index: bool = False):
  """``x <- tanh(W x + b) * s`` for exactly ``STEPS`` steps (a counter in the carry stops it), with
  ``W``, ``b`` and ``s`` as params: smooth, so derivatives compare with an unrolled reference."""
  c = sc.sym("c", 4)
  k = sc.sym("k", (), dtype="int64")
  w, b, s = sc.sym("w", (3, 3)), sc.sym("b", 3), sc.sym("s", ())
  x, count = c[:3], c[3]
  scale = s * (1.0 + 0.1 * k.cast("float64")) if index else s
  nxt = sc.concat([(w @ x + b).tanh() * scale, (count + 1.0).reshape((1,))])
  inputs = [c, k, w, b, s] if index else [c, w, b, s]
  body = sc.Function._from_exprs(f"wp_body_{tag}", inputs, [nxt], [str(e.name) for e in inputs], ["n"])
  cond = sc.Function._from_exprs(f"wp_cond_{tag}", [c, w, b, s], [sc.less(c[3], float(STEPS))], ["c", "w", "b", "s"], ["go"])
  return cond, body


def _unrolled(x0: Expr, w: Expr, b: Expr, s: Expr, *, index: bool = False) -> Expr:
  x = x0
  for k in range(STEPS):
    x = (w @ x + b).tanh() * (s * (1.0 + 0.1 * k) if index else s)
  return x


@pytest.mark.parametrize("index", [False, True])
def test_values_and_every_derivative_match_the_unrolled_loop(index: bool) -> None:
  cond, body = _counted_loop(f"d{int(index)}", index=index)
  x0, w, b, s = sc.sym("x0", 3), sc.sym("W", (3, 3)), sc.sym("B", 3), sc.sym("S", ())
  out, n = sc.while_loop(cond, body, sc.concat([x0, sc.const(np.zeros(1))]), max_iter=20, index=index, params=(w, b, s))
  loop = out[:3]
  ref = _unrolled(x0, w, b, s, index=index)
  f_loop, f_ref = (loop * loop).sum() + loop[0] * s, (ref * ref).sum() + ref[0] * s
  exprs = []
  for y, fy in ((loop, f_loop), (ref, f_ref)):
    exprs += [y, sc.jacobian(y, w), sc.jacobian(y, s.reshape((1,))), sc.gradient(fy, w), sc.gradient(fy, b), sc.gradient(fy, s), sc.hessian(fy, b)]
    exprs.append(sc.jvp(y, b, sc.const(np.array([0.3, -0.2, 0.1]))))
  fn = sc.Function._from_exprs(f"wp_all_{int(index)}", [x0, w, b, s], [*exprs, n], ["x0", "W", "B", "S"], [f"o{i}" for i in range(len(exprs) + 1)])
  rng = np.random.default_rng(0)
  vals = fn((rng.standard_normal(3), 0.4 * rng.standard_normal((3, 3)), rng.standard_normal(3), np.array(0.8)))
  half = len(exprs) // 2
  for got, want in zip(vals[:half], vals[half:-1], strict=True):
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-13)
  assert vals[-1] == STEPS


def test_only_params_that_depend_on_wrt_get_cotangents() -> None:
  """A gradient in the initial carry alone does not form the params' cotangents, and one in a param
  alone does not need the initial carry's."""
  cond, body = _counted_loop("act")
  x0, w, b, s = sc.sym("x0", 3), sc.sym("W", (3, 3)), sc.sym("B", 3), sc.sym("S", ())
  out, _ = sc.while_loop(cond, body, sc.concat([x0, sc.const(np.zeros(1))]), max_iter=20, params=(w, b, s))
  f = sc.sumsqr(out[:3])
  g_x0, g_b = sc.gradient(f, x0), sc.gradient(f, b)
  ref = _unrolled(x0, w, b, s)
  fn = sc.Function._from_exprs(
    "wp_act", [x0, w, b, s], [g_x0, g_b, sc.gradient(sc.sumsqr(ref), x0), sc.gradient(sc.sumsqr(ref), b)], ["x0", "W", "B", "S"], ["a", "b", "c", "d"]
  )
  rng = np.random.default_rng(1)
  a, bb, c, d = fn((rng.standard_normal(3), 0.3 * rng.standard_normal((3, 3)), rng.standard_normal(3), np.array(1.1)))
  np.testing.assert_allclose(a, c, rtol=1e-12)
  np.testing.assert_allclose(bb, d, rtol=1e-12)
  # The backward scan's carry holds the carry's cotangent and only b's.
  adj = next(e for e in topo([g_b]) if e.op == ExprOp.SCAN)
  assert adj.args[0].size == 4 + 3


def test_sparsity_brings_the_params_in_at_every_step() -> None:
  c, a = sc.sym("c", 4), sc.sym("a", 4)
  body = sc.Function._from_exprs("wp_sp_body", [c, a], [0.5 * c + a * a], ["c", "a"], ["n"])
  cond = sc.Function._from_exprs("wp_sp_cond", [c, a], [sc.less(sc.norm_inf(c), 10.0)], ["c", "a"], ["go"])
  x0, p = sc.sym("x0", 4), sc.sym("p", 4)
  out, _ = sc.while_loop(cond, body, x0, max_iter=6, params=(p,))
  np.testing.assert_array_equal(sc.jacobian_sparsity(out, p).to_mask(), np.eye(4, dtype=bool))
  np.testing.assert_array_equal(sc.jacobian_sparsity(out, x0).to_mask(), np.eye(4, dtype=bool))
  # A loop reading nothing that depends on wrt has no pattern at all.
  q = sc.sym("q", 4)
  assert sc.jacobian_sparsity(out, q).nnz == 0


def test_params_are_passed_not_copied() -> None:
  """The loop's store holds the carry only; a 4000-entry param reaches the calls by pointer."""
  c, big = sc.sym("c", 2), sc.sym("big", 4000)
  body = sc.Function._from_exprs("wp_nc_body", [c, big], [c * 0.5 + big[:2]], ["c", "big"], ["n"])
  cond = sc.Function._from_exprs("wp_nc_cond", [c, big], [sc.greater(c[0], big[2])], ["c", "big"], ["go"])
  x, bigv = sc.sym("x", 2), sc.sym("B", 4000)
  out, n = sc.while_loop(cond, body, x, max_iter=50, params=(bigv,))
  fn = sc.Function._from_exprs("wp_nc", [x, bigv], [out, n], ["x", "B"], ["out", "n"])
  src = render_c_module(fn).body
  sizes = {int(m) for m in re.findall(r"\[(\d+)\]", src)}
  assert not any(size >= 4000 for size in sizes)  # no buffer of the param's size
  bv = np.zeros(4000)
  bv[:3] = [0.1, 0.1, 0.25]
  got, steps = fn((np.array([8.0, 1.0]), bv))
  x_ref = np.array([8.0, 1.0])
  count = 0
  while x_ref[0] > 0.25 and count < 50:
    x_ref = 0.5 * x_ref + 0.1
    count += 1
  np.testing.assert_allclose(got, x_ref, rtol=1e-14)
  assert steps == count


def test_an_integer_constant_param_serves_the_in_place_proof() -> None:
  """``put`` at indices read from a constant int64 param: the proof reads the table as the same value
  every step, and the loop updates its carry in place."""
  c, idx = sc.sym("c", 6), sc.sym("idx", 2, dtype="int64")
  body = sc.Function._from_exprs("wp_ip_body", [c, idx], [sc.put_add(c, idx, sc.stack([c[0], c[0]]) * 0.0 + 1.0)], ["c", "idx"], ["n"])
  cond = sc.Function._from_exprs("wp_ip_cond", [c, idx], [sc.less(c[4], 3.0)], ["c", "idx"], ["go"])
  x = sc.sym("x", 6)
  out, _ = sc.while_loop(cond, body, x, max_iter=10, params=(sc.const(np.array([4, 5]), dtype="int64"),))
  fn = sc.Function._from_exprs("wp_ip", [x], [out], ["x"], ["out"])
  assert "wp_ip_body_inplace" in render_c_module(fn).body
  np.testing.assert_array_equal(fn(np.zeros(6)), [0, 0, 0, 0, 3, 3])


def _loops(fn: sc.Function) -> list[Expr]:
  """Every while-loop node in ``fn``'s graph and its callees'."""
  seen, out, todo = set(), [], [fn]
  while todo:
    f = todo.pop()
    for e in topo(f.outputs):
      if e.op == ExprOp.WHILE:
        out.append(e)
      for callee in callees_of(e) if e.op in CALLEE_OPS else ():
        if id(callee) not in seen:
          seen.add(id(callee))
          todo.append(callee)
  return out


def test_adaptive_refinement_carries_only_x_and_r() -> None:
  """SparseLDL's adaptive refinement, the first user: the factor, K and b are params, and the loop
  still updates its carry [x | r] in place."""
  k = np.array([[4.0, 1.0, 0.0], [1.0, -3.0, 0.5], [0.0, 0.5, 2.0]])
  mat = SparseMatrix.symbol("K", np.tril(k) != 0)
  bsym = sc.sym("b", 3)
  fact = SparseLDL(mat, schedule="scan", name="wp_refine")
  fn = sc.Function._from_exprs("wp_refine_fn", [mat.values, bsym], [fact.solve(bsym, refine=4, tol=1e-15)], ["K", "b"], ["x"])
  assert "_refine_inplace" in render_c_module(fn).body
  (loop,) = _loops(fn)
  assert loop.args[0].size == 6 and len(loop.args) == 5  # [x | r], then f, K, b and the threshold
  rows, cols = mat.coordinates()
  bv = np.array([1.0, 2.0, -1.0])
  np.testing.assert_allclose(fn((k[rows, cols], bv)), np.linalg.solve(k, bv), rtol=1e-14)


def test_validation_and_the_verifier() -> None:
  c, p = sc.sym("c", 2), sc.sym("p", 3)
  body = sc.Function._from_exprs("wp_v_body", [c, p], [c + p[:2]], ["c", "p"], ["n"])
  cond = sc.Function._from_exprs("wp_v_cond", [c, p], [sc.less(c[0], p[2])], ["c", "p"], ["go"])
  plain_cond = sc.Function._from_exprs("wp_v_cond0", [c], [sc.less(c[0], 1.0)], ["c"], ["go"])
  x = sc.sym("x", 2)
  with pytest.raises(ValueError, match="body takes the carry and 0 params"):
    sc.while_loop(cond, body, x, max_iter=3)
  with pytest.raises(ValueError, match="param 0 is float64\\(4,\\), but body takes float64\\(3,\\)"):
    sc.while_loop(cond, body, x, max_iter=3, params=(sc.sym("q", 4),))
  with pytest.raises(ValueError, match="cond the carry and the params"):
    sc.while_loop(plain_cond, body, x, max_iter=3, params=(sc.sym("q", 3),))
  out, _ = sc.while_loop(cond, body, x, max_iter=3, params=(sc.sym("q", 3),))
  verify_expr(out)
  bad = Expr(ExprOp.WHILE, (x, sc.sym("q4", 4)), out.type, attrs=dict(out.attrs))
  with pytest.raises(Exception, match="WHILE param 0"):
    verify_expr(bad)


def test_two_loops_that_differ_only_in_their_params_stay_apart() -> None:
  cond, body = _counted_loop("two")
  x0, w = sc.sym("x0", 3), sc.sym("W", (3, 3))
  init = sc.concat([x0, sc.const(np.zeros(1))])
  first, _ = sc.while_loop(cond, body, init, max_iter=20, params=(w, sc.const(np.zeros(3)), sc.const(1.0)))
  second, _ = sc.while_loop(cond, body, init, max_iter=20, params=(w, sc.const(np.ones(3)), sc.const(1.0)))
  fn = sc.Function._from_exprs("wp_two", [x0, w], [first[:3], second[:3]], ["x0", "W"], ["a", "b"])
  xv, wv = np.array([0.1, 0.2, 0.3]), 0.3 * np.eye(3)
  a, b = fn((xv, wv))
  ra, rb = xv.copy(), xv.copy()
  for _ in range(STEPS):
    ra, rb = np.tanh(wv @ ra), np.tanh(wv @ rb + 1.0)
  np.testing.assert_allclose(a, ra, rtol=1e-14)
  np.testing.assert_allclose(b, rb, rtol=1e-14)


def test_a_constant_carry_still_depends_on_its_params() -> None:
  cond, body = _counted_loop("const")
  w, b, s = sc.sym("W", (3, 3)), sc.sym("B", 3), sc.sym("S", ())
  out, _ = sc.while_loop(cond, body, sc.const(np.array([0.5, -0.5, 0.25, 0.0])), max_iter=20, params=(w, b, s))
  assert out.type.diff
  ref = _unrolled(sc.const(np.array([0.5, -0.5, 0.25])), w, b, s)
  fn = sc.Function._from_exprs(
    "wp_const", [w, b, s], [sc.gradient(sc.sumsqr(out[:3]), b), sc.gradient(sc.sumsqr(ref), b)], ["W", "B", "S"], ["g", "r"]
  )
  g, r = fn((0.3 * np.eye(3), np.array([0.1, 0.2, 0.3]), np.array(1.0)))
  assert np.abs(g).max() > 0.1
  np.testing.assert_allclose(g, r, rtol=1e-12)


def test_the_printer_shows_the_params() -> None:
  c, p = sc.sym("c", 2), sc.sym("p", 3)
  body = sc.Function._from_exprs("wp_pr_body", [c, p], [c + p[:2]], ["c", "p"], ["n"])
  cond = sc.Function._from_exprs("wp_pr_cond", [c, p], [sc.less(c[0], p[2])], ["c", "p"], ["go"])
  out, _ = sc.while_loop(cond, body, sc.sym("x", 2), max_iter=3, params=(sc.sym("q", 3),))
  assert re.search(r"while\[3\] wp_pr_cond wp_pr_body\[0\]\(%\w+, %\w+\)", format_expr([out]))
