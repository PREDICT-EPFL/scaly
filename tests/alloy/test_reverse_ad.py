from __future__ import annotations

import numpy as np

import alloy as al


def test_vjp_scalar_output_matches_gradient() -> None:
  x = al.sym("x", 3)
  y = (x.sin() + x * x).sum()
  (grad_x,) = al.vjp((y,), (x,), (al.const(1.0),))
  f = al.Function("vjp", [x], [grad_x], ["x"], ["grad_x"])
  xv = np.array([0.1, 0.4, 0.9])

  np.testing.assert_allclose(f(xv), np.cos(xv) + 2 * xv)


def test_vjp_vector_output_uses_cotangent_seed() -> None:
  x = al.sym("x", 2)
  y = al.stack([x[0] * x[1], x[0].sin()])
  seed = al.sym("seed", 2)
  (grad_x,) = al.vjp((y,), (x,), (seed,))
  f = al.Function("vjp", [x, seed], [grad_x], ["x", "seed"], ["grad_x"])
  xv = np.array([0.3, 2.0])
  sv = np.array([1.5, -0.25])

  np.testing.assert_allclose(f(xv, sv), np.array([sv[0] * xv[1] + sv[1] * np.cos(xv[0]), sv[0] * xv[0]]))


def test_vjp_broadcast_and_multi_output_accumulates_adjoint() -> None:
  x = al.sym("x", (2, 3))
  b = al.sym("b", 3)
  y0 = (x + b).sum()
  y1 = (x * b).sum()
  (grad_x, grad_b) = al.vjp((y0, y1), (x, b), (al.const(2.0), al.const(-0.5)))
  f = al.Function("vjp", [x, b], [grad_x, grad_b], ["x", "b"], ["grad_x", "grad_b"])
  xv = np.arange(6.0).reshape(2, 3)
  bv = np.array([0.5, 1.5, 2.5])

  gx, gb = f(xv, bv)
  np.testing.assert_allclose(gx, np.broadcast_to(2.0 - 0.5 * bv, xv.shape))
  np.testing.assert_allclose(gb, np.full(3, 4.0) - 0.5 * xv.sum(axis=0))


def test_vjp_through_structural_ops_and_matmul() -> None:
  x = al.sym("x", (2, 2))
  a = al.sym("a", (2, 2))
  y = al.concat([x.T, a @ x], axis=1)
  seed = al.sym("seed", (2, 4))
  (grad_x,) = al.vjp((y,), (x,), (seed,))
  f = al.Function("vjp", [x, a, seed], [grad_x], ["x", "a", "seed"], ["grad_x"])
  xv = np.array([[1.0, 2.0], [3.0, 4.0]])
  av = np.array([[2.0, -1.0], [0.5, 3.0]])
  sv = np.arange(8.0).reshape(2, 4)

  expected = sv[:, :2].T + av.T @ sv[:, 2:]
  np.testing.assert_allclose(f(xv, av, sv), expected)


def test_jvp_many_uses_leading_seed_axis() -> None:
  x = al.sym("x", 3)
  y = al.stack([x[0] * x[1], x[2].sin() + x[0]])
  seeds = al.sym("seeds", (2, 3))
  dy = al.jvp_many(y, x, seeds)
  f = al.Function("jvp_many", [x, seeds], [dy], ["x", "seeds"], ["dy"])
  xv = np.array([0.3, 1.2, 0.7])
  sv = np.array([[1.5, -0.25, 0.4], [-0.5, 2.0, 1.25]])
  jac = np.array([[xv[1], xv[0], 0.0], [1.0, 0.0, np.cos(xv[2])]])

  np.testing.assert_allclose(f(xv, sv), sv @ jac.T)


def test_jvp_many_broadcast_scalar_tangent_over_vector() -> None:
  x = al.sym("x", 3)
  scale = x.sum()
  seeds = al.sym("seeds", (2, 3))
  dy = al.jvp_many(scale * x, x, seeds)
  f = al.Function("jvp_many_broadcast_scalar", [x, seeds], [dy], ["x", "seeds"], ["dy"])
  xv = np.array([0.3, 1.2, -0.4])
  sv = np.array([[1.5, -0.25, 0.4], [-0.5, 2.0, 1.25]])

  np.testing.assert_allclose(f(xv, sv), sv.sum(axis=1, keepdims=True) * xv + xv.sum() * sv)


def test_jvp_many_structural_rank_mismatch_corner_cases() -> None:
  # every rule of the structural batched JVP must return tangents shaped (nseed, *expr.shape), also under rank-mismatched broadcasts
  from alloy.ad import _jvp_many_structural

  rng = np.random.default_rng(3)
  x = al.sym("x", 9)
  seeds = al.sym("seeds", (4, 9))
  xv, sv = rng.normal(size=9), rng.normal(size=(4, 9))
  cases = {
    "mat_times_vec": ((x[:6].reshape((2, 3)) * x[6:9]).reshape((6,)), lambda v: (v[:6].reshape(2, 3) * v[6:9]).ravel()),
    "vec_div_scalar_sum": (x[:3] / x.sum(), lambda v: v[:3] / v.sum()),
    "scalar_div_vec": (x.sum() / x[:3], lambda v: v.sum() / v[:3]),
    "scalar_sum_plus_vec": (x.sum() + x[:3] * x[:3], lambda v: v.sum() + v[:3] * v[:3]),
    "vec_minus_scalar_sum": (x[:3] - x.sum(), lambda v: v[:3] - v.sum()),
    "scalar_plus_const_vec": (x.sum() + al.const(np.arange(3.0)), lambda v: v.sum() + np.arange(3.0)),
    "vec_pow_scalar_sym": ((x[:3] * x[:3] + 1.0) ** x[8], lambda v: (v[:3] * v[:3] + 1.0) ** v[8]),
    "scalar_pow_const_vec": (x[8] ** al.const(np.array([2.0, 3.0, 4.0])), lambda v: v[8] ** np.array([2.0, 3.0, 4.0])),
  }
  for name, (expr, np_fn) in cases.items():
    dy = _jvp_many_structural(expr, x, seeds, {}, {})
    assert dy.shape == (4, *expr.shape), f"{name}: tangent shape {dy.shape}"
    f = al.Function(f"jvp_many_{name}", [x, seeds], [dy], ["x", "seeds"], ["dy"])
    eps = 1e-6
    fd = np.stack([(np_fn(xv + eps * sv[i]) - np_fn(xv - eps * sv[i])) / (2 * eps) for i in range(4)])
    np.testing.assert_allclose(f(xv, sv), fd, rtol=1e-6, atol=1e-8, err_msg=name)


def test_jvp_many_vec_dot_vec_keeps_seed_axis() -> None:
  # (dx * y).sum() in the 1-D matmul JVP branch also contracted the seed axis, silently summing per-seed derivatives
  x = al.sym("x", 6)
  expr = x[:3] @ x[3:6] + x[0] * x[1]
  fn = al.Function("vec_dot_vec", [x], [al.stack([expr])], ["x"], ["y"])
  jf = fn.factory("vec_dot_vec_jac", ["x"], ["jac:y:x"])
  xv = np.random.default_rng(11).normal(size=6)
  expected = np.concatenate([xv[3:6] + np.array([xv[1], xv[0], 0.0]), xv[:3]])[None, :]
  np.testing.assert_allclose(jf(xv), expected, rtol=1e-12, atol=1e-12)


def test_sparse_jacobian_colored_scalar_plus_vector() -> None:
  z = al.sym("z", 6)
  sj = al.sparse_jacobian_colored(z[5] + z[:5] * z[:5], z)
  f = al.Function("spjac_scalar_plus_vec", [z], [sj.to_dense()], ["z"], ["dense"])
  zv = np.random.default_rng(5).normal(size=6)
  dense = np.zeros((5, 6))
  dense[:, 5] = 1.0
  dense[np.arange(5), np.arange(5)] += 2 * zv[:5]
  np.testing.assert_allclose(f(zv), dense)


def test_vjp_many_uses_leading_seed_axis_and_multiple_outputs() -> None:
  x = al.sym("x", 2)
  y0 = x * x
  y1 = x.sum()
  c0 = al.sym("c0", (2, 2))
  c1 = al.sym("c1", 2)
  (grad_x,) = al.vjp_many((y0, y1), (x,), (c0, c1))
  f = al.Function("vjp_many", [x, c0, c1], [grad_x], ["x", "c0", "c1"], ["grad_x"])
  xv = np.array([0.3, 1.2])
  c0v = np.array([[1.5, -0.25], [-0.5, 2.0]])
  c1v = np.array([0.75, -1.25])

  np.testing.assert_allclose(f(xv, c0v, c1v), c0v * (2 * xv) + c1v[:, None])


def test_multi_seed_shape_errors() -> None:
  x = al.sym("x", 2)
  y = x * x

  try:
    _ = al.jvp_many(y, x, al.sym("bad", 2))
  except ValueError as e:
    assert "multi-seed JVP expects seeds shape" in str(e)
  else:  # pragma: no cover
    raise AssertionError("bad multi-seed JVP shape should fail")

  try:
    _ = al.vjp_many((y,), (x,), (al.sym("bad", 2),))
  except ValueError as e:
    assert "multi-seed VJP expects cotangent shape" in str(e)
  else:  # pragma: no cover
    raise AssertionError("bad multi-seed VJP shape should fail")


def test_vjp_through_call_node_inlines_callee_reverse_graph() -> None:
  x = al.sym("x", 2)
  inner = al.Function("inner", [x], [x.sin() * x], ["x"], ["y"])
  z = al.sym("z", 2)
  (inner_z,) = inner.call([z])
  seed = al.sym("seed", 2)
  (grad_z,) = al.vjp((inner_z,), (z,), (seed,))
  outer = al.Function("outer", [z, seed], [grad_z], ["z", "seed"], ["grad_z"])
  zv = np.array([0.2, 0.7])
  sv = np.array([3.0, -1.0])

  np.testing.assert_allclose(outer(zv, sv), sv * (np.sin(zv) + zv * np.cos(zv)))


def test_ad_skips_nonsmooth_parameter_terms_independent_of_wrt() -> None:
  x = al.sym("x", 2)
  p = al.sym("p", 2, diff=False)
  seed = al.sym("seed", 2)
  y = (x * x + p.floor()).sum()
  dy = al.jvp(y, x, seed)
  grad = al.expr_gradient(y, x)
  f = al.Function("smooth_wrt_x", [x, p, seed], [dy, grad], ["x", "p", "seed"], ["dy", "grad"])
  xv = np.array([0.3, 1.2])
  pv = np.array([1.1, 2.9])
  sv = np.array([1.5, -0.25])

  dyv, gradv = f(xv, pv, sv)
  np.testing.assert_allclose(dyv, np.dot(2 * xv, sv))
  np.testing.assert_allclose(gradv, 2 * xv)

  try:
    _ = al.jvp(x.floor().sum(), x, seed)
  except NotImplementedError as e:
    assert "nonsmooth" in str(e)
  else:  # pragma: no cover
    raise AssertionError("nonsmooth JVP through wrt should still fail")
