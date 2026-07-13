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
