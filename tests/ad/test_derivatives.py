from __future__ import annotations

import numpy as np
import pytest

import alloy as al
from alloy.ad import _jvp_many_structural, _jvp_many_unrolled, finite_difference
from alloy.ir.expr import topo


def _vmap_vjp_piece(name: str, nargs: int = 1) -> al.Function:
  input_names = [f"x{i}" for i in range(nargs)]
  inputs = [al.sym(input_name, 2) for input_name in input_names]
  out = inputs[0] * inputs[0] + inputs[0].sin()
  for inp in inputs[1:]:
    out = out + inputs[0] * inp + inp.sin()
  return al.Function._from_exprs(name, inputs, [out], input_names, ["y"])


def _assert_vmap_vjp_matches_unrolled_and_fd(
  name: str, callee: al.Function, z: al.Expr, length: int, specs: list[tuple[al.Expr, int, int]], zv: np.ndarray
) -> None:
  mapped = al.vmap(callee, length, specs)
  unrolled = al.concat(
    [
      callee(
        callee.input_tree.unflatten(
          tuple(
            outer[start + it * stride : start + it * stride + formal.size]
            for formal, (outer, start, stride) in zip(callee.inputs, specs, strict=True)
          )
        )
      )
      for it in range(length)
    ]
  )
  lam = al.sym("lam", mapped.size)
  mapped_obj, unrolled_obj = al.dot(lam, mapped), al.dot(lam, unrolled)
  (mapped_grad,) = al.vjp((mapped_obj,), (z,), (al.const(1.0),))
  (unrolled_grad,) = al.vjp((unrolled_obj,), (z,), (al.const(1.0),))
  grad_fn = al.Function._from_exprs(f"{name}_grads", [z, lam], [mapped_grad, unrolled_grad], ["z", "lam"], ["mapped", "unrolled"])
  obj_fn = al.Function._from_exprs(f"{name}_objective", [z, lam], [mapped_obj], ["z", "lam"], ["objective"])
  lamv = np.random.default_rng(10).normal(size=mapped.size)

  mapped_value, unrolled_value = grad_fn((zv, lamv))
  fd = finite_difference(lambda value: obj_fn((value, lamv)), zv).reshape(-1)
  np.testing.assert_allclose(mapped_value, unrolled_value, rtol=1e-10, atol=1e-10)
  np.testing.assert_allclose(mapped_value, fd, rtol=1e-6, atol=1e-7)


def test_vjp_scalar_output_matches_gradient() -> None:
  x = al.sym("x", 3)
  y = (x.sin() + x * x).sum()
  (grad_x,) = al.vjp((y,), (x,), (al.const(1.0),))
  f = al.Function._from_exprs("vjp", [x], [grad_x], ["x"], ["grad_x"])
  xv = np.array([0.1, 0.4, 0.9])

  np.testing.assert_allclose(f(xv), np.cos(xv) + 2 * xv)


def test_vjp_vector_output_uses_cotangent_seed() -> None:
  x = al.sym("x", 2)
  y = al.stack([x[0] * x[1], x[0].sin()])
  seed = al.sym("seed", 2)
  (grad_x,) = al.vjp((y,), (x,), (seed,))
  f = al.Function._from_exprs("vjp", [x, seed], [grad_x], ["x", "seed"], ["grad_x"])
  xv = np.array([0.3, 2.0])
  sv = np.array([1.5, -0.25])

  np.testing.assert_allclose(f((xv, sv)), np.array([sv[0] * xv[1] + sv[1] * np.cos(xv[0]), sv[0] * xv[0]]))


def test_vjp_broadcast_and_multi_output_accumulates_adjoint() -> None:
  x = al.sym("x", (2, 3))
  b = al.sym("b", 3)
  y0 = (x + b).sum()
  y1 = (x * b).sum()
  (grad_x, grad_b) = al.vjp((y0, y1), (x, b), (al.const(2.0), al.const(-0.5)))
  f = al.Function._from_exprs("vjp", [x, b], [grad_x, grad_b], ["x", "b"], ["grad_x", "grad_b"])
  xv = np.arange(6.0).reshape(2, 3)
  bv = np.array([0.5, 1.5, 2.5])

  gx, gb = f((xv, bv))
  np.testing.assert_allclose(gx, np.broadcast_to(2.0 - 0.5 * bv, xv.shape))
  np.testing.assert_allclose(gb, np.full(3, 4.0) - 0.5 * xv.sum(axis=0))


def test_vjp_through_structural_ops_and_matmul() -> None:
  x = al.sym("x", (2, 2))
  a = al.sym("a", (2, 2))
  y = al.concat([x.T, a @ x], axis=1)
  seed = al.sym("seed", (2, 4))
  (grad_x,) = al.vjp((y,), (x,), (seed,))
  f = al.Function._from_exprs("vjp", [x, a, seed], [grad_x], ["x", "a", "seed"], ["grad_x"])
  xv = np.array([[1.0, 2.0], [3.0, 4.0]])
  av = np.array([[2.0, -1.0], [0.5, 3.0]])
  sv = np.arange(8.0).reshape(2, 4)

  expected = sv[:, :2].T + av.T @ sv[:, 2:]
  np.testing.assert_allclose(f((xv, av, sv)), expected)


def test_vjp_nested_calls_shared_symbol_matches_fd_and_jvp() -> None:
  # The CALL VJP inlines the callee adjoint and substitutes formals with actuals. The caller here
  # reuses the callee's formal symbol `x`, so the substitution must not rewrite occurrences of `x`
  # inside the incoming cotangent (which the second call's adjoint injects into the first call's).
  x = al.sym("x", 1)
  f = al.Function._from_exprs("sq", [x], [x * x], ["x"], ["y"])
  k1 = f(2 * x)
  k2 = f(x + k1)
  (grad_rev,) = al.vjp((k2,), (x,), (al.const(np.ones(1)),))
  jac_fwd = al.jacobian(k2, x).reshape((1,))
  fn = al.Function._from_exprs("nested_sq", [x], [grad_rev, jac_fwd], ["x"], ["rev", "fwd"])

  rev, fwd = fn(np.array([1.0]))
  np.testing.assert_allclose(rev, [90.0], rtol=1e-12)  # d/dx (x + 4x^2)^2 at x=1
  np.testing.assert_allclose(rev, fwd, rtol=1e-12)


def test_vjp_nested_call_rk4_matches_jvp_transpose_and_fd() -> None:
  # RK4 body calling an inner Function 4x with shared symbols: reverse must equal J^T lam and FD,
  # and the exact Hessian of lam^T rk4(x) must come out symmetric.
  from alloy.ad import hessian  # noqa: PLC0415

  rng = np.random.default_rng(7)
  x, u = al.sym("x", 2), al.sym("u", 1)
  ode = al.Function._from_exprs("ode2", [x, u], [al.stack([x[0] * x[1] + u[0], x[0].tanh() - x[1] * x[1]])], ["x", "u"], ["f"])
  dt = 0.1
  k1 = ode((x, u))
  k2 = ode((x + (dt / 2) * k1, u))
  k3 = ode((x + (dt / 2) * k2, u))
  k4 = ode((x + dt * k3, u))
  xnext = x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
  lam = al.sym("lam", 2)
  (grad_rev,) = al.vjp((xnext,), (x,), (lam,))
  jac_t_lam = al.jacobian(xnext, x).transpose((1, 0)) @ lam
  hess = hessian(al.dot(lam, xnext), x)
  fn = al.Function._from_exprs("rk4_adj", [x, u, lam], [grad_rev, jac_t_lam, hess], ["x", "u", "lam"], ["rev", "fwd", "hess"])
  obj = al.Function._from_exprs("rk4_obj", [x, u, lam], [al.dot(lam, xnext)], ["x", "u", "lam"], ["obj"])
  xv, uv, lamv = rng.normal(size=2), rng.normal(size=1), rng.normal(size=2)

  rev, fwd, hv = fn((xv, uv, lamv))
  fd = finite_difference(lambda v: obj((v, uv, lamv)), xv).reshape(-1)
  np.testing.assert_allclose(rev, fwd, rtol=1e-12, atol=1e-12)
  np.testing.assert_allclose(rev, fd, rtol=1e-6, atol=1e-8)
  np.testing.assert_allclose(hv, hv.T, rtol=1e-12, atol=1e-12)


def test_jvp_many_uses_leading_seed_axis() -> None:
  x = al.sym("x", 3)
  y = al.stack([x[0] * x[1], x[2].sin() + x[0]])
  seeds = al.sym("seeds", (2, 3))
  dy = al.jvp_many(y, x, seeds)
  f = al.Function._from_exprs("jvp_many", [x, seeds], [dy], ["x", "seeds"], ["dy"])
  xv = np.array([0.3, 1.2, 0.7])
  sv = np.array([[1.5, -0.25, 0.4], [-0.5, 2.0, 1.25]])
  jac = np.array([[xv[1], xv[0], 0.0], [1.0, 0.0, np.cos(xv[2])]])

  np.testing.assert_allclose(f((xv, sv)), sv @ jac.T)


def test_erf_forward_reverse_jacobian_and_sparse_hessian(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv("ALLOY_STRICT_JVP_MANY", "1")
  x = al.sym("x", 3)
  seed = al.sym("seed", 3)
  y = x.erf()
  jvp = al.jvp(y, x, seed)
  (vjp,) = al.vjp((y,), (x,), (seed,))
  ad = al.Function._from_exprs("erf_seed_ad", [x, seed], [jvp, vjp], ["x", "seed"], ["jvp", "vjp"])
  weights = np.array([0.5, -1.25, 2.0])
  base = al.Function._from_exprs("erf_derivatives", [x], [y, (al.const(weights) * y).sum()], ["x"], ["y", "cost"])
  derivatives = base.factory("erf_jac_sphess", ["x"], [al.factory.Jac("y", "x"), al.factory.SpHess("cost", "x")])

  xv = np.array([-1.2, 0.25, 2.1])
  seedv = np.array([0.3, -0.7, 1.4])
  first = 2 / np.sqrt(np.pi) * np.exp(-(xv**2))
  jvp_value, vjp_value = ad((xv, seedv))
  jac_value, sphess_value = derivatives(xv)

  np.testing.assert_allclose(jvp_value, seedv * first, rtol=1e-12, atol=1e-12)
  np.testing.assert_allclose(vjp_value, seedv * first, rtol=1e-12, atol=1e-12)
  np.testing.assert_allclose(jac_value, np.diag(first), rtol=1e-12, atol=1e-12)
  sparsity = derivatives.output_sparsities[1]
  assert sparsity is not None
  assert sparsity.rows == (0, 1, 2)
  assert sparsity.cols == (0, 1, 2)
  second = weights * (-4 * xv / np.sqrt(np.pi)) * np.exp(-(xv**2))
  np.testing.assert_allclose(sphess_value, second, rtol=1e-12, atol=1e-12)


def test_jvp_many_broadcast_scalar_tangent_over_vector() -> None:
  x = al.sym("x", 3)
  scale = x.sum()
  seeds = al.sym("seeds", (2, 3))
  dy = al.jvp_many(scale * x, x, seeds)
  f = al.Function._from_exprs("jvp_many_broadcast_scalar", [x, seeds], [dy], ["x", "seeds"], ["dy"])
  xv = np.array([0.3, 1.2, -0.4])
  sv = np.array([[1.5, -0.25, 0.4], [-0.5, 2.0, 1.25]])

  np.testing.assert_allclose(f((xv, sv)), sv.sum(axis=1, keepdims=True) * xv + xv.sum() * sv)


def test_jvp_many_sum_batches_seeds_without_unrolling() -> None:
  x = al.sym("x", (2, 3))
  expr = x.sum()
  seeds = al.sym("seeds", (4, 2, 3))
  structural = _jvp_many_structural(expr, x, seeds, {}, {})
  reference = _jvp_many_unrolled(expr, x, seeds)
  one_seed = _jvp_many_structural(expr, x, al.sym("one_seed", (1, 2, 3)), {}, {})

  assert structural.shape == (4,)
  nodes = topo((structural,))
  assert len(nodes) == len(topo((one_seed,)))
  assert sum(node.op == al.ExprOp.MATMUL for node in nodes) == 1
  assert sum(node.op == al.ExprOp.RESHAPE for node in nodes) == 1
  assert not any(node.op == al.ExprOp.STACK for node in nodes)

  fn = al.Function._from_exprs("jvp_many_sum", [x, seeds], [structural, reference], ["x", "seeds"], ["structural", "reference"])
  xv = np.random.default_rng(16).normal(size=(2, 3))
  seedv = np.random.default_rng(17).normal(size=(4, 2, 3))
  actual, expected = fn((xv, seedv))
  np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_float32_reduction_derivative_paths(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv("ALLOY_STRICT_JVP_MANY", "1")
  x = al.sym("float32_x", 3, dtype=al.dtypes.float32)
  expr = x.sum()
  seeds = al.sym("float32_seeds", (2, 3), dtype=al.dtypes.float32)

  jvp_many = al.jvp_many(expr, x, seeds)
  jacobian = al.jacobian(expr, x)
  hessian = al.hessian(expr, x)
  sparse_jacobian = al.sparse_jacobian(expr, x)
  sparse_hessian = al.sparse_hessian(expr, x)

  assert jvp_many.shape == (2,)
  assert jvp_many.type.dtype == al.dtypes.float32
  assert jacobian.shape == (1, 3)
  assert hessian.shape == (3, 3)
  assert sparse_jacobian.sparsity.shape == (1, 3)
  assert sparse_hessian.sparsity.shape == (3, 3)


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
    f = al.Function._from_exprs(f"jvp_many_{name}", [x, seeds], [dy], ["x", "seeds"], ["dy"])
    eps = 1e-6
    fd = np.stack([(np_fn(xv + eps * sv[i]) - np_fn(xv - eps * sv[i])) / (2 * eps) for i in range(4)])
    np.testing.assert_allclose(f((xv, sv)), fd, rtol=1e-6, atol=1e-8, err_msg=name)


def test_jvp_many_vec_dot_vec_keeps_seed_axis() -> None:
  # (dx * y).sum() in the 1-D matmul JVP branch also contracted the seed axis, silently summing per-seed derivatives
  x = al.sym("x", 6)
  expr = x[:3] @ x[3:6] + x[0] * x[1]
  fn = al.Function._from_exprs("vec_dot_vec", [x], [al.stack([expr])], ["x"], ["y"])
  jf = fn.factory("vec_dot_vec_jac", ["x"], [al.factory.Jac("y", "x")])
  xv = np.random.default_rng(11).normal(size=6)
  expected = np.concatenate([xv[3:6] + np.array([xv[1], xv[0], 0.0]), xv[:3]])[None, :]
  np.testing.assert_allclose(jf(xv), expected, rtol=1e-12, atol=1e-12)


def test_jvp_many_scatter_and_gather_stays_structural() -> None:
  x = al.sym("x", 6)
  seeds = al.sym("seeds", (3, 6))
  expr = al.scatter(al.gather(x, np.array([[5, 1], [4, 0]])), np.array([[0, 4], [7, 8]]), (3, 3))
  structural = _jvp_many_structural(expr, x, seeds, {}, {})
  reference = _jvp_many_unrolled(expr, x, seeds)

  assert structural.shape == (3, *expr.shape)
  nodes = topo((structural,))
  assert sum(node.op == al.ExprOp.GATHER for node in nodes) == 1
  assert sum(node.op == al.ExprOp.SCATTER for node in nodes) == 1
  fn = al.Function._from_exprs("jvp_many_scatter_gather", [x, seeds], [structural, reference], ["x", "seeds"], ["structural", "reference"])
  xv = np.random.default_rng(12).normal(size=6)
  seedv = np.random.default_rng(13).normal(size=(3, 6))
  actual, expected = fn((xv, seedv))
  np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_jvp_many_transpose_stays_structural() -> None:
  x = al.sym("x", 12)
  seeds = al.sym("seeds", (4, 12))
  expr = x.reshape((2, 3, 2)).transpose((2, 0, 1))
  structural = _jvp_many_structural(expr, x, seeds, {}, {})
  reference = _jvp_many_unrolled(expr, x, seeds)

  assert structural.shape == (4, *expr.shape)
  assert sum(node.op == al.ExprOp.TRANSPOSE for node in topo((structural,))) == 1
  fn = al.Function._from_exprs("jvp_many_transpose", [x, seeds], [structural, reference], ["x", "seeds"], ["structural", "reference"])
  xv = np.random.default_rng(14).normal(size=12)
  seedv = np.random.default_rng(15).normal(size=(4, 12))
  actual, expected = fn((xv, seedv))
  np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_jvp_many_rank4_transpose_falls_back_and_strict_raises(monkeypatch: pytest.MonkeyPatch) -> None:
  # A rank-4 primal TRANSPOSE would need a rank-5 tangent TRANSPOSE, beyond the rank-4 lowering
  # limit; the structural rule must decline (unrolled fallback lowers fine) instead of building an
  # un-lowerable graph that only fails later at compile time.
  x = al.sym("x", 12)
  expr = (x * x).reshape((2, 3, 1, 2)).transpose((3, 1, 0, 2))
  seeds = al.sym("seeds", (2, 12))
  fn = al.Function._from_exprs("rank4_transpose_jvp", [x, seeds], [al.jvp_many(expr, x, seeds).reshape((24,))], ["x", "seeds"], ["tan"])
  rng = np.random.default_rng(7)
  xv, sv = rng.normal(size=12), rng.normal(size=(2, 12))
  expected = np.concatenate([(2.0 * xv * sv[i]).reshape(2, 3, 1, 2).transpose(3, 1, 0, 2).reshape(-1) for i in range(2)])
  np.testing.assert_allclose(fn((xv, sv)), expected, rtol=1e-12, atol=1e-12)
  monkeypatch.setenv("ALLOY_STRICT_JVP_MANY", "1")
  with pytest.raises(NotImplementedError, match="structural jvp_many does not support"):
    al.jvp_many(expr, x, seeds)


def test_jvp_many_strict_mode_raises_on_unsupported_structural_rule(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv("ALLOY_STRICT_JVP_MANY", "1")
  x = al.sym("x", 2)
  with pytest.raises(NotImplementedError, match="structural jvp_many does not support"):
    al.jvp_many(x.abs(), x, al.const(np.eye(2)))


def test_jvp_many_strict_mode_raises_on_structural_shape_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
  import alloy.ad.forward as ad

  monkeypatch.setenv("ALLOY_STRICT_JVP_MANY", "1")
  x = al.sym("x", 2)
  monkeypatch.setattr(ad, "_jvp_many_structural", lambda *_args: al.const(np.zeros((1, 2))))
  with pytest.raises(NotImplementedError, match=r"structural jvp_many returned shape \(1, 2\).+expected \(2, 2\)"):
    ad.jvp_many(x * x, x, al.const(np.eye(2)))


def test_matmul_vjp_all_shape_cases_match_finite_differences() -> None:
  rng = np.random.default_rng(16)
  for index, (x_shape, y_shape) in enumerate([((4,), (4,)), ((3, 4), (4,)), ((4,), (4, 2)), ((3, 4), (4, 2))]):
    x, y = al.sym("x", x_shape), al.sym("y", y_shape)
    out = x @ y
    cot = al.sym("cot", out.shape)
    objective = al.dot(cot, out)
    gx, gy = al.vjp((objective,), (x, y), (al.const(1.0),))
    grad_fn = al.Function._from_exprs(f"matmul_vjp_{index}", [x, y, cot], [gx, gy], ["x", "y", "cot"], ["gx", "gy"])
    objective_fn = al.Function._from_exprs(f"matmul_vjp_objective_{index}", [x, y, cot], [objective], ["x", "y", "cot"], ["objective"])
    xv, yv, cotv = rng.normal(size=x_shape), rng.normal(size=y_shape), rng.normal(size=out.shape)

    actual_x, actual_y = grad_fn((xv, yv, cotv))
    expected_x = finite_difference(lambda value: objective_fn((value, yv, cotv)), xv).reshape(x_shape)
    expected_y = finite_difference(lambda value: objective_fn((xv, value, cotv)), yv).reshape(y_shape)
    np.testing.assert_allclose(actual_x, expected_x, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(actual_y, expected_y, rtol=1e-6, atol=1e-8)


def test_matrix_vector_vjp_graph_uses_tensor_ops() -> None:
  x, y, cot = al.sym("x", (17, 19)), al.sym("y", 19), al.sym("cot", 17)
  gx, gy = al.vjp((x @ y,), (x, y), (cot,))
  nodes = topo((gx, gy))

  assert sum(node.op == al.ExprOp.MATMUL for node in nodes) == 2
  assert sum(node.op == al.ExprOp.TRANSPOSE for node in nodes) == 1
  assert all(node.op != al.ExprOp.STACK for node in nodes)
  assert len(nodes) <= 10


def test_sparse_jacobian_colored_scalar_plus_vector() -> None:
  z = al.sym("z", 6)
  sj = al.sparse_jacobian_colored(z[5] + z[:5] * z[:5], z)
  f = al.Function._from_exprs("spjac_scalar_plus_vec", [z], [sj.to_dense()], ["z"], ["dense"])
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
  f = al.Function._from_exprs("vjp_many", [x, c0, c1], [grad_x], ["x", "c0", "c1"], ["grad_x"])
  xv = np.array([0.3, 1.2])
  c0v = np.array([[1.5, -0.25], [-0.5, 2.0]])
  c1v = np.array([0.75, -1.25])

  np.testing.assert_allclose(f((xv, c0v, c1v)), c0v * (2 * xv) + c1v[:, None])


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
  inner = al.Function._from_exprs("inner", [x], [x.sin() * x], ["x"], ["y"])
  z = al.sym("z", 2)
  inner_z = inner(z)
  seed = al.sym("seed", 2)
  (grad_z,) = al.vjp((inner_z,), (z,), (seed,))
  outer = al.Function._from_exprs("outer", [z, seed], [grad_z], ["z", "seed"], ["grad_z"])
  zv = np.array([0.2, 0.7])
  sv = np.array([3.0, -1.0])

  np.testing.assert_allclose(outer((zv, sv)), sv * (np.sin(zv) + zv * np.cos(zv)))


def test_vjp_through_vmap_partitioned_stride_with_offset_and_zero_fill() -> None:
  piece = _vmap_vjp_piece("vmap_vjp_partitioned_piece")
  z = al.sym("z", 10)
  _assert_vmap_vjp_matches_unrolled_and_fd("vmap_vjp_partitioned", piece, z, 3, [(z, 2, 2)], np.linspace(-0.7, 0.8, 10))


def test_vjp_through_vmap_broadcast_stride_accumulates() -> None:
  piece = _vmap_vjp_piece("vmap_vjp_broadcast_piece")
  z = al.sym("z", 5)
  _assert_vmap_vjp_matches_unrolled_and_fd("vmap_vjp_broadcast", piece, z, 4, [(z, 1, 0)], np.linspace(-0.4, 0.6, 5))


def test_vjp_through_vmap_cross_formal_overlap_accumulates() -> None:
  piece = _vmap_vjp_piece("vmap_vjp_cross_formal_piece", nargs=2)
  z = al.sym("z", 8)
  _assert_vmap_vjp_matches_unrolled_and_fd("vmap_vjp_cross_formal", piece, z, 3, [(z, 0, 2), (z, 2, 2)], np.linspace(-0.5, 0.9, 8))


def test_vjp_through_vmap_single_formal_overlap_uses_grouped_scatter() -> None:
  x = al.sym("x", 3)
  piece = al.Function._from_exprs("vmap_vjp_grouped_piece", [x], [x * x + x.sin()], ["x"], ["y"])
  z = al.sym("z", 7)
  _assert_vmap_vjp_matches_unrolled_and_fd("vmap_vjp_grouped", piece, z, 3, [(z, 0, 2)], np.linspace(-0.8, 0.7, 7))


def test_vjp_through_vmap_outer_slice_of_wrt() -> None:
  piece = _vmap_vjp_piece("vmap_vjp_outer_slice_piece")
  z = al.sym("z", 10)
  _assert_vmap_vjp_matches_unrolled_and_fd("vmap_vjp_outer_slice", piece, z, 3, [(z[1:9], 1, 2)], np.linspace(-0.6, 0.75, 10))


def test_vjp_through_vmap_duplicate_outer_expr_bound_to_two_formals() -> None:
  piece = _vmap_vjp_piece("vmap_vjp_duplicate_piece", nargs=2)
  z = al.sym("z", 6)
  _assert_vmap_vjp_matches_unrolled_and_fd("vmap_vjp_duplicate", piece, z, 3, [(z, 0, 2), (z, 0, 2)], np.linspace(-0.5, 0.5, 6))


def test_vjp_through_vmap_broadcast_full_outer_skips_scatter() -> None:
  piece = _vmap_vjp_piece("vmap_vjp_broadcast_full_piece")
  z = al.sym("z", 2)
  _assert_vmap_vjp_matches_unrolled_and_fd("vmap_vjp_broadcast_full", piece, z, 3, [(z, 0, 0)], np.array([-0.3, 0.55]))


def test_vjp_through_vmap_adjoint_names_disambiguate_active_formal_sets() -> None:
  # {a_b} and {a, b} would both suffix to "a_b" if adjoints were named by joined formal names;
  # lowering dedupes callees by name, so the two maps would silently share one proc body.
  a, a_b, b = al.sym("a", 2), al.sym("a_b", 2), al.sym("b", 2)
  piece = al.Function._from_exprs("vmap_vjp_collision_piece", [a, a_b, b], [a * a_b.sin() + b * a_b + a * b], ["a", "a_b", "b"], ["y"])
  z = al.sym("z", 8)
  c0, c1 = al.const(np.array([0.3, -0.7, 1.1, 0.2])), al.const(np.array([0.9, 0.4, -0.5, 1.3]))
  m1 = al.vmap(piece, 2, [(c0, 0, 2), (z[0:4], 0, 2), (c1, 0, 2)])
  m2 = al.vmap(piece, 2, [(z[0:4], 0, 2), (c0, 0, 2), (z[4:8], 0, 2)])
  obj = m1.sum() + m2.sum()
  (grad_z,) = al.vjp((obj,), (z,), (al.const(1.0),))
  grad_fn = al.Function._from_exprs("vmap_vjp_collision_grads", [z], [grad_z], ["z"], ["grad_z"])
  obj_fn = al.Function._from_exprs("vmap_vjp_collision_obj", [z], [obj], ["z"], ["objective"])
  zv = np.random.default_rng(3).normal(size=8)

  fd = finite_difference(obj_fn, zv).reshape(-1)
  np.testing.assert_allclose(grad_fn(zv), fd, rtol=1e-6, atol=1e-7)


def test_ad_skips_nonsmooth_parameter_terms_independent_of_wrt() -> None:
  x = al.sym("x", 2)
  p = al.sym("p", 2, diff=False)
  seed = al.sym("seed", 2)
  y = (x * x + p.floor()).sum()
  dy = al.jvp(y, x, seed)
  grad = al.gradient(y, x)
  f = al.Function._from_exprs("smooth_wrt_x", [x, p, seed], [dy, grad], ["x", "p", "seed"], ["dy", "grad"])
  xv = np.array([0.3, 1.2])
  pv = np.array([1.1, 2.9])
  sv = np.array([1.5, -0.25])

  dyv, gradv = f((xv, pv, sv))
  np.testing.assert_allclose(dyv, np.dot(2 * xv, sv))
  np.testing.assert_allclose(gradv, 2 * xv)

  try:
    _ = al.jvp(x.floor().sum(), x, seed)
  except NotImplementedError as e:
    assert "nonsmooth" in str(e)
  else:  # pragma: no cover
    raise AssertionError("nonsmooth JVP through wrt should still fail")
