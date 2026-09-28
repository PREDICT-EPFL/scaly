from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

import scaly as sc
from scaly.ad import finite_difference
from scaly.ad.forward import _jvp_many_structural, _jvp_many_unrolled  # jvp_many's two paths, checked against each other
from scaly.ir.expr import topo


def _vmap_vjp_piece(name: str, nargs: int = 1) -> sc.Function:
  leaves = [sc.L(f"x{i}", 2) for i in range(nargs)]

  @sc.function(leaves[0] if nargs == 1 else sc.G(*leaves), sc.L("y", ...), name=name)
  def piece(inputs):
    first, *rest = (inputs,) if nargs == 1 else inputs
    out = first * first + first.sin()
    for inp in rest:
      out = out + first * inp + inp.sin()
    return out

  return piece


def _assert_vmap_vjp_matches_unrolled_and_fd(
  name: str, callee: sc.Function, length: int, specs: Callable[[sc.Expr], list[tuple[sc.Expr, int, int]]], zv: np.ndarray
) -> None:
  def objectives(z: sc.Expr, lam: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
    bound = specs(z)
    mapped = sc.vmap(callee, length, bound)
    unrolled = sc.concat(
      [
        callee(
          callee.input_tree.unflatten(
            tuple(
              outer[start + it * stride : start + it * stride + formal.size]
              for formal, (outer, start, stride) in zip(callee.inputs, bound, strict=True)
            )
          )
        )
        for it in range(length)
      ]
    )
    return sc.dot(lam, mapped), sc.dot(lam, unrolled)

  lam_size = length * callee.outputs[0].size
  inputs = sc.G(sc.L("z", zv.size), sc.L("lam", lam_size))

  @sc.function(inputs, sc.G(sc.L("mapped", ...), sc.L("unrolled", ...)), name=f"{name}_grads")
  def grad_fn(inputs):
    z, lam = inputs
    mapped_obj, unrolled_obj = objectives(z, lam)
    (mapped_grad,) = sc.vjp((mapped_obj,), (z,), (sc.const(1.0),))
    (unrolled_grad,) = sc.vjp((unrolled_obj,), (z,), (sc.const(1.0),))
    return mapped_grad, unrolled_grad

  @sc.function(inputs, sc.L("objective", ...), name=f"{name}_objective")
  def obj_fn(inputs):
    return objectives(*inputs)[0]

  lamv = np.random.default_rng(10).normal(size=lam_size)

  mapped_value, unrolled_value = grad_fn((zv, lamv))
  fd = finite_difference(lambda value: obj_fn((value, lamv)), zv).reshape(-1)
  np.testing.assert_allclose(mapped_value, unrolled_value, rtol=1e-10, atol=1e-10)
  np.testing.assert_allclose(mapped_value, fd, rtol=1e-6, atol=1e-7)


def test_vjp_scalar_output_matches_gradient() -> None:
  @sc.function(sc.L("x", 3), sc.L("grad_x", ...), name="vjp")
  def f(x):
    y = (x.sin() + x * x).sum()
    (grad_x,) = sc.vjp((y,), (x,), (sc.const(1.0),))
    return grad_x

  xv = np.array([0.1, 0.4, 0.9])

  np.testing.assert_allclose(f(xv), np.cos(xv) + 2 * xv)


def test_vjp_vector_output_uses_cotangent_seed() -> None:
  @sc.function(sc.G(sc.L("x", 2), sc.L("seed", 2)), sc.L("grad_x", ...), name="vjp")
  def f(inputs):
    x, seed = inputs
    y = sc.stack([x[0] * x[1], x[0].sin()])
    (grad_x,) = sc.vjp((y,), (x,), (seed,))
    return grad_x

  xv = np.array([0.3, 2.0])
  sv = np.array([1.5, -0.25])

  np.testing.assert_allclose(f((xv, sv)), np.array([sv[0] * xv[1] + sv[1] * np.cos(xv[0]), sv[0] * xv[0]]))


def test_vjp_broadcast_and_multi_output_accumulates_adjoint() -> None:
  @sc.function(sc.G(sc.L("x", (2, 3)), sc.L("b", 3)), sc.G(sc.L("grad_x", ...), sc.L("grad_b", ...)), name="vjp")
  def f(inputs):
    x, b = inputs
    y0 = (x + b).sum()
    y1 = (x * b).sum()
    (grad_x, grad_b) = sc.vjp((y0, y1), (x, b), (sc.const(2.0), sc.const(-0.5)))
    return grad_x, grad_b

  xv = np.arange(6.0).reshape(2, 3)
  bv = np.array([0.5, 1.5, 2.5])

  gx, gb = f((xv, bv))
  np.testing.assert_allclose(gx, np.broadcast_to(2.0 - 0.5 * bv, xv.shape))
  np.testing.assert_allclose(gb, np.full(3, 4.0) - 0.5 * xv.sum(axis=0))


def test_vjp_through_structural_ops_and_matmul() -> None:
  @sc.function(sc.G(sc.L("x", (2, 2)), sc.L("a", (2, 2)), sc.L("seed", (2, 4))), sc.L("grad_x", ...), name="vjp")
  def f(inputs):
    x, a, seed = inputs
    y = sc.concat([x.T, a @ x], axis=1)
    (grad_x,) = sc.vjp((y,), (x,), (seed,))
    return grad_x

  xv = np.array([[1.0, 2.0], [3.0, 4.0]])
  av = np.array([[2.0, -1.0], [0.5, 3.0]])
  sv = np.arange(8.0).reshape(2, 4)

  expected = sv[:, :2].T + av.T @ sv[:, 2:]
  np.testing.assert_allclose(f((xv, av, sv)), expected)


def test_vjp_nested_calls_shared_symbol_matches_fd_and_jvp() -> None:
  # The CALL VJP inlines the callee adjoint and substitutes formals with actuals. The caller here
  # reuses the callee's formal symbol `x`, so the substitution must not rewrite occurrences of `x`
  # inside the incoming cotangent (which the second call's adjoint injects into the first call's).
  @sc.function(sc.L("x", 1), sc.L("y", ...), name="sq")
  def f(x):
    return x * x

  @sc.function(sc.L("x", 1), sc.G(sc.L("rev", ...), sc.L("fwd", ...)), name="nested_sq")
  def fn(x):
    k1 = f(2 * x)
    k2 = f(x + k1)
    (grad_rev,) = sc.vjp((k2,), (x,), (sc.const(np.ones(1)),))
    return grad_rev, sc.jacobian(k2, x).reshape((1,))

  assert fn.inputs == f.inputs

  rev, fwd = fn(np.array([1.0]))
  np.testing.assert_allclose(rev, [90.0], rtol=1e-12)  # d/dx (x + 4x^2)^2 at x=1
  np.testing.assert_allclose(rev, fwd, rtol=1e-12)


def test_vjp_nested_call_rk4_matches_jvp_transpose_and_fd() -> None:
  # RK4 body calling an inner Function 4x with shared symbols: reverse must equal J^T lam and FD,
  # and the exact Hessian of lam^T rk4(x) must come out symmetric.
  from scaly.ad import hessian  # noqa: PLC0415

  rng = np.random.default_rng(7)

  @sc.function(sc.G(sc.L("x", 2), sc.L("u", 1)), sc.L("f", ...), name="ode2")
  def ode(inputs):
    x, u = inputs
    return sc.stack([x[0] * x[1] + u[0], x[0].tanh() - x[1] * x[1]])

  def rk4(x: sc.Expr, u: sc.Expr) -> sc.Expr:
    dt = 0.1
    k1 = ode((x, u))
    k2 = ode((x + (dt / 2) * k1, u))
    k3 = ode((x + (dt / 2) * k2, u))
    k4 = ode((x + dt * k3, u))
    return x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

  inputs = sc.G(sc.L("x", 2), sc.L("u", 1), sc.L("lam", 2))

  @sc.function(inputs, sc.G(sc.L("rev", ...), sc.L("fwd", ...), sc.L("hess", ...)), name="rk4_adj")
  def fn(inputs):
    x, u, lam = inputs
    xnext = rk4(x, u)
    (grad_rev,) = sc.vjp((xnext,), (x,), (lam,))
    jac_t_lam = sc.jacobian(xnext, x).transpose((1, 0)) @ lam
    return grad_rev, jac_t_lam, hessian(sc.dot(lam, xnext), x)

  @sc.function(inputs, sc.L("obj", ...), name="rk4_obj")
  def obj(inputs):
    x, u, lam = inputs
    return sc.dot(lam, rk4(x, u))

  xv, uv, lamv = rng.normal(size=2), rng.normal(size=1), rng.normal(size=2)

  rev, fwd, hv = fn((xv, uv, lamv))
  fd = finite_difference(lambda v: obj((v, uv, lamv)), xv).reshape(-1)
  np.testing.assert_allclose(rev, fwd, rtol=1e-12, atol=1e-12)
  np.testing.assert_allclose(rev, fd, rtol=1e-6, atol=1e-8)
  np.testing.assert_allclose(hv, hv.T, rtol=1e-12, atol=1e-12)


def test_jvp_many_uses_leading_seed_axis() -> None:
  @sc.function(sc.G(sc.L("x", 3), sc.L("seeds", (2, 3))), sc.L("dy", ...), name="jvp_many")
  def f(inputs):
    x, seeds = inputs
    return sc.jvp_many(sc.stack([x[0] * x[1], x[2].sin() + x[0]]), x, seeds)

  xv = np.array([0.3, 1.2, 0.7])
  sv = np.array([[1.5, -0.25, 0.4], [-0.5, 2.0, 1.25]])
  jac = np.array([[xv[1], xv[0], 0.0], [1.0, 0.0, np.cos(xv[2])]])

  np.testing.assert_allclose(f((xv, sv)), sv @ jac.T)


def test_erf_forward_reverse_jacobian_and_sparse_hessian(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")

  @sc.function(sc.G(sc.L("x", 3), sc.L("seed", 3)), sc.G(sc.L("jvp", ...), sc.L("vjp", ...)), name="erf_seed_ad")
  def ad(inputs):
    x, seed = inputs
    y = x.erf()
    (vjp,) = sc.vjp((y,), (x,), (seed,))
    return sc.jvp(y, x, seed), vjp

  weights = np.array([0.5, -1.25, 2.0])

  @sc.function(sc.L("x", 3), sc.G(sc.L("y", ...), sc.L("cost", ...)), name="erf_derivatives")
  def base(x):
    y = x.erf()
    return y, (sc.const(weights) * y).sum()

  derivatives = base.factory("erf_jac_sphess", ["x"], [sc.factory.Jac("y", "x"), sc.factory.SpHess("cost", "x")])

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
  @sc.function(sc.G(sc.L("x", 3), sc.L("seeds", (2, 3))), sc.L("dy", ...), name="jvp_many_broadcast_scalar")
  def f(inputs):
    x, seeds = inputs
    return sc.jvp_many(x.sum() * x, x, seeds)

  xv = np.array([0.3, 1.2, -0.4])
  sv = np.array([[1.5, -0.25, 0.4], [-0.5, 2.0, 1.25]])

  np.testing.assert_allclose(f((xv, sv)), sv.sum(axis=1, keepdims=True) * xv + xv.sum() * sv)


def test_jvp_many_sum_batches_seeds_without_unrolling() -> None:
  @sc.function(sc.G(sc.L("x", (2, 3)), sc.L("seeds", (4, 2, 3))), sc.G(sc.L("structural", ...), sc.L("reference", ...)), name="jvp_many_sum")
  def fn(inputs):
    x, seeds = inputs
    return _jvp_many_structural(x.sum(), x, seeds, {}, {}), _jvp_many_unrolled(x.sum(), x, seeds)

  x, _ = fn.inputs
  structural, _ = fn.outputs
  expr = x.sum()
  one_seed = _jvp_many_structural(expr, x, sc.sym("one_seed", (1, 2, 3)), {}, {})

  assert structural.shape == (4,)
  nodes = topo((structural,))
  assert len(nodes) == len(topo((one_seed,)))
  assert sum(node.op == sc.ExprOp.MATMUL for node in nodes) == 1
  assert sum(node.op == sc.ExprOp.RESHAPE for node in nodes) == 1
  assert not any(node.op == sc.ExprOp.STACK for node in nodes)

  xv = np.random.default_rng(16).normal(size=(2, 3))
  seedv = np.random.default_rng(17).normal(size=(4, 2, 3))
  actual, expected = fn((xv, seedv))
  np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_float32_reduction_derivative_paths(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  x = sc.sym("float32_x", 3, dtype=sc.dtypes.float32)
  expr = x.sum()
  seeds = sc.sym("float32_seeds", (2, 3), dtype=sc.dtypes.float32)

  jvp_many = sc.jvp_many(expr, x, seeds)
  jacobian = sc.jacobian(expr, x)
  hessian = sc.hessian(expr, x)
  sparse_jacobian = sc.sparse_jacobian(expr, x)
  sparse_hessian = sc.sparse_hessian(expr, x)

  assert jvp_many.shape == (2,)
  assert jvp_many.type.dtype == sc.dtypes.float32
  assert jacobian.shape == (1, 3)
  assert hessian.shape == (3, 3)
  assert sparse_jacobian.sparsity.shape == (1, 3)
  assert sparse_hessian.sparsity.shape == (3, 3)


def test_jvp_many_structural_rank_mismatch_corner_cases() -> None:
  # every rule of the structural batched JVP must return tangents shaped (nseed, *expr.shape), also under rank-mismatched broadcasts

  rng = np.random.default_rng(3)
  xv, sv = rng.normal(size=9), rng.normal(size=(4, 9))
  cases = {
    "mat_times_vec": (lambda x: (x[:6].reshape((2, 3)) * x[6:9]).reshape((6,)), lambda v: (v[:6].reshape(2, 3) * v[6:9]).ravel()),
    "vec_div_scalar_sum": (lambda x: x[:3] / x.sum(), lambda v: v[:3] / v.sum()),
    "scalar_div_vec": (lambda x: x.sum() / x[:3], lambda v: v.sum() / v[:3]),
    "scalar_sum_plus_vec": (lambda x: x.sum() + x[:3] * x[:3], lambda v: v.sum() + v[:3] * v[:3]),
    "vec_minus_scalar_sum": (lambda x: x[:3] - x.sum(), lambda v: v[:3] - v.sum()),
    "scalar_plus_const_vec": (lambda x: x.sum() + sc.const(np.arange(3.0)), lambda v: v.sum() + np.arange(3.0)),
    "vec_pow_scalar_sym": (lambda x: (x[:3] * x[:3] + 1.0) ** x[8], lambda v: (v[:3] * v[:3] + 1.0) ** v[8]),
    "scalar_pow_const_vec": (lambda x: x[8] ** sc.const(np.array([2.0, 3.0, 4.0])), lambda v: v[8] ** np.array([2.0, 3.0, 4.0])),
  }
  for name, (build, np_fn) in cases.items():

    @sc.function(sc.G(sc.L("x", 9), sc.L("seeds", (4, 9))), sc.L("dy", ...), name=f"jvp_many_{name}")
    def f(inputs):
      x, seeds = inputs
      return _jvp_many_structural(build(x), x, seeds, {}, {})

    (dy,) = f.outputs
    assert dy.shape == (4, *build(f.inputs[0]).shape), f"{name}: tangent shape {dy.shape}"
    eps = 1e-6
    fd = np.stack([(np_fn(xv + eps * sv[i]) - np_fn(xv - eps * sv[i])) / (2 * eps) for i in range(4)])
    np.testing.assert_allclose(f((xv, sv)), fd, rtol=1e-6, atol=1e-8, err_msg=name)


def test_jvp_many_vec_dot_vec_keeps_seed_axis() -> None:
  # (dx * y).sum() in the 1-D matmul JVP branch also contracted the seed axis, silently summing per-seed derivatives
  @sc.function(sc.L("x", 6), sc.L("y", ...), name="vec_dot_vec")
  def fn(x):
    return sc.stack([x[:3] @ x[3:6] + x[0] * x[1]])

  jf = fn.factory("vec_dot_vec_jac", ["x"], [sc.factory.Jac("y", "x")])
  xv = np.random.default_rng(11).normal(size=6)
  expected = np.concatenate([xv[3:6] + np.array([xv[1], xv[0], 0.0]), xv[:3]])[None, :]
  np.testing.assert_allclose(jf(xv), expected, rtol=1e-12, atol=1e-12)


def test_jvp_many_scatter_and_gather_stays_structural() -> None:
  @sc.function(sc.G(sc.L("x", 6), sc.L("seeds", (3, 6))), sc.G(sc.L("structural", ...), sc.L("reference", ...)), name="jvp_many_scatter_gather")
  def fn(inputs):
    x, seeds = inputs
    expr = sc.scatter(sc.gather(x, np.array([[5, 1], [4, 0]])), np.array([[0, 4], [7, 8]]), (3, 3))
    return _jvp_many_structural(expr, x, seeds, {}, {}), _jvp_many_unrolled(expr, x, seeds)

  structural, _ = fn.outputs
  assert structural.shape == (3, 3, 3)
  nodes = topo((structural,))
  assert sum(node.op == sc.ExprOp.GATHER for node in nodes) == 1
  assert sum(node.op == sc.ExprOp.SCATTER for node in nodes) == 1
  xv = np.random.default_rng(12).normal(size=6)
  seedv = np.random.default_rng(13).normal(size=(3, 6))
  actual, expected = fn((xv, seedv))
  np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_jvp_many_transpose_stays_structural() -> None:
  @sc.function(sc.G(sc.L("x", 12), sc.L("seeds", (4, 12))), sc.G(sc.L("structural", ...), sc.L("reference", ...)), name="jvp_many_transpose")
  def fn(inputs):
    x, seeds = inputs
    expr = x.reshape((2, 3, 2)).transpose((2, 0, 1))
    return _jvp_many_structural(expr, x, seeds, {}, {}), _jvp_many_unrolled(expr, x, seeds)

  structural, _ = fn.outputs
  assert structural.shape == (4, 2, 2, 3)
  assert sum(node.op == sc.ExprOp.TRANSPOSE for node in topo((structural,))) == 1
  xv = np.random.default_rng(14).normal(size=12)
  seedv = np.random.default_rng(15).normal(size=(4, 12))
  actual, expected = fn((xv, seedv))
  np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_jvp_many_rank4_transpose_falls_back_and_strict_raises(monkeypatch: pytest.MonkeyPatch) -> None:
  # A rank-4 primal TRANSPOSE would need a rank-5 tangent TRANSPOSE, beyond the rank-4 lowering
  # limit; the structural rule must decline (unrolled fallback lowers fine) instead of building an
  # un-lowerable graph that only fails later at compile time.
  def rank4(x: sc.Expr) -> sc.Expr:
    return (x * x).reshape((2, 3, 1, 2)).transpose((3, 1, 0, 2))

  @sc.function(sc.G(sc.L("x", 12), sc.L("seeds", (2, 12))), sc.L("tan", ...), name="rank4_transpose_jvp")
  def fn(inputs):
    x, seeds = inputs
    return sc.jvp_many(rank4(x), x, seeds).reshape((24,))

  x, seeds = fn.inputs
  rng = np.random.default_rng(7)
  xv, sv = rng.normal(size=12), rng.normal(size=(2, 12))
  expected = np.concatenate([(2.0 * xv * sv[i]).reshape(2, 3, 1, 2).transpose(3, 1, 0, 2).reshape(-1) for i in range(2)])
  np.testing.assert_allclose(fn((xv, sv)), expected, rtol=1e-12, atol=1e-12)
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  with pytest.raises(NotImplementedError, match="structural jvp_many does not support"):
    sc.jvp_many(rank4(x), x, seeds)


def test_jvp_many_strict_mode_raises_on_unsupported_structural_rule(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  x = sc.sym("x", 2)
  with pytest.raises(NotImplementedError, match="structural jvp_many does not support"):
    sc.jvp_many(x.abs(), x, sc.const(np.eye(2)))


def test_jvp_many_strict_mode_raises_on_structural_shape_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
  import scaly.ad.forward as ad

  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  x = sc.sym("x", 2)
  monkeypatch.setattr(ad, "_jvp_many_structural", lambda *_args: sc.const(np.zeros((1, 2))))
  with pytest.raises(NotImplementedError, match=r"structural jvp_many returned shape \(1, 2\).+expected \(2, 2\)"):
    ad.jvp_many(x * x, x, sc.const(np.eye(2)))


def test_matmul_vjp_all_shape_cases_match_finite_differences() -> None:
  rng = np.random.default_rng(16)
  for index, (x_shape, y_shape) in enumerate([((4,), (4,)), ((3, 4), (4,)), ((4,), (4, 2)), ((3, 4), (4, 2))]):
    out_shape = np.broadcast_shapes(x_shape[:-1] + y_shape[1:])
    inputs = sc.G(sc.L("x", x_shape), sc.L("y", y_shape), sc.L("cot", out_shape))

    @sc.function(inputs, sc.L("objective", ...), name=f"matmul_vjp_objective_{index}")
    def objective_fn(inputs):
      x, y, cot = inputs
      return sc.dot(cot, x @ y)

    @sc.function(inputs, sc.G(sc.L("gx", ...), sc.L("gy", ...)), name=f"matmul_vjp_{index}")
    def grad_fn(inputs):
      x, y, _ = inputs
      return sc.vjp((objective_fn(inputs),), (x, y), (sc.const(1.0),))

    xv, yv, cotv = rng.normal(size=x_shape), rng.normal(size=y_shape), rng.normal(size=out_shape)

    actual_x, actual_y = grad_fn((xv, yv, cotv))
    expected_x = finite_difference(lambda value: objective_fn((value, yv, cotv)), xv).reshape(x_shape)
    expected_y = finite_difference(lambda value: objective_fn((xv, value, cotv)), yv).reshape(y_shape)
    np.testing.assert_allclose(actual_x, expected_x, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(actual_y, expected_y, rtol=1e-6, atol=1e-8)


def test_matrix_vector_vjp_graph_uses_tensor_ops() -> None:
  x, y, cot = sc.sym("x", (17, 19)), sc.sym("y", 19), sc.sym("cot", 17)
  gx, gy = sc.vjp((x @ y,), (x, y), (cot,))
  nodes = topo((gx, gy))

  assert sum(node.op == sc.ExprOp.MATMUL for node in nodes) == 2
  assert sum(node.op == sc.ExprOp.TRANSPOSE for node in nodes) == 1
  assert all(node.op != sc.ExprOp.STACK for node in nodes)
  assert len(nodes) <= 10


def test_sparse_jacobian_colored_scalar_plus_vector() -> None:
  @sc.function(sc.L("z", 6), sc.L("dense", ...), name="spjac_scalar_plus_vec")
  def f(z):
    return sc.sparse_jacobian_colored(z[5] + z[:5] * z[:5], z).to_dense()

  zv = np.random.default_rng(5).normal(size=6)
  dense = np.zeros((5, 6))
  dense[:, 5] = 1.0
  dense[np.arange(5), np.arange(5)] += 2 * zv[:5]
  np.testing.assert_allclose(f(zv), dense)


def test_vjp_many_uses_leading_seed_axis_and_multiple_outputs() -> None:
  @sc.function(sc.G(sc.L("x", 2), sc.L("c0", (2, 2)), sc.L("c1", 2)), sc.L("grad_x", ...), name="vjp_many")
  def f(inputs):
    x, c0, c1 = inputs
    (grad_x,) = sc.vjp_many((x * x, x.sum()), (x,), (c0, c1))
    return grad_x

  xv = np.array([0.3, 1.2])
  c0v = np.array([[1.5, -0.25], [-0.5, 2.0]])
  c1v = np.array([0.75, -1.25])

  np.testing.assert_allclose(f((xv, c0v, c1v)), c0v * (2 * xv) + c1v[:, None])


def test_multi_seed_shape_errors() -> None:
  x = sc.sym("x", 2)
  y = x * x

  try:
    _ = sc.jvp_many(y, x, sc.sym("bad", 2))
  except ValueError as e:
    assert "multi-seed JVP expects seeds shape" in str(e)
  else:  # pragma: no cover
    raise AssertionError("bad multi-seed JVP shape should fail")

  try:
    _ = sc.vjp_many((y,), (x,), (sc.sym("bad", 2),))
  except ValueError as e:
    assert "multi-seed VJP expects cotangent shape" in str(e)
  else:  # pragma: no cover
    raise AssertionError("bad multi-seed VJP shape should fail")


def test_vjp_through_call_node_inlines_callee_reverse_graph() -> None:
  @sc.function(sc.L("x", 2), sc.L("y", ...))
  def inner(x):
    return x.sin() * x

  @sc.function(sc.G(sc.L("z", 2), sc.L("seed", 2)), sc.L("grad_z", ...))
  def outer(inputs):
    z, seed = inputs
    (grad_z,) = sc.vjp((inner(z),), (z,), (seed,))
    return grad_z

  zv = np.array([0.2, 0.7])
  sv = np.array([3.0, -1.0])

  np.testing.assert_allclose(outer((zv, sv)), sv * (np.sin(zv) + zv * np.cos(zv)))


def test_vjp_through_vmap_partitioned_stride_with_offset_and_zero_fill() -> None:
  piece = _vmap_vjp_piece("vmap_vjp_partitioned_piece")
  _assert_vmap_vjp_matches_unrolled_and_fd("vmap_vjp_partitioned", piece, 3, lambda z: [(z, 2, 2)], np.linspace(-0.7, 0.8, 10))


def test_vjp_through_vmap_broadcast_stride_accumulates() -> None:
  piece = _vmap_vjp_piece("vmap_vjp_broadcast_piece")
  _assert_vmap_vjp_matches_unrolled_and_fd("vmap_vjp_broadcast", piece, 4, lambda z: [(z, 1, 0)], np.linspace(-0.4, 0.6, 5))


def test_vjp_through_vmap_cross_formal_overlap_accumulates() -> None:
  piece = _vmap_vjp_piece("vmap_vjp_cross_formal_piece", nargs=2)
  _assert_vmap_vjp_matches_unrolled_and_fd("vmap_vjp_cross_formal", piece, 3, lambda z: [(z, 0, 2), (z, 2, 2)], np.linspace(-0.5, 0.9, 8))


def test_vjp_through_vmap_single_formal_overlap_uses_grouped_scatter() -> None:
  @sc.function(sc.L("x", 3), sc.L("y", ...), name="vmap_vjp_grouped_piece")
  def piece(x):
    return x * x + x.sin()

  _assert_vmap_vjp_matches_unrolled_and_fd("vmap_vjp_grouped", piece, 3, lambda z: [(z, 0, 2)], np.linspace(-0.8, 0.7, 7))


def test_vjp_through_vmap_outer_slice_of_wrt() -> None:
  piece = _vmap_vjp_piece("vmap_vjp_outer_slice_piece")
  _assert_vmap_vjp_matches_unrolled_and_fd("vmap_vjp_outer_slice", piece, 3, lambda z: [(z[1:9], 1, 2)], np.linspace(-0.6, 0.75, 10))


def test_vjp_through_vmap_duplicate_outer_expr_bound_to_two_formals() -> None:
  piece = _vmap_vjp_piece("vmap_vjp_duplicate_piece", nargs=2)
  _assert_vmap_vjp_matches_unrolled_and_fd("vmap_vjp_duplicate", piece, 3, lambda z: [(z, 0, 2), (z, 0, 2)], np.linspace(-0.5, 0.5, 6))


def test_vjp_through_vmap_broadcast_full_outer_skips_scatter() -> None:
  piece = _vmap_vjp_piece("vmap_vjp_broadcast_full_piece")
  _assert_vmap_vjp_matches_unrolled_and_fd("vmap_vjp_broadcast_full", piece, 3, lambda z: [(z, 0, 0)], np.array([-0.3, 0.55]))


def test_vjp_through_vmap_adjoint_names_disambiguate_active_formal_sets() -> None:
  # {a_b} and {a, b} would both suffix to "a_b" if adjoints were named by joined formal names;
  # lowering dedupes callees by name, so the two maps would silently share one proc body.
  @sc.function(sc.G(sc.L("a", 2), sc.L("a_b", 2), sc.L("b", 2)), sc.L("y", ...), name="vmap_vjp_collision_piece")
  def piece(inputs):
    a, a_b, b = inputs
    return a * a_b.sin() + b * a_b + a * b

  c0, c1 = sc.const(np.array([0.3, -0.7, 1.1, 0.2])), sc.const(np.array([0.9, 0.4, -0.5, 1.3]))

  @sc.function(sc.L("z", 8), sc.L("objective", ...), name="vmap_vjp_collision_obj")
  def obj_fn(z):
    m1 = sc.vmap(piece, 2, [(c0, 0, 2), (z[0:4], 0, 2), (c1, 0, 2)])
    m2 = sc.vmap(piece, 2, [(z[0:4], 0, 2), (c0, 0, 2), (z[4:8], 0, 2)])
    return m1.sum() + m2.sum()

  @sc.function(sc.L("z", 8), sc.L("grad_z", ...), name="vmap_vjp_collision_grads")
  def grad_fn(z):
    (grad_z,) = sc.vjp((obj_fn(z),), (z,), (sc.const(1.0),))
    return grad_z

  zv = np.random.default_rng(3).normal(size=8)

  fd = finite_difference(obj_fn, zv).reshape(-1)
  np.testing.assert_allclose(grad_fn(zv), fd, rtol=1e-6, atol=1e-7)


def test_ad_skips_nonsmooth_parameter_terms_independent_of_wrt() -> None:
  @sc.function(
    sc.G(sc.L("x", 2), sc.L("p", sc.TensorType((2,), diff=False)), sc.L("seed", 2)), sc.G(sc.L("dy", ...), sc.L("grad", ...)), name="smooth_wrt_x"
  )
  def f(inputs):
    x, p, seed = inputs
    y = (x * x + p.floor()).sum()
    return sc.jvp(y, x, seed), sc.gradient(y, x)

  x, _, seed = f.inputs
  xv = np.array([0.3, 1.2])
  pv = np.array([1.1, 2.9])
  sv = np.array([1.5, -0.25])

  dyv, gradv = f((xv, pv, sv))
  np.testing.assert_allclose(dyv, np.dot(2 * xv, sv))
  np.testing.assert_allclose(gradv, 2 * xv)

  try:
    _ = sc.jvp(x.floor().sum(), x, seed)
  except NotImplementedError as e:
    assert "nonsmooth" in str(e)
  else:  # pragma: no cover
    raise AssertionError("nonsmooth JVP through wrt should still fail")
