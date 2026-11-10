from __future__ import annotations

from collections.abc import Callable

import numpy as np

from scaly.function.model import as_concrete
from scaly.function.sugar import _mapped_call
import scaly as sc
from scaly.ad import finite_difference
from scaly.ir.expr import topo


def _vmap_vjp_piece(name: str, nargs: int = 1) -> sc.Function:
  leaves = [sc.arg(f"x{i}", 2) for i in range(nargs)]

  @sc.function(leaves[0] if nargs == 1 else sc.group(*leaves), outputs=sc.arg("y"), name=name)
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
    mapped = _mapped_call(callee, length, bound)
    unrolled = sc.concat(
      [
        callee(
          *as_concrete(callee).input_tree.unflatten(
            tuple(
              outer[start + it * stride : start + it * stride + formal.size]
              for formal, (outer, start, stride) in zip(as_concrete(callee).inputs, bound, strict=True)
            )
          )
        )
        for it in range(length)
      ]
    )
    return sc.dot(lam, mapped), sc.dot(lam, unrolled)

  lam_size = length * as_concrete(callee).outputs[0].size
  inputs = sc.group(sc.arg("z", zv.size), sc.arg("lam", lam_size))

  @sc.function(inputs, outputs=sc.group(sc.arg("mapped", zv.shape), sc.arg("unrolled", zv.shape)), name=f"{name}_grads")
  def grad_fn(inputs):
    z, lam = inputs
    mapped_obj, unrolled_obj = objectives(z, lam)
    (mapped_grad,) = sc.vjp((mapped_obj,), (z,), (sc.const(1.0),))
    (unrolled_grad,) = sc.vjp((unrolled_obj,), (z,), (sc.const(1.0),))
    return mapped_grad, unrolled_grad

  @sc.function(inputs, outputs=sc.arg("objective"), name=f"{name}_objective")
  def obj_fn(inputs):
    return objectives(*inputs)[0]

  lamv = np.random.default_rng(10).normal(size=lam_size)

  mapped_value, unrolled_value = grad_fn((zv, lamv))
  fd = finite_difference(lambda value: obj_fn((value, lamv)), zv).reshape(-1)
  np.testing.assert_allclose(mapped_value, unrolled_value, rtol=1e-10, atol=1e-10)
  np.testing.assert_allclose(mapped_value, fd, rtol=1e-6, atol=1e-7)


def test_vjp_broadcast_and_multi_output_accumulates_adjoint() -> None:
  @sc.function(sc.group(sc.arg("x", (2, 3)), sc.arg("b", 3)), outputs=sc.group(sc.arg("grad_x", (2, 3)), sc.arg("grad_b", 3)), name="vjp")
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


def test_vjp_nested_calls_shared_symbol_matches_fd_and_jvp() -> None:
  # The CALL VJP inlines the callee adjoint and substitutes formals with actuals. The caller here
  # reuses the callee's formal symbol `x`, so the substitution must not rewrite occurrences of `x`
  # inside the incoming cotangent (which the second call's adjoint injects into the first call's).
  @sc.function(sc.arg("x", 1), outputs=sc.arg("y"), name="sq")
  def f(x):
    return x * x

  @sc.function(sc.arg("x", 1), outputs=sc.group(sc.arg("rev", 1), sc.arg("fwd", 1)), name="nested_sq")
  def fn(x):
    k1 = f(2 * x)
    k2 = f(x + k1)
    (grad_rev,) = sc.vjp((k2,), (x,), (sc.const(np.ones(1)),))
    return grad_rev, sc.jacobian(k2, x).reshape((1,))

  assert as_concrete(fn).inputs == as_concrete(f).inputs

  rev, fwd = fn(np.array([1.0]))
  np.testing.assert_allclose(rev, [90.0], rtol=1e-12)  # d/dx (x + 4x^2)^2 at x=1
  np.testing.assert_allclose(rev, fwd, rtol=1e-12)


def test_vjp_nested_call_rk4_matches_jvp_transpose_and_fd() -> None:
  # RK4 body calling an inner Function 4x with shared symbols: reverse must equal J^T lam and FD,
  # and the exact Hessian of lam^T rk4(x) must come out symmetric.
  from scaly.ad import hessian  # noqa: PLC0415

  rng = np.random.default_rng(7)

  @sc.function(sc.group(sc.arg("x", 2), sc.arg("u", 1)), outputs=sc.arg("f"), name="ode2")
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

  inputs = sc.group(sc.arg("x", 2), sc.arg("u", 1), sc.arg("lam", 2))

  @sc.function(inputs, outputs=sc.group(sc.arg("rev", 2), sc.arg("fwd", 2), sc.arg("hess", (2, 2))), name="rk4_adj")
  def fn(inputs):
    x, u, lam = inputs
    xnext = rk4(x, u)
    (grad_rev,) = sc.vjp((xnext,), (x,), (lam,))
    jac_t_lam = sc.jacobian(xnext, x).transpose((1, 0)) @ lam
    return grad_rev, jac_t_lam, hessian(sc.dot(lam, xnext), x)

  @sc.function(inputs, outputs=sc.arg("obj"), name="rk4_obj")
  def obj(inputs):
    x, u, lam = inputs
    return sc.dot(lam, rk4(x, u))

  xv, uv, lamv = rng.normal(size=2), rng.normal(size=1), rng.normal(size=2)

  rev, fwd, hv = fn((xv, uv, lamv))
  fd = finite_difference(lambda v: obj((v, uv, lamv)), xv).reshape(-1)
  np.testing.assert_allclose(rev, fwd, rtol=1e-12, atol=1e-12)
  np.testing.assert_allclose(rev, fd, rtol=1e-6, atol=1e-8)
  np.testing.assert_allclose(hv, hv.T, rtol=1e-12, atol=1e-12)


def test_erf_forward_reverse_jacobian_and_sparse_hessian() -> None:

  @sc.function(sc.group(sc.arg("x", 3), sc.arg("seed", 3)), outputs=sc.group(sc.arg("jvp", 3), sc.arg("vjp", 3)), name="erf_seed_ad")
  def ad(inputs):
    x, seed = inputs
    y = x.erf()
    (vjp,) = sc.vjp((y,), (x,), (seed,))
    return sc.jvp(y, x, seed), vjp

  weights = np.array([0.5, -1.25, 2.0])

  @sc.function(sc.arg("x", 3), outputs=sc.group(sc.arg("y"), sc.arg("cost")), name="erf_derivatives")
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
  sparsity = as_concrete(derivatives).output_sparsities[1]
  assert sparsity is not None
  assert sparsity.rows == (0, 1, 2)
  assert sparsity.cols == (0, 1, 2)
  second = weights * (-4 * xv / np.sqrt(np.pi)) * np.exp(-(xv**2))
  np.testing.assert_allclose(sphess_value, second, rtol=1e-12, atol=1e-12)


def test_jvp_many_sum_batches_seeds_without_unrolling() -> None:
  @sc.function(
    sc.group(sc.arg("x", (2, 3)), sc.arg("seeds", (4, 2, 3))),
    outputs=sc.arg("tangent", 4),
    name="jvp_many_sum",
  )
  def fn(inputs):
    x, seeds = inputs
    return sc.jvp_many(x.sum(), x, seeds)

  x, _ = as_concrete(fn).inputs
  (structural,) = as_concrete(fn).outputs
  expr = x.sum()
  one_seed = sc.jvp_many(expr, x, sc.sym("one_seed", (1, 2, 3)))

  assert structural.shape == (4,)
  nodes = topo((structural,))
  assert len(nodes) == len(topo((one_seed,)))
  assert sum(node.op == sc.ExprOp.MATMUL for node in nodes) == 1
  assert sum(node.op == sc.ExprOp.RESHAPE for node in nodes) == 1
  assert not any(node.op == sc.ExprOp.STACK for node in nodes)

  xv = np.random.default_rng(16).normal(size=(2, 3))
  seedv = np.random.default_rng(17).normal(size=(4, 2, 3))
  np.testing.assert_allclose(fn((xv, seedv)), seedv.sum(axis=(1, 2)), rtol=1e-12, atol=1e-12)


def test_float32_reduction_derivative_paths() -> None:
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


def test_jvp_many_scatter_and_gather_stays_structural() -> None:
  @sc.function(
    sc.group(sc.arg("x", 6), sc.arg("seeds", (3, 6))),
    outputs=sc.arg("tangent", (3, 3, 3)),
    name="jvp_many_scatter_gather",
  )
  def fn(inputs):
    x, seeds = inputs
    expr = sc.scatter(sc.gather(x, np.array([[5, 1], [4, 0]])), np.array([[0, 4], [7, 8]]), (3, 3))
    return sc.jvp_many(expr, x, seeds)

  (structural,) = as_concrete(fn).outputs
  assert structural.shape == (3, 3, 3)
  nodes = topo((structural,))
  assert sum(node.op == sc.ExprOp.GATHER for node in nodes) == 1
  assert sum(node.op == sc.ExprOp.SCATTER for node in nodes) == 1
  xv = np.random.default_rng(12).normal(size=6)
  seedv = np.random.default_rng(13).normal(size=(3, 6))
  expected = np.zeros((3, 9))
  expected[:, [0, 4, 7, 8]] = seedv[:, [5, 1, 4, 0]]
  np.testing.assert_allclose(fn((xv, seedv)), expected.reshape(3, 3, 3), rtol=1e-12, atol=1e-12)


def test_jvp_many_transpose_stays_structural() -> None:
  @sc.function(
    sc.group(sc.arg("x", 12), sc.arg("seeds", (4, 12))),
    outputs=sc.arg("tangent", (4, 2, 2, 3)),
    name="jvp_many_transpose",
  )
  def fn(inputs):
    x, seeds = inputs
    expr = x.reshape((2, 3, 2)).transpose((2, 0, 1))
    return sc.jvp_many(expr, x, seeds)

  (structural,) = as_concrete(fn).outputs
  assert structural.shape == (4, 2, 2, 3)
  assert sum(node.op == sc.ExprOp.TRANSPOSE for node in topo((structural,))) == 1
  xv = np.random.default_rng(14).normal(size=12)
  seedv = np.random.default_rng(15).normal(size=(4, 12))
  expected = seedv.reshape(4, 2, 3, 2).transpose(0, 3, 1, 2)
  np.testing.assert_allclose(fn((xv, seedv)), expected, rtol=1e-12, atol=1e-12)


def test_matrix_vector_vjp_graph_uses_tensor_ops() -> None:
  x, y, cot = sc.sym("x", (17, 19)), sc.sym("y", 19), sc.sym("cot", 17)
  gx, gy = sc.vjp((x @ y,), (x, y), (cot,))
  nodes = topo((gx, gy))

  assert sum(node.op == sc.ExprOp.MATMUL for node in nodes) == 2
  assert sum(node.op == sc.ExprOp.TRANSPOSE for node in nodes) == 1
  assert all(node.op != sc.ExprOp.STACK for node in nodes)
  assert len(nodes) <= 10


def test_sparse_jacobian_colored_scalar_plus_vector() -> None:
  @sc.function(sc.arg("z", 6), outputs=sc.arg("dense", (5, 6)), name="spjac_scalar_plus_vec")
  def f(z):
    return sc.sparse_jacobian_colored(z[5] + z[:5] * z[:5], z).to_dense()

  zv = np.random.default_rng(5).normal(size=6)
  dense = np.zeros((5, 6))
  dense[:, 5] = 1.0
  dense[np.arange(5), np.arange(5)] += 2 * zv[:5]
  np.testing.assert_allclose(f(zv), dense)


def test_multi_seed_shape_errors() -> None:
  x = sc.sym("x", 2)
  y = x * x

  try:
    _ = sc.jvp_many(y, x, sc.sym("bad", 2))
  except ValueError as e:
    assert "multi-seed JVP expects seeds shape" in str(e)
  else:  # pragma: no cover
    raise AssertionError("bad multi-seed JVP shape should fail")


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
  @sc.function(sc.arg("x", 3), outputs=sc.arg("y"), name="vmap_vjp_grouped_piece")
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
  @sc.function(sc.group(sc.arg("a", 2), sc.arg("a_b", 2), sc.arg("b", 2)), outputs=sc.arg("y"), name="vmap_vjp_collision_piece")
  def piece(inputs):
    a, a_b, b = inputs
    return a * a_b.sin() + b * a_b + a * b

  c0, c1 = sc.const(np.array([0.3, -0.7, 1.1, 0.2])), sc.const(np.array([0.9, 0.4, -0.5, 1.3]))

  @sc.function(sc.arg("z", 8), outputs=sc.arg("objective"), name="vmap_vjp_collision_obj")
  def obj_fn(z):
    m1 = _mapped_call(piece, 2, [(c0, 0, 2), (z[0:4], 0, 2), (c1, 0, 2)])
    m2 = _mapped_call(piece, 2, [(z[0:4], 0, 2), (c0, 0, 2), (z[4:8], 0, 2)])
    return m1.sum() + m2.sum()

  @sc.function(sc.arg("z", 8), outputs=sc.arg("grad_z", 8), name="vmap_vjp_collision_grads")
  def grad_fn(z):
    (grad_z,) = sc.vjp((obj_fn(z),), (z,), (sc.const(1.0),))
    return grad_z

  zv = np.random.default_rng(3).normal(size=8)

  fd = finite_difference(obj_fn, zv).reshape(-1)
  np.testing.assert_allclose(grad_fn(zv), fd, rtol=1e-6, atol=1e-7)


def test_ad_skips_nonsmooth_parameter_terms_independent_of_wrt() -> None:
  @sc.function(
    sc.group(sc.arg("x", 2), sc.arg("p", sc.TensorType((2,), diff=False)), sc.arg("seed", 2)),
    outputs=sc.group(sc.arg("dy", ()), sc.arg("grad", 2)),
    name="smooth_wrt_x",
  )
  def f(inputs):
    x, p, seed = inputs
    y = (x * x + p.floor()).sum()
    return sc.jvp(y, x, seed), sc.gradient(y, x)

  x, _, seed = as_concrete(f).inputs
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
