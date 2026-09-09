from __future__ import annotations

import numpy as np

import alloy as al


@al.function(al.G(al.L("x", 3), al.L("p", 3)), al.L("y", ...), name="scale_add")
def scale_add(inputs):
  x, p = inputs
  return 2.0 * x + p


def test_jvp_many_of_vmap_matches_unrolled_jvp() -> None:
  N = 4
  z = al.sym("z", 3 * N)
  p = al.sym("p", 3 * N)

  mapped = al.vmap(scale_add, N, [(z, 0, 3), (p, 0, 3)])
  unrolled = al.concat([scale_add((z[i * 3 : (i + 1) * 3], p[i * 3 : (i + 1) * 3])) for i in range(N)])

  seeds = al.const(np.eye(3 * N))
  jvp_vmap = al.jvp_many(mapped, z, seeds)
  jvp_ref = al.jvp_many(unrolled, z, seeds)

  fn_vmap = al.Function._from_exprs("jvp_vmap", [z, p], [jvp_vmap], ["z", "p"], ["dy"])
  fn_ref = al.Function._from_exprs("jvp_ref", [z, p], [jvp_ref], ["z", "p"], ["dy"])

  rng = np.random.default_rng(0)
  zv = rng.normal(size=3 * N)
  pv = rng.normal(size=3 * N)

  np.testing.assert_allclose(fn_vmap((zv, pv)), fn_ref((zv, pv)), rtol=1e-10, atol=1e-10)


def test_jacobian_of_vmap_matches_finite_differences() -> None:
  N = 5
  z = al.sym("z", 3 * N)
  p = al.sym("p", 3 * N)
  mapped = al.vmap(scale_add, N, [(z, 0, 3), (p, 0, 3)])
  fn = al.Function._from_exprs("mapped", [z, p], [mapped], ["z", "p"], ["y"])

  seeds = al.const(np.eye(3 * N))
  jacobian = al.jvp_many(mapped, z, seeds).T  # (3N, 3N)
  jac_fn = al.Function._from_exprs("jac", [z, p], [jacobian], ["z", "p"], ["jac"])

  rng = np.random.default_rng(1)
  zv = rng.normal(size=3 * N)
  pv = rng.normal(size=3 * N)
  jac = jac_fn((zv, pv))

  eps = 1e-7
  y0 = fn((zv, pv))
  assert isinstance(y0, np.ndarray)
  fd = np.zeros((3 * N, 3 * N))
  for j in range(3 * N):
    zp = zv.copy()
    zp[j] += eps
    yj = fn((zp, pv))
    assert isinstance(yj, np.ndarray)
    fd[:, j] = (yj - y0) / eps
  np.testing.assert_allclose(jac, fd, atol=1e-5)


def test_grad_factory_over_vmap_matches_unrolled_and_finite_difference() -> None:
  from alloy.ad import finite_difference
  from alloy.ir.expr import topo

  @al.function(al.L("x", 2), al.L("y", ...), name="vmap_grad_piece")
  def piece(x):
    return x * x + x.sin()

  N = 4
  z = al.sym("z", 2 * N)
  mapped = al.vmap(piece, N, [(z, 0, 2)])
  unrolled = al.concat([piece(z[2 * it : 2 * (it + 1)]) for it in range(N)])
  mapped_fn = al.Function._from_exprs("vmap_grad_factory", [z], [mapped], ["z"], ["y"])
  unrolled_fn = al.Function._from_exprs("vmap_grad_unrolled", [z], [unrolled], ["z"], ["y"])
  mapped_grad = mapped_fn.factory("vmap_grad_factory_grad", ["z", "lam:y"], [al.factory.Grad("gamma", "z")], aux={"gamma": ["y"]})
  unrolled_grad = unrolled_fn.factory("vmap_grad_unrolled_grad", ["z", "lam:y"], [al.factory.Grad("gamma", "z")], aux={"gamma": ["y"]})
  vmap_nodes = [node for node in topo(mapped_grad.outputs) if node.op == al.ExprOp.VMAP]
  assert len(vmap_nodes) == 1
  assert "_adj0_0" in vmap_nodes[0].attrs["callee"].name

  zv = np.random.default_rng(7).normal(size=2 * N)
  lamv = np.random.default_rng(8).normal(size=2 * N)
  np.testing.assert_allclose(mapped_grad((zv, lamv)), unrolled_grad((zv, lamv)), rtol=1e-10, atol=1e-10)
  np.testing.assert_allclose(
    mapped_grad((zv, lamv)), finite_difference(lambda value: np.dot(lamv, np.asarray(mapped_fn(value))), zv).reshape(-1), rtol=1e-6, atol=1e-7
  )


@al.function(al.G(al.L("x", 3), al.L("q", 2)), al.L("y", ...), name="vmap_duality_piece")
def duality_piece(inputs):
  x, q = inputs
  return al.stack([x[0] * x[1] * q[0].sin() + x[2].exp(), (1.0 + al.dot(x, x)).sqrt() * q[1] + x[0] * x[2]])


def test_forward_and_adjoint_of_vmap_match_unrolled_and_are_dual() -> None:
  N = 5
  z, q = al.sym("z", 3 * N), al.sym("q", 2 * N)
  mapped = al.vmap(duality_piece, N, [(z, 0, 3), (q, 0, 2)])
  unrolled = al.concat([duality_piece((z[3 * k : 3 * (k + 1)], q[2 * k : 2 * (k + 1)])) for k in range(N)])
  fn_vmap = al.Function._from_exprs("duality_vmap", [z, q], [mapped], ["z", "q"], ["y"])
  fn_unroll = al.Function._from_exprs("duality_unroll", [z, q], [unrolled], ["z", "q"], ["y"])

  rng = np.random.default_rng(5)
  zv, qv = rng.normal(size=3 * N), rng.normal(size=2 * N)
  v, w = rng.normal(size=3 * N), rng.normal(size=2 * N)
  # Inputs are unit normal and the callee holds exp and products of three, so |y|, |J| stay around 10.
  fwd = {name: np.asarray(al.forward(fn, "y", "z")(((zv, qv), v))) for name, fn in (("vmap", fn_vmap), ("unroll", fn_unroll))}
  adj = {name: np.asarray(al.adjoint(fn, "y", "z")(((zv, qv), w))) for name, fn in (("vmap", fn_vmap), ("unroll", fn_unroll))}
  jac = np.asarray(al.jacobian(fn_unroll, "y", "z")((zv, qv)))
  np.testing.assert_allclose(fwd["vmap"], fwd["unroll"], rtol=1e-12, atol=1e-12)
  np.testing.assert_allclose(adj["vmap"], adj["unroll"], rtol=1e-12, atol=1e-12)
  np.testing.assert_allclose(fwd["vmap"], jac @ v, rtol=1e-10, atol=1e-10)
  np.testing.assert_allclose(adj["vmap"], jac.T @ w, rtol=1e-10, atol=1e-10)
  # Forward/reverse duality: <J v, w> == <v, J^T w>.
  np.testing.assert_allclose(w @ fwd["vmap"], v @ adj["vmap"], rtol=1e-12)
