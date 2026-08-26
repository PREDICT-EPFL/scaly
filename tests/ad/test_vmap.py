from __future__ import annotations

import numpy as np

import alloy as al


@al.function("scale_add", {"x": 3, "p": 3})
def scale_add(x, p):
  return {"y": 2.0 * x + p}


def test_jvp_many_of_vmap_matches_unrolled_jvp() -> None:
  N = 4
  z = al.sym("z", 3 * N)
  p = al.sym("p", 3 * N)

  mapped = al.vmap(scale_add, N, [(z, 0, 3), (p, 0, 3)])
  unrolled = al.concat([scale_add.call([z[i * 3 : (i + 1) * 3], p[i * 3 : (i + 1) * 3]])[0] for i in range(N)])

  seeds = al.const(np.eye(3 * N))
  jvp_vmap = al.jvp_many(mapped, z, seeds)
  jvp_ref = al.jvp_many(unrolled, z, seeds)

  fn_vmap = al.Function("jvp_vmap", [z, p], [jvp_vmap], ["z", "p"], ["dy"])
  fn_ref = al.Function("jvp_ref", [z, p], [jvp_ref], ["z", "p"], ["dy"])

  rng = np.random.default_rng(0)
  zv = rng.normal(size=3 * N)
  pv = rng.normal(size=3 * N)

  np.testing.assert_allclose(fn_vmap(zv, pv), fn_ref(zv, pv), rtol=1e-10, atol=1e-10)


def test_jacobian_of_vmap_matches_finite_differences() -> None:
  N = 5
  z = al.sym("z", 3 * N)
  p = al.sym("p", 3 * N)
  mapped = al.vmap(scale_add, N, [(z, 0, 3), (p, 0, 3)])
  fn = al.Function("mapped", [z, p], [mapped], ["z", "p"], ["y"])

  seeds = al.const(np.eye(3 * N))
  jacobian = al.jvp_many(mapped, z, seeds).T  # (3N, 3N)
  jac_fn = al.Function("jac", [z, p], [jacobian], ["z", "p"], ["jac"])

  rng = np.random.default_rng(1)
  zv = rng.normal(size=3 * N)
  pv = rng.normal(size=3 * N)
  jac = jac_fn(zv, pv)

  eps = 1e-7
  y0 = fn(zv, pv)
  assert isinstance(y0, np.ndarray)
  fd = np.zeros((3 * N, 3 * N))
  for j in range(3 * N):
    zp = zv.copy()
    zp[j] += eps
    yj = fn(zp, pv)
    assert isinstance(yj, np.ndarray)
    fd[:, j] = (yj - y0) / eps
  np.testing.assert_allclose(jac, fd, atol=1e-5)


def test_grad_factory_over_vmap_matches_unrolled_and_finite_difference() -> None:
  from alloy.ad import finite_difference
  from alloy.ir.expr import topo

  @al.function("vmap_grad_piece", {"x": 2})
  def piece(x):
    return {"y": x * x + x.sin()}

  N = 4
  z = al.sym("z", 2 * N)
  mapped = al.vmap(piece, N, [(z, 0, 2)])
  unrolled = al.concat([piece.call([z[2 * it : 2 * (it + 1)]])[0] for it in range(N)])
  mapped_fn = al.Function("vmap_grad_factory", [z], [mapped], ["z"], ["y"])
  unrolled_fn = al.Function("vmap_grad_unrolled", [z], [unrolled], ["z"], ["y"])
  mapped_grad = mapped_fn.factory("vmap_grad_factory_grad", ["z", "lam:y"], [al.factory.Grad("gamma", "z")], aux={"gamma": ["y"]})
  unrolled_grad = unrolled_fn.factory("vmap_grad_unrolled_grad", ["z", "lam:y"], [al.factory.Grad("gamma", "z")], aux={"gamma": ["y"]})
  vmap_nodes = [node for node in topo(mapped_grad.outputs) if node.op == al.ExprOp.VMAP]
  assert len(vmap_nodes) == 1
  assert "_adj0_0" in vmap_nodes[0].attrs["callee"].name

  zv = np.random.default_rng(7).normal(size=2 * N)
  lamv = np.random.default_rng(8).normal(size=2 * N)
  np.testing.assert_allclose(mapped_grad(zv, lamv), unrolled_grad(zv, lamv), rtol=1e-10, atol=1e-10)
  np.testing.assert_allclose(
    mapped_grad(zv, lamv), finite_difference(lambda value: np.dot(lamv, np.asarray(mapped_fn(value))), zv).reshape(-1), rtol=1e-6, atol=1e-7
  )
