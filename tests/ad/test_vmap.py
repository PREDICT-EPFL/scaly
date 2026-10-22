from __future__ import annotations

import numpy as np
import pytest

from scaly.function.model import as_concrete
from scaly.function.sugar import _mapped_call
import scaly as sc


@sc.function(sc.group(sc.arg("x", 3), sc.arg("p", 3)), outputs=sc.arg("y"), name="scale_add")
def scale_add(inputs: tuple[sc.Expr, sc.Expr]) -> sc.Expr:
  x, p = inputs
  return 2.0 * x + p


def test_jvp_many_of_vmap_matches_unrolled_jvp() -> None:
  N = 4

  @sc.function(sc.group(sc.arg("z", 3 * N), sc.arg("p", 3 * N)), outputs=sc.arg("dy"), name="jvp_vmap")
  def fn_vmap(inputs):
    z, p = inputs
    return sc.jvp_many(_mapped_call(scale_add, N, [(z, 0, 3), (p, 0, 3)]), z, sc.const(np.eye(3 * N)))

  @sc.function(sc.group(sc.arg("z", 3 * N), sc.arg("p", 3 * N)), outputs=sc.arg("dy"), name="jvp_ref")
  def fn_ref(inputs):
    z, p = inputs
    unrolled = sc.concat([scale_add((z[i * 3 : (i + 1) * 3], p[i * 3 : (i + 1) * 3])) for i in range(N)])
    return sc.jvp_many(unrolled, z, sc.const(np.eye(3 * N)))

  rng = np.random.default_rng(0)
  zv = rng.normal(size=3 * N)
  pv = rng.normal(size=3 * N)

  np.testing.assert_allclose(fn_vmap((zv, pv)), fn_ref((zv, pv)), rtol=1e-10, atol=1e-10)


def test_jacobian_of_vmap_matches_finite_differences() -> None:
  N = 5

  @sc.function(sc.group(sc.arg("z", 3 * N), sc.arg("p", 3 * N)), outputs=sc.arg("y"), name="mapped")
  def fn(inputs):
    z, p = inputs
    return _mapped_call(scale_add, N, [(z, 0, 3), (p, 0, 3)])

  @sc.function(sc.group(sc.arg("z", 3 * N), sc.arg("p", 3 * N)), outputs=sc.arg("jac"))
  def jac_fn(inputs):
    z, p = inputs
    return sc.jvp_many(_mapped_call(scale_add, N, [(z, 0, 3), (p, 0, 3)]), z, sc.const(np.eye(3 * N))).T

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
  from scaly.ad import finite_difference
  from scaly.ir.expr import topo

  @sc.function(sc.arg("x", 2), outputs=sc.arg("y"), name="vmap_grad_piece")
  def piece(x: sc.Expr) -> sc.Expr:
    return x * x + x.sin()

  N = 4

  @sc.function(sc.arg("z", 2 * N), outputs=sc.arg("y"), name="vmap_grad_factory")
  def mapped_fn(z):
    return _mapped_call(piece, N, [(z, 0, 2)])

  @sc.function(sc.arg("z", 2 * N), outputs=sc.arg("y"), name="vmap_grad_unrolled")
  def unrolled_fn(z):
    return sc.concat([piece(z[2 * it : 2 * (it + 1)]) for it in range(N)])

  mapped_grad = mapped_fn.factory("vmap_grad_factory_grad", ["z", "lam:y"], [sc.factory.Grad("gamma", "z")], aux={"gamma": ["y"]})
  unrolled_grad = unrolled_fn.factory("vmap_grad_unrolled_grad", ["z", "lam:y"], [sc.factory.Grad("gamma", "z")], aux={"gamma": ["y"]})
  vmap_nodes = [node for node in topo(as_concrete(mapped_grad).outputs) if node.op == sc.ExprOp.VMAP]
  assert len(vmap_nodes) == 1
  assert "_adj_" in vmap_nodes[0].attrs["callee"].name

  zv = np.random.default_rng(7).normal(size=2 * N)
  lamv = np.random.default_rng(8).normal(size=2 * N)
  np.testing.assert_allclose(mapped_grad(*(zv, lamv)), unrolled_grad(*(zv, lamv)), rtol=1e-10, atol=1e-10)
  np.testing.assert_allclose(
    mapped_grad(*(zv, lamv)), finite_difference(lambda value: np.dot(lamv, np.asarray(mapped_fn(value))), zv).reshape(-1), rtol=1e-6, atol=1e-7
  )


@sc.function(sc.group(sc.arg("x", 3), sc.arg("q", 2)), outputs=sc.arg("y"), name="vmap_duality_piece")
def duality_piece(inputs: tuple[sc.Expr, sc.Expr]) -> sc.Expr:
  x, q = inputs
  return sc.stack([x[0] * x[1] * q[0].sin() + x[2].exp(), (1.0 + sc.dot(x, x)).sqrt() * q[1] + x[0] * x[2]])


def test_forward_and_adjoint_of_vmap_match_unrolled_and_are_dual() -> None:
  N = 5

  @sc.function(sc.group(sc.arg("z", 3 * N), sc.arg("q", 2 * N)), outputs=sc.arg("y"), name="duality_vmap")
  def fn_vmap(inputs):
    z, q = inputs
    return _mapped_call(duality_piece, N, [(z, 0, 3), (q, 0, 2)])

  @sc.function(sc.group(sc.arg("z", 3 * N), sc.arg("q", 2 * N)), outputs=sc.arg("y"), name="duality_unroll")
  def fn_unroll(inputs):
    z, q = inputs
    return sc.concat([duality_piece((z[3 * k : 3 * (k + 1)], q[2 * k : 2 * (k + 1)])) for k in range(N)])

  rng = np.random.default_rng(5)
  zv, qv = rng.normal(size=3 * N), rng.normal(size=2 * N)
  v, w = rng.normal(size=3 * N), rng.normal(size=2 * N)
  # Inputs are unit normal and the callee holds exp and products of three, so |y|, |J| stay around 1.
  fwd = {name: np.asarray(sc.forward(fn, "y", "z")(*((zv, qv), v))) for name, fn in (("vmap", fn_vmap), ("unroll", fn_unroll))}
  adj = {name: np.asarray(sc.adjoint(fn, "y", "z")(*((zv, qv), w))) for name, fn in (("vmap", fn_vmap), ("unroll", fn_unroll))}
  jac = np.asarray(sc.jacobian(fn_unroll, "y", "z")((zv, qv)))
  np.testing.assert_allclose(fwd["vmap"], fwd["unroll"], rtol=1e-12, atol=1e-12)
  np.testing.assert_allclose(adj["vmap"], adj["unroll"], rtol=1e-12, atol=1e-12)
  np.testing.assert_allclose(fwd["vmap"], jac @ v, rtol=1e-10, atol=1e-10)
  np.testing.assert_allclose(adj["vmap"], jac.T @ w, rtol=1e-10, atol=1e-10)
  # Forward/reverse duality: <J v, w> == <v, J^T w>.
  np.testing.assert_allclose(w @ fwd["vmap"], v @ adj["vmap"], rtol=1e-12)


@pytest.mark.parametrize("length", [0, 1, 4])
@pytest.mark.parametrize("stride", [0, 1, 2])
@pytest.mark.parametrize(("seed_kind", "nseed"), [("runtime", 4), ("runtime", 1), ("periodic", 3)], ids=["local", "generic", "periodic"])
def test_mapped_windows_and_broadcast_formals_against_numpy(length, stride, seed_kind, nseed):

  @sc.function(sc.group(sc.arg("window", 3), sc.arg("bias", 2)), outputs=sc.arg("out"), name="window_stage")
  def stage(inputs):
    x, p = inputs
    return sc.stack([x[0] * x[1] + p[0].sin(), x[1] * x[2] + p[1] * x[0]])

  size = 4 + max(0, length - 1) * stride

  rng = np.random.default_rng(12)
  z, seed_values, cot = rng.normal(size=size), rng.normal(size=(nseed, size)), rng.normal(size=2 * length)
  if seed_kind == "periodic":
    seed_values = np.tile([1.0, -0.4], (nseed, (size + 1) // 2))[:, :size] * np.arange(1, nseed + 1)[:, None]
  if nseed > 1:
    seed_values[1] = 0

  @sc.function(
    sc.group(sc.arg("windows_z", size), sc.arg("windows_seeds", (nseed, size)), sc.arg("windows_cot", 2 * length)),
    outputs=sc.group(sc.arg("value"), sc.arg("many"), sc.arg("single"), sc.arg("adj")),
    name="window_products",
  )
  def products(inputs):
    z, seeds, cot = inputs
    seeds = sc.const(seed_values) if seed_kind == "periodic" else seeds
    mapped = _mapped_call(stage, length, [(z, 1, stride), (z, 0, 0)])
    return mapped, sc.jvp_many(mapped, z, seeds), sc.stack([sc.jvp(mapped, z, seeds[i]) for i in range(nseed)]), sc.vjp((mapped,), (z,), (cot,))[0]

  def reference(z):
    blocks = []
    for it in range(length):
      x = z[1 + it * stride : 4 + it * stride]
      p = z[:2]
      blocks.extend([x[0] * x[1] + np.sin(p[0]), x[1] * x[2] + p[1] * x[0]])
    return np.asarray(blocks)

  value, many, single, adj = products((z, seed_values, cot))
  expected = np.stack([(reference(z + 1e-5 * seed) - reference(z - 1e-5 * seed)) / 2e-5 for seed in seed_values])
  np.testing.assert_allclose(value, reference(z), atol=1e-12, rtol=1e-12)
  np.testing.assert_allclose(many, expected, atol=1e-9, rtol=1e-8)
  np.testing.assert_allclose(single, expected, atol=1e-9, rtol=1e-8)
  np.testing.assert_allclose(many @ cot, seed_values @ adj, atol=1e-11, rtol=1e-11)
