from __future__ import annotations

import numpy as np

from scaly.function.model import as_concrete
import scaly as sc
from typing import Any, cast
from scaly.ad import finite_difference


@sc.function(sc.group(sc.arg("x", 3), sc.arg("p", 3)), outputs=sc.arg("y"), name="scale_add")
def scale_add(inputs: tuple[sc.Expr, sc.Expr]) -> sc.Expr:
  x, p = inputs
  return 2.0 * x + p


def test_race_car_eq_primal_source_is_constant_in_horizon() -> None:
  """Rewriting the race-car fixture with `sc.vmap` yields C source whose size does not grow with N."""

  import re

  from scaly.codegen import render_c_source

  NX = 4
  NU = 2
  NZ = NX + NU
  WHEELBASE = 0.3
  DT = 0.05
  M = 3.47
  C_M0 = 11.0
  C_R0 = 0.1
  C_R1 = 0.01
  C_R2 = 0.001

  def cont(x: sc.Expr, u: sc.Expr) -> sc.Expr:
    phi, v = x[2], x[3]
    throttle, delta = u[0], u[1]
    beta = 0.5 * delta
    vx = v * beta.cos()
    lr = 0.5 * WHEELBASE
    return sc.stack(
      [
        v * (phi + beta).cos(),
        v * (phi + beta).sin(),
        v * beta.sin() / lr,
        (C_M0 * throttle - (C_R0 + C_R1 * vx + C_R2 * vx * vx) * (10 * vx).tanh()) / M,
      ]
    )

  def rk4(x: sc.Expr, u: sc.Expr) -> sc.Expr:
    k1 = cont(x, u)
    k2 = cont(x + DT / 2 * k1, u)
    k3 = cont(x + DT / 2 * k2, u)
    k4 = cont(x + DT * k3, u)
    return x + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

  @sc.function(sc.group(sc.arg("z", NZ), sc.arg("p", NX)), outputs=sc.arg("eq"), name="race_car_eq_initial")
  def eq_initial(inputs: tuple[sc.Expr, sc.Expr]) -> sc.Expr:
    z, p = inputs
    return z[:NX] - p[:NX]

  @sc.function(sc.group(sc.arg("z", NZ), sc.arg("znext", NZ), sc.arg("p", NX)), outputs=sc.arg("eq"), name="race_car_eq_interstage")
  def eq_interstage(inputs: tuple[sc.Expr, sc.Expr, sc.Expr]) -> sc.Expr:
    z, znext, p = inputs
    return rk4(z[:NX], z[NX : NX + NU]) - znext[:NX]

  def build(N: int) -> sc.Function:
    @sc.function(
      sc.group(sc.arg("z", NZ * (N + 1)), sc.arg("p", sc.TensorType((NX * (N + 1),), diff=False))),
      outputs=sc.arg("eq"),
      name=f"race_car_eq_vmap_N{N}",
    )
    def fn(inputs: tuple[sc.Expr, sc.Expr]) -> sc.Expr:
      z, p = inputs
      initial = eq_initial((z[:NZ], p[:NX]))
      mapped = sc.vmap(eq_interstage, N)((sc.window(z, 0, NZ), sc.window(z, NZ, NZ), sc.window(p, NX, NX))).vec()
      return sc.concat([initial, mapped])

    return fn

  fn_a = build(50)
  fn_b = build(100)

  rng = np.random.default_rng(0)
  for fn in (fn_a, fn_b):
    N = (as_concrete(fn).inputs[0].size // NZ) - 1
    zv = rng.normal(size=NZ * (N + 1))
    pv = rng.normal(size=NX * (N + 1))

    # Sanity: the primal numerically matches the unrolled concat-of-call equivalent.
    @sc.function(cast(Any, as_concrete(fn).input_tree).parts[0], outputs=sc.arg("eq"), name=f"race_car_eq_ref_N{N}")
    def ref(inputs: tuple[sc.Expr, sc.Expr]) -> sc.Expr:
      z, p = inputs
      parts = [eq_initial((z[:NZ], p[:NX]))]
      for i in range(N):
        parts.append(eq_interstage((z[i * NZ : (i + 1) * NZ], z[(i + 1) * NZ : (i + 2) * NZ], p[(i + 1) * NX : (i + 2) * NX])))
      return sc.concat(parts)

    np.testing.assert_allclose(fn((zv, pv)), ref((zv, pv)), rtol=1e-12, atol=1e-12)

  # Ignore changing numeric literals and generated identifiers, but require the same C structure.
  for lanes in (1, "auto"):
    src_a = render_c_source(fn_a, lanes=lanes)
    src_b = render_c_source(fn_b, lanes=lanes)
    assert re.sub(r"\d+", "N", src_a) == re.sub(r"\d+", "N", src_b)


def test_sparse_jacobian_of_race_car_vmap_matches_unrolled_concat() -> None:
  """End-to-end: sparse_jacobian on a VMAP-based race-car fixture matches the unrolled-concat fixture
  numerically and structurally."""

  NX, NU, NZ = 4, 2, 6
  WHEELBASE, DT, M = 0.3, 0.05, 3.47
  C_M0, C_R0, C_R1, C_R2 = 11.0, 0.1, 0.01, 0.001

  def cont(x: sc.Expr, u: sc.Expr) -> sc.Expr:
    phi, v = x[2], x[3]
    throttle, delta = u[0], u[1]
    beta = 0.5 * delta
    vx = v * beta.cos()
    lr = 0.5 * WHEELBASE
    return sc.stack(
      [
        v * (phi + beta).cos(),
        v * (phi + beta).sin(),
        v * beta.sin() / lr,
        (C_M0 * throttle - (C_R0 + C_R1 * vx + C_R2 * vx * vx) * (10 * vx).tanh()) / M,
      ]
    )

  def rk4(x: sc.Expr, u: sc.Expr) -> sc.Expr:
    k1 = cont(x, u)
    k2 = cont(x + DT / 2 * k1, u)
    k3 = cont(x + DT / 2 * k2, u)
    k4 = cont(x + DT * k3, u)
    return x + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

  @sc.function(sc.group(sc.arg("z", NZ), sc.arg("p", NX)), outputs=sc.arg("eq"), name="race_car_eq_initial2")
  def eq_initial(inputs: tuple[sc.Expr, sc.Expr]) -> sc.Expr:
    z, p = inputs
    return z[:NX] - p[:NX]

  @sc.function(sc.group(sc.arg("z", NZ), sc.arg("znext", NZ), sc.arg("p", NX)), outputs=sc.arg("eq"), name="race_car_eq_interstage2")
  def eq_interstage(inputs: tuple[sc.Expr, sc.Expr, sc.Expr]) -> sc.Expr:
    z, znext, p = inputs
    return rk4(z[:NX], z[NX : NX + NU]) - znext[:NX]

  def inputs_tree(N: int):
    return sc.group(sc.arg("z", NZ * (N + 1)), sc.arg("p", sc.TensorType((NX * (N + 1),), diff=False)))

  def build_vmap(N: int) -> sc.Function:
    @sc.function(inputs_tree(N), outputs=sc.arg("eq"), name=f"tr_vmap_N{N}")
    def fn(inputs: tuple[sc.Expr, sc.Expr]) -> sc.Expr:
      z, p = inputs
      initial = eq_initial((z[:NZ], p[:NX]))
      mapped = sc.vmap(eq_interstage, N)((sc.window(z, 0, NZ), sc.window(z, NZ, NZ), sc.window(p, NX, NX))).vec()
      return sc.concat([initial, mapped])

    return fn

  def build_unroll(N: int) -> sc.Function:
    @sc.function(inputs_tree(N), outputs=sc.arg("eq"), name=f"tr_unroll_N{N}")
    def fn(inputs: tuple[sc.Expr, sc.Expr]) -> sc.Expr:
      z, p = inputs
      parts = [eq_initial((z[:NZ], p[:NX]))]
      for i in range(N):
        parts.append(eq_interstage((z[i * NZ : (i + 1) * NZ], z[(i + 1) * NZ : (i + 2) * NZ], p[(i + 1) * NX : (i + 2) * NX])))
      return sc.concat(parts)

    return fn

  N = 4
  fn_vmap = build_vmap(N)
  fn_unroll = build_unroll(N)

  spj_vmap = sc.sparse_jacobian(fn_vmap, "eq", "z")
  spj_unroll = sc.sparse_jacobian(fn_unroll, "eq", "z")

  # The nnz orderings differ (the VMAP path emits per-formal disjoint partitions), but the
  # patterns must agree as coordinate sets and the densified values must match exactly.
  sp_m, sp_u = as_concrete(spj_vmap).output_sparsities[0], as_concrete(spj_unroll).output_sparsities[0]
  assert sp_m is not None and sp_u is not None
  assert sp_m.shape == sp_u.shape and set(zip(sp_m.rows, sp_m.cols)) == set(zip(sp_u.rows, sp_u.cols))

  rng = np.random.default_rng(2)
  zv = rng.normal(size=NZ * (N + 1))
  pv = rng.normal(size=NX * (N + 1))

  dense_m, dense_u = np.zeros(sp_m.shape), np.zeros(sp_u.shape)
  dense_m[np.asarray(sp_m.rows), np.asarray(sp_m.cols)] = np.asarray(spj_vmap((zv, pv)), dtype=np.float64).reshape(-1)
  dense_u[np.asarray(sp_u.rows), np.asarray(sp_u.cols)] = np.asarray(spj_unroll((zv, pv)), dtype=np.float64).reshape(-1)
  np.testing.assert_allclose(dense_m, dense_u, rtol=1e-10, atol=1e-10)


# --- pairwise-barrier composition: gather-fed VMAP, VMAP -> gather -> VMAP, concat of two VMAPs ------
# Shape of a centralized one-step barrier filter: one VMAP advances every body, index tables gather
# the (i, j) pair operands out of that VMAP's output, a second VMAP evaluates the pairwise rows, and a
# third VMAP evaluates the per-body rows. `slack` rides along as a stride-0 broadcast argument.

NB, NS, NU = 3, 3, 2
PAIRS = [(i, j) for i in range(NB) for j in range(i + 1, NB)]


@sc.function(sc.group(sc.arg("s", NS), sc.arg("u", NU)), outputs=sc.arg("next"), name="pairs_step")
def pairs_step(inputs: tuple[sc.Expr, sc.Expr]) -> sc.Expr:
  s, u = inputs
  return sc.stack([s[0] + 0.1 * s[2].cos() * u[0], s[1] + 0.1 * s[2].sin() * u[1], s[2] + 0.1 * (u[0] - u[1])])


@sc.function(
  sc.group(sc.arg("prev_i", NS), sc.arg("prev_j", NS), sc.arg("si", NS), sc.arg("sj", NS), sc.arg("slack", 1)),
  outputs=sc.arg("h"),
  name="pairs_barrier",
)
def pairs_barrier(inputs: tuple[sc.Expr, sc.Expr, sc.Expr, sc.Expr, sc.Expr]) -> sc.Expr:
  # The trailing p-norm term mirrors the smooth-max a velocity-margin barrier uses; it is what
  # brings integer POW, a *non-integer* POW (the shape of such a barrier's braking envelope,
  # `c * d^q`) and a nested sqrt into the second-order path through the VMAP. As in the real
  # barrier, the non-integer power's base is bounded away from zero, where its slope diverges.
  prev_i, prev_j, si, sj, slack = inputs
  d, dprev = si[:2] - sj[:2], prev_i[:2] - prev_j[:2]
  envelope = 1.1 * (1.0 + sc.dot(dprev, dprev)) ** 0.84
  soft_max = (sc.dot(d, d) ** 2 + envelope**4).sqrt().sqrt()
  return sc.stack([(sc.dot(d, d).sqrt() - 0.5 * (1.0 + sc.dot(dprev, dprev)).log() + soft_max + slack[0])])


@sc.function(sc.group(sc.arg("s", NS), sc.arg("snext", NS), sc.arg("slack", 1)), outputs=sc.arg("h"), name="pairs_wall")
def pairs_wall(inputs: tuple[sc.Expr, sc.Expr, sc.Expr]) -> sc.Expr:
  s, snext, slack = inputs
  return sc.stack([(snext[0] - 0.5 * s[0] + slack[0]), (1.0 - snext[1].exp() + slack[0])])


def _pair_index_table(bodies: list[int]) -> np.ndarray:
  return np.concatenate([np.arange(NS, dtype=np.int64) + k * NS for k in bodies])


def _build_pairs_fn(mapped: bool) -> sc.Function:
  @sc.function(
    sc.group(sc.arg("u", NU * NB + 1), sc.arg("p", sc.TensorType((NS * NB,), diff=False))),
    outputs=sc.arg("h"),
    name=f"pairs_{'vmap' if mapped else 'unroll'}",
  )
  def fn(inputs: tuple[sc.Expr, sc.Expr]) -> sc.Expr:
    u, p = inputs
    uu, slack = u[: NU * NB], u[NU * NB : NU * NB + 1]
    if mapped:
      nxt = sc.vmap(pairs_step, NB)((sc.window(p, 0, NS), sc.window(uu, 0, NU))).vec()
      idx_i, idx_j = _pair_index_table([i for i, _ in PAIRS]), _pair_index_table([j for _, j in PAIRS])
      pair_rows = sc.vmap(pairs_barrier, len(PAIRS))(
        (
          sc.window(sc.gather(p, idx_i), 0, NS),
          sc.window(sc.gather(p, idx_j), 0, NS),
          sc.window(sc.gather(nxt, idx_i), 0, NS),
          sc.window(sc.gather(nxt, idx_j), 0, NS),
          sc.window(slack, 0, 0),
        )
      ).vec()
      body_rows = sc.vmap(pairs_wall, NB)((sc.window(p, 0, NS), sc.window(nxt, 0, NS), sc.window(slack, 0, 0))).vec()
      return sc.concat([pair_rows, body_rows])
    nxt = sc.concat([pairs_step((p[NS * k : NS * (k + 1)], uu[NU * k : NU * (k + 1)])) for k in range(NB)])
    sl = lambda e, k: e[NS * k : NS * (k + 1)]  # noqa: E731
    rows = [pairs_barrier((sl(p, i), sl(p, j), sl(nxt, i), sl(nxt, j), slack)) for i, j in PAIRS]
    rows += [pairs_wall((sl(p, k), sl(nxt, k), slack)) for k in range(NB)]
    return sc.concat(rows)

  return fn


def _pairs_sample() -> tuple[np.ndarray, np.ndarray]:
  rng = np.random.default_rng(11)
  uv = rng.normal(scale=0.3, size=NU * NB + 1)
  uv[-1] = 0.05
  return uv, rng.normal(scale=0.5, size=NS * NB) + np.tile([1.0, 2.0, 0.3], NB)


def test_gather_fed_chained_vmaps_match_unrolled_calls() -> None:
  fn_vmap, fn_unroll = _build_pairs_fn(True), _build_pairs_fn(False)
  uv, pv = _pairs_sample()
  np.testing.assert_allclose(fn_vmap((uv, pv)), fn_unroll((uv, pv)), rtol=1e-12, atol=1e-12)

  jac_vmap = fn_vmap.factory("pairs_vmap_jac", ["u", "p"], [sc.factory.Jac("h", "u")])(*(uv, pv))
  jac_unroll = fn_unroll.factory("pairs_unroll_jac", ["u", "p"], [sc.factory.Jac("h", "u")])(*(uv, pv))
  np.testing.assert_allclose(jac_vmap, jac_unroll, rtol=1e-10, atol=1e-10)


def test_gather_fed_chained_vmaps_spjac_and_sphess_match_dense() -> None:
  fn = _build_pairs_fn(True)
  uv, pv = _pairs_sample()
  dense = fn.factory("pairs_vmap_jac2", ["u", "p"], [sc.factory.Jac("h", "u")])(*(uv, pv))
  assert isinstance(dense, np.ndarray)
  spjf = fn.factory("pairs_vmap_spjac", ["u", "p"], [sc.factory.SpJac("h", "u")])
  sp = as_concrete(spjf).output_sparsities[0]
  assert sp is not None
  flat = np.asarray(sp.rows) * dense.shape[1] + np.asarray(sp.cols)
  np.testing.assert_allclose(spjf(*(uv, pv)), np.ravel(dense)[flat], rtol=1e-10, atol=1e-10)
  # Coloring must not claim structural zeros that the dense Jacobian disagrees with.
  assert not np.any(np.abs(dense[~sp.to_mask()]) > 1e-12)

  # Second order through the same composition, against the unrolled reference.
  lam = np.arange(1.0, as_concrete(fn).outputs[0].shape[0] + 1.0)
  hess = {
    name: fn_.factory(f"pairs_{name}_sphess", ["u", "p", "lam:h"], [sc.factory.SpHess("gamma", "u")], aux={"gamma": ["h"]})
    for name, fn_ in (("vmap", fn), ("unroll", _build_pairs_fn(False)))
  }
  dense_hess = {}
  for name, hf in hess.items():
    hsp = as_concrete(hf).output_sparsities[0]
    assert hsp is not None
    dense_hess[name] = np.zeros(hsp.shape)
    dense_hess[name][np.asarray(hsp.rows), np.asarray(hsp.cols)] = np.asarray(hf(uv, pv, lam), dtype=np.float64).reshape(-1)
  np.testing.assert_allclose(dense_hess["vmap"], dense_hess["unroll"], rtol=1e-9, atol=1e-9)

  # The mapped Hessian must also be the derivative of the mapped Lagrangian gradient, so a wrong
  # entry shared by both forms cannot hide behind their agreement. Entries are below 0.2 and the
  # barrier is smooth at this sample, so central differences at 1e-6 land within 1e-9 of them.
  grad_l = sc.adjoint(fn, "h", "u")
  fd_hess = finite_difference(lambda value: np.asarray(grad_l(*((value, pv), lam))), uv)
  np.testing.assert_allclose(dense_hess["vmap"], fd_hess, rtol=1e-7, atol=1e-7)


def test_matmul_inside_vmap_callee_differentiates() -> None:
  """Differentiate a dense layer inside a mapped callee against a NumPy reference."""
  w = np.array([[0.4, -0.2, 0.7], [0.1, 0.9, -0.3]])
  b = np.array([0.05, -0.15])

  @sc.function(sc.arg("s", 3), outputs=sc.arg("y"), name="vmap_dense_layer")
  def layer(s: sc.Expr) -> sc.Expr:
    phi = sc.stack([s[0], s[1], s[2]])
    h = sc.const(w) @ phi + sc.const(b)
    return sc.stack([(h * h).sum()])

  N = 4

  @sc.function(sc.arg("z", 3 * N), outputs=sc.arg("y"), name="vmap_dense")
  def fn(z: sc.Expr) -> sc.Expr:
    return sc.vmap(layer, N)(sc.window(z, 0, 3)).vec()

  zv = np.random.default_rng(3).normal(size=3 * N)
  expected = np.zeros((N, 3 * N))
  for k in range(N):
    h = w @ zv[3 * k : 3 * k + 3] + b
    expected[k, 3 * k : 3 * k + 3] = 2.0 * h @ w
  np.testing.assert_allclose(fn.factory("vmap_dense_jac", ["z"], [sc.factory.Jac("y", "z")])(zv), expected, rtol=1e-10, atol=1e-10)


def test_weighted_mapped_residual_cost_matches_unrolled_derivatives() -> None:
  @sc.function(sc.group(sc.arg("x", 2), sc.arg("ref", 2), sc.arg("scale", 1)), outputs=sc.arg("r"))
  def residual(inputs: tuple[sc.Expr, sc.Expr, sc.Expr]) -> sc.Expr:
    x, ref, scale = inputs
    return sc.stack([x[0] - ref[0], ref[1].cos() * x[1] - scale[0].tanh()])

  n = 4
  weights = sc.const(np.array([0.0, 0.3, 1.2, 0.3, 1.2, 0.3, 5.0, 0.3]))
  rng = np.random.default_rng(21)
  inputs = rng.normal(size=2 * n), rng.normal(size=2 * n + 1)
  values = []
  for name in ("mapped", "unrolled"):

    @sc.function(sc.group(sc.arg("x", 2 * n), sc.arg("p", sc.TensorType((2 * n + 1,), diff=False))), outputs=sc.arg("f"), name=name)
    def fn(inputs: tuple[sc.Expr, sc.Expr]) -> sc.Expr:
      x, p = inputs
      if name == "mapped":
        r = sc.vmap(residual, n)((sc.window(x, 0, 2), sc.window(p, 0, 2), sc.window(p, 2 * n, 0))).vec()
      else:
        r = sc.concat([residual((x[2 * i : 2 * i + 2], p[2 * i : 2 * i + 2], p[-1:])) for i in range(n)])
      return sc.dot(weights, r**2)

    derivatives = fn.factory(name + "_derivatives", ["x", "p"], ["f", sc.factory.Grad("f", "x"), sc.factory.SpHess("f", "x")])
    values.append(derivatives(*inputs))
  for actual, expected in zip(values[0], values[1], strict=True):
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
