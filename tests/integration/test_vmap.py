from __future__ import annotations

import numpy as np

import alloy as al


@al.function(al.G(al.L("x", 3), al.L("p", 3)), al.L("y", ...), name="scale_add")
def scale_add(inputs):
  x, p = inputs
  return 2.0 * x + p


def test_race_car_eq_primal_source_is_constant_in_horizon() -> None:
  """Rewriting the race-car fixture with `al.vmap` yields C source whose size does not grow with N."""

  from alloy.codegen import render_c_source

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

  def cont(x, u):
    phi, v = x[2], x[3]
    throttle, delta = u[0], u[1]
    beta = 0.5 * delta
    vx = v * beta.cos()
    lr = 0.5 * WHEELBASE
    return al.stack(
      [
        v * (phi + beta).cos(),
        v * (phi + beta).sin(),
        v * beta.sin() / lr,
        (C_M0 * throttle - (C_R0 + C_R1 * vx + C_R2 * vx * vx) * (10 * vx).tanh()) / M,
      ]
    )

  def rk4(x, u):
    k1 = cont(x, u)
    k2 = cont(x + DT / 2 * k1, u)
    k3 = cont(x + DT / 2 * k2, u)
    k4 = cont(x + DT * k3, u)
    return x + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

  @al.function(al.G(al.L("z", NZ), al.L("p", NX)), al.L("eq", ...), name="race_car_eq_initial")
  def eq_initial(inputs):
    z, p = inputs
    return z[:NX] - p[:NX]

  @al.function(al.G(al.L("z", NZ), al.L("znext", NZ), al.L("p", NX)), al.L("eq", ...), name="race_car_eq_interstage")
  def eq_interstage(inputs):
    z, znext, p = inputs
    return rk4(z[:NX], z[NX : NX + NU]) - znext[:NX]

  def build(N: int) -> al.Function:
    z = al.sym("z", NZ * (N + 1))
    p = al.sym("p", NX * (N + 1), diff=False)
    initial = eq_initial((z[:NZ], p[:NX]))
    mapped = al.vmap(eq_interstage, length=N, inputs={"z": (z, 0, NZ), "znext": (z, NZ, NZ), "p": (p, NX, NX)})
    return al.Function._from_exprs(f"race_car_eq_vmap_N{N}", [z, p], [al.concat([initial, mapped])], ["z", "p"], ["eq"])

  fn_a = build(50)
  fn_b = build(100)

  rng = np.random.default_rng(0)
  for fn in (fn_a, fn_b):
    N = (fn.inputs[0].size // NZ) - 1
    zv = rng.normal(size=NZ * (N + 1))
    pv = rng.normal(size=NX * (N + 1))
    # Sanity: the primal numerically matches the unrolled concat-of-call equivalent.
    parts = [eq_initial((fn.inputs[0][:NZ], fn.inputs[1][:NX]))]
    for i in range(N):
      zi = fn.inputs[0][i * NZ : (i + 1) * NZ]
      znext = fn.inputs[0][(i + 1) * NZ : (i + 2) * NZ]
      pi = fn.inputs[1][(i + 1) * NX : (i + 2) * NX]
      parts.append(eq_interstage((zi, znext, pi)))
    ref = al.Function._from_exprs(f"race_car_eq_ref_N{N}", [fn.inputs[0], fn.inputs[1]], [al.concat(parts)], ["z", "p"], ["eq"])
    np.testing.assert_allclose(fn((zv, pv)), ref((zv, pv)), rtol=1e-12, atol=1e-12)

  src_a = render_c_source(fn_a)
  src_b = render_c_source(fn_b)
  # Source size grows only with the literal-N loop bound differences; the loop body is shared.
  diff = abs(len(src_a) - len(src_b))
  assert diff < 200, f"primal source grew by {diff} bytes from N=50 to N=100, expected near-constant"


def test_sparse_jacobian_of_race_car_vmap_matches_unrolled_concat() -> None:
  """End-to-end: sparse_jacobian on a VMAP-based race-car fixture matches the unrolled-concat fixture
  numerically and structurally."""

  NX, NU, NZ = 4, 2, 6
  WHEELBASE, DT, M = 0.3, 0.05, 3.47
  C_M0, C_R0, C_R1, C_R2 = 11.0, 0.1, 0.01, 0.001

  def cont(x, u):
    phi, v = x[2], x[3]
    throttle, delta = u[0], u[1]
    beta = 0.5 * delta
    vx = v * beta.cos()
    lr = 0.5 * WHEELBASE
    return al.stack(
      [
        v * (phi + beta).cos(),
        v * (phi + beta).sin(),
        v * beta.sin() / lr,
        (C_M0 * throttle - (C_R0 + C_R1 * vx + C_R2 * vx * vx) * (10 * vx).tanh()) / M,
      ]
    )

  def rk4(x, u):
    k1 = cont(x, u)
    k2 = cont(x + DT / 2 * k1, u)
    k3 = cont(x + DT / 2 * k2, u)
    k4 = cont(x + DT * k3, u)
    return x + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

  @al.function(al.G(al.L("z", NZ), al.L("p", NX)), al.L("eq", ...), name="race_car_eq_initial2")
  def eq_initial(inputs):
    z, p = inputs
    return z[:NX] - p[:NX]

  @al.function(al.G(al.L("z", NZ), al.L("znext", NZ), al.L("p", NX)), al.L("eq", ...), name="race_car_eq_interstage2")
  def eq_interstage(inputs):
    z, znext, p = inputs
    return rk4(z[:NX], z[NX : NX + NU]) - znext[:NX]

  def build_vmap(N: int) -> al.Function:
    z = al.sym("z", NZ * (N + 1))
    p = al.sym("p", NX * (N + 1), diff=False)
    initial = eq_initial((z[:NZ], p[:NX]))
    mapped = al.vmap(eq_interstage, length=N, inputs={"z": (z, 0, NZ), "znext": (z, NZ, NZ), "p": (p, NX, NX)})
    return al.Function._from_exprs(f"tr_vmap_N{N}", [z, p], [al.concat([initial, mapped])], ["z", "p"], ["eq"])

  def build_unroll(N: int) -> al.Function:
    z = al.sym("z", NZ * (N + 1))
    p = al.sym("p", NX * (N + 1), diff=False)
    parts = [eq_initial((z[:NZ], p[:NX]))]
    for i in range(N):
      parts.append(eq_interstage((z[i * NZ : (i + 1) * NZ], z[(i + 1) * NZ : (i + 2) * NZ], p[(i + 1) * NX : (i + 2) * NX])))
    return al.Function._from_exprs(f"tr_unroll_N{N}", [z, p], [al.concat(parts)], ["z", "p"], ["eq"])

  N = 4
  fn_vmap = build_vmap(N)
  fn_unroll = build_unroll(N)

  spj_vmap = al.sparse_jacobian(fn_vmap, "eq", "z")
  spj_unroll = al.sparse_jacobian(fn_unroll, "eq", "z")

  # The nnz orderings differ (the VMAP path emits per-formal disjoint partitions), but the
  # patterns must agree as coordinate sets and the densified values must match exactly.
  sp_m, sp_u = spj_vmap.output_sparsities[0], spj_unroll.output_sparsities[0]
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


@al.function(al.G(al.L("s", NS), al.L("u", NU)), al.L("next", ...), name="pairs_step")
def pairs_step(inputs):
  s, u = inputs
  return al.stack([s[0] + 0.1 * s[2].cos() * u[0], s[1] + 0.1 * s[2].sin() * u[1], s[2] + 0.1 * (u[0] - u[1])])


@al.function(al.G(al.L("prev_i", NS), al.L("prev_j", NS), al.L("si", NS), al.L("sj", NS), al.L("slack", 1)), al.L("h", ...), name="pairs_barrier")
def pairs_barrier(inputs):
  # The trailing p-norm term mirrors the smooth-max a velocity-margin barrier uses; it is what
  # brings integer POW, a *non-integer* POW (the shape of such a barrier's braking envelope,
  # `c * d^q`) and a nested sqrt into the second-order path through the VMAP. As in the real
  # barrier, the non-integer power's base is bounded away from zero, where its slope diverges.
  prev_i, prev_j, si, sj, slack = inputs
  d, dprev = si[:2] - sj[:2], prev_i[:2] - prev_j[:2]
  envelope = 1.1 * (1.0 + al.dot(dprev, dprev)) ** 0.84
  soft_max = (al.dot(d, d) ** 2 + envelope**4).sqrt().sqrt()
  return al.stack([(al.dot(d, d).sqrt() - 0.5 * (1.0 + al.dot(dprev, dprev)).log() + soft_max + slack[0])])


@al.function(al.G(al.L("s", NS), al.L("snext", NS), al.L("slack", 1)), al.L("h", ...), name="pairs_wall")
def pairs_wall(inputs):
  s, snext, slack = inputs
  return al.stack([(snext[0] - 0.5 * s[0] + slack[0]), (1.0 - snext[1].exp() + slack[0])])


def _pair_index_table(bodies: list[int]) -> np.ndarray:
  return np.concatenate([np.arange(NS, dtype=np.int64) + k * NS for k in bodies])


def _build_pairs_fn(mapped: bool) -> al.Function:
  u = al.sym("u", NU * NB + 1)
  p = al.sym("p", NS * NB, diff=False)
  uu, slack = u[: NU * NB], u[NU * NB : NU * NB + 1]
  if mapped:
    nxt = al.vmap(pairs_step, NB, [(p, 0, NS), (uu, 0, NU)])
    idx_i, idx_j = _pair_index_table([i for i, _ in PAIRS]), _pair_index_table([j for _, j in PAIRS])
    pair_rows = al.vmap(
      pairs_barrier,
      len(PAIRS),
      [(al.gather(p, idx_i), 0, NS), (al.gather(p, idx_j), 0, NS), (al.gather(nxt, idx_i), 0, NS), (al.gather(nxt, idx_j), 0, NS), (slack, 0, 0)],
    )
    body_rows = al.vmap(pairs_wall, NB, [(p, 0, NS), (nxt, 0, NS), (slack, 0, 0)])
    h = al.concat([pair_rows, body_rows])
  else:
    nxt = al.concat([pairs_step((p[NS * k : NS * (k + 1)], uu[NU * k : NU * (k + 1)])) for k in range(NB)])
    sl = lambda e, k: e[NS * k : NS * (k + 1)]  # noqa: E731
    rows = [pairs_barrier((sl(p, i), sl(p, j), sl(nxt, i), sl(nxt, j), slack)) for i, j in PAIRS]
    rows += [pairs_wall((sl(p, k), sl(nxt, k), slack)) for k in range(NB)]
    h = al.concat(rows)
  return al.Function._from_exprs(f"pairs_{'vmap' if mapped else 'unroll'}", [u, p], [h], ["u", "p"], ["h"])


def _pairs_sample() -> tuple[np.ndarray, np.ndarray]:
  rng = np.random.default_rng(11)
  uv = rng.normal(scale=0.3, size=NU * NB + 1)
  uv[-1] = 0.05
  return uv, rng.normal(scale=0.5, size=NS * NB) + np.tile([1.0, 2.0, 0.3], NB)


def test_gather_fed_chained_vmaps_match_unrolled_calls() -> None:
  fn_vmap, fn_unroll = _build_pairs_fn(True), _build_pairs_fn(False)
  uv, pv = _pairs_sample()
  np.testing.assert_allclose(fn_vmap((uv, pv)), fn_unroll((uv, pv)), rtol=1e-12, atol=1e-12)

  jac_vmap = fn_vmap.factory("pairs_vmap_jac", ["u", "p"], [al.factory.Jac("h", "u")])((uv, pv))
  jac_unroll = fn_unroll.factory("pairs_unroll_jac", ["u", "p"], [al.factory.Jac("h", "u")])((uv, pv))
  np.testing.assert_allclose(jac_vmap, jac_unroll, rtol=1e-10, atol=1e-10)


def test_gather_fed_chained_vmaps_spjac_and_sphess_match_dense() -> None:
  fn = _build_pairs_fn(True)
  uv, pv = _pairs_sample()
  dense = fn.factory("pairs_vmap_jac2", ["u", "p"], [al.factory.Jac("h", "u")])((uv, pv))
  assert isinstance(dense, np.ndarray)
  spjf = fn.factory("pairs_vmap_spjac", ["u", "p"], [al.factory.SpJac("h", "u")])
  sp = spjf.output_sparsities[0]
  assert sp is not None
  flat = np.asarray(sp.rows) * dense.shape[1] + np.asarray(sp.cols)
  np.testing.assert_allclose(spjf((uv, pv)), np.ravel(dense)[flat], rtol=1e-10, atol=1e-10)
  # Coloring must not claim structural zeros that the dense Jacobian disagrees with.
  assert not np.any(np.abs(dense[~sp.to_mask()]) > 1e-12)

  # Second order through the same composition, against the unrolled reference.
  lam = np.arange(1.0, fn.outputs[0].shape[0] + 1.0)
  hess = {
    name: fn_.factory(f"pairs_{name}_sphess", ["u", "p", "lam:h"], [al.factory.SpHess("gamma", "u")], aux={"gamma": ["h"]})
    for name, fn_ in (("vmap", fn), ("unroll", _build_pairs_fn(False)))
  }
  dense_hess = {}
  for name, hf in hess.items():
    hsp = hf.output_sparsities[0]
    assert hsp is not None
    dense_hess[name] = np.zeros(hsp.shape)
    dense_hess[name][np.asarray(hsp.rows), np.asarray(hsp.cols)] = np.asarray(hf((uv, pv, lam)), dtype=np.float64).reshape(-1)
  np.testing.assert_allclose(dense_hess["vmap"], dense_hess["unroll"], rtol=1e-9, atol=1e-9)


def test_matmul_inside_vmap_callee_differentiates() -> None:
  """Differentiate a dense layer inside a mapped callee against a NumPy reference."""
  w = np.array([[0.4, -0.2, 0.7], [0.1, 0.9, -0.3]])
  b = np.array([0.05, -0.15])

  @al.function(al.L("s", 3), al.L("y", ...), name="vmap_dense_layer")
  def layer(s):
    phi = al.stack([s[0], s[1], s[2]])
    h = al.const(w) @ phi + al.const(b)
    return al.stack([(h * h).sum()])

  N = 4
  z = al.sym("z", 3 * N)
  fn = al.Function._from_exprs("vmap_dense", [z], [al.vmap(layer, N, [(z, 0, 3)])], ["z"], ["y"])

  zv = np.random.default_rng(3).normal(size=3 * N)
  expected = np.zeros((N, 3 * N))
  for k in range(N):
    h = w @ zv[3 * k : 3 * k + 3] + b
    expected[k, 3 * k : 3 * k + 3] = 2.0 * h @ w
  np.testing.assert_allclose(fn.factory("vmap_dense_jac", ["z"], [al.factory.Jac("y", "z")])(zv), expected, rtol=1e-10, atol=1e-10)


def test_weighted_mapped_residual_cost_matches_unrolled_derivatives() -> None:
  @al.function(al.G(al.L("x", 2), al.L("ref", 2), al.L("scale", 1)), al.L("r", ...))
  def residual(inputs):
    x, ref, scale = inputs
    return al.stack([x[0] - ref[0], ref[1].cos() * x[1] - scale[0].tanh()])

  n = 4
  x, p = al.sym("x", 2 * n), al.sym("p", 2 * n + 1, diff=False)
  weights = al.const(np.array([0.0, 0.3, 1.2, 0.3, 1.2, 0.3, 5.0, 0.3]))
  mapped = al.vmap(residual, n, [(x, 0, 2), (p, 0, 2), (p, 2 * n, 0)])
  unrolled = al.concat([residual((x[2 * i : 2 * i + 2], p[2 * i : 2 * i + 2], p[-1:])) for i in range(n)])
  rng = np.random.default_rng(21)
  inputs = rng.normal(size=x.size), rng.normal(size=p.size)
  values = []
  for name, r in (("mapped", mapped), ("unrolled", unrolled)):
    fn = al.Function._from_exprs(name, [x, p], [al.dot(weights, r**2)], ["x", "p"], ["f"])
    derivatives = fn.factory(name + "_derivatives", ["x", "p"], ["f", al.factory.Grad("f", "x"), al.factory.SpHess("f", "x")])
    values.append(derivatives(inputs))
  for actual, expected in zip(values[0], values[1], strict=True):
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
