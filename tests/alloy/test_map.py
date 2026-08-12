from __future__ import annotations

import ctypes
import re
import shutil
import subprocess
import sys

import numpy as np
import pytest

import alloy as al


@al.function("scale_add", {"x": 3, "p": 3})
def scale_add(x, p):
  return {"y": 2.0 * x + p}


def test_map_eval_matches_unrolled_concat_of_call() -> None:
  N = 4
  z = al.sym("z", 3 * N)
  p = al.sym("p", 3 * N)

  mapped = al.map_(scale_add, N, [(z, 0, 3), (p, 0, 3)])
  unrolled = al.concat([scale_add.call([z[i * 3 : (i + 1) * 3], p[i * 3 : (i + 1) * 3]])[0] for i in range(N)])

  fn_map = al.Function("scaled_map", [z, p], [mapped], ["z", "p"], ["y"])
  fn_concat = al.Function("scaled_concat", [z, p], [unrolled], ["z", "p"], ["y"])

  rng = np.random.default_rng(0)
  zv = rng.normal(size=3 * N)
  pv = rng.normal(size=3 * N)

  np.testing.assert_allclose(fn_map(zv, pv), fn_concat(zv, pv))


def test_map_overlapping_strided_slices_match_unrolled() -> None:
  NZ = 6
  NX = 4
  N = 3

  @al.function("step", {"z": NZ, "znext": NZ, "p": NX})
  def step(z, znext, p):
    return {"eq": (z[:NX] - znext[:NX]) + p}

  z = al.sym("z", NZ * (N + 1))
  p = al.sym("p", NX * (N + 1))

  mapped = al.map_(step, N, [(z, 0, NZ), (z, NZ, NZ), (p, NX, NX)])
  parts = []
  for i in range(N):
    parts.append(step.call([z[i * NZ : (i + 1) * NZ], z[(i + 1) * NZ : (i + 2) * NZ], p[(i + 1) * NX : (i + 2) * NX]])[0])
  unrolled = al.concat(parts)

  fn_map = al.Function("step_map", [z, p], [mapped], ["z", "p"], ["eq"])
  fn_concat = al.Function("step_concat", [z, p], [unrolled], ["z", "p"], ["eq"])

  rng = np.random.default_rng(1)
  zv = rng.normal(size=NZ * (N + 1))
  pv = rng.normal(size=NX * (N + 1))

  np.testing.assert_allclose(fn_map(zv, pv), fn_concat(zv, pv))


def test_map_zero_length_returns_empty() -> None:
  z = al.sym("z", 3)
  p = al.sym("p", 3)
  empty = al.map_(scale_add, 0, [(z, 0, 0), (p, 0, 0)])
  fn = al.Function("empty_map", [z, p], [empty], ["z", "p"], ["y"])
  out = fn(np.zeros(3), np.zeros(3))
  assert isinstance(out, np.ndarray)
  assert out.shape == (0,)


def test_map_broadcast_stride_zero_repeats_same_slice() -> None:
  N = 3
  z = al.sym("z", 3)
  p = al.sym("p", 3)
  mapped = al.map_(scale_add, N, [(z, 0, 0), (p, 0, 0)])
  fn = al.Function("broadcast_map", [z, p], [mapped], ["z", "p"], ["y"])
  zv = np.array([1.0, 2.0, 3.0])
  pv = np.array([0.5, -1.0, 0.25])
  expected = np.tile(2.0 * zv + pv, N)
  np.testing.assert_allclose(fn(zv, pv), expected)


def test_map_jit_matches_unrolled_numpy() -> None:
  N = 2
  z = al.sym("z", 6)
  p = al.sym("p", 6)
  mapped = al.map_(scale_add, N, [(z, 0, 3), (p, 0, 3)])
  fn = al.Function("eval_path", [z, p], [mapped], ["z", "p"], ["y"])
  zv = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  pv = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
  expected = np.concatenate([2.0 * zv[i * 3 : (i + 1) * 3] + pv[i * 3 : (i + 1) * 3] for i in range(N)])
  np.testing.assert_allclose(fn(zv, pv), expected)


def test_map_rejects_bad_input_specs() -> None:
  z = al.sym("z", 6)
  p = al.sym("p", 6)

  with pytest.raises(ValueError, match="map length"):
    al.map_(scale_add, -1, [(z, 0, 3), (p, 0, 3)])

  with pytest.raises(ValueError, match="reads past outer tensor"):
    al.map_(scale_add, 3, [(z, 0, 3), (p, 0, 3)])  # would need size 9 > 6

  with pytest.raises(ValueError, match=r"expects 2 input specs"):
    al.map_(scale_add, 2, [(z, 0, 3)])

  with pytest.raises(ValueError, match="stride must be non-negative"):
    al.map_(scale_add, 2, [(z, 0, -1), (p, 0, 3)])


def test_map_structural_key_matches_for_equal_constructions() -> None:
  # Construction-time interning collapses two structurally-identical ``al.map_`` calls to
  # the same Expr instance, so ``a is b`` and the structural-equality contract is preserved.
  z = al.sym("z", 6)
  p = al.sym("p", 6)
  a = al.map_(scale_add, 2, [(z, 0, 3), (p, 0, 3)])
  b = al.map_(scale_add, 2, [(z, 0, 3), (p, 0, 3)])
  assert a is b
  assert a.structurally_equal(b)


def test_map_c_source_loop_size_is_independent_of_length() -> None:
  from alloy.codegen import render_c_source

  def render(N: int) -> str:
    z = al.sym("z", 3 * N)
    p = al.sym("p", 3 * N)
    fn = al.Function(f"scale_map_{N}", [z, p], [al.map_(scale_add, N, [(z, 0, 3), (p, 0, 3)])], ["z", "p"], ["y"])
    return render_c_source(fn)

  # Past the 32-element threshold where the trailing copy loop also folds, the rendered source
  # must be identical except for the loop bound and the function name.
  src_a = render(20)
  src_b = render(100)
  # Renderer-agnostic: the MAP body is one loop whose bound scales with N (not unrolled) and the
  # source LOC stays constant in N. The legacy renderer emits `for (int it = 0; it < N; ++it)`,
  # the Program IR renderer `for (long long it_y = 0; it_y < N; ++it_y)` — both carry the `< N;` bound.
  assert "< 20;" in src_a
  assert "< 100;" in src_b
  assert src_a.count("\n") == src_b.count("\n")


def test_mapped_sparse_hessian_c_source_is_constant_in_length(monkeypatch: pytest.MonkeyPatch) -> None:
  from alloy.codegen import render_c_source
  from alloy.expr import topo

  monkeypatch.setenv("ALLOY_STRICT_JVP_MANY", "1")
  x = al.sym("x", 2)
  hidden = al.stack([x[0] * x[1], x[0] - 0.4 * x[1]])
  piece = al.Function("map_sphess_codegen_piece", [x], [al.stack([(hidden.tanh() ** 2).sum()])], ["x"], ["g"])

  def render(length: int) -> tuple[str, tuple[str, ...], int, dict[str, int]]:
    z = al.sym("z", 2 * length)
    mapped = al.map_(piece, length, [(z, 0, 2)])
    base = al.Function(f"map_sphess_codegen_base_{length}", [z], [(z * z).sum(), mapped], ["z"], ["f", "g"])
    sphess = base.factory(f"map_sphess_codegen_{length}", ["z", "lam:f", "lam:g"], ["sphess:gamma:z:z"], aux={"gamma": ["f", "g"]})
    map_nodes = [node for node in topo(sphess.outputs) if node.op == al.Ops.MAP]
    mapped_callees = sorted({node.attrs["callee"].name for node in map_nodes})
    second_order = tuple(name for name in mapped_callees if "_adj" in name and "_fwd" in name)
    source = render_c_source(sphess)
    copies = {name: source.count(f"{name.replace(':', '_')}_raw(") for name in mapped_callees}
    return source, second_order, len(map_nodes), copies

  rendered = [render(length) for length in (2, 8, 32)]
  assert len({source.count("\n") for source, _, _, _ in rendered}) == 1
  assert all(names for _, names, _, _ in rendered)
  assert len({names for _, names, _, _ in rendered}) == 1
  # The sphess graph's MAP-node count is a property of (#formals x #local-color-groups), never of
  # the map length; every mapped callee (primal, adjoint, second-order) renders one definition and
  # a length-independent number of call sites.
  assert len({map_count for _, _, map_count, _ in rendered}) == 1
  assert len({tuple(sorted(copies.items())) for _, _, _, copies in rendered}) == 1
  for source, names, _, copies in rendered:
    for name, count in copies.items():
      assert count >= 2, f"{name} rendered without a call site"
    for name in names:
      c_name = name.replace(":", "_")
      assert copies[name] == 2  # one definition and one call in one MAP loop
      assert len(re.findall(rf"for \([^\n]+\) \{{\n\s+{re.escape(c_name)}_raw\(", source)) == 1


def test_callee_formal_named_w_avoids_workspace_collision() -> None:
  # The rendered callee signature appends the `double* w` workspace tail; a formal named `w` used
  # to redefine that parameter and fail to compile.
  x, w = al.sym("x", 3), al.sym("w", 3)
  piece = al.Function("w_name_piece", [x, w], [x * w + w.sin()], ["x", "w"], ["y"])
  z, wv = al.sym("z", 6), al.sym("w", 3)
  fn = al.Function("w_name_map", [z, wv], [al.map_(piece, 2, [(z, 0, 3), (wv, 0, 0)])], ["z", "w"], ["y"])
  zval = np.arange(6.0)
  wval = np.array([0.3, -0.2, 0.8])
  expected = np.concatenate([zval[3 * i : 3 * i + 3] * wval + np.sin(wval) for i in range(2)])
  np.testing.assert_allclose(fn(zval, wval), expected)


def test_map_compiled_c_matches_unrolled_concat(tmp_path) -> None:
  cc = shutil.which("cc")
  if cc is None:
    pytest.skip("cc is required for generated C smoke test")

  from alloy.codegen import render_c_module

  N = 5
  z = al.sym("z", 3 * N)
  p = al.sym("p", 3 * N)
  fn = al.Function("scale_map_compiled", [z, p], [al.map_(scale_add, N, [(z, 0, 3), (p, 0, 3)])], ["z", "p"], ["y"])
  module = render_c_module(fn)
  (tmp_path / module.header_name).write_text(module.header)
  source = tmp_path / module.source_name
  source.write_text(module.source)
  lib_path = tmp_path / ("libmap.dylib" if sys.platform == "darwin" else "libmap.so")
  cmd = [cc, "-fPIC", str(source), "-lm", "-o", str(lib_path)]
  cmd.insert(1, "-dynamiclib" if sys.platform == "darwin" else "-shared")
  subprocess.run(cmd, check=True)

  lib = ctypes.CDLL(str(lib_path))
  c_double_p = ctypes.POINTER(ctypes.c_double)
  lib.scale_map_compiled.argtypes = [
    ctypes.POINTER(c_double_p),
    ctypes.POINTER(c_double_p),
    ctypes.POINTER(ctypes.c_int),
    c_double_p,
    ctypes.c_void_p,
  ]
  lib.scale_map_compiled.restype = ctypes.c_int

  rng = np.random.default_rng(0)
  zv = rng.normal(size=3 * N)
  pv = rng.normal(size=3 * N)

  z_buf = (ctypes.c_double * (3 * N))(*zv)
  p_buf = (ctypes.c_double * (3 * N))(*pv)
  y_buf = (ctypes.c_double * (3 * N))()
  w_buf = (ctypes.c_double * max(lib.scale_map_compiled_sz_w(), 1))()
  args = (c_double_p * 2)(ctypes.cast(z_buf, c_double_p), ctypes.cast(p_buf, c_double_p))
  res = (c_double_p * 1)(ctypes.cast(y_buf, c_double_p))

  assert lib.scale_map_compiled(args, res, None, w_buf, None) == 0
  np.testing.assert_allclose(np.array(y_buf), 2.0 * zv + pv)


def test_race_car_eq_primal_source_is_constant_in_horizon() -> None:
  """Rewriting the race-car fixture with `al.scan` yields C source whose size does not grow with N."""

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

  @al.function("race_car_eq_initial", {"z": NZ, "p": NX})
  def eq_initial(z, p):
    return {"eq": z[:NX] - p[:NX]}

  @al.function("race_car_eq_interstage", {"z": NZ, "znext": NZ, "p": NX})
  def eq_interstage(z, znext, p):
    return {"eq": rk4(z[:NX], z[NX : NX + NU]) - znext[:NX]}

  def build(N: int) -> al.Function:
    z = al.sym("z", NZ * (N + 1))
    p = al.sym("p", NX * (N + 1), diff=False)
    initial = eq_initial.call([z[:NZ], p[:NX]])[0]
    mapped = al.scan(eq_interstage, length=N, inputs={"z": (z, 0, NZ), "znext": (z, NZ, NZ), "p": (p, NX, NX)})
    return al.Function(f"race_car_eq_map_N{N}", [z, p], [al.concat([initial, mapped])], ["z", "p"], ["eq"])

  fn_a = build(50)
  fn_b = build(100)

  rng = np.random.default_rng(0)
  for fn in (fn_a, fn_b):
    N = (fn.inputs[0].size // NZ) - 1
    zv = rng.normal(size=NZ * (N + 1))
    pv = rng.normal(size=NX * (N + 1))
    # Sanity: the primal numerically matches the unrolled concat-of-call equivalent.
    parts = [eq_initial.call([fn.inputs[0][:NZ], fn.inputs[1][:NX]])[0]]
    for i in range(N):
      zi = fn.inputs[0][i * NZ : (i + 1) * NZ]
      znext = fn.inputs[0][(i + 1) * NZ : (i + 2) * NZ]
      pi = fn.inputs[1][(i + 1) * NX : (i + 2) * NX]
      parts.append(eq_interstage.call([zi, znext, pi])[0])
    ref = al.Function(f"race_car_eq_ref_N{N}", [fn.inputs[0], fn.inputs[1]], [al.concat(parts)], ["z", "p"], ["eq"])
    np.testing.assert_allclose(fn(zv, pv), ref(zv, pv), rtol=1e-12, atol=1e-12)

  src_a = render_c_source(fn_a)
  src_b = render_c_source(fn_b)
  # Source size grows only with the literal-N loop bound differences; the loop body is shared.
  diff = abs(len(src_a) - len(src_b))
  assert diff < 200, f"primal source grew by {diff} bytes from N=50 to N=100, expected near-constant"


def test_jvp_many_of_map_matches_unrolled_jvp() -> None:
  N = 4
  z = al.sym("z", 3 * N)
  p = al.sym("p", 3 * N)

  mapped = al.map_(scale_add, N, [(z, 0, 3), (p, 0, 3)])
  unrolled = al.concat([scale_add.call([z[i * 3 : (i + 1) * 3], p[i * 3 : (i + 1) * 3]])[0] for i in range(N)])

  seeds = al.const(np.eye(3 * N))
  jvp_map = al.jvp_many(mapped, z, seeds)
  jvp_ref = al.jvp_many(unrolled, z, seeds)

  fn_map = al.Function("jvp_map", [z, p], [jvp_map], ["z", "p"], ["dy"])
  fn_ref = al.Function("jvp_ref", [z, p], [jvp_ref], ["z", "p"], ["dy"])

  rng = np.random.default_rng(0)
  zv = rng.normal(size=3 * N)
  pv = rng.normal(size=3 * N)

  np.testing.assert_allclose(fn_map(zv, pv), fn_ref(zv, pv), rtol=1e-10, atol=1e-10)


def test_jacobian_of_map_matches_finite_differences() -> None:
  N = 5
  z = al.sym("z", 3 * N)
  p = al.sym("p", 3 * N)
  mapped = al.map_(scale_add, N, [(z, 0, 3), (p, 0, 3)])
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


def test_grad_factory_over_map_matches_unrolled_and_finite_difference() -> None:
  from alloy.ad import finite_difference
  from alloy.expr import topo

  @al.function("map_grad_piece", {"x": 2})
  def piece(x):
    return {"y": x * x + x.sin()}

  N = 4
  z = al.sym("z", 2 * N)
  mapped = al.map_(piece, N, [(z, 0, 2)])
  unrolled = al.concat([piece.call([z[2 * it : 2 * (it + 1)]])[0] for it in range(N)])
  mapped_fn = al.Function("map_grad_factory", [z], [mapped], ["z"], ["y"])
  unrolled_fn = al.Function("map_grad_unrolled", [z], [unrolled], ["z"], ["y"])
  mapped_grad = mapped_fn.factory("map_grad_factory_grad", ["z", "lam:y"], ["grad:gamma:z"], aux={"gamma": ["y"]})
  unrolled_grad = unrolled_fn.factory("map_grad_unrolled_grad", ["z", "lam:y"], ["grad:gamma:z"], aux={"gamma": ["y"]})
  map_nodes = [node for node in topo(mapped_grad.outputs) if node.op == al.Ops.MAP]
  assert len(map_nodes) == 1
  assert "_adj0_0" in map_nodes[0].attrs["callee"].name

  zv = np.random.default_rng(7).normal(size=2 * N)
  lamv = np.random.default_rng(8).normal(size=2 * N)
  np.testing.assert_allclose(mapped_grad(zv, lamv), unrolled_grad(zv, lamv), rtol=1e-10, atol=1e-10)
  np.testing.assert_allclose(
    mapped_grad(zv, lamv), finite_difference(lambda value: np.dot(lamv, np.asarray(mapped_fn(value))), zv).reshape(-1), rtol=1e-6, atol=1e-7
  )


def test_spjacobian_of_race_car_map_matches_unrolled_concat() -> None:
  """End-to-end: spjacobian on a MAP-based race-car fixture matches the unrolled-concat fixture
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

  @al.function("race_car_eq_initial2", {"z": NZ, "p": NX})
  def eq_initial(z, p):
    return {"eq": z[:NX] - p[:NX]}

  @al.function("race_car_eq_interstage2", {"z": NZ, "znext": NZ, "p": NX})
  def eq_interstage(z, znext, p):
    return {"eq": rk4(z[:NX], z[NX : NX + NU]) - znext[:NX]}

  def build_map(N: int) -> al.Function:
    z = al.sym("z", NZ * (N + 1))
    p = al.sym("p", NX * (N + 1), diff=False)
    initial = eq_initial.call([z[:NZ], p[:NX]])[0]
    mapped = al.scan(eq_interstage, length=N, inputs={"z": (z, 0, NZ), "znext": (z, NZ, NZ), "p": (p, NX, NX)})
    return al.Function(f"tr_map_N{N}", [z, p], [al.concat([initial, mapped])], ["z", "p"], ["eq"])

  def build_unroll(N: int) -> al.Function:
    z = al.sym("z", NZ * (N + 1))
    p = al.sym("p", NX * (N + 1), diff=False)
    parts = [eq_initial.call([z[:NZ], p[:NX]])[0]]
    for i in range(N):
      parts.append(eq_interstage.call([z[i * NZ : (i + 1) * NZ], z[(i + 1) * NZ : (i + 2) * NZ], p[(i + 1) * NX : (i + 2) * NX]])[0])
    return al.Function(f"tr_unroll_N{N}", [z, p], [al.concat(parts)], ["z", "p"], ["eq"])

  N = 4
  fn_map = build_map(N)
  fn_unroll = build_unroll(N)

  spj_map = al.spjacobian(fn_map, "z", "eq")
  spj_unroll = al.spjacobian(fn_unroll, "z", "eq")

  # The nnz orderings differ (the map path emits per-formal disjoint partitions), but the
  # patterns must agree as coordinate sets and the densified values must match exactly.
  sp_m, sp_u = spj_map.output_sparsities[0], spj_unroll.output_sparsities[0]
  assert sp_m is not None and sp_u is not None
  assert sp_m.shape == sp_u.shape and set(zip(sp_m.rows, sp_m.cols)) == set(zip(sp_u.rows, sp_u.cols))

  rng = np.random.default_rng(2)
  zv = rng.normal(size=NZ * (N + 1))

  dense_m, dense_u = np.zeros(sp_m.shape), np.zeros(sp_u.shape)
  dense_m[np.asarray(sp_m.rows), np.asarray(sp_m.cols)] = np.asarray(spj_map(zv), dtype=np.float64).reshape(-1)
  dense_u[np.asarray(sp_u.rows), np.asarray(sp_u.cols)] = np.asarray(spj_unroll(zv), dtype=np.float64).reshape(-1)
  np.testing.assert_allclose(dense_m, dense_u, rtol=1e-10, atol=1e-10)


RK4_NX, RK4_NU, RK4_NZ, RK4_N_PARAMS = 4, 2, 6, 7


def _rk4_bicycle_eq_map(horizon: int) -> al.Function:
  """Scan-based RK4 bicycle stage transcription with a symbolic parameter tail in ``p``.

  Self-contained on purpose: this is the realistic shape that produces a piece-ordered
  (non-row-major) COO sparsity and exercises the transposed-concat peephole, and the tests
  below must keep covering it whether or not any benchmark problem still uses it.
  """
  n_param = RK4_NX * (horizon + 1) + RK4_N_PARAMS

  def ode(x, u, params):
    wheelbase, _, mass, c_m0, c_r0, c_r1, c_r2 = [params[i] for i in range(RK4_N_PARAMS)]
    beta = 0.5 * u[1]
    vx = x[3] * beta.cos()
    return al.stack(
      [
        x[3] * (x[2] + beta).cos(),
        x[3] * (x[2] + beta).sin(),
        x[3] * beta.sin() / (0.5 * wheelbase),
        (c_m0 * u[0] - (c_r0 + c_r1 * vx + c_r2 * vx * vx) * (10 * vx).tanh()) / mass,
      ]
    )

  def rk4(x, u, params):
    dt = params[1]
    k1 = ode(x, u, params)
    k2 = ode(x + dt / 2 * k1, u, params)
    k3 = ode(x + dt / 2 * k2, u, params)
    k4 = ode(x + dt * k3, u, params)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

  @al.function("rk4_bicycle_initial", {"z": RK4_NZ, "p": RK4_NX})
  def eq_initial(z, p):
    return {"eq": z[:RK4_NX] - p[:RK4_NX]}

  @al.function("rk4_bicycle_interstage", {"z": RK4_NZ, "znext": RK4_NZ, "params": RK4_N_PARAMS})
  def eq_interstage(z, znext, params):
    return {"eq": rk4(z[:RK4_NX], z[RK4_NX : RK4_NX + RK4_NU], params) - znext[:RK4_NX]}

  z = al.sym("z", RK4_NZ * (horizon + 1))
  p = al.sym("p", n_param, diff=False)
  initial = eq_initial.call([z[:RK4_NZ], p[:RK4_NX]])[0]
  mapped = al.scan(
    eq_interstage,
    length=horizon,
    inputs={"z": (z, 0, RK4_NZ), "znext": (z, RK4_NZ, RK4_NZ), "params": (p, RK4_NX * (horizon + 1), 0)},
  )
  return al.Function(f"rk4_bicycle_eq_map_N{horizon}", [z, p], [al.concat([initial, mapped])], ["z", "p"], ["eq"])


def test_csr_csc_header_tables_carry_value_perm_for_non_row_major_coo() -> None:
  """The rendered header's CSR/CSC index tables are sorted, but the compact value buffer stays in
  COO order — piece-ordered on the merged map path, i.e. NOT row-major — so the header must also
  emit ``*_csr_val_perm`` / ``*_csc_val_perm`` tables (``values_csr[k] = values[csr_val_perm[k]]``).
  Reconstructing the dense Jacobian through both compressed formats must match the COO scatter."""

  import re

  from alloy.codegen.c import render_c_module

  N = 3
  fn = _rk4_bicycle_eq_map(N)
  spjf = fn.factory(f"trk_map_valperm_N{N}", ["z", "p"], ["spjac:eq:z"])
  sp = spjf.output_sparsities[0]
  assert sp is not None
  coo = list(zip(sp.rows, sp.cols))
  assert coo != sorted(coo), "test needs a non-row-major COO ordering to be meaningful"

  module = render_c_module(spjf, header_name="t.h", source_name="t.c", typed_buffers=False)

  def table(name: str) -> np.ndarray:
    m = re.search(rf"_{name}\[\d+\] = \{{([^}}]*)\}};", module.header)
    assert m is not None, f"header table {name} missing"
    return np.array([int(x) for x in m.group(1).split(",")], dtype=np.int64)

  rng = np.random.default_rng(3)
  zv, pv = rng.normal(size=RK4_NZ * (N + 1)), rng.normal(size=RK4_NX * (N + 1) + RK4_N_PARAMS)
  values = np.asarray(spjf(zv, pv), dtype=np.float64).reshape(-1)
  dense_ref = np.zeros(sp.shape)
  dense_ref[np.asarray(sp.rows), np.asarray(sp.cols)] = values

  row_ptr, col_ind, csr_perm = table("csr_row_ptr"), table("csr_col_ind"), table("csr_val_perm")
  dense_csr = np.zeros(sp.shape)
  for r in range(sp.shape[0]):
    for k in range(row_ptr[r], row_ptr[r + 1]):
      dense_csr[r, col_ind[k]] = values[csr_perm[k]]
  np.testing.assert_allclose(dense_csr, dense_ref, rtol=0, atol=0)

  col_ptr, row_ind, csc_perm = table("csc_col_ptr"), table("csc_row_ind"), table("csc_val_perm")
  dense_csc = np.zeros(sp.shape)
  for c in range(sp.shape[1]):
    for k in range(col_ptr[c], col_ptr[c + 1]):
      dense_csc[row_ind[k], c] = values[csc_perm[k]]
  np.testing.assert_allclose(dense_csc, dense_ref, rtol=0, atol=0)


def test_spjac_keeps_constant_loc_on_rk4_race_car_map() -> None:
  """End-to-end: even on the realistic RK4 race-car case (no periodic coloring), the rendered spjac
  C source stays at constant LOC across horizons because the transposed-concat peephole now emits
  per-block loops with a static `idx[]` table when the per-block group is large."""

  from alloy.codegen import render_c_source

  def loc(N: int) -> int:
    fn = _rk4_bicycle_eq_map(N)
    spj = fn.factory(f"rk4_bicycle_eq_map_N{N}_spjac_eq_z", ["z", "p"], ["spjac:eq:z"])
    return render_c_source(spj).count("\n")

  loc_a = loc(10)
  loc_b = loc(50)
  # LOC is bounded by a tiny constant — variation comes only from whether the workspace
  # spill threshold is crossed, which adds one wrapper line for the SZ_W null check.
  assert abs(loc_a - loc_b) <= 2, f"expected constant RK4 race-car-map LOC, got {loc_a} -> {loc_b}"


def test_simple_banded_map_spjac_has_constant_loc() -> None:
  """Simple banded MAP spjac C source stays at constant LOC across N. With the structured
  MAP-aware sparse Jacobian, the loop comes from per-formal const-seed JVPs wrapped in MAP,
  plus a constant scatter index table; without it, the tile-strided gather peephole would
  fire instead. Either way the LOC must not grow with N."""

  from alloy.codegen import render_c_source

  NX, NZ = 4, 6

  @al.function("eq_initial_t", {"z": NZ})
  def eq_initial(z):
    return {"eq": z[:NX] * 2.0}

  @al.function("eq_interstage_t", {"z": NZ, "znext": NZ})
  def eq_interstage(z, znext):
    return {"eq": z[:NX] * 1.5 - znext[:NX]}

  def build(N: int) -> al.Function:
    z = al.sym("z", NZ * (N + 1))
    initial = eq_initial.call([z[:NZ]])[0]
    mapped = al.scan(eq_interstage, length=N, inputs={"z": (z, 0, NZ), "znext": (z, NZ, NZ)})
    return al.Function(f"banded_N{N}", [z], [al.concat([initial, mapped])], ["z"], ["eq"])

  loc_a = render_c_source(al.spjacobian(build(10), "z", "eq")).count("\n")
  loc_b = render_c_source(al.spjacobian(build(50), "z", "eq")).count("\n")
  # LOC is bounded by a tiny constant — variation comes only from whether the workspace
  # spill threshold is crossed, which adds one wrapper line for the SZ_W null check.
  assert abs(loc_a - loc_b) <= 2, f"expected constant LOC, got {loc_a} -> {loc_b}"
  src_b = render_c_source(al.spjacobian(build(50), "z", "eq"))
  # Renderer-agnostic: the inner work stays loop-based (the constant LOC above already rules out a
  # per-iteration unroll), and the assembly renders as a for-loop under either renderer. The legacy
  # renderer uses a `static const int tile`/`idx` gather table; Program IR uses a const index buffer.
  assert "for (" in src_b
  assert "static const int tile" in src_b or "static const int idx" in src_b or "static const int64_t" in src_b


def test_map_accepts_input_dict_keyed_by_name() -> None:
  N = 4
  z = al.sym("z", 3 * N)
  p = al.sym("p", 3 * N)
  positional = al.map_(scale_add, N, [(z, 0, 3), (p, 0, 3)])
  by_name = al.scan(scale_add, length=N, inputs={"x": (z, 0, 3), "p": (p, 0, 3)})
  assert positional.structurally_equal(by_name)

  with pytest.raises(ValueError, match="unknown callee input names"):
    al.scan(scale_add, length=N, inputs={"x": (z, 0, 3), "q": (p, 0, 3)})
  with pytest.raises(ValueError, match="missing entries for callee inputs"):
    al.scan(scale_add, length=N, inputs={"x": (z, 0, 3)})


# --- pairwise-barrier composition: gather-fed MAP, MAP -> gather -> MAP, concat of two MAPs ------
# Shape of a centralized one-step barrier filter: one MAP advances every body, index tables gather
# the (i, j) pair operands out of that MAP's output, a second MAP evaluates the pairwise rows, and a
# third MAP evaluates the per-body rows. `slack` rides along as a stride-0 broadcast argument.

NB, NS, NU = 3, 3, 2
PAIRS = [(i, j) for i in range(NB) for j in range(i + 1, NB)]


@al.function("pairs_step", {"s": NS, "u": NU})
def pairs_step(s, u):
  return {"next": al.stack([s[0] + 0.1 * s[2].cos() * u[0], s[1] + 0.1 * s[2].sin() * u[1], s[2] + 0.1 * (u[0] - u[1])])}


@al.function("pairs_barrier", {"prev_i": NS, "prev_j": NS, "si": NS, "sj": NS, "slack": 1})
def pairs_barrier(prev_i, prev_j, si, sj, slack):
  # The trailing p-norm term mirrors the smooth-max a velocity-margin barrier uses; it is what
  # brings integer POW, a *non-integer* POW (the shape of such a barrier's braking envelope,
  # `c * d^q`) and a nested sqrt into the second-order path through the MAP. As in the real
  # barrier, the non-integer power's base is bounded away from zero, where its slope diverges.
  d, dprev = si[:2] - sj[:2], prev_i[:2] - prev_j[:2]
  envelope = 1.1 * (1.0 + al.dot(dprev, dprev)) ** 0.84
  soft_max = (al.dot(d, d) ** 2 + envelope**4).sqrt().sqrt()
  return {"h": al.stack([(al.dot(d, d).sqrt() - 0.5 * (1.0 + al.dot(dprev, dprev)).log() + soft_max + slack[0]).scalar()])}


@al.function("pairs_wall", {"s": NS, "snext": NS, "slack": 1})
def pairs_wall(s, snext, slack):
  return {"h": al.stack([(snext[0] - 0.5 * s[0] + slack[0]).scalar(), (1.0 - snext[1].exp() + slack[0]).scalar()])}


def _pair_index_table(bodies: list[int]) -> np.ndarray:
  return np.concatenate([np.arange(NS, dtype=np.int64) + k * NS for k in bodies])


def _build_pairs_fn(mapped: bool) -> al.Function:
  u = al.sym("u", NU * NB + 1)
  p = al.sym("p", NS * NB, diff=False)
  uu, slack = u[: NU * NB], u[NU * NB : NU * NB + 1]
  if mapped:
    nxt = al.map_(pairs_step, NB, [(p, 0, NS), (uu, 0, NU)])
    idx_i, idx_j = _pair_index_table([i for i, _ in PAIRS]), _pair_index_table([j for _, j in PAIRS])
    pair_rows = al.map_(
      pairs_barrier,
      len(PAIRS),
      [(al.gather(p, idx_i), 0, NS), (al.gather(p, idx_j), 0, NS), (al.gather(nxt, idx_i), 0, NS), (al.gather(nxt, idx_j), 0, NS), (slack, 0, 0)],
    )
    body_rows = al.map_(pairs_wall, NB, [(p, 0, NS), (nxt, 0, NS), (slack, 0, 0)])
    h = al.concat([pair_rows, body_rows])
  else:
    nxt = al.concat([pairs_step.call([p[NS * k : NS * (k + 1)], uu[NU * k : NU * (k + 1)]])[0] for k in range(NB)])
    sl = lambda e, k: e[NS * k : NS * (k + 1)]  # noqa: E731
    rows = [pairs_barrier.call([sl(p, i), sl(p, j), sl(nxt, i), sl(nxt, j), slack])[0] for i, j in PAIRS]
    rows += [pairs_wall.call([sl(p, k), sl(nxt, k), slack])[0] for k in range(NB)]
    h = al.concat(rows)
  return al.Function(f"pairs_{'map' if mapped else 'unroll'}", [u, p], [h.scalar()], ["u", "p"], ["h"])


def _pairs_sample() -> tuple[np.ndarray, np.ndarray]:
  rng = np.random.default_rng(11)
  uv = rng.normal(scale=0.3, size=NU * NB + 1)
  uv[-1] = 0.05
  return uv, rng.normal(scale=0.5, size=NS * NB) + np.tile([1.0, 2.0, 0.3], NB)


def test_gather_fed_chained_maps_match_unrolled_calls() -> None:
  fn_map, fn_unroll = _build_pairs_fn(True), _build_pairs_fn(False)
  uv, pv = _pairs_sample()
  np.testing.assert_allclose(fn_map(uv, pv), fn_unroll(uv, pv), rtol=1e-12, atol=1e-12)

  jac_map = fn_map.factory("pairs_map_jac", ["u", "p"], ["jac:h:u"])(uv, pv)
  jac_unroll = fn_unroll.factory("pairs_unroll_jac", ["u", "p"], ["jac:h:u"])(uv, pv)
  np.testing.assert_allclose(jac_map, jac_unroll, rtol=1e-10, atol=1e-10)


def test_gather_fed_chained_maps_spjac_and_sphess_match_dense() -> None:
  fn = _build_pairs_fn(True)
  uv, pv = _pairs_sample()
  dense = fn.factory("pairs_map_jac2", ["u", "p"], ["jac:h:u"])(uv, pv)
  assert isinstance(dense, np.ndarray)
  spjf = fn.factory("pairs_map_spjac", ["u", "p"], ["spjac:h:u"])
  sp = spjf.output_sparsities[0]
  assert sp is not None
  flat = np.asarray(sp.rows) * dense.shape[1] + np.asarray(sp.cols)
  np.testing.assert_allclose(spjf(uv, pv), np.ravel(dense)[flat], rtol=1e-10, atol=1e-10)
  # Coloring must not claim structural zeros that the dense Jacobian disagrees with.
  assert not np.any(np.abs(dense[~sp.to_mask()]) > 1e-12)

  # Second order through the same composition, against the unrolled reference.
  lam = np.arange(1.0, fn.outputs[0].shape[0] + 1.0)
  hess = {
    name: fn_.factory(f"pairs_{name}_sphess", ["u", "p", "lam:h"], ["sphess:gamma:u:u"], aux={"gamma": ["h"]})
    for name, fn_ in (("map", fn), ("unroll", _build_pairs_fn(False)))
  }
  dense_hess = {}
  for name, hf in hess.items():
    hsp = hf.output_sparsities[0]
    assert hsp is not None
    dense_hess[name] = np.zeros(hsp.shape)
    dense_hess[name][np.asarray(hsp.rows), np.asarray(hsp.cols)] = np.asarray(hf(uv, pv, lam), dtype=np.float64).reshape(-1)
  np.testing.assert_allclose(dense_hess["map"], dense_hess["unroll"], rtol=1e-9, atol=1e-9)


def test_block_lowered_matmul_inside_map_callee_infers_block_and_differentiates() -> None:
  """A dense layer inside a MAP callee — the shape of an MLP-in-the-loop dynamics model. A MATMUL
  with two block-marked operands infers `block` itself, while the scalar assembly wrapped around
  the MAP stays `scalar`. Marking a formal directly (`s.block()`) would clone the INPUT node, so
  the input side is marked on the packed vector, as real fixtures do."""
  from alloy.expr import topo

  w = np.array([[0.4, -0.2, 0.7], [0.1, 0.9, -0.3]])
  b = np.array([0.05, -0.15])

  @al.function("map_block_layer", {"s": 3})
  def layer(s):
    phi = al.stack([s[0], s[1], s[2]]).block()
    h = (al.const(w).block() @ phi + al.const(b)).block()
    return {"y": al.stack([(h * h).sum().scalar()])}

  assert any(node.op == al.Ops.MATMUL and node.lowering == "block" for node in topo(layer.outputs))

  N = 4
  z = al.sym("z", 3 * N)
  fn = al.Function("map_block", [z], [al.map_(layer, N, [(z, 0, 3)]).scalar()], ["z"], ["y"])
  assert fn.outputs[0].lowering == "scalar"

  zv = np.random.default_rng(3).normal(size=3 * N)
  expected = np.zeros((N, 3 * N))
  for k in range(N):
    h = w @ zv[3 * k : 3 * k + 3] + b
    expected[k, 3 * k : 3 * k + 3] = 2.0 * h @ w
  np.testing.assert_allclose(fn.factory("map_block_jac", ["z"], ["jac:y:z"])(zv), expected, rtol=1e-10, atol=1e-10)
