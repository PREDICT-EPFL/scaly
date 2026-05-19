from __future__ import annotations

import ctypes
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


def test_map_recursive_eval_path_matches_tape() -> None:
  N = 2
  z = al.sym("z", 6)
  p = al.sym("p", 6)
  mapped = al.map_(scale_add, N, [(z, 0, 3), (p, 0, 3)])
  fn = al.Function("eval_path", [z, p], [mapped], ["z", "p"], ["y"])
  zv = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  pv = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
  np.testing.assert_allclose(mapped.eval({"z": zv, "p": pv}), fn(zv, pv))


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
  z = al.sym("z", 6)
  p = al.sym("p", 6)
  a = al.map_(scale_add, 2, [(z, 0, 3), (p, 0, 3)])
  b = al.map_(scale_add, 2, [(z, 0, 3), (p, 0, 3)])
  assert a.id != b.id
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
  assert "for (int it = 0; it < 20; ++it)" in src_a
  assert "for (int it = 0; it < 100; ++it)" in src_b
  assert src_a.count("\n") == src_b.count("\n")


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


def test_tracking_eq_primal_source_is_constant_in_horizon() -> None:
  """Rewriting the tracking fixture with `al.scan` yields C source whose size does not grow with N."""

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

  @al.function("tracking_eq_initial", {"z": NZ, "p": NX})
  def eq_initial(z, p):
    return {"eq": z[:NX] - p[:NX]}

  @al.function("tracking_eq_interstage", {"z": NZ, "znext": NZ, "p": NX})
  def eq_interstage(z, znext, p):
    return {"eq": rk4(z[:NX], z[NX : NX + NU]) - znext[:NX]}

  def build(N: int) -> al.Function:
    z = al.sym("z", NZ * (N + 1))
    p = al.sym("p", NX * (N + 1), diff=False)
    initial = eq_initial.call([z[:NZ], p[:NX]])[0]
    mapped = al.scan(eq_interstage, length=N, inputs={"z": (z, 0, NZ), "znext": (z, NZ, NZ), "p": (p, NX, NX)})
    return al.Function(f"tracking_eq_map_N{N}", [z, p], [al.concat([initial, mapped])], ["z", "p"], ["eq"])

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
    ref = al.Function(f"tracking_eq_ref_N{N}", [fn.inputs[0], fn.inputs[1]], [al.concat(parts)], ["z", "p"], ["eq"])
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


def test_spjacobian_of_tracking_map_matches_unrolled_concat() -> None:
  """End-to-end: spjacobian on a MAP-based tracking fixture matches the unrolled-concat fixture
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

  @al.function("tracking_eq_initial2", {"z": NZ, "p": NX})
  def eq_initial(z, p):
    return {"eq": z[:NX] - p[:NX]}

  @al.function("tracking_eq_interstage2", {"z": NZ, "znext": NZ, "p": NX})
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

  assert spj_map.output_sparsities[0] == spj_unroll.output_sparsities[0]

  rng = np.random.default_rng(2)
  zv = rng.normal(size=NZ * (N + 1))

  np.testing.assert_allclose(spj_map(zv), spj_unroll(zv), rtol=1e-10, atol=1e-10)


def test_spjac_keeps_constant_loc_on_rk4_tracking_map() -> None:
  """End-to-end: even on the realistic RK4 tracking case (no periodic coloring), the rendered spjac
  C source stays at constant LOC across horizons because the transposed-concat peephole now emits
  per-block loops with a static `idx[]` table when the per-block group is large."""

  import sys
  from pathlib import Path

  fixture_path = Path(__file__).parent / "test_tracking_workload.py"
  import importlib.util

  spec = importlib.util.spec_from_file_location("alloy_tracking_workload_for_loc", fixture_path)
  assert spec is not None and spec.loader is not None
  mod = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = mod
  spec.loader.exec_module(mod)

  from alloy.codegen import render_c_source

  def loc(N: int) -> int:
    fn = mod.tracking_eq_function_map(N)
    spj = al.spjacobian(fn, "z", "eq")
    return render_c_source(spj).count("\n")

  loc_a = loc(10)
  loc_b = loc(50)
  # LOC is bounded by a tiny constant — variation comes only from whether the workspace
  # spill threshold is crossed, which adds one wrapper line for the SZ_W null check.
  assert abs(loc_a - loc_b) <= 2, f"expected constant RK4 tracking-map LOC, got {loc_a} -> {loc_b}"


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
  # Either path keeps the inner work in for-loops, not a per-iteration unroll.
  assert "static const int tile" in src_b or "static const int idx" in src_b
  assert "for (int it = 0;" in src_b


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
