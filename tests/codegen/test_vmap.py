from __future__ import annotations

import ctypes
import re
import shutil
import subprocess
import sys

import numpy as np
import pytest

import scaly as sc


@sc.function(sc.G(sc.L("x", 3), sc.L("p", 3)), output=sc.L("y", ...), name="scale_add")
def scale_add(inputs):
  x, p = inputs
  return 2.0 * x + p


def test_vmap_c_source_loop_size_is_independent_of_length() -> None:
  from scaly.codegen import render_c_source

  def render(N: int) -> str:
    z = sc.sym("z", 3 * N)
    p = sc.sym("p", 3 * N)
    fn = sc.Function.from_exprs(f"scale_vmap_{N}", [z, p], [sc.vmap(scale_add, N, [(z, 0, 3), (p, 0, 3)])], ["z", "p"], ["y"])
    return render_c_source(fn)

  # Past the 32-element threshold where the trailing copy loop also folds, the rendered source
  # must be identical except for the loop bound and the function name.
  src_a = render(20)
  src_b = render(100)
  # Renderer-agnostic: the VMAP body is one loop whose bound scales with N (not unrolled) and the
  # source LOC stays constant in N. The legacy renderer emits `for (int it = 0; it < N; ++it)`,
  # the Program IR renderer `for (long long it_y = 0; it_y < N; ++it_y)` — both carry the `< N;` bound.
  assert "< 20;" in src_a
  assert "< 100;" in src_b
  assert src_a.count("\n") == src_b.count("\n")


def test_vmap_sparse_hessian_c_source_is_constant_in_length(monkeypatch: pytest.MonkeyPatch) -> None:
  from scaly.codegen import render_c_source
  from scaly.ir.expr import topo

  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  x = sc.sym("x", 2)
  hidden = sc.stack([x[0] * x[1], x[0] - 0.4 * x[1]])
  piece = sc.Function.from_exprs("vmap_sphess_codegen_piece", [x], [sc.stack([(hidden.tanh() ** 2).sum()])], ["x"], ["g"])

  def render(length: int) -> tuple[str, tuple[str, ...], int, dict[str, int]]:
    z = sc.sym("z", 2 * length)
    mapped = sc.vmap(piece, length, [(z, 0, 2)])
    base = sc.Function.from_exprs(f"vmap_sphess_codegen_base_{length}", [z], [(z * z).sum(), mapped], ["z"], ["f", "g"])
    sphess = base.factory(f"vmap_sphess_codegen_{length}", ["z", "lam:f", "lam:g"], [sc.factory.SpHess("gamma", "z")], aux={"gamma": ["f", "g"]})
    vmap_nodes = [node for node in topo(sphess.outputs) if node.op == sc.ExprOp.VMAP]
    mapped_callees = sorted({node.attrs["callee"].name for node in vmap_nodes})
    second_order = tuple(name for name in mapped_callees if "_adj" in name and "_fwd" in name)
    source = render_c_source(sphess)
    copies = {name: source.count(f"{name.replace(':', '_')}_raw(") for name in mapped_callees}
    return source, second_order, len(vmap_nodes), copies

  rendered = [render(length) for length in (2, 8, 32)]
  assert len({source.count("\n") for source, _, _, _ in rendered}) == 1
  assert all(names for _, names, _, _ in rendered)
  assert len({names for _, names, _, _ in rendered}) == 1
  # The sphess graph's VMAP-node count is a property of (#formals x #local-color-groups), never of
  # the vmap length; every mapped callee (primal, adjoint, second-order) renders one definition and
  # a length-independent number of call sites.
  assert len({vmap_count for _, _, vmap_count, _ in rendered}) == 1
  assert len({tuple(sorted(copies.items())) for _, _, _, copies in rendered}) == 1
  for source, names, _, copies in rendered:
    for name, count in copies.items():
      assert count >= 2, f"{name} rendered without a call site"
    for name in names:
      c_name = name.replace(":", "_")
      assert copies[name] == 2  # one definition and one call in one VMAP loop
      assert len(re.findall(rf"for \([^\n]+\) \{{\n\s+{re.escape(c_name)}_raw\(", source)) == 1


def test_sparse_hessian_triangle_c_source_has_no_full_nnz_buffer(monkeypatch: pytest.MonkeyPatch) -> None:
  from scaly.codegen import render_c_source
  from scaly.ir.expr import topo

  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  piece_x = sc.sym("triangle_shared_piece_x", 2)
  shared = sc.sym("triangle_shared_piece_shared", 1)
  hidden = sc.stack([piece_x[0] * piece_x[1] + shared[0] * piece_x[0], piece_x[0] - 0.4 * piece_x[1] + shared[0] * piece_x[1]])
  piece = sc.Function.from_exprs(
    "triangle_shared_piece",
    [piece_x, shared],
    [sc.stack([(hidden.tanh() ** 2).sum()])],
    ["x", "shared"],
    ["g"],
  )
  length = 3
  z = sc.sym("triangle_shared_vmap_z", 2 * length + 1)
  mapped = sc.vmap(piece, length, [(z, 0, 2), (z, 2 * length, 0)])
  f = (z * z).sum()
  base = sc.Function.from_exprs("triangle_shared_vmap_base", [z], [f, mapped], ["z"], ["f", "g"])
  full = base.factory(
    "triangle_shared_vmap_full",
    ["z", "lam:f", "lam:g"],
    [sc.factory.SpHess("gamma", "z")],
    aux={"gamma": ["f", "g"]},
  )
  lower = base.factory(
    "triangle_shared_vmap_lower",
    ["z", "lam:f", "lam:g"],
    [sc.factory.SpHess("gamma", "z", triangle="lower")],
    aux={"gamma": ["f", "g"]},
  )
  full_sp, lower_sp = full.output_sparsities[0], lower.output_sparsities[0]
  assert full_sp is not None and lower_sp is not None
  assert lower_sp.nnz < full_sp.nnz
  shared_index = 2 * length
  assert any(row == shared_index and col < shared_index for row, col in zip(full_sp.rows, full_sp.cols, strict=True))
  assert any(node.op == sc.ExprOp.VMAP for node in topo(lower.outputs))

  source = render_c_source(lower)
  declaration_lengths = {
    int(match[1]) for match in re.finditer(r"^\s*(?:static\s+)?(?:const\s+)?(?:double|int64_t)\s+\w+\[(\d+)\]", source, re.MULTILINE)
  }
  assert full_sp.nnz not in declaration_lengths


def test_callee_formal_named_w_avoids_workspace_collision() -> None:
  # The rendered callee signature appends the `double* w` workspace tail; a formal named `w` used
  # to redefine that parameter and fail to compile.
  x, w = sc.sym("x", 3), sc.sym("w", 3)
  piece = sc.Function.from_exprs("w_name_piece", [x, w], [x * w + w.sin()], ["x", "w"], ["y"])
  z, wv = sc.sym("z", 6), sc.sym("w", 3)
  fn = sc.Function.from_exprs("w_name_vmap", [z, wv], [sc.vmap(piece, 2, [(z, 0, 3), (wv, 0, 0)])], ["z", "w"], ["y"])
  zval = np.arange(6.0)
  wval = np.array([0.3, -0.2, 0.8])
  expected = np.concatenate([zval[3 * i : 3 * i + 3] * wval + np.sin(wval) for i in range(2)])
  np.testing.assert_allclose(fn((zval, wval)), expected)


def test_vmap_compiled_c_matches_unrolled_concat(tmp_path) -> None:
  cc = shutil.which("cc")
  if cc is None:
    pytest.skip("cc is required for generated C smoke test")

  from scaly.codegen import render_c_module

  N = 5
  z = sc.sym("z", 3 * N)
  p = sc.sym("p", 3 * N)
  fn = sc.Function.from_exprs("scale_vmap_compiled", [z, p], [sc.vmap(scale_add, N, [(z, 0, 3), (p, 0, 3)])], ["z", "p"], ["y"])
  module = render_c_module(fn)
  (tmp_path / module.header_name).write_text(module.header)
  source = tmp_path / module.source_name
  source.write_text(module.source)
  lib_path = tmp_path / ("libvmap.dylib" if sys.platform == "darwin" else "libvmap.so")
  cmd = [cc, "-fPIC", str(source), "-lm", "-o", str(lib_path)]
  cmd.insert(1, "-dynamiclib" if sys.platform == "darwin" else "-shared")
  subprocess.run(cmd, check=True)

  lib = ctypes.CDLL(str(lib_path))
  c_double_p = ctypes.POINTER(ctypes.c_double)
  lib.scale_vmap_compiled.argtypes = [
    ctypes.POINTER(c_double_p),
    ctypes.POINTER(c_double_p),
    ctypes.POINTER(ctypes.c_int),
    c_double_p,
    ctypes.c_int,
  ]
  lib.scale_vmap_compiled.restype = ctypes.c_int

  rng = np.random.default_rng(0)
  zv = rng.normal(size=3 * N)
  pv = rng.normal(size=3 * N)

  z_buf = (ctypes.c_double * (3 * N))(*zv)
  p_buf = (ctypes.c_double * (3 * N))(*pv)
  y_buf = (ctypes.c_double * (3 * N))()
  w_buf = (ctypes.c_double * max(module.workspace_size, 1))()
  args = (c_double_p * 2)(ctypes.cast(z_buf, c_double_p), ctypes.cast(p_buf, c_double_p))
  res = (c_double_p * 1)(ctypes.cast(y_buf, c_double_p))

  assert lib.scale_vmap_compiled(args, res, None, w_buf, 0) == 0
  np.testing.assert_allclose(np.array(y_buf), 2.0 * zv + pv)


RK4_NX, RK4_NU, RK4_NZ, RK4_N_PARAMS = 4, 2, 6, 7


def _rk4_bicycle_eq_vmap(horizon: int) -> sc.Function:
  """VMAP-based RK4 bicycle stage transcription with a symbolic parameter tail in ``p``.

  Self-contained on purpose: this is the realistic shape that produces a piece-ordered
  (non-row-major) COO sparsity and exercises the transposed-concat peephole, and the tests
  below must keep covering it whether or not any benchmark problem still uses it.
  """
  n_param = RK4_NX * (horizon + 1) + RK4_N_PARAMS

  def ode(x, u, params):
    wheelbase, _, mass, c_m0, c_r0, c_r1, c_r2 = [params[i] for i in range(RK4_N_PARAMS)]
    beta = 0.5 * u[1]
    vx = x[3] * beta.cos()
    return sc.stack(
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

  @sc.function(sc.G(sc.L("z", RK4_NZ), sc.L("p", RK4_NX)), output=sc.L("eq", ...), name="rk4_bicycle_initial")
  def eq_initial(inputs):
    z, p = inputs
    return z[:RK4_NX] - p[:RK4_NX]

  @sc.function(sc.G(sc.L("z", RK4_NZ), sc.L("znext", RK4_NZ), sc.L("params", RK4_N_PARAMS)), output=sc.L("eq", ...), name="rk4_bicycle_interstage")
  def eq_interstage(inputs):
    z, znext, params = inputs
    return rk4(z[:RK4_NX], z[RK4_NX : RK4_NX + RK4_NU], params) - znext[:RK4_NX]

  z = sc.sym("z", RK4_NZ * (horizon + 1))
  p = sc.sym("p", n_param, diff=False)
  initial = eq_initial((z[:RK4_NZ], p[:RK4_NX]))
  mapped = sc.vmap(
    eq_interstage,
    length=horizon,
    inputs={"z": (z, 0, RK4_NZ), "znext": (z, RK4_NZ, RK4_NZ), "params": (p, RK4_NX * (horizon + 1), 0)},
  )
  return sc.Function.from_exprs(f"rk4_bicycle_eq_vmap_N{horizon}", [z, p], [sc.concat([initial, mapped])], ["z", "p"], ["eq"])


def test_csr_csc_header_tables_carry_value_perm_for_non_row_major_coo() -> None:
  """The rendered header's CSR/CSC index tables are sorted, but the compact value buffer stays in
  COO order — piece-ordered on the merged vmap path, i.e. NOT row-major — so the header must also
  emit ``*_csr_val_perm`` / ``*_csc_val_perm`` tables (``values_csr[k] = values[csr_val_perm[k]]``).
  Reconstructing the dense Jacobian through both compressed formats must match the COO scatter."""

  import re

  from scaly.codegen.aot import render_c_module

  N = 3
  fn = _rk4_bicycle_eq_vmap(N)
  spjf = fn.factory(f"trk_vmap_valperm_N{N}", ["z", "p"], [sc.factory.SpJac("eq", "z")])
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
  values = np.asarray(spjf((zv, pv)), dtype=np.float64).reshape(-1)
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


def test_spjac_keeps_constant_loc_on_rk4_race_car_vmap() -> None:
  """End-to-end: even on the realistic RK4 race-car case (no periodic coloring), the rendered spjac
  C source stays at constant LOC across horizons because the transposed-concat peephole now emits
  per-block loops with a static `idx[]` table when the per-block group is large."""

  from scaly.codegen import render_c_source

  def loc(N: int) -> int:
    fn = _rk4_bicycle_eq_vmap(N)
    spj = fn.factory(f"rk4_bicycle_eq_vmap_N{N}_spjac_eq_z", ["z", "p"], [sc.factory.SpJac("eq", "z")])
    return render_c_source(spj).count("\n")

  loc_a = loc(10)
  loc_b = loc(50)
  # LOC is bounded by a tiny constant — variation comes only from whether the workspace
  # spill threshold is crossed, which adds one wrapper line for the SZ_W null check.
  assert abs(loc_a - loc_b) <= 2, f"expected constant RK4 race-car-vmap LOC, got {loc_a} -> {loc_b}"


def test_simple_banded_vmap_spjac_has_constant_loc() -> None:
  """Simple banded VMAP spjac C source stays at constant LOC across N. With the structured
  VMAP-aware sparse Jacobian, the loop comes from per-formal const-seed JVPs wrapped in VMAP,
  plus a constant scatter index table; without it, the tile-strided gather peephole would
  fire instead. Either way the LOC must not grow with N."""

  from scaly.codegen import render_c_source

  NX, NZ = 4, 6

  @sc.function(sc.L("z", NZ), output=sc.L("eq", ...), name="eq_initial_t")
  def eq_initial(z):
    return z[:NX] * 2.0

  @sc.function(sc.G(sc.L("z", NZ), sc.L("znext", NZ)), output=sc.L("eq", ...), name="eq_interstage_t")
  def eq_interstage(inputs):
    z, znext = inputs
    return z[:NX] * 1.5 - znext[:NX]

  def build(N: int) -> sc.Function:
    z = sc.sym("z", NZ * (N + 1))
    initial = eq_initial(z[:NZ])
    mapped = sc.vmap(eq_interstage, length=N, inputs={"z": (z, 0, NZ), "znext": (z, NZ, NZ)})
    return sc.Function.from_exprs(f"banded_N{N}", [z], [sc.concat([initial, mapped])], ["z"], ["eq"])

  loc_a = render_c_source(sc.sparse_jacobian(build(10), "eq", "z")).count("\n")
  loc_b = render_c_source(sc.sparse_jacobian(build(50), "eq", "z")).count("\n")
  # LOC is bounded by a tiny constant — variation comes only from whether the workspace
  # spill threshold is crossed, which adds one wrapper line for the SZ_W null check.
  assert abs(loc_a - loc_b) <= 2, f"expected constant LOC, got {loc_a} -> {loc_b}"
  src_b = render_c_source(sc.sparse_jacobian(build(50), "eq", "z"))
  # The inner work stays loop-based (the constant LOC above already rules out a per-iteration
  # unroll) and the assembly renders as a for-loop.
  assert "for (" in src_b


def test_vmap_jit_matches_unrolled_numpy() -> None:
  N = 2
  z = sc.sym("z", 6)
  p = sc.sym("p", 6)
  mapped = sc.vmap(scale_add, N, [(z, 0, 3), (p, 0, 3)])
  fn = sc.Function.from_exprs("eval_path", [z, p], [mapped], ["z", "p"], ["y"])
  zv = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  pv = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
  expected = np.concatenate([2.0 * zv[i * 3 : (i + 1) * 3] + pv[i * 3 : (i + 1) * 3] for i in range(N)])
  np.testing.assert_allclose(fn((zv, pv)), expected)
