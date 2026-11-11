from __future__ import annotations

import re

import numpy as np
import pytest

from scaly.function.model import as_concrete
from scaly.function.sugar import _mapped_call
import scaly as sc
from scaly.codegen.toolchain import LaneCount


@pytest.mark.parametrize("lanes", [1, "auto"])
def test_vmap_sparse_hessian_c_source_is_constant_in_length(lanes: LaneCount) -> None:
  from scaly.codegen import render_c_source
  from scaly.ir.expr import topo

  @sc.function(sc.arg("x", 2), outputs=sc.arg("g"), name="vmap_sphess_codegen_piece")
  def piece(x: sc.Expr) -> sc.Expr:
    hidden = sc.stack([x[0] * x[1], x[0] - 0.4 * x[1]])
    return sc.stack([(hidden.tanh() ** 2).sum()])

  def render(length: int) -> tuple[str, tuple[str, ...], int, dict[str, int]]:
    @sc.function(sc.arg("z", 2 * length), outputs=sc.group(sc.arg("f"), sc.arg("g")), name=f"vmap_sphess_codegen_base_{length}")
    def base(z: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
      return ((z * z).sum(), _mapped_call(piece, length, [(z, 0, 2)]))

    sphess = base.factory(f"vmap_sphess_codegen_{length}", ["z", "lam:f", "lam:g"], [sc.factory.SpHess("gamma", "z")], aux={"gamma": ["f", "g"]})
    vmap_nodes = [node for node in topo(as_concrete(sphess).outputs) if node.op == sc.ExprOp.VMAP]
    mapped_callees = sorted({node.attrs["callee"].name for node in vmap_nodes})
    second_order = tuple(name for name in mapped_callees if "_adj" in name and "_fwd" in name)
    source = render_c_source(sphess, lanes=lanes)
    copies = {name: source.count(f"{name.replace(':', '_')}_raw(") for name in mapped_callees}
    return source, second_order, len(vmap_nodes), copies

  rendered = [render(length) for length in (2, 8, 32)]
  assert len({source.count("\n") for source, _, _, _ in rendered}) == 1
  assert all(names for _, names, _, _ in rendered)
  assert len({names for _, names, _, _ in rendered}) == 1
  assert len({vmap_count for _, _, vmap_count, _ in rendered}) == 1
  for source, _, _, copies in rendered:
    assert all(count == 0 for count in copies.values())
    if lanes == 1:
      assert source.count("for (") == 1
    else:
      assert "SCALY_LANES" in source
  assert len({source.count("for (") for source, _, _, _ in rendered}) == 1


def test_sparse_hessian_triangle_c_source_has_no_full_nnz_buffer() -> None:
  from scaly.codegen import render_c_source
  from scaly.ir.expr import topo

  @sc.function(sc.group(sc.arg("x", 2), sc.arg("shared", 1)), outputs=sc.arg("g"), name="triangle_shared_piece")
  def piece(inputs: tuple[sc.Expr, sc.Expr]) -> sc.Expr:
    piece_x, shared = inputs
    hidden = sc.stack([piece_x[0] * piece_x[1] + shared[0] * piece_x[0], piece_x[0] - 0.4 * piece_x[1] + shared[0] * piece_x[1]])
    return sc.stack([(hidden.tanh() ** 2).sum()])

  length = 3

  @sc.function(sc.arg("z", 2 * length + 1), outputs=sc.group(sc.arg("f"), sc.arg("g")), name="triangle_shared_vmap_base")
  def base(z: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
    return ((z * z).sum(), _mapped_call(piece, length, [(z, 0, 2), (z, 2 * length, 0)]))

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
  full_sp, lower_sp = as_concrete(full).outputs[0].type.pattern, as_concrete(lower).outputs[0].type.pattern
  assert full_sp is not None and lower_sp is not None
  assert lower_sp.nnz < full_sp.nnz
  shared_index = 2 * length
  assert any(row == shared_index and col < shared_index for row, col in zip(full_sp.rows, full_sp.cols, strict=True))
  assert any(node.op == sc.ExprOp.VMAP for node in topo(as_concrete(lower).outputs))

  source = render_c_source(lower)
  declaration_lengths = {
    int(match[1]) for match in re.finditer(r"^\s*(?:static\s+)?(?:const\s+)?(?:double|int64_t)\s+\w+\[(\d+)\]", source, re.MULTILINE)
  }
  assert full_sp.nnz not in declaration_lengths


def test_callee_formal_named_w_avoids_workspace_collision() -> None:
  # The rendered callee signature appends the `double* w` workspace tail; a formal named `w` used
  # to redefine that parameter and fail to compile.
  @sc.function(sc.group(sc.arg("x", 3), sc.arg("w", 3)), outputs=sc.arg("y"), name="w_name_piece")
  def piece(inputs: tuple[sc.Expr, sc.Expr]) -> sc.Expr:
    x, w = inputs
    return x * w + w.sin()

  @sc.function(sc.group(sc.arg("z", 6), sc.arg("w", 3)), outputs=sc.arg("y", 6), name="w_name_vmap")
  def fn(inputs: tuple[sc.Expr, sc.Expr]) -> sc.Expr:
    z, wv = inputs
    return _mapped_call(piece, 2, [(z, 0, 3), (wv, 0, 0)])

  zval = np.arange(6.0)
  wval = np.array([0.3, -0.2, 0.8])
  expected = np.concatenate([zval[3 * i : 3 * i + 3] * wval + np.sin(wval) for i in range(2)])
  np.testing.assert_allclose(fn((zval, wval)), expected)


RK4_NX, RK4_NU, RK4_NZ, RK4_N_PARAMS = 4, 2, 6, 7


def _rk4_bicycle_eq_vmap(horizon: int) -> sc.Function:
  """VMAP-based RK4 bicycle stage transcription with a symbolic parameter tail in ``p``.

  Self-contained on purpose: this is the realistic shape that produces a piece-ordered
  (non-row-major) COO sparsity and exercises the transposed-concat peephole, and the tests
  below must keep covering it whether or not any benchmark problem still uses it.
  """
  n_param = RK4_NX * (horizon + 1) + RK4_N_PARAMS

  def ode(x: sc.Expr, u: sc.Expr, params: sc.Expr) -> sc.Expr:
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

  def rk4(x: sc.Expr, u: sc.Expr, params: sc.Expr) -> sc.Expr:
    dt = params[1]
    k1 = ode(x, u, params)
    k2 = ode(x + dt / 2 * k1, u, params)
    k3 = ode(x + dt / 2 * k2, u, params)
    k4 = ode(x + dt * k3, u, params)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

  @sc.function(sc.group(sc.arg("z", RK4_NZ), sc.arg("p", RK4_NX)), outputs=sc.arg("eq"), name="rk4_bicycle_initial")
  def eq_initial(inputs: tuple[sc.Expr, sc.Expr]) -> sc.Expr:
    z, p = inputs
    return z[:RK4_NX] - p[:RK4_NX]

  @sc.function(
    sc.group(sc.arg("z", RK4_NZ), sc.arg("znext", RK4_NZ), sc.arg("params", RK4_N_PARAMS)), outputs=sc.arg("eq"), name="rk4_bicycle_interstage"
  )
  def eq_interstage(inputs: tuple[sc.Expr, sc.Expr, sc.Expr]) -> sc.Expr:
    z, znext, params = inputs
    return rk4(z[:RK4_NX], z[RK4_NX : RK4_NX + RK4_NU], params) - znext[:RK4_NX]

  @sc.function(
    sc.group(sc.arg("z", RK4_NZ * (horizon + 1)), sc.arg("p", sc.TensorType((n_param,), diff=False))),
    outputs=sc.arg("eq"),
    name=f"rk4_bicycle_eq_vmap_N{horizon}",
  )
  def eq(inputs: tuple[sc.Expr, sc.Expr]) -> sc.Expr:
    z, p = inputs
    initial = eq_initial((z[:RK4_NZ], p[:RK4_NX]))
    mapped = _mapped_call(
      eq_interstage,
      length=horizon,
      inputs=[(z, 0, RK4_NZ), (z, RK4_NZ, RK4_NZ), (p, RK4_NX * (horizon + 1), 0)],
    )
    return sc.concat([initial, mapped])

  return eq


def test_mapped_spjac_values_follow_the_header_tables_of_their_layout() -> None:
  """On the merged map path the values come straight out in the declared layout, so walking the
  header's compressed tables over the value buffer rebuilds the dense Jacobian with no permutation."""

  import re

  from scaly.codegen.aot import render_c_module
  from scaly.ir.expr import topo

  N, layout = 3, "csc"
  fn = _rk4_bicycle_eq_vmap(N)
  spjf = fn.factory(f"trk_vmap_{layout}_N{N}", ["z", "p"], [sc.factory.SpJac("eq", "z", layout=layout)])
  dense_fn = fn.factory(f"trk_vmap_dense_N{N}", ["z", "p"], [sc.factory.Jac("eq", "z")])
  assert any(node.op == sc.ExprOp.VMAP for node in topo(as_concrete(spjf).outputs))
  module = render_c_module(spjf, header_name="t.h", source_name="t.c")
  assert "val_perm" not in module.header

  def table(name: str) -> np.ndarray:
    m = re.search(rf"_{name}\[\d+\] = \{{([^}}]*)\}};", module.header)
    assert m is not None, f"header table {name} missing"
    return np.array([int(x) for x in m.group(1).split(",")], dtype=np.int64)

  rng = np.random.default_rng(3)
  zv, pv = rng.normal(size=RK4_NZ * (N + 1)), rng.normal(size=RK4_NX * (N + 1) + RK4_N_PARAMS)
  result = spjf(*(zv, pv))
  dense_ref = np.asarray(dense_fn(*(zv, pv)))
  ptr, ind = (table("csr_row_ptr"), table("csr_col_ind")) if layout == "csr" else (table("csc_col_ptr"), table("csc_row_ind"))
  rebuilt = np.zeros(dense_ref.shape)
  for outer in range(len(ptr) - 1):
    for k in range(ptr[outer], ptr[outer + 1]):
      rebuilt[(outer, ind[k]) if layout == "csr" else (ind[k], outer)] = result.data[k]
  np.testing.assert_allclose(rebuilt, dense_ref, rtol=1e-12, atol=1e-12)
  np.testing.assert_allclose(result.toarray(), dense_ref, rtol=1e-12, atol=1e-12)


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
  # The line count can change when spilling adds the SZ_W null check to the wrapper.
  assert abs(loc_a - loc_b) <= 2, f"expected constant RK4 race-car-vmap LOC, got {loc_a} -> {loc_b}"
