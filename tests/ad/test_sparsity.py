from __future__ import annotations

from typing import cast

import numpy as np
import pytest

import scaly as sc
from scaly.ad import finite_difference
from scaly.ad.sparse import Triangle


def _mapped_sphess_fixture(length: int, *, shared: bool = False) -> tuple[sc.Function, sc.Function]:
  x = sc.sym("x", 2)
  if shared:
    s = sc.sym("s", 1)
    hidden = sc.stack([x[0] * x[1] + s[0] * x[0], x[0] - 0.4 * x[1] + s[0] * x[1]])
    piece = sc.Function._from_exprs("mapped_sphess_shared_piece", [x, s], [sc.stack([(hidden.tanh() ** 2).sum()])], ["x", "s"], ["g"])
  else:
    hidden = sc.stack([x[0] * x[1], x[0] - 0.4 * x[1]])
    piece = sc.Function._from_exprs("mapped_sphess_piece", [x], [sc.stack([(hidden.tanh() ** 2).sum()])], ["x"], ["g"])

  z = sc.sym("z", 2 * length + int(shared))
  specs = [(z, 0, 2), *(((z, 2 * length, 0),) if shared else ())]
  mapped = sc.vmap(piece, length, specs)
  calls = []
  for it in range(length):
    args = [z[2 * it : 2 * (it + 1)], *((z[2 * length : 2 * length + 1],) if shared else ())]
    calls.append(piece(*piece.input_tree.unflatten(tuple(args))))
  unrolled = sc.concat(calls)
  f = (z * z).sum()
  return (
    sc.Function._from_exprs(f"mapped_sphess_{length}_{int(shared)}", [z], [f, mapped], ["z"], ["f", "g"]),
    sc.Function._from_exprs(f"unrolled_sphess_{length}_{int(shared)}", [z], [f, unrolled], ["z"], ["f", "g"]),
  )


def _scatter_sparse(values: np.ndarray, sparsity: sc.SparsityType) -> np.ndarray:
  dense = np.zeros(sparsity.shape)
  dense[np.asarray(sparsity.rows), np.asarray(sparsity.cols)] = values
  return dense


def test_sparsity_type_roundtrip_and_bounds() -> None:
  mask = np.array([[True, False, True], [False, True, False]])
  sp = sc.SparsityType.from_mask(mask)

  assert sp.shape == (2, 3)
  assert sp.nnz == 3
  np.testing.assert_array_equal(sp.to_mask(), mask)

  try:
    _ = sc.SparsityType((2, 2), (0, 2), (0, 1))
  except ValueError as e:
    assert "out of bounds" in str(e)
  else:  # pragma: no cover
    raise AssertionError("invalid sparsity should fail")

  try:
    _ = sc.SparsityType((2, 2), (0, 0), (1, 1))
  except ValueError as e:
    assert "sparsity indices must be unique" in str(e)
  else:  # pragma: no cover
    raise AssertionError("duplicate sparsity coordinates should fail")


def test_sparsity_type_csr_csc_conversions() -> None:
  sp = sc.SparsityType((3, 4), (2, 0, 1, 1), (3, 2, 0, 3))

  row_ptr, col_ind, csr_perm = sp.to_csr()
  assert row_ptr == (0, 1, 3, 4)
  assert col_ind == (2, 0, 3, 3)
  # val_perm maps CSR slot -> COO position: sorted (row, col) order of the COO pattern above.
  assert csr_perm == (1, 2, 3, 0)
  assert tuple((sp.rows[i], sp.cols[i]) for i in csr_perm) == ((0, 2), (1, 0), (1, 3), (2, 3))
  np.testing.assert_array_equal(sc.SparsityType.from_csr(sp.shape, row_ptr, col_ind).to_mask(), sp.to_mask())

  col_ptr, row_ind, csc_perm = sp.to_csc()
  assert col_ptr == (0, 1, 1, 2, 4)
  assert row_ind == (1, 0, 1, 2)
  assert csc_perm == (2, 1, 3, 0)
  assert tuple((sp.cols[i], sp.rows[i]) for i in csc_perm) == ((0, 1), (2, 0), (3, 1), (3, 2))
  np.testing.assert_array_equal(sc.SparsityType.from_csc(sp.shape, col_ptr, row_ind).to_mask(), sp.to_mask())


def test_sparsity_type_compressed_format_errors() -> None:
  try:
    _ = sc.SparsityType.from_csr((2, 3), (0, 1), (0,))
  except ValueError as e:
    assert "row_ptr must have length 3" in str(e)
  else:  # pragma: no cover
    raise AssertionError("short CSR row pointer should fail")

  try:
    _ = sc.SparsityType.from_csc((2, 3), (0, 2, 1, 1), (0,))
  except ValueError as e:
    assert "col_ptr must be nondecreasing" in str(e)
  else:  # pragma: no cover
    raise AssertionError("invalid CSC column pointer should fail")

  try:
    _ = sc.SparsityType.from_csr((2, 3), (0, 1, 1), (3,))
  except ValueError as e:
    assert "CSR column indices out of bounds" in str(e)
  else:  # pragma: no cover
    raise AssertionError("out-of-bounds CSR column should fail")


def test_jacobian_sparsity_for_structural_ops() -> None:
  x = sc.sym("x", 4)
  y = sc.stack([x[0], x[2:4].sum(), sc.gather(x, [1, 3]).sum()])
  sp = sc.jacobian_sparsity(y, x)

  np.testing.assert_array_equal(
    sp.to_mask(),
    np.array(
      [
        [True, False, False, False],
        [False, False, True, True],
        [False, True, False, True],
      ]
    ),
  )


def test_jacobian_sparsity_tracks_concat_axis_layout() -> None:
  x = sc.sym("x", (2, 2))
  y = sc.concat([x[:, :1], x[:, 1:2]], axis=1)
  sp = sc.jacobian_sparsity(y, x)

  np.testing.assert_array_equal(sp.to_mask(), np.eye(4, dtype=bool))


def test_sparse_hessian_factory_returns_compact_values_with_sparsity_metadata() -> None:
  x = sc.sym("x", 3)
  y = x[0] * x[0] + x[1] * x[2]
  f = sc.Function._from_exprs("f", [x], [y], ["x"], ["y"])
  shf = sc.sparse_hessian(f, "y", "x")
  xv = np.array([2.0, 3.0, 4.0])

  assert shf.output_names == ("sphess_y_x_x",)
  assert shf.output_coloring_widths == (2,)
  assert shf.output_sparsities[0] is not None
  assert shf.output_sparsities[0].rows == (0, 1, 2)
  assert shf.output_sparsities[0].cols == (0, 2, 1)
  np.testing.assert_allclose(shf(xv), np.array([2.0, 1.0, 1.0]))


@pytest.mark.parametrize("triangle", ("lower", "upper"))
def test_sparse_hessian_triangle_matches_masked_full(triangle: Triangle) -> None:
  x = sc.sym("triangle_x", 4)
  y = (x[0] * x[1]).sin() + (x[2] * x[3]).sin()
  full_expr = sc.sparse_hessian(y, x)
  triangle_expr = sc.sparse_hessian(y, x, triangle=triangle)
  selected_expr = full_expr.triangle(triangle)
  full_sp, triangle_sp = full_expr.sparsity, triangle_expr.sparsity
  keep = np.asarray(full_sp.rows) >= np.asarray(full_sp.cols) if triangle == "lower" else np.asarray(full_sp.rows) <= np.asarray(full_sp.cols)

  assert triangle_sp.rows == tuple(np.asarray(full_sp.rows)[keep])
  assert triangle_sp.cols == tuple(np.asarray(full_sp.cols)[keep])
  assert triangle_expr.coloring_width == selected_expr.coloring_width == full_expr.coloring_width
  assert selected_expr._compressed is full_expr._compressed
  np.testing.assert_array_equal(selected_expr._recovery, triangle_expr._recovery)
  value_fn = sc.Function._from_exprs(
    f"triangle_expr_{triangle}",
    [x],
    [full_expr.values, triangle_expr.values, selected_expr.values],
    ["triangle_x"],
    ["full", "triangle", "selected"],
  )
  full_values, triangle_values, selected_values = value_fn(np.array([0.2, 0.7, -0.3, 1.1]))
  np.testing.assert_allclose(triangle_values, full_values[keep], rtol=1e-10, atol=1e-10)
  np.testing.assert_allclose(selected_values, triangle_values, rtol=1e-10, atol=1e-10)

  fn = sc.Function._from_exprs("triangle_fn", [x], [y], ["triangle_x"], ["y"])
  full_fn = sc.sparse_hessian(fn, "y", "triangle_x", name=f"triangle_fn_full_{triangle}")
  triangle_fn = sc.sparse_hessian(fn, "y", "triangle_x", name=f"triangle_fn_{triangle}", triangle=triangle)
  assert full_fn.output_names == triangle_fn.output_names == ("sphess_y_triangle_x_triangle_x",)
  assert full_fn.output_coloring_widths == triangle_fn.output_coloring_widths
  full_fn_sp, triangle_fn_sp = full_fn.output_sparsities[0], triangle_fn.output_sparsities[0]
  assert full_fn_sp is not None and triangle_fn_sp is not None
  assert triangle_fn_sp.rows == tuple(np.asarray(full_fn_sp.rows)[keep])
  assert triangle_fn_sp.cols == tuple(np.asarray(full_fn_sp.cols)[keep])
  np.testing.assert_allclose(
    triangle_fn(np.array([0.2, 0.7, -0.3, 1.1])),
    np.asarray(full_fn(np.array([0.2, 0.7, -0.3, 1.1])))[keep],
    rtol=1e-10,
    atol=1e-10,
  )


def test_sparse_lagrangian_hessian_uses_aux_output() -> None:
  x = sc.sym("x", 2)
  f_expr = x[0] * x[0]
  g_expr = sc.stack([x[0] * x[1], x[1] * x[1]])
  nlp = sc.Function._from_exprs("nlp", [x], [f_expr, g_expr], ["x"], ["f", "g"])
  shf = sc.sparse_lagrangian_hessian(nlp, "x")

  assert shf.input_names == ("x", "lam:f", "lam:g")
  assert shf.output_names == ("sphess_gamma_x_x",)
  assert shf.output_coloring_widths == (2,)
  assert shf.output_sparsities[0] is not None
  assert shf.output_sparsities[0].rows == (0, 0, 1, 1)
  assert shf.output_sparsities[0].cols == (0, 1, 0, 1)
  np.testing.assert_allclose(shf(np.array([2.0, 3.0]), (np.array(1.5), np.array([0.25, -0.5]))), np.array([3.0, 0.25, 0.25, -1.0]))


def test_sparse_lagrangian_hessian_through_vmap_matches_unrolled_dense_and_fd(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  mapped, unrolled = _mapped_sphess_fixture(3)
  mapped_sphess = mapped.factory("mapped_sphess_exact", ["z", "lam:f", "lam:g"], [sc.factory.SpHess("gamma", "z")], aux={"gamma": ["f", "g"]})
  unrolled_sphess = unrolled.factory("unrolled_sphess_exact", ["z", "lam:f", "lam:g"], [sc.factory.SpHess("gamma", "z")], aux={"gamma": ["f", "g"]})
  unrolled_hess = unrolled.factory("unrolled_hess_dense", ["z", "lam:f", "lam:g"], [sc.factory.Hess("gamma", "z")], aux={"gamma": ["f", "g"]})
  unrolled_grad = unrolled.factory("unrolled_grad_for_fd", ["z", "lam:f", "lam:g"], [sc.factory.Grad("gamma", "z")], aux={"gamma": ["f", "g"]})

  mapped_sp, unrolled_sp = mapped_sphess.output_sparsities[0], unrolled_sphess.output_sparsities[0]
  assert mapped_sp is not None and unrolled_sp is not None
  np.testing.assert_array_equal(mapped_sp.to_mask(), unrolled_sp.to_mask())
  np.testing.assert_array_equal(mapped_sp.to_mask(), mapped_sp.to_mask().T)

  zv = np.array([-0.7, 0.2, 0.4, -0.5, 0.8, 0.3])
  lam_f, lam_g = np.array(0.6), np.array([0.3, -0.8, 1.1])
  mapped_dense = _scatter_sparse(np.asarray(mapped_sphess((zv, lam_f, lam_g))), mapped_sp)
  unrolled_dense = _scatter_sparse(np.asarray(unrolled_sphess((zv, lam_f, lam_g))), unrolled_sp)
  np.testing.assert_allclose(mapped_dense, unrolled_dense, rtol=1e-10, atol=1e-10)
  np.testing.assert_allclose(mapped_dense, unrolled_hess((zv, lam_f, lam_g)), rtol=1e-10, atol=1e-10)
  np.testing.assert_allclose(mapped_dense, finite_difference(lambda value: unrolled_grad((value, lam_f, lam_g)), zv), rtol=2e-5, atol=2e-6)


def test_mapped_sparse_hessian_multiplier_weighting_and_shared_fill(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  mapped, _ = _mapped_sphess_fixture(3)
  sphess = mapped.factory("mapped_sphess_weighting", ["z", "lam:f", "lam:g"], [sc.factory.SpHess("gamma", "z")], aux={"gamma": ["f", "g"]})
  sparsity = sphess.output_sparsities[0]
  assert sparsity is not None
  zv = np.array([-0.7, 0.2, 0.4, -0.5, 0.8, 0.3])
  lam_f, lam_g = np.array(0.6), np.array([0.3, -0.8, 1.1])
  base = _scatter_sparse(np.asarray(sphess((zv, lam_f, lam_g))), sparsity)
  changed_lam = lam_g.copy()
  changed_lam[1] += 0.7
  changed = _scatter_sparse(np.asarray(sphess((zv, lam_f, changed_lam))), sparsity)
  delta = changed - base
  np.testing.assert_allclose(delta[:2], 0.0, atol=1e-12)
  np.testing.assert_allclose(delta[4:], 0.0, atol=1e-12)
  assert np.any(np.abs(delta[2:4, 2:4]) > 1e-9)

  shared, shared_unrolled = _mapped_sphess_fixture(3, shared=True)
  shared_sphess = shared.factory("mapped_sphess_shared_fill", ["z", "lam:f", "lam:g"], [sc.factory.SpHess("gamma", "z")], aux={"gamma": ["f", "g"]})
  shared_unrolled_sphess = shared_unrolled.factory(
    "unrolled_sphess_shared_fill", ["z", "lam:f", "lam:g"], [sc.factory.SpHess("gamma", "z")], aux={"gamma": ["f", "g"]}
  )
  shared_sp, shared_unrolled_sp = shared_sphess.output_sparsities[0], shared_unrolled_sphess.output_sparsities[0]
  assert shared_sp is not None and shared_unrolled_sp is not None
  np.testing.assert_array_equal(shared_sp.to_mask(), shared_unrolled_sp.to_mask())
  shared_index = 6
  mask = shared_sp.to_mask()
  assert np.all(mask[shared_index, :shared_index])
  assert np.all(mask[:shared_index, shared_index])
  # Values, not just pattern: a wrong-iteration sum or wrong per-instance multiplier in the
  # stride-0 adjoint reduction would keep the same mask, so pin the numbers with nonuniform lam:g.
  zv_shared = np.array([-0.7, 0.2, 0.4, -0.5, 0.8, 0.3, 0.9])
  lam_g_shared = np.array([0.7, -1.3, 0.45])
  shared_dense = _scatter_sparse(np.asarray(shared_sphess((zv_shared, lam_f, lam_g_shared))), shared_sp)
  shared_unrolled_dense = _scatter_sparse(np.asarray(shared_unrolled_sphess((zv_shared, lam_f, lam_g_shared))), shared_unrolled_sp)
  np.testing.assert_allclose(shared_dense, shared_unrolled_dense, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("triangle", ("lower", "upper"))
def test_sparse_lagrangian_hessian_triangle_matches_masked_full_on_shared_vmap(monkeypatch: pytest.MonkeyPatch, triangle: Triangle) -> None:
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  mapped, unrolled = _mapped_sphess_fixture(3, shared=True)
  full = mapped.factory(
    "shared_triangle_full",
    ["z", "lam:f", "lam:g"],
    [sc.factory.SpHess("gamma", "z")],
    aux={"gamma": ["f", "g"]},
  )
  triangle_fn = mapped.factory(
    f"shared_triangle_{triangle}",
    ["z", "lam:f", "lam:g"],
    [sc.factory.SpHess("gamma", "z", triangle=triangle)],
    aux={"gamma": ["f", "g"]},
  )
  unrolled_triangle = unrolled.factory(
    f"unrolled_triangle_{triangle}",
    ["z", "lam:f", "lam:g"],
    [sc.factory.SpHess("gamma", "z", triangle=triangle)],
    aux={"gamma": ["f", "g"]},
  )
  full_sp, triangle_sp = full.output_sparsities[0], triangle_fn.output_sparsities[0]
  assert full_sp is not None and triangle_sp is not None
  rows, cols = np.asarray(full_sp.rows), np.asarray(full_sp.cols)
  keep = rows >= cols if triangle == "lower" else rows <= cols
  assert triangle_sp.rows == tuple(rows[keep])
  assert triangle_sp.cols == tuple(cols[keep])
  assert triangle_fn.output_coloring_widths == full.output_coloring_widths

  unrolled_sp = unrolled_triangle.output_sparsities[0]
  assert unrolled_sp is not None
  assert unrolled_sp.rows == triangle_sp.rows
  assert unrolled_sp.cols == triangle_sp.cols

  zv = np.array([-0.7, 0.2, 0.4, -0.5, 0.8, 0.3, 0.9])
  lam_f = np.array(0.6)
  lam_g = np.array([0.7, -1.3, 0.45])
  full_values = np.asarray(full((zv, lam_f, lam_g)))
  triangle_values = np.asarray(triangle_fn((zv, lam_f, lam_g)))
  unrolled_values = np.asarray(unrolled_triangle((zv, lam_f, lam_g)))
  np.testing.assert_allclose(triangle_values, full_values[keep], rtol=1e-10, atol=1e-10)
  np.testing.assert_allclose(triangle_values, unrolled_values, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("triangle", ("diagonal", None, 1))
def test_sparse_hessian_rejects_invalid_triangle(triangle: object) -> None:
  x = sc.sym("invalid_triangle_x", 2)
  y = x[0] * x[1]
  invalid_triangle = cast(Triangle, triangle)
  with pytest.raises(ValueError, match="triangle must be one of"):
    sc.sparse_hessian(y, x, triangle=invalid_triangle)
  fn = sc.Function._from_exprs("invalid_triangle", [x], [y], ["invalid_triangle_x"], ["y"])
  with pytest.raises(ValueError, match="triangle must be one of"):
    sc.sparse_hessian(fn, "y", "invalid_triangle_x", triangle=invalid_triangle)


def test_spjac_factory_returns_compact_values_with_sparsity_metadata() -> None:
  x = sc.sym("x", 4)
  y = sc.stack([x[0], x[2:4].sum(), x[1]])
  f = sc.Function._from_exprs("f", [x], [y], ["x"], ["y"])
  spjf = sc.sparse_jacobian(f, "y", "x")
  xv = np.array([1.0, 2.0, 3.0, 4.0])

  assert spjf.output_names == ("spjac_y_x",)
  assert spjf.output_coloring_widths == (2,)
  assert spjf.output_sparsities[0] is not None
  assert spjf.output_sparsities[0].rows == (0, 1, 1, 2)
  assert spjf.output_sparsities[0].cols == (0, 2, 3, 1)
  np.testing.assert_allclose(spjf(xv), np.ones(4))


def test_function_rejects_sparse_output_metadata_size_mismatch() -> None:
  x = sc.sym("x", 2)
  try:
    _ = sc.Function._from_exprs("bad", [x], [x], ["x"], ["sp"], output_sparsities=[sc.SparsityType.dense((2, 2))])
  except ValueError as e:
    assert "sparse output metadata for 'sp' has 4 nonzeros" in str(e)
    assert "output shape (2,) has 2 entries" in str(e)
  else:  # pragma: no cover
    raise AssertionError("sparse output metadata size mismatch should fail")


def test_sparse_jacobian_preserves_constructed_local_coloring_width() -> None:
  a = sc.sym("a", 1)
  b = sc.sym("b", 1)
  piece = sc.Function._from_exprs("two_formal_piece", [a, b], [sc.stack([a, b])], ["a", "b"], ["y"])
  z = sc.sym("z", 5)
  mapped_expr = sc.vmap(piece, 4, [(z, 0, 1), (z, 1, 1)])
  mapped = sc.Function._from_exprs("two_formal_mapped", [z], [mapped_expr], ["z"], ["y"])
  sj = sc.sparse_jacobian(mapped_expr, z)

  np.testing.assert_array_equal(sj.sparsity.to_mask(), sc.jacobian_sparsity(mapped_expr, z).to_mask())
  assert sj.coloring_width == 2
  assert max(sc.column_coloring(sj.sparsity), default=-1) + 1 == 1

  built = mapped.factory("two_formal_spjac", ["z"], [sc.factory.SpJac("y", "z")])
  assert built.output_coloring_widths == (2,)


def test_colored_sparse_jacobian_matches_dense_gather_reference() -> None:
  x = sc.sym("x", 4)
  y = sc.stack([x[0] * x[2], x[2:4].sum(), x[1].sin()])
  colored = sc.sparse_jacobian_colored(y, x)
  reference = sc.sparse_jacobian_reference(y, x)
  f = sc.Function._from_exprs("sj_compare", [x], [colored.values, reference.values], ["x"], ["colored", "reference"])
  xv = np.array([1.0, 2.0, 3.0, 4.0])

  assert colored.sparsity == reference.sparsity
  colored_values, reference_values = f(xv)
  np.testing.assert_allclose(colored_values, reference_values)


def test_sparse_jacobian_values_round_trip_to_dense() -> None:
  x = sc.sym("x", 4)
  y = sc.stack([x[0], x[2:4].sum(), x[1]])
  sj = sc.sparse_jacobian(y, x)
  f = sc.Function._from_exprs("sj", [x], [sj.values, sj.to_dense()], ["x"], ["values", "dense"])
  xv = np.array([1.0, 2.0, 3.0, 4.0])

  assert sj.sparsity.rows == (0, 1, 1, 2)
  assert sj.sparsity.cols == (0, 2, 3, 1)
  values, dense = f(xv)
  np.testing.assert_allclose(values, np.ones(4))
  np.testing.assert_allclose(
    dense,
    np.array(
      [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 1.0, 0.0, 0.0],
      ]
    ),
  )


def test_column_coloring_groups_nonoverlapping_columns() -> None:
  sp = sc.SparsityType.from_mask(
    np.array(
      [
        [True, False, True, False],
        [False, True, False, True],
        [False, False, True, False],
      ]
    )
  )
  colors = sc.column_coloring(sp)

  assert colors == (0, 0, 1, 1)
  assert sc.color_groups(colors) == ((0, 1), (2, 3))


def test_jacobian_sparsity_for_matmul_and_call_chain_rule() -> None:
  x = sc.sym("x", 3)
  a = sc.const(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
  y = a @ x
  np.testing.assert_array_equal(sc.jacobian_sparsity(y, x).to_mask(), np.ones((2, 3), dtype=bool))

  u = sc.sym("u", 2)
  inner = sc.Function._from_exprs("inner", [u], [sc.stack([u[0], u[0] + u[1]])], ["u"], ["y"])
  z = sc.sym("z", 3)
  inner_z = inner(sc.gather(z, [2, 0]))

  np.testing.assert_array_equal(
    sc.jacobian_sparsity(inner_z, z).to_mask(),
    np.array(
      [
        [False, False, True],
        [True, False, True],
      ]
    ),
  )


def test_dependency_composition_keeps_exactly_256_shared_paths() -> None:
  u = sc.sym("u", 256)
  inner = sc.Function._from_exprs("shared_256_inner", [u], [u.sum()], ["u"], ["y"])
  x = sc.sym("x", 1)
  y = inner(x + np.zeros(256))

  np.testing.assert_array_equal(sc.jacobian_sparsity(y, x).to_mask(), np.ones((1, 1), dtype=bool))
  colored = sc.sparse_jacobian_colored(y, x)
  reference = sc.sparse_jacobian_reference(y, x)
  f = sc.Function._from_exprs("shared_256_jac", [x], [colored.values, reference.values], ["x"], ["colored", "reference"])
  colored_values, reference_values = f(np.array([2.0]))
  np.testing.assert_allclose(colored_values, np.array([256.0]))
  np.testing.assert_allclose(colored_values, reference_values)


def test_column_coloring_detects_exactly_256_shared_rows() -> None:
  sparsity = sc.SparsityType.from_mask(np.ones((256, 2), dtype=bool))

  assert sc.column_coloring(sparsity) == (0, 1)


def test_mapped_sparsity_storage_grows_with_nonzeros_not_global_mask() -> None:
  from scaly.ad.sparsity import _jac_mask

  def mapped_mask(length: int):
    u = sc.sym(f"u_{length}", 2)
    piece = sc.Function._from_exprs(f"storage_piece_{length}", [u], [sc.stack([u.sum()])], ["u"], ["y"])
    z = sc.sym(f"z_{length}", 2 * length)
    return _jac_mask(sc.vmap(piece, length, [(z, 0, 2)]), z, {})

  small = mapped_mask(32)
  large = mapped_mask(128)
  small_bytes = small.data.nbytes + small.indices.nbytes + small.indptr.nbytes
  large_bytes = large.data.nbytes + large.indices.nbytes + large.indptr.nbytes

  assert small.shape == (32, 64) and small.nnz == 64
  assert large.shape == (128, 256) and large.nnz == 256
  assert large_bytes < 5 * small_bytes


def test_star_coloring_is_not_distance_two_coloring() -> None:
  sparsity = sc.SparsityType.from_mask(
    np.array(
      [
        [True, False, True],
        [False, True, True],
        [True, True, True],
      ]
    )
  )

  colors = sc.star_coloring(sparsity)

  assert colors[0] == colors[1]
  assert colors[0] != colors[2]


def test_star_coloring_rejects_bicolored_four_vertex_path() -> None:
  path = sc.SparsityType.from_mask(
    np.array(
      [
        [True, True, False, False],
        [True, True, True, False],
        [False, True, True, True],
        [False, False, True, True],
      ]
    )
  )

  colors = sc.star_coloring(path)

  assert colors == (0, 1, 0, 2)
  assert colors != (0, 1, 0, 1)
  # The only simple three-edge path is 0--1--2--3. A star coloring cannot use
  # just two alternating colors on that path.
  assert not (colors[0] == colors[2] and colors[1] == colors[3] and colors[0] != colors[1])


def test_star_coloring_rejects_non_square_and_symmetrizes_asymmetric_graphs() -> None:
  with pytest.raises(ValueError, match="square"):
    sc.star_coloring(sc.SparsityType.from_mask(np.ones((2, 3), dtype=bool)))

  from scaly.ad.sparsity import _symmetrize_sparsity

  asymmetric = sc.SparsityType.from_mask(
    np.array(
      [
        [True, True, False],
        [False, True, True],
        [False, False, True],
      ]
    )
  )
  symmetric = _symmetrize_sparsity(asymmetric)
  np.testing.assert_array_equal(
    symmetric.to_mask(),
    np.array(
      [
        [True, True, False],
        [True, True, True],
        [False, True, True],
      ]
    ),
  )
  colors = sc.star_coloring(asymmetric)
  assert colors[0] == colors[2] != colors[1]


def test_star_recovery_rejects_ambiguous_orientation() -> None:
  from scaly.ad.sparse import _star_recovery_indices

  sparsity = sc.SparsityType.from_mask(np.ones((3, 3), dtype=bool))
  with pytest.raises(ValueError, match="cannot recover Hessian entry"):
    _star_recovery_indices(sparsity, (0, 0, 0))


def test_shared_fill_star_hessian_matches_one_sided_and_dense(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  star_widths: list[int] = []
  one_sided_widths: list[int] = []

  for length in (1, 2, 4):
    mapped, _ = _mapped_sphess_fixture(length, shared=True)
    z = mapped.inputs[0]
    lam_f = sc.sym(f"shared_fill_lam_f_{length}")
    lam_g = sc.sym(f"shared_fill_lam_g_{length}", length)
    lagrangian = mapped.outputs[0] * lam_f + (mapped.outputs[1] * lam_g).sum()
    gradient = sc.gradient(lagrangian, z).reshape((z.size,))
    one_sided = sc.sparse_jacobian_colored(gradient, z)
    star = sc.sparse_hessian(lagrangian, z)
    dense = sc.hessian(lagrangian, z)

    assert one_sided.sparsity == star.sparsity
    assert star.coloring_width is not None
    assert one_sided.coloring_width is not None
    star_widths.append(star.coloring_width)
    one_sided_widths.append(one_sided.coloring_width)
    fn = sc.Function._from_exprs(
      f"shared_fill_differential_{length}",
      [z, lam_f, lam_g],
      [star.values, one_sided.values, star.to_dense(), one_sided.to_dense(), dense],
      ["z", "lam_f", "lam_g"],
      ["star_values", "one_sided_values", "star", "one_sided", "dense"],
    )
    zv = np.random.default_rng(length).normal(size=z.size)
    lam_fv = np.array(0.6)
    lam_gv = np.random.default_rng(length + 10).normal(size=length)
    star_values, one_sided_values, star_dense, one_sided_dense, dense_value = fn((zv, lam_fv, lam_gv))
    np.testing.assert_allclose(star_values, one_sided_values, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(star_dense, one_sided_dense, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(star_dense, dense_value, rtol=1e-10, atol=1e-10)

  assert len(set(star_widths)) == 1
  assert star_widths[0] == 3
  assert one_sided_widths[-1] > one_sided_widths[0]


def _dense(sp: sc.SparsityType) -> np.ndarray:
  out = np.zeros(sp.shape, dtype=bool)
  out[list(sp.rows), list(sp.cols)] = True
  return out


def test_select_pattern_is_the_branch_union_and_predicates_add_nothing() -> None:
  x = sc.sym("x", 4)
  # Branch a reads x[0] and x[1]; branch b reads x[3]; the condition reads x[2], which must not appear.
  a = sc.stack([x[0], x[1], x[0], x[1]])
  b = x[3] * sc.const(np.ones(4))
  y = sc.where(x[2] > 0.0, a, b)
  mask = _dense(sc.jacobian_sparsity(y, x))
  expected = np.zeros((4, 4), dtype=bool)
  expected[[0, 2], 0] = expected[[1, 3], 1] = True
  expected[:, 3] = True
  np.testing.assert_array_equal(mask, expected)
  assert sc.jacobian_sparsity(sc.cast(x < 1.0, "float64"), x).nnz == 0
  assert sc.jacobian_sparsity(sc.cast(x, "float32"), x).nnz == 4
  # copysign's sign operand never contributes a derivative.
  np.testing.assert_array_equal(_dense(sc.jacobian_sparsity(sc.copysign(x[:2], x[2:]), x)), np.eye(2, 4, dtype=bool))


def test_reduction_extrema_depend_on_every_entry() -> None:
  x = sc.sym("x", 5)
  y = sc.stack([x[:3].max(), x[2:].min(), sc.norm_inf(x[1:2])])
  np.testing.assert_array_equal(_dense(sc.jacobian_sparsity(y, x)), [[1, 1, 1, 0, 0], [0, 0, 1, 1, 1], [0, 1, 0, 0, 0]])


def test_accumulating_scatter_and_segment_extrema_patterns() -> None:
  x = sc.sym("x", 4)
  ids = np.array([1, 0, 1, 1])
  expected = np.array([[0, 1, 0, 0], [1, 0, 1, 1]], dtype=bool)
  np.testing.assert_array_equal(_dense(sc.jacobian_sparsity(sc.scatter(x, ids, 2), x)), expected)
  np.testing.assert_array_equal(_dense(sc.jacobian_sparsity(sc.segment_max(x, ids, 2), x)), expected)
  np.testing.assert_array_equal(_dense(sc.jacobian_sparsity(sc.segment_sum(x * x, ids, 3), x)), np.vstack([expected, np.zeros((1, 4), bool)]))


def test_scan_patterns_equal_the_unrolled_patterns() -> None:
  c, u = sc.sym("c", 3), sc.sym("u", 1)
  # A shift register: the carry moves one slot per step, so a dependence takes steps to arrive.
  body = sc.Function._from_exprs("shift_step", [c, u], [sc.stack([u[0], c[0] * 2.0, c[1] + c[2]]), c[2:] * u[0]], ["c", "u"], ["n", "y"])
  c0, us = sc.sym("c0", 3), sc.sym("us", 6)
  from scaly.function.sugar import _scan_node

  scanned = [*sc.scan(body, c0, [(us, 0, 1)], length=6), _scan_node(body, c0, (us,), (0,), (1,), 6, -1)]
  z, ys, carries = c0, [], []
  for k in range(6):
    carries.append(z)
    z, y = body._flat_symbolic_call([z, us[k : k + 1]])
    ys.append(y)
  unrolled = [z, sc.concat(ys), sc.concat(carries)]
  for got, want in zip(scanned, unrolled, strict=True):
    for wrt in (c0, us):
      np.testing.assert_array_equal(_dense(sc.jacobian_sparsity(got, wrt)), _dense(sc.jacobian_sparsity(want, wrt)))
  # With no input that moves from step to step the patterns cycle, and the walk reads the answer off
  # the cycle instead of taking 100 000 steps: a rotation, so step 100 000 looks like step 1.
  w = sc.sym("w", 1)
  bcast = sc.Function._from_exprs("bcast_step", [c, w], [sc.stack([c[1], c[2], c[0] + w[0]])], ["c", "w"], ["n"])
  (final,) = sc.scan(bcast, c0, [(w, 0, 0)], length=100_000)
  import time

  t0 = time.perf_counter()
  np.testing.assert_array_equal(_dense(sc.jacobian_sparsity(final, c0)), np.eye(3, k=1, dtype=bool) | np.eye(3, k=-2, dtype=bool))
  assert sc.jacobian_sparsity(final, w).nnz == 3
  assert time.perf_counter() - t0 < 1.0


def test_while_loop_pattern_is_the_closure_of_the_step() -> None:
  c = sc.sym("c", 4)
  # A two-step shift: entry i reaches entry i - 1 each step, so after enough steps every lower entry.
  body = sc.Function._from_exprs("wl_shift", [c], [sc.stack([c[0], c[0] + c[1], c[1] * c[2], c[3]])], ["c"], ["cn"])
  cond = sc.Function._from_exprs("wl_go", [c], [c[0] < 1.0], ["c"], ["go"])
  c0 = sc.sym("c0", 4)
  final, count = sc.while_loop(cond, body, c0, max_iter=50)
  expected = np.array([[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [0, 0, 0, 1]], dtype=bool)
  np.testing.assert_array_equal(_dense(sc.jacobian_sparsity(final, c0)), expected)
  assert sc.jacobian_sparsity(count, c0).nnz == 0
  one, _ = sc.while_loop(cond, body, c0, max_iter=1)
  np.testing.assert_array_equal(
    _dense(sc.jacobian_sparsity(one, c0)), expected & ~np.array([[0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]], dtype=bool)
  )


def _mask(expr: sc.Expr, wrt: sc.Expr) -> np.ndarray:
  return sc.jacobian_sparsity(expr, wrt).to_mask()


def test_custom_sparsity_replaces_the_body_pattern_in_calls_maps_and_loops() -> None:
  """A body whose structural pattern is dense (a sum couples every entry) declared diagonal: the
  declared pattern is used wherever the Function is applied."""
  x = sc.sym("x", 3)
  body = sc.Function._from_exprs("cs_dense", [x], [x + 1e-30 * x.sum()], ["x"], ["y"])
  declared = sc.custom_derivative(body, sparsity=lambda out, k: np.eye(3, dtype=bool))
  q = sc.sym("q", 3)
  assert _mask(body(q), q).all()
  np.testing.assert_array_equal(_mask(declared(q), q), np.eye(3, dtype=bool))
  qs = sc.sym("qs", 6)
  np.testing.assert_array_equal(_mask(sc.vmap(declared, 2, [(qs, 0, 3)]), qs), np.eye(6, dtype=bool))
  (looped,) = sc.scan(declared, q, length=4)
  np.testing.assert_array_equal(_mask(looped, q), np.eye(3, dtype=bool))
  (walked, _) = sc.while_loop(sc.Function._from_exprs("cs_go", [x], [x[0] < 0.0], ["x"], ["go"]), declared, q, max_iter=3)
  np.testing.assert_array_equal(_mask(walked, q), np.eye(3, dtype=bool))
  # Copies keep the declaration unless given their own; the derivative rules are untouched.
  assert sc.custom_derivative(declared).custom_sparsity is declared.custom_sparsity
  rule = sc.Function._from_exprs("cs_rule", [x, sc.sym("t", 3)], [sc.sym("t", 3) * 2.0], ["x", "t"], ["dy"])
  assert sc.custom_derivative(declared, jvp=rule).custom_sparsity is None  # new rules, no stale pattern
  np.testing.assert_allclose(
    sc.jacobian(sc.Function._from_exprs("cs_host", [q], [declared(q)], ["q"], ["y"]), "y", "q")(np.ones(3)), np.eye(3) + 1e-30
  )


@pytest.mark.parametrize(
  "pattern",
  [None, sc.SparsityType((2, 3), (0, 1), (2, 0)), np.array([[0, 0, 1], [1, 0, 0]], dtype=bool)],
  ids=["none", "sparsity-type", "mask"],
)
def test_custom_sparsity_forms(pattern) -> None:
  x, y = sc.sym("x", 3), sc.sym("y", 1)
  fn = sc.Function._from_exprs("cs_forms", [x, y], [sc.stack([x.sum(), y[0]])], ["x", "y"], ["z"])
  declared = sc.custom_derivative(fn, sparsity=lambda out, k: pattern if k == 0 else None)
  q, r = sc.sym("q", 3), sc.sym("r", 1)
  z = declared.symbolic_call((q, r))
  expected = np.zeros((2, 3), dtype=bool) if pattern is None else np.array([[0, 0, 1], [1, 0, 0]], dtype=bool)
  np.testing.assert_array_equal(_mask(z, q), expected)
  assert not _mask(z, r).any()  # None: no dependence


def test_custom_sparsity_of_the_wrong_shape_raises() -> None:
  x = sc.sym("x", 3)
  fn = sc.custom_derivative(sc.Function._from_exprs("cs_bad", [x], [x * 2.0], ["x"], ["y"]), sparsity=lambda out, k: np.eye(2, dtype=bool))
  with pytest.raises(ValueError, match=r"shape \(2, 2\) does not fit .* \(3, 3\)"):
    sc.jacobian_sparsity(fn(sc.sym("q", 3)), sc.sym("q", 3))


def test_loop_patterns_with_a_large_carry_and_no_dependence() -> None:
  """A carry of 300 entries (the cycle key once held its shape in single bytes), and a loop that
  reads nothing depending on ``wrt``, whose pattern is empty without walking its steps."""
  c, u = sc.sym("c", 300), sc.sym("u", 300)
  body = sc.Function._from_exprs("big_step", [c], [sc.concat([c[1:], c[:1]])], ["c"], ["cn"])
  (out,) = sc.scan(body, u, length=7)
  expected = np.roll(np.eye(300, dtype=bool), 7, axis=1)
  np.testing.assert_array_equal(_mask(out, u), expected)
  other = sc.sym("other", 2)
  assert sc.jacobian_sparsity(out, other).shape == (300, 2) and sc.jacobian_sparsity(out, other).nnz == 0
