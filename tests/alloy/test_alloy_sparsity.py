from __future__ import annotations

import numpy as np

import alloy as al


def test_sparsity_type_roundtrip_and_bounds() -> None:
  mask = np.array([[True, False, True], [False, True, False]])
  sp = al.SparsityType.from_mask(mask)

  assert sp.shape == (2, 3)
  assert sp.nnz == 3
  np.testing.assert_array_equal(sp.to_mask(), mask)

  try:
    _ = al.SparsityType((2, 2), (0, 2), (0, 1))
  except ValueError as e:
    assert "out of bounds" in str(e)
  else:  # pragma: no cover
    raise AssertionError("invalid sparsity should fail")

  try:
    _ = al.SparsityType((2, 2), (0, 0), (1, 1))
  except ValueError as e:
    assert "sparsity indices must be unique" in str(e)
  else:  # pragma: no cover
    raise AssertionError("duplicate sparsity coordinates should fail")


def test_sparsity_type_csr_csc_conversions() -> None:
  sp = al.SparsityType((3, 4), (2, 0, 1, 1), (3, 2, 0, 3))

  row_ptr, col_ind, csr_perm = sp.to_csr()
  assert row_ptr == (0, 1, 3, 4)
  assert col_ind == (2, 0, 3, 3)
  # val_perm maps CSR slot -> COO position: sorted (row, col) order of the COO pattern above.
  assert csr_perm == (1, 2, 3, 0)
  assert tuple((sp.rows[i], sp.cols[i]) for i in csr_perm) == ((0, 2), (1, 0), (1, 3), (2, 3))
  np.testing.assert_array_equal(al.SparsityType.from_csr(sp.shape, row_ptr, col_ind).to_mask(), sp.to_mask())

  col_ptr, row_ind, csc_perm = sp.to_csc()
  assert col_ptr == (0, 1, 1, 2, 4)
  assert row_ind == (1, 0, 1, 2)
  assert csc_perm == (2, 1, 3, 0)
  assert tuple((sp.cols[i], sp.rows[i]) for i in csc_perm) == ((0, 1), (2, 0), (3, 1), (3, 2))
  np.testing.assert_array_equal(al.SparsityType.from_csc(sp.shape, col_ptr, row_ind).to_mask(), sp.to_mask())


def test_sparsity_type_compressed_format_errors() -> None:
  try:
    _ = al.SparsityType.from_csr((2, 3), (0, 1), (0,))
  except ValueError as e:
    assert "row_ptr must have length 3" in str(e)
  else:  # pragma: no cover
    raise AssertionError("short CSR row pointer should fail")

  try:
    _ = al.SparsityType.from_csc((2, 3), (0, 2, 1, 1), (0,))
  except ValueError as e:
    assert "col_ptr must be nondecreasing" in str(e)
  else:  # pragma: no cover
    raise AssertionError("invalid CSC column pointer should fail")

  try:
    _ = al.SparsityType.from_csr((2, 3), (0, 1, 1), (3,))
  except ValueError as e:
    assert "CSR column indices out of bounds" in str(e)
  else:  # pragma: no cover
    raise AssertionError("out-of-bounds CSR column should fail")


def test_jacobian_sparsity_for_structural_ops() -> None:
  x = al.sym("x", 4)
  y = al.stack([x[0], x[2:4].sum(), al.gather(x, [1, 3]).sum()])
  sp = al.jacobian_sparsity(y, x)

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
  x = al.sym("x", (2, 2))
  y = al.concat([x[:, :1], x[:, 1:2]], axis=1)
  sp = al.jacobian_sparsity(y, x)

  np.testing.assert_array_equal(sp.to_mask(), np.eye(4, dtype=bool))


def test_sphessian_factory_returns_compact_values_with_sparsity_metadata() -> None:
  x = al.sym("x", 3)
  y = x[0] * x[0] + x[1] * x[2]
  f = al.Function("f", [x], [y], ["x"], ["y"])
  shf = al.sphessian(f, "x", "y")
  xv = np.array([2.0, 3.0, 4.0])

  assert shf.output_names == ("sphess_y_x_x",)
  assert shf.output_sparsities[0] is not None
  assert shf.output_sparsities[0].rows == (0, 1, 2)
  assert shf.output_sparsities[0].cols == (0, 2, 1)
  np.testing.assert_allclose(shf(xv), np.array([2.0, 1.0, 1.0]))


def test_sparse_lagrangian_hessian_uses_aux_output() -> None:
  x = al.sym("x", 2)
  f_expr = x[0] * x[0]
  g_expr = al.stack([x[0] * x[1], x[1] * x[1]])
  nlp = al.Function("nlp", [x], [f_expr, g_expr], ["x"], ["f", "g"])
  shf = al.sparse_lagrangian_hessian(nlp, "x", ["f", "g"])

  assert shf.input_names == ("x", "lam:f", "lam:g")
  assert shf.output_names == ("sphess_gamma_x_x",)
  assert shf.output_sparsities[0] is not None
  assert shf.output_sparsities[0].rows == (0, 0, 1, 1)
  assert shf.output_sparsities[0].cols == (0, 1, 0, 1)
  np.testing.assert_allclose(shf(np.array([2.0, 3.0]), np.array(1.5), np.array([0.25, -0.5])), np.array([3.0, 0.25, 0.25, -1.0]))


def test_spjac_factory_returns_compact_values_with_sparsity_metadata() -> None:
  x = al.sym("x", 4)
  y = al.stack([x[0], x[2:4].sum(), x[1]])
  f = al.Function("f", [x], [y], ["x"], ["y"])
  spjf = al.spjacobian(f, "x", "y")
  xv = np.array([1.0, 2.0, 3.0, 4.0])

  assert spjf.output_names == ("spjac_y_x",)
  assert spjf.output_sparsities[0] is not None
  assert spjf.output_sparsities[0].rows == (0, 1, 1, 2)
  assert spjf.output_sparsities[0].cols == (0, 2, 3, 1)
  np.testing.assert_allclose(spjf(xv), np.ones(4))


def test_function_rejects_sparse_output_metadata_size_mismatch() -> None:
  x = al.sym("x", 2)
  try:
    _ = al.Function("bad", [x], [x], ["x"], ["sp"], output_sparsities=[al.SparsityType.dense((2, 2))])
  except ValueError as e:
    assert "sparse output metadata for 'sp' has 4 nonzeros" in str(e)
    assert "output shape (2,) has 2 entries" in str(e)
  else:  # pragma: no cover
    raise AssertionError("sparse output metadata size mismatch should fail")


def test_colored_sparse_jacobian_matches_dense_gather_reference() -> None:
  x = al.sym("x", 4)
  y = al.stack([x[0] * x[2], x[2:4].sum(), x[1].sin()])
  colored = al.sparse_jacobian_colored(y, x)
  reference = al.sparse_jacobian_reference(y, x)
  f = al.Function("sj_compare", [x], [colored.values, reference.values], ["x"], ["colored", "reference"])
  xv = np.array([1.0, 2.0, 3.0, 4.0])

  assert colored.sparsity == reference.sparsity
  colored_values, reference_values = f(xv)
  np.testing.assert_allclose(colored_values, reference_values)


def test_sparse_jacobian_values_round_trip_to_dense() -> None:
  x = al.sym("x", 4)
  y = al.stack([x[0], x[2:4].sum(), x[1]])
  sj = al.sparse_jacobian(y, x)
  f = al.Function("sj", [x], [sj.values, sj.to_dense()], ["x"], ["values", "dense"])
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
  sp = al.SparsityType.from_mask(
    np.array(
      [
        [True, False, True, False],
        [False, True, False, True],
        [False, False, True, False],
      ]
    )
  )
  colors = al.column_coloring(sp)

  assert colors == (0, 0, 1, 1)
  assert al.color_groups(colors) == ((0, 1), (2, 3))


def test_jacobian_sparsity_for_matmul_and_call_chain_rule() -> None:
  x = al.sym("x", 3)
  a = al.const(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
  y = a @ x
  np.testing.assert_array_equal(al.jacobian_sparsity(y, x).to_mask(), np.ones((2, 3), dtype=bool))

  u = al.sym("u", 2)
  inner = al.Function("inner", [u], [al.stack([u[0], u[0] + u[1]])], ["u"], ["y"])
  z = al.sym("z", 3)
  (inner_z,) = inner.call([al.gather(z, [2, 0])])

  np.testing.assert_array_equal(
    al.jacobian_sparsity(inner_z, z).to_mask(),
    np.array(
      [
        [False, False, True],
        [True, False, True],
      ]
    ),
  )
