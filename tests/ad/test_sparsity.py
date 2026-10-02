from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from scaly.function.model import as_concrete
from scaly.function.concrete import ConcreteFunction
from scaly.function.sugar import _mapped_call
import scaly as sc
from scaly.ad import finite_difference
from scaly.ad.sparse import Triangle


def _mapped_sphess_fixture(length: int, *, shared: bool = False) -> tuple[sc.Function, sc.Function]:
  if shared:

    @sc.function(sc.group(sc.arg("x", 2), sc.arg("s", 1)), outputs=sc.arg("g", ...), name="mapped_sphess_shared_piece")
    def piece(inputs):
      x, s = inputs
      hidden = sc.stack([x[0] * x[1] + s[0] * x[0], x[0] - 0.4 * x[1] + s[0] * x[1]])
      return sc.stack([(hidden.tanh() ** 2).sum()])
  else:

    @sc.function(sc.arg("x", 2), outputs=sc.arg("g", ...), name="mapped_sphess_piece")
    def piece(x):
      hidden = sc.stack([x[0] * x[1], x[0] - 0.4 * x[1]])
      return sc.stack([(hidden.tanh() ** 2).sum()])

  output_tree = sc.group(sc.arg("f", ...), sc.arg("g", ...))

  @sc.function(sc.arg("z", 2 * length + int(shared)), outputs=output_tree, name=f"mapped_sphess_{length}_{int(shared)}")
  def mapped_fn(z):
    specs = [(z, 0, 2), *(((z, 2 * length, 0),) if shared else ())]
    return (z * z).sum(), _mapped_call(piece, length, specs)

  @sc.function(sc.arg("z", 2 * length + int(shared)), outputs=output_tree, name=f"unrolled_sphess_{length}_{int(shared)}")
  def unrolled_fn(z):
    calls = []
    for it in range(length):
      args = [z[2 * it : 2 * (it + 1)], *((z[2 * length : 2 * length + 1],) if shared else ())]
      calls.append(piece(*as_concrete(piece).input_tree.unflatten(tuple(args))))
    return (z * z).sum(), sc.concat(calls)

  return mapped_fn, unrolled_fn


def _scatter_sparse(values: np.ndarray, sparsity: sc.SparsityPattern) -> np.ndarray:
  dense = np.zeros(sparsity.shape)
  dense[np.asarray(sparsity.rows), np.asarray(sparsity.cols)] = values
  return dense


def test_sparsity_type_roundtrip_and_bounds() -> None:
  mask = np.array([[True, False, True], [False, True, False]])
  sp = sc.SparsityPattern.from_mask(mask)

  assert sp.shape == (2, 3)
  assert sp.nnz == 3
  np.testing.assert_array_equal(sp.to_mask(), mask)

  try:
    _ = sc.SparsityPattern((2, 2), (0, 2), (0, 1))
  except ValueError as e:
    assert "out of bounds" in str(e)
  else:  # pragma: no cover
    raise AssertionError("invalid sparsity should fail")

  try:
    _ = sc.SparsityPattern((2, 2), (0, 0), (1, 1))
  except ValueError as e:
    assert "sparsity indices must be unique" in str(e)
  else:  # pragma: no cover
    raise AssertionError("duplicate sparsity coordinates should fail")


def test_sparsity_type_csr_csc_conversions() -> None:
  sp = sc.SparsityPattern((3, 4), (2, 0, 1, 1), (3, 2, 0, 3))

  row_ptr, col_ind, csr_perm = sp.to_csr()
  assert row_ptr == (0, 1, 3, 4)
  assert col_ind == (2, 0, 3, 3)
  # val_perm maps CSR slot -> COO position: sorted (row, col) order of the COO pattern above.
  assert csr_perm == (1, 2, 3, 0)
  assert tuple((sp.rows[i], sp.cols[i]) for i in csr_perm) == ((0, 2), (1, 0), (1, 3), (2, 3))
  np.testing.assert_array_equal(sc.SparsityPattern.from_csr(sp.shape, row_ptr, col_ind).to_mask(), sp.to_mask())

  col_ptr, row_ind, csc_perm = sp.to_csc()
  assert col_ptr == (0, 1, 1, 2, 4)
  assert row_ind == (1, 0, 1, 2)
  assert csc_perm == (2, 1, 3, 0)
  assert tuple((sp.cols[i], sp.rows[i]) for i in csc_perm) == ((0, 1), (2, 0), (3, 1), (3, 2))
  np.testing.assert_array_equal(sc.SparsityPattern.from_csc(sp.shape, col_ptr, row_ind).to_mask(), sp.to_mask())


def test_sparsity_type_compressed_format_errors() -> None:
  try:
    _ = sc.SparsityPattern.from_csr((2, 3), (0, 1), (0,))
  except ValueError as e:
    assert "row_ptr must have length 3" in str(e)
  else:  # pragma: no cover
    raise AssertionError("short CSR row pointer should fail")

  try:
    _ = sc.SparsityPattern.from_csc((2, 3), (0, 2, 1, 1), (0,))
  except ValueError as e:
    assert "col_ptr must be nondecreasing" in str(e)
  else:  # pragma: no cover
    raise AssertionError("invalid CSC column pointer should fail")

  try:
    _ = sc.SparsityPattern.from_csr((2, 3), (0, 1, 1), (3,))
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
  @sc.function(sc.arg("x", 3), outputs=sc.arg("y", ...), name="f")
  def f(x):
    return x[0] * x[0] + x[1] * x[2]

  shf = sc.sparse_hessian(f, "y", "x")
  xv = np.array([2.0, 3.0, 4.0])

  assert as_concrete(shf).output_names == ("sphess_y_x_x",)
  assert as_concrete(shf).output_coloring_widths == (2,)
  shf_sparsity = as_concrete(shf).output_sparsities[0]
  assert shf_sparsity is not None
  assert shf_sparsity.rows == (0, 1, 2)
  assert shf_sparsity.cols == (0, 2, 1)
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

  @sc.function(
    sc.arg("triangle_x", 4),
    outputs=sc.group(sc.arg("full", ...), sc.arg("triangle", ...), sc.arg("selected", ...)),
    name=f"triangle_expr_{triangle}",
  )
  def value_fn(x):
    y = (x[0] * x[1]).sin() + (x[2] * x[3]).sin()
    full = sc.sparse_hessian(y, x)
    return full.values, sc.sparse_hessian(y, x, triangle=triangle).values, full.triangle(triangle).values

  full_values, triangle_values, selected_values = value_fn(np.array([0.2, 0.7, -0.3, 1.1]))
  np.testing.assert_allclose(triangle_values, full_values[keep], rtol=1e-10, atol=1e-10)
  np.testing.assert_allclose(selected_values, triangle_values, rtol=1e-10, atol=1e-10)

  @sc.function(sc.arg("triangle_x", 4), outputs=sc.arg("y", ...), name="triangle_fn")
  def fn(x):
    return (x[0] * x[1]).sin() + (x[2] * x[3]).sin()

  full_fn = sc.sparse_hessian(fn, "y", "triangle_x", name=f"triangle_fn_full_{triangle}")
  triangle_fn = sc.sparse_hessian(fn, "y", "triangle_x", name=f"triangle_fn_{triangle}", triangle=triangle)
  assert as_concrete(full_fn).output_names == as_concrete(triangle_fn).output_names == ("sphess_y_triangle_x_triangle_x",)
  assert as_concrete(full_fn).output_coloring_widths == as_concrete(triangle_fn).output_coloring_widths
  full_fn_sp, triangle_fn_sp = as_concrete(full_fn).output_sparsities[0], as_concrete(triangle_fn).output_sparsities[0]
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
  @sc.function(sc.arg("x", 2), outputs=sc.group(sc.arg("f", ...), sc.arg("g", ...)), name="nlp")
  def nlp(x):
    return x[0] * x[0], sc.stack([x[0] * x[1], x[1] * x[1]])

  shf = sc.sparse_lagrangian_hessian(nlp, "x")

  assert as_concrete(shf).input_names == ("x", "lam:f", "lam:g")
  assert as_concrete(shf).output_names == ("sphess_gamma_x_x",)
  assert as_concrete(shf).output_coloring_widths == (2,)
  shf_sparsity = as_concrete(shf).output_sparsities[0]
  assert shf_sparsity is not None
  assert shf_sparsity.rows == (0, 0, 1, 1)
  assert shf_sparsity.cols == (0, 1, 0, 1)
  np.testing.assert_allclose(shf(*(np.array([2.0, 3.0]), (np.array(1.5), np.array([0.25, -0.5])))), np.array([3.0, 0.25, 0.25, -1.0]))


def test_sparse_lagrangian_hessian_through_vmap_matches_unrolled_dense_and_fd(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  mapped, unrolled = _mapped_sphess_fixture(3)
  mapped_sphess = mapped.factory("mapped_sphess_exact", ["z", "lam:f", "lam:g"], [sc.factory.SpHess("gamma", "z")], aux={"gamma": ["f", "g"]})
  unrolled_sphess = unrolled.factory("unrolled_sphess_exact", ["z", "lam:f", "lam:g"], [sc.factory.SpHess("gamma", "z")], aux={"gamma": ["f", "g"]})
  unrolled_hess = unrolled.factory("unrolled_hess_dense", ["z", "lam:f", "lam:g"], [sc.factory.Hess("gamma", "z")], aux={"gamma": ["f", "g"]})
  unrolled_grad = unrolled.factory("unrolled_grad_for_fd", ["z", "lam:f", "lam:g"], [sc.factory.Grad("gamma", "z")], aux={"gamma": ["f", "g"]})

  mapped_sp, unrolled_sp = as_concrete(mapped_sphess).output_sparsities[0], as_concrete(unrolled_sphess).output_sparsities[0]
  assert mapped_sp is not None and unrolled_sp is not None
  np.testing.assert_array_equal(mapped_sp.to_mask(), unrolled_sp.to_mask())
  np.testing.assert_array_equal(mapped_sp.to_mask(), mapped_sp.to_mask().T)

  zv = np.array([-0.7, 0.2, 0.4, -0.5, 0.8, 0.3])
  lam_f, lam_g = np.array(0.6), np.array([0.3, -0.8, 1.1])
  mapped_dense = _scatter_sparse(np.asarray(mapped_sphess(*(zv, lam_f, lam_g))), mapped_sp)
  unrolled_dense = _scatter_sparse(np.asarray(unrolled_sphess(*(zv, lam_f, lam_g))), unrolled_sp)
  np.testing.assert_allclose(mapped_dense, unrolled_dense, rtol=1e-10, atol=1e-10)
  np.testing.assert_allclose(mapped_dense, unrolled_hess(*(zv, lam_f, lam_g)), rtol=1e-10, atol=1e-10)
  np.testing.assert_allclose(mapped_dense, finite_difference(lambda value: unrolled_grad(*(value, lam_f, lam_g)), zv), rtol=2e-5, atol=2e-6)


def test_mapped_sparse_hessian_multiplier_weighting_and_shared_fill(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  mapped, _ = _mapped_sphess_fixture(3)
  sphess = mapped.factory("mapped_sphess_weighting", ["z", "lam:f", "lam:g"], [sc.factory.SpHess("gamma", "z")], aux={"gamma": ["f", "g"]})
  sparsity = as_concrete(sphess).output_sparsities[0]
  assert sparsity is not None
  zv = np.array([-0.7, 0.2, 0.4, -0.5, 0.8, 0.3])
  lam_f, lam_g = np.array(0.6), np.array([0.3, -0.8, 1.1])
  base = _scatter_sparse(np.asarray(sphess(*(zv, lam_f, lam_g))), sparsity)
  changed_lam = lam_g.copy()
  changed_lam[1] += 0.7
  changed = _scatter_sparse(np.asarray(sphess(*(zv, lam_f, changed_lam))), sparsity)
  delta = changed - base
  np.testing.assert_allclose(delta[:2], 0.0, atol=1e-12)
  np.testing.assert_allclose(delta[4:], 0.0, atol=1e-12)
  assert np.any(np.abs(delta[2:4, 2:4]) > 1e-9)

  shared, shared_unrolled = _mapped_sphess_fixture(3, shared=True)
  shared_sphess = shared.factory("mapped_sphess_shared_fill", ["z", "lam:f", "lam:g"], [sc.factory.SpHess("gamma", "z")], aux={"gamma": ["f", "g"]})
  shared_unrolled_sphess = shared_unrolled.factory(
    "unrolled_sphess_shared_fill", ["z", "lam:f", "lam:g"], [sc.factory.SpHess("gamma", "z")], aux={"gamma": ["f", "g"]}
  )
  shared_sp, shared_unrolled_sp = as_concrete(shared_sphess).output_sparsities[0], as_concrete(shared_unrolled_sphess).output_sparsities[0]
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
  shared_dense = _scatter_sparse(np.asarray(shared_sphess(*(zv_shared, lam_f, lam_g_shared))), shared_sp)
  shared_unrolled_dense = _scatter_sparse(np.asarray(shared_unrolled_sphess(*(zv_shared, lam_f, lam_g_shared))), shared_unrolled_sp)
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
  full_sp, triangle_sp = as_concrete(full).output_sparsities[0], as_concrete(triangle_fn).output_sparsities[0]
  assert full_sp is not None and triangle_sp is not None
  rows, cols = np.asarray(full_sp.rows), np.asarray(full_sp.cols)
  keep = rows >= cols if triangle == "lower" else rows <= cols
  assert triangle_sp.rows == tuple(rows[keep])
  assert triangle_sp.cols == tuple(cols[keep])
  assert as_concrete(triangle_fn).output_coloring_widths == as_concrete(full).output_coloring_widths

  unrolled_sp = as_concrete(unrolled_triangle).output_sparsities[0]
  assert unrolled_sp is not None
  assert unrolled_sp.rows == triangle_sp.rows
  assert unrolled_sp.cols == triangle_sp.cols

  zv = np.array([-0.7, 0.2, 0.4, -0.5, 0.8, 0.3, 0.9])
  lam_f = np.array(0.6)
  lam_g = np.array([0.7, -1.3, 0.45])
  full_values = np.asarray(full(*(zv, lam_f, lam_g)))
  triangle_values = np.asarray(triangle_fn(*(zv, lam_f, lam_g)))
  unrolled_values = np.asarray(unrolled_triangle(*(zv, lam_f, lam_g)))
  np.testing.assert_allclose(triangle_values, full_values[keep], rtol=1e-10, atol=1e-10)
  np.testing.assert_allclose(triangle_values, unrolled_values, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("triangle", ("diagonal", None, 1))
def test_sparse_hessian_rejects_invalid_triangle(triangle: object) -> None:
  x = sc.sym("invalid_triangle_x", 2)
  y = x[0] * x[1]
  invalid_triangle = cast(Triangle, triangle)
  with pytest.raises(ValueError, match="triangle must be one of"):
    sc.sparse_hessian(y, x, triangle=invalid_triangle)

  @sc.function(sc.arg("invalid_triangle_x", 2), outputs=sc.arg("y", ...), name="invalid_triangle")
  def fn(x):
    return x[0] * x[1]

  with pytest.raises(ValueError, match="triangle must be one of"):
    sc.sparse_hessian(fn, "y", "invalid_triangle_x", triangle=invalid_triangle)


def test_spjac_factory_returns_compact_values_with_sparsity_metadata() -> None:
  @sc.function(sc.arg("x", 4), outputs=sc.arg("y", ...), name="f")
  def f(x):
    return sc.stack([x[0], x[2:4].sum(), x[1]])

  spjf = sc.sparse_jacobian(f, "y", "x")
  xv = np.array([1.0, 2.0, 3.0, 4.0])

  assert as_concrete(spjf).output_names == ("spjac_y_x",)
  assert as_concrete(spjf).output_coloring_widths == (2,)
  spjf_sparsity = as_concrete(spjf).output_sparsities[0]
  assert spjf_sparsity is not None
  assert spjf_sparsity.rows == (0, 1, 1, 2)
  assert spjf_sparsity.cols == (0, 2, 3, 1)
  np.testing.assert_allclose(spjf(xv), np.ones(4))


def test_function_rejects_sparse_output_metadata_size_mismatch() -> None:
  x = sc.sym("x", 2)
  try:
    _ = ConcreteFunction._from_exprs("bad", [x], [x], ["x"], ["sp"], output_sparsities=[sc.SparsityPattern.dense((2, 2))])
  except ValueError as e:
    assert "sparse output metadata for 'sp' has 4 nonzeros" in str(e)
    assert "output shape (2,) has 2 entries" in str(e)
  else:  # pragma: no cover
    raise AssertionError("sparse output metadata size mismatch should fail")


def test_sparse_jacobian_preserves_constructed_local_coloring_width() -> None:
  @sc.function(sc.group(sc.arg("a", 1), sc.arg("b", 1)), outputs=sc.arg("y", ...), name="two_formal_piece")
  def piece(inputs):
    return sc.stack(inputs)

  z = sc.sym("z", 5)
  mapped_expr = _mapped_call(piece, 4, [(z, 0, 1), (z, 1, 1)])

  @sc.function(sc.arg("z", 5), outputs=sc.arg("y", ...), name="two_formal_mapped")
  def mapped(z):
    return _mapped_call(piece, 4, [(z, 0, 1), (z, 1, 1)])

  sj = sc.sparse_jacobian(mapped_expr, z)

  np.testing.assert_array_equal(sj.sparsity.to_mask(), sc.jacobian_sparsity(mapped_expr, z).to_mask())
  assert sj.coloring_width == 2
  assert max(sc.column_coloring(sj.sparsity), default=-1) + 1 == 1

  built = mapped.factory("two_formal_spjac", ["z"], [sc.factory.SpJac("y", "z")])
  assert as_concrete(built).output_coloring_widths == (2,)


def test_colored_sparse_jacobian_matches_dense_gather_reference() -> None:
  x = sc.sym("x", 4)
  y = sc.stack([x[0] * x[2], x[2:4].sum(), x[1].sin()])
  colored = sc.sparse_jacobian_colored(y, x)
  reference = sc.sparse_jacobian_reference(y, x)

  @sc.function(sc.arg("x", 4), outputs=sc.group(sc.arg("colored", ...), sc.arg("reference", ...)), name="sj_compare")
  def f(x):
    y = sc.stack([x[0] * x[2], x[2:4].sum(), x[1].sin()])
    return sc.sparse_jacobian_colored(y, x).values, sc.sparse_jacobian_reference(y, x).values

  xv = np.array([1.0, 2.0, 3.0, 4.0])

  assert colored.sparsity == reference.sparsity
  colored_values, reference_values = f(xv)
  np.testing.assert_allclose(colored_values, reference_values)


def test_sparse_jacobian_values_round_trip_to_dense() -> None:
  x = sc.sym("x", 4)
  y = sc.stack([x[0], x[2:4].sum(), x[1]])
  sj = sc.sparse_jacobian(y, x)

  @sc.function(sc.arg("x", 4), outputs=sc.group(sc.arg("values", ...), sc.arg("dense", ...)), name="sj")
  def f(x):
    y = sc.stack([x[0], x[2:4].sum(), x[1]])
    sj = sc.sparse_jacobian(y, x)
    return sj.values, sj.to_dense()

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
  sp = sc.SparsityPattern.from_mask(
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

  @sc.function(sc.arg("u", 2), outputs=sc.arg("y", ...), name="inner")
  def inner(u):
    return sc.stack([u[0], u[0] + u[1]])

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
  @sc.function(sc.arg("u", 256), outputs=sc.arg("y", ...), name="shared_256_inner")
  def inner(u):
    return u.sum()

  x = sc.sym("x", 1)
  y = inner(x + np.zeros(256))

  np.testing.assert_array_equal(sc.jacobian_sparsity(y, x).to_mask(), np.ones((1, 1), dtype=bool))

  @sc.function(sc.arg("x", 1), outputs=sc.group(sc.arg("colored", ...), sc.arg("reference", ...)), name="shared_256_jac")
  def f(x):
    y = inner(x + np.zeros(256))
    return sc.sparse_jacobian_colored(y, x).values, sc.sparse_jacobian_reference(y, x).values

  colored_values, reference_values = f(np.array([2.0]))
  np.testing.assert_allclose(colored_values, np.array([256.0]))
  np.testing.assert_allclose(colored_values, reference_values)


def test_column_coloring_detects_exactly_256_shared_rows() -> None:
  sparsity = sc.SparsityPattern.from_mask(np.ones((256, 2), dtype=bool))

  assert sc.column_coloring(sparsity) == (0, 1)


def test_mapped_sparsity_storage_grows_with_nonzeros_not_global_mask() -> None:
  def mapped_mask(length: int):
    @sc.function(sc.arg("u", 2), outputs=sc.arg("y", ...), name=f"storage_piece_{length}")
    def piece(u):
      return sc.stack([u.sum()])

    z = sc.sym(f"z_{length}", 2 * length)
    return sc.jacobian_sparsity(_mapped_call(piece, length, [(z, 0, 2)]), z)

  small = mapped_mask(32)
  large = mapped_mask(128)
  small_bytes = sum(np.asarray(indices, dtype=np.int64).nbytes for indices in (small.rows, small.cols))
  large_bytes = sum(np.asarray(indices, dtype=np.int64).nbytes for indices in (large.rows, large.cols))

  assert small.shape == (32, 64) and small.nnz == 64
  assert large.shape == (128, 256) and large.nnz == 256
  assert large_bytes < 5 * small_bytes


def test_star_coloring_is_not_distance_two_coloring() -> None:
  sparsity = sc.SparsityPattern.from_mask(
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
  path = sc.SparsityPattern.from_mask(
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
    sc.star_coloring(sc.SparsityPattern.from_mask(np.ones((2, 3), dtype=bool)))

  from scaly.ad.sparsity import _symmetrize_sparsity

  asymmetric = sc.SparsityPattern.from_mask(
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

  sparsity = sc.SparsityPattern.from_mask(np.ones((3, 3), dtype=bool))
  with pytest.raises(ValueError, match="cannot recover Hessian entry"):
    _star_recovery_indices(sparsity, (0, 0, 0))


def test_shared_fill_star_hessian_matches_one_sided_and_dense(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  star_widths: list[int] = []
  one_sided_widths: list[int] = []

  for length in (1, 2, 4):
    mapped, _ = _mapped_sphess_fixture(length, shared=True)
    z = as_concrete(mapped).inputs[0]
    lam_f = sc.sym(f"shared_fill_lam_f_{length}")
    lam_g = sc.sym(f"shared_fill_lam_g_{length}", length)
    lagrangian = as_concrete(mapped).outputs[0] * lam_f + (as_concrete(mapped).outputs[1] * lam_g).sum()
    gradient = sc.gradient(lagrangian, z).reshape((z.size,))
    one_sided = sc.sparse_jacobian_colored(gradient, z)
    star = sc.sparse_hessian(lagrangian, z)

    assert one_sided.sparsity == star.sparsity
    assert star.coloring_width is not None
    assert one_sided.coloring_width is not None
    star_widths.append(star.coloring_width)
    one_sided_widths.append(one_sided.coloring_width)

    @sc.function(
      sc.group(sc.arg("z", z.size), sc.arg("lam_f", ()), sc.arg("lam_g", length)),
      outputs=sc.group(
        sc.arg("star_values", ...), sc.arg("one_sided_values", ...), sc.arg("star", ...), sc.arg("one_sided", ...), sc.arg("dense", ...)
      ),
      name=f"shared_fill_differential_{length}",
    )
    def fn(inputs):
      z, lam_f, lam_g = inputs
      mapped_f, mapped_g = mapped(z)
      lagrangian = mapped_f * lam_f + (mapped_g * lam_g).sum()
      one_sided = sc.sparse_jacobian_colored(sc.gradient(lagrangian, z).reshape((z.size,)), z)
      star = sc.sparse_hessian(lagrangian, z)
      return star.values, one_sided.values, star.to_dense(), one_sided.to_dense(), sc.hessian(lagrangian, z)

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


@pytest.mark.parametrize("mapped", [False, True])
def test_nested_callee_sparsity_is_analyzed_once_per_variable(monkeypatch, mapped: bool) -> None:
  from collections import Counter

  from scaly.ad import sparsity as analysis

  @sc.function(sc.arg("x", 2), sc.arg("y", 2), outputs=sc.arg("out", 2), name="mask_leaf")
  def leaf(x: sc.Expr, y: sc.Expr) -> sc.Expr:
    return sc.stack([x[0] * y[1], x[1]])

  @sc.function(sc.arg("x", 2), sc.arg("y", 2), outputs=sc.arg("out", 2), name="mask_nested")
  def nested(x: sc.Expr, y: sc.Expr) -> sc.Expr:
    return leaf(x, y) + leaf(y, x)

  z = sc.sym("mask_z", 8)
  output = _mapped_call(nested, 2, [(z, 0, 4), (z, 2, 4)]) if mapped else sc.concat([nested(*(z[:2], z[2:4])), nested(*(z[4:6], z[6:8]))])
  visits = Counter()
  original = analysis._jac_mask_uncached

  def counted(expr, wrt, memo):
    visits[expr.id, wrt.id] += 1
    return original(expr, wrt, memo)

  monkeypatch.setattr(analysis, "_jac_mask_uncached", counted)
  mask = analysis.jacobian_sparsity(output, z).to_mask()
  block = np.array([[True, True, True, True], [False, True, False, True]])
  expected = np.zeros((4, 8), dtype=bool)
  expected[:2, :4] = block
  expected[2:, 4:] = block
  np.testing.assert_array_equal(mask, expected)
  assert max(visits.values()) == 1
  values = np.arange(1.0, 9.0)

  @sc.function(sc.arg("z", 8), outputs=sc.arg("out", 4), name="nested_mask_values")
  def fn(actual: sc.Expr) -> sc.Expr:
    return (
      sc.vmap(nested, 2)(sc.window(actual, 0, 4), sc.window(actual, 2, 4)).vec()
      if mapped
      else sc.concat([nested(actual[:2], actual[2:4]), nested(actual[4:6], actual[6:8])])
    )

  reference = np.array([[a * d + c * b, b + d] for a, b, c, d in values.reshape(2, 4)])
  np.testing.assert_array_equal(fn(values), reference.reshape(-1))
